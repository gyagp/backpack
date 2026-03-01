"""
Qwen3.5-27B inference on WebGPU via Triton.

Hybrid Mamba-2 / Transformer architecture:
  - 64 layers total: 48 Mamba-2 SSM (linear attention) + 16 full GQA
  - Pattern: 3 linear_attention + 1 full_attention, repeating
  - Full attention: GQA with QK-norm, partial RoPE (25%), output gate
  - Linear attention: Mamba-2 SSM with conv1d, gated projections
  - RMSNorm, SwiGLU MLP, separate lm_head

Optimizations:
  - 4-bit weight quantization (INT4 per-group with fp16 scales)
  - fp16 storage for embeddings/norms
  - Per-layer weight streaming to save VRAM
  - Pre-allocated KV cache for full attention layers
  - Pre-computed RoPE cos/sin tables
  - Vectorized multi-head attention

Usage:
    python models/qwen-3.5/model.py --verify
    python models/qwen-3.5/model.py --quantize
    python models/qwen-3.5/model.py --prompt "Hello"

Requirements:
    pip install requests tokenizers
"""
import os
import sys
import time
from typing import Dict, Tuple, List

_t_script_start = time.perf_counter_ns()

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_SCRIPT_DIR))

import numpy as np

from common.model_base import WebGPUModel
from common.utils import (
    load_weights, download_weights, load_tokenizer, generate,
)

_t_imports_done = time.perf_counter_ns()


# ---------------------------------------------------------------------------
# 4-bit quantization utilities (reuse Phi4 pattern)
# ---------------------------------------------------------------------------

Q4_GROUP_SIZE = 128


def quantize_int4(weight: np.ndarray, group_size: int = Q4_GROUP_SIZE):
    """Quantize a 2D weight matrix to INT4 with per-group fp16 scales/zeros."""
    N, K = weight.shape
    K_pad = ((K + group_size - 1) // group_size) * group_size
    if K_pad != K:
        w = np.zeros((N, K_pad), dtype=np.float32)
        w[:, :K] = weight
    else:
        w = weight.copy()

    n_groups = K_pad // group_size
    w_grouped = w.reshape(N * n_groups, group_size)

    w_min = w_grouped.min(axis=1, keepdims=True)
    w_max = w_grouped.max(axis=1, keepdims=True)
    w_range = w_max - w_min
    w_range = np.where(w_range == 0, 1.0, w_range)

    scales = (w_range / 15.0).astype(np.float16)
    zeros = w_min.astype(np.float16)

    scales_f32 = scales.astype(np.float32)
    zeros_f32 = zeros.astype(np.float32)
    q = np.clip(np.round((w_grouped - zeros_f32) / scales_f32), 0, 15).astype(
        np.uint8)

    q_even = q[:, 0::2]
    q_odd = q[:, 1::2]
    q_packed = (q_odd << 4) | q_even

    return (q_packed.reshape(N, -1),
            scales.reshape(N, n_groups),
            zeros.reshape(N, n_groups))


def dequantize_int4(q_packed, scales, zeros, K_orig,
                    group_size=Q4_GROUP_SIZE, dtype=np.float32):
    """Dequantize INT4 packed weights."""
    N = q_packed.shape[0]
    q_low = (q_packed & 0x0F).astype(np.float32)
    q_high = (q_packed >> 4).astype(np.float32)
    K_pad_half = q_packed.shape[1]
    q = np.empty((N, K_pad_half * 2), dtype=np.float32)
    q[:, 0::2] = q_low
    q[:, 1::2] = q_high

    n_groups = scales.shape[1]
    q_grouped = q.reshape(N * n_groups, group_size)
    s = scales.reshape(N * n_groups, 1).astype(np.float32)
    z = zeros.reshape(N * n_groups, 1).astype(np.float32)
    w = q_grouped * s + z

    K_pad = n_groups * group_size
    w = w.reshape(N, K_pad)[:, :K_orig]
    if dtype != np.float32:
        w = w.astype(dtype)
    return w


# ---------------------------------------------------------------------------
# Architecture constants
# ---------------------------------------------------------------------------

# Layer type pattern: 3 linear + 1 full, repeating for 64 layers
LAYER_TYPES = []
for i in range(64):
    if (i + 1) % 4 == 0:  # layers 3,7,11,...,63 are full attention
        LAYER_TYPES.append("full_attention")
    else:
        LAYER_TYPES.append("linear_attention")

FULL_ATTN_LAYERS = [i for i, t in enumerate(LAYER_TYPES)
                     if t == "full_attention"]
LINEAR_ATTN_LAYERS = [i for i, t in enumerate(LAYER_TYPES)
                       if t == "linear_attention"]

QWEN35_CONFIG = {
    "n_layer": 64,
    "n_head": 24,
    "n_kv_heads": 4,
    "n_embd": 5120,
    "intermediate_size": 17408,
    "n_vocab": 248320,
    "rope_theta": 10000000.0,
    "rms_norm_eps": 1e-6,
    "head_dim": 256,
    "partial_rotary_factor": 0.25,
    "attn_output_gate": True,
    # Mamba-2 SSM parameters for linear attention layers
    "ssm_n_heads": 48,
    "ssm_head_dim": 128,
    "ssm_n_groups": 16,
    "ssm_d_state": 128,
    "ssm_conv_kernel": 4,
    "hf_repo": "Qwen/Qwen3.5-27B",
    # Q proj uses 2× heads (48 Q heads for double-Q with QK-norm)
    "n_q_heads": 48,
}


class Qwen35WebGPU(WebGPUModel):
    """Qwen3.5-27B inference on WebGPU via Triton kernels.

    Hybrid Mamba-2 / Transformer architecture with 64 layers:
      - 48 Mamba-2 SSM layers (linear attention)
      - 16 full GQA layers (every 4th layer)

    Full attention features:
      - QK-norm (RMSNorm on Q and K before attention)
      - Partial RoPE (25% of head_dim rotated)
      - Output gate (sigmoid gate on attention output)
      - GQA (24 Q heads, 4 KV heads)

    Linear attention (Mamba-2 SSM):
      - Fused QKV projection → conv1d → SSM scan
      - Gated output via in_proj_z + sigmoid
      - Causal recurrence via diagonal A matrix
    """

    MAX_SEQ_LEN = 2048

    def __init__(self, weights: Dict[str, np.ndarray],
                 n_layer: int = 64, n_head: int = 24,
                 n_kv_heads: int = 4, n_embd: int = 5120,
                 intermediate_size: int = 17408,
                 n_vocab: int = 248320,
                 rope_theta: float = 10000000.0,
                 rms_norm_eps: float = 1e-6,
                 head_dim: int = 256,
                 partial_rotary_factor: float = 0.25,
                 attn_output_gate: bool = True,
                 ssm_n_heads: int = 48,
                 ssm_head_dim: int = 128,
                 ssm_n_groups: int = 16,
                 ssm_d_state: int = 128,
                 ssm_conv_kernel: int = 4,
                 max_seq_len: int = 512,
                 quantized: bool = False,
                 n_q_heads: int = None):
        self.partial_rotary_factor = partial_rotary_factor
        self.rotary_dim = int(head_dim * partial_rotary_factor)
        self.attn_output_gate = attn_output_gate
        self.MAX_SEQ_LEN = max_seq_len
        self._quantized = quantized
        # Q heads can differ from output heads (e.g. 48 Q heads, 24 output heads)
        self.n_q_heads = n_q_heads if n_q_heads is not None else n_head

        # Mamba-2 SSM params
        self.ssm_n_heads = ssm_n_heads
        self.ssm_head_dim = ssm_head_dim
        self.ssm_n_groups = ssm_n_groups
        self.ssm_d_state = ssm_d_state
        self.ssm_conv_kernel = ssm_conv_kernel
        # SSM intermediate dimension = n_heads * head_dim
        self.ssm_dim = ssm_n_heads * ssm_head_dim  # 48 * 128 = 6144

        # GatedDeltaNet dimensions:
        #   num_v_heads = ssm_n_heads = 48
        #   num_k_heads = ssm_n_groups = 16
        #   head_k_dim = ssm_d_state = 128
        #   head_v_dim = ssm_head_dim = 128
        #   key_dim = num_k_heads * head_k_dim = 16 * 128 = 2048
        #   value_dim = num_v_heads * head_v_dim = 48 * 128 = 6144
        #   conv_dim = key_dim * 2 + value_dim = 10240
        # in_proj_qkv: (conv_dim=10240, hidden_size=5120) → Q(2048) + K(2048) + V(6144)
        self.gdn_n_v_heads = ssm_n_heads     # 48
        self.gdn_n_k_heads = ssm_n_groups    # 16
        self.gdn_head_k_dim = ssm_d_state    # 128
        self.gdn_head_v_dim = ssm_head_dim   # 128
        self.gdn_key_dim = ssm_n_groups * ssm_d_state  # 2048
        self.gdn_value_dim = ssm_n_heads * ssm_head_dim  # 6144
        self.gdn_v_per_k = ssm_n_heads // ssm_n_groups  # 3

        self.ssm_qkv_dim = self.gdn_key_dim * 2 + self.gdn_value_dim  # 10240

        # Full attention output dimension (o_proj: n_head × head_dim)
        self.full_attn_qo_dim = n_head * head_dim  # 24 * 256 = 6144
        # Q projection dimension (may differ: n_q_heads × head_dim)
        self.full_attn_q_dim = self.n_q_heads * head_dim  # 48 * 256 = 12288

        # Fused QKV output dimension for full attention
        self.full_attn_qkv_dim = (
            self.full_attn_q_dim + 2 * n_kv_heads * head_dim)

        # Fused gate_up output dimension for MLP
        self.gate_up_out = 2 * intermediate_size

        # Collect all K dimensions for kernel compilation
        k_dims = {
            n_embd,              # most projections: K=5120
            intermediate_size,   # down_proj: K=17408
            self.ssm_qkv_dim,    # ssm in_proj_qkv output: 10240
            self.ssm_dim,        # ssm out_proj, in_proj_z: K=6144
            self.full_attn_qo_dim,  # attn o_proj: K=6144
        }

        # Fuse QKV and gate_up weights before base class init
        self._fuse_qkv_weights(weights, n_layer, n_head, n_kv_heads,
                               head_dim, n_embd)
        self._fuse_gate_up_weights(weights, n_layer, intermediate_size,
                                   n_embd)

        super().__init__(
            weights, n_layer=n_layer, n_head=n_head, n_embd=n_embd,
            n_vocab=n_vocab,
            n_kv_heads=n_kv_heads,
            intermediate_size=intermediate_size,
            head_dim=head_dim,
            rope_theta=rope_theta,
            rms_norm_eps=rms_norm_eps,
            k_dimensions=k_dims,
        )
        self._precompute_rope_tables()
        self._upload_weights_to_gpu()
        self._init_ssm_state()
        self._init_gpu_kv_cache()

    @staticmethod
    def _fuse_qkv_weights(weights, n_layer, n_head, n_kv_heads,
                          head_dim, n_embd):
        """Fuse Q, K, V projection weights into single QKV matrix.

        Only for full-attention layers.  Reduces 3 GPU dispatches to 1.
        Handles both fp32 and INT4-quantized weight formats.
        """
        E = n_embd
        qo_dim = n_head * head_dim
        kv_dim = n_kv_heads * head_dim
        for i in range(n_layer):
            if LAYER_TYPES[i] != "full_attention":
                continue
            pfx = f"layers.{i}.self_attn."
            q_key = pfx + "q_proj.weight"
            k_key = pfx + "k_proj.weight"
            v_key = pfx + "v_proj.weight"
            fused_key = pfx + "qkv_proj.weight"

            if q_key in weights:
                # fp32 unfused path
                q_w = weights[q_key].reshape(qo_dim, E)
                k_w = weights[k_key].reshape(kv_dim, E)
                v_w = weights[v_key].reshape(kv_dim, E)
                weights[fused_key] = np.concatenate(
                    [q_w, k_w, v_w], axis=0)
                del weights[q_key], weights[k_key], weights[v_key]
            elif (q_key + ".q4") in weights:
                # INT4-quantized: dequant, fuse, re-quantize
                for suffix in (".q4", ".scales", ".zeros", ".K"):
                    parts = []
                    for src in (q_key, k_key, v_key):
                        parts.append(weights[src + suffix])
                        del weights[src + suffix]
                    if suffix == ".K":
                        # K values should all be the same (= n_embd)
                        weights[fused_key + suffix] = parts[0]
                    else:
                        weights[fused_key + suffix] = np.concatenate(
                            parts, axis=0)

    @staticmethod
    def _fuse_gate_up_weights(weights, n_layer, intermediate_size, n_embd):
        """Fuse gate_proj + up_proj into single gate_up matrix.

        Reduces 2 GPU dispatches to 1 per layer.
        Handles both fp32 and INT4-quantized weight formats.
        """
        E = n_embd
        IM = intermediate_size
        for i in range(n_layer):
            pfx = f"layers.{i}.mlp."
            g_key = pfx + "gate_proj.weight"
            u_key = pfx + "up_proj.weight"
            fused_key = pfx + "gate_up_proj.weight"

            if g_key in weights:
                g_w = weights[g_key].reshape(IM, E)
                u_w = weights[u_key].reshape(IM, E)
                weights[fused_key] = np.concatenate(
                    [g_w, u_w], axis=0)
                del weights[g_key], weights[u_key]
            elif (g_key + ".q4") in weights:
                for suffix in (".q4", ".scales", ".zeros", ".K"):
                    parts = []
                    for src in (g_key, u_key):
                        parts.append(weights[src + suffix])
                        del weights[src + suffix]
                    if suffix == ".K":
                        weights[fused_key + suffix] = parts[0]
                    else:
                        weights[fused_key + suffix] = np.concatenate(
                            parts, axis=0)

    def _compile_model_kernels(self):
        """Compile Qwen3.5-specific kernels."""
        self._compile_rms_norm()
        self._compile_silu_mul()
        self._compile_sigmoid()
        self._compile_gdn_kernel()

    def _compile_gdn_kernel(self):
        """Compile the GatedDeltaNet recurrence WGSL kernel."""
        from common.gdn_kernel import WGSL_GDN_KERNEL, GDN_BINDINGS, GDN_WORKGROUP_SIZE
        runner = self.cache.runner
        self._gdn_pipeline, self._gdn_bgl = runner.get_pipeline_info(
            WGSL_GDN_KERNEL, GDN_BINDINGS, [])
        self._gdn_bindings = GDN_BINDINGS

    def _proj(self, x, weight_name, bias_key, out_features,
              K=None, gpu_out=False):
        """Linear projection using best available kernel (Q4/fp16/fp32)."""
        from common.model_base import GPUBuffer
        # INT4 GPU path
        q4_key = weight_name + ".q4.gpu"
        if q4_key in self._gpu_weights:
            w_q4 = self._gpu_weights[q4_key]
            scales = self._gpu_weights[weight_name + ".scales.gpu"]
            zeros = self._gpu_weights[weight_name + ".zeros.gpu"]
            bias = self._gpu_weights[bias_key]
            N = out_features

            # Use wide kernel for large N (> MAX_DISPATCH_DIM)
            if N > self.MAX_DISPATCH_DIM:
                x_is_gpu = isinstance(x, GPUBuffer)
                if x_is_gpu:
                    T = x.shape[0] if x.shape else 1
                    if K is None:
                        K = x.shape[1] if (x.shape and len(x.shape) > 1) else self.n_embd
                else:
                    T = x.shape[0]
                    if K is None:
                        K = x.shape[1]
                n_groups = K // 128
                stride_w_q4 = K // 8
                grid_y = self.MAX_DISPATCH_DIM
                grid_x = (N + grid_y - 1) // grid_y
                out = self.cache.run(
                    self._linear_q4_wide_result, grid=(grid_x, grid_y),
                    buffers={
                        'X': x if x_is_gpu else x.ravel(),
                        'W_Q4': w_q4, 'Scales': scales, 'Zeros': zeros,
                        'Bias': bias,
                        'Y': np.zeros(N, dtype=np.float32),
                    },
                    scalars={'K': K, 'stride_x': K,
                             'stride_w_q4': stride_w_q4,
                             'n_groups': n_groups, 'N': N,
                             'grid_y': grid_y},
                    gpu_outputs={'Y'} if gpu_out else None)
                if gpu_out:
                    gpu_buf = out['Y']
                    gpu_buf.shape = (1, N)
                    return gpu_buf
                return out['Y'].reshape(1, N)

            return self._linear_q4(
                x, w_q4, scales, zeros, bias, N,
                K=K, gpu_out=gpu_out)
        # fp16 path
        fp16_name = weight_name + ".fp16"
        if fp16_name in self._gpu_weights and hasattr(self, '_linear_fp16w_result'):
            return self._linear_fp16w(
                x, self._gpu_weights[fp16_name],
                self._gpu_weights[bias_key], out_features,
                K=K, gpu_out=gpu_out)
        # fp32 fallback
        return self._linear(
            x, self._gpu_weights[weight_name],
            self._gpu_weights[bias_key], out_features,
            gpu_out=gpu_out)

    def _precompute_rope_tables(self):
        """Pre-compute partial RoPE cos/sin tables."""
        rotary_dim = self.rotary_dim  # 64 (25% of head_dim=256)
        half_rot = rotary_dim // 2
        inv_freq = 1.0 / (self.rope_theta ** (
            np.arange(0, rotary_dim, 2, dtype=np.float32) / rotary_dim))
        positions = np.arange(self.MAX_SEQ_LEN, dtype=np.float32)
        angles = positions[:, None] * inv_freq[None, :]
        self._rope_cos = np.cos(angles).astype(np.float32)
        self._rope_sin = np.sin(angles).astype(np.float32)

    def _apply_partial_rope_fast(self, x, positions, rotary_dim=None):
        """Apply partial RoPE using pre-computed tables."""
        if rotary_dim is None:
            rotary_dim = self.rotary_dim
        half_rot = rotary_dim // 2
        cos_v = self._rope_cos[positions][:, None, :]
        sin_v = self._rope_sin[positions][:, None, :]

        x1 = x[..., :half_rot]
        x2 = x[..., half_rot:rotary_dim]
        out = np.empty_like(x)
        out[..., :half_rot] = x1 * cos_v - x2 * sin_v
        out[..., half_rot:rotary_dim] = x2 * cos_v + x1 * sin_v
        out[..., rotary_dim:] = x[..., rotary_dim:]
        return out

    def _init_ssm_state(self):
        """Initialize Mamba-2 SSM recurrent state for each linear_attn layer."""
        self._ssm_states = {}
        self._ssm_conv_states = {}
        self._ssm_gpu_states = {}  # GPU SSM state buffers
        self._ssm_gpu_conv_states = {}  # GPU conv state buffers
        # Pre-cache fp32 conversion of SSM/attention weights used per decode
        self._fp32_cache = {}

        runner = self.cache.runner
        n_v = self.gdn_n_v_heads  # 48
        hk = self.gdn_head_k_dim  # 128
        hv = self.gdn_head_v_dim  # 128
        conv_dim = self.ssm_qkv_dim  # 10240
        conv_hist = self.ssm_conv_kernel - 1  # 3

        for i in range(self.n_layer):
            pfx = f"layers.{i}."
            if LAYER_TYPES[i] == "full_attention":
                for wn in ("self_attn.q_norm.weight", "self_attn.k_norm.weight"):
                    w = self.weights.get(pfx + wn)
                    if w is not None and w.dtype == np.float16:
                        self._fp32_cache[pfx + wn] = w.astype(np.float32)
            else:
                la = pfx + "linear_attn."

                # Allocate GPU SSM state: (n_v, hk, hv) fp32
                state_size = n_v * hk * hv
                self._ssm_gpu_states[i] = runner.upload_to_gpu(
                    np.zeros(state_size, dtype=np.float32), f"ssm_state_{i}")

                # Allocate GPU conv state: (conv_hist, conv_dim) fp32
                conv_size = conv_hist * conv_dim
                self._ssm_gpu_conv_states[i] = runner.upload_to_gpu(
                    np.zeros(conv_size, dtype=np.float32), f"ssm_conv_state_{i}")

                # Upload SSM weights to GPU
                for wn in ("A_log", "dt_bias", "norm.weight"):
                    w = self.weights.get(la + wn)
                    if w is not None:
                        w_fp32 = w.astype(np.float32) if w.dtype != np.float32 else w
                        self._fp32_cache[la + wn] = w_fp32
                        self._gpu_weights[la + wn] = runner.upload_to_gpu(
                            w_fp32.ravel(), la + wn)

                # Conv1d weight: reshape to (conv_dim, conv_kernel) and upload
                conv_w = self.weights.get(la + "conv1d.weight")
                if conv_w is not None:
                    cw = conv_w.astype(np.float32) if conv_w.dtype == np.float16 else conv_w
                    cw_reshaped = cw.reshape(conv_dim, self.ssm_conv_kernel)
                    self._fp32_cache[la + "conv1d.weight"] = cw
                    self._fp32_cache[la + "conv1d.weight.reshaped"] = cw_reshaped
                    self._gpu_weights[la + "conv1d.weight"] = runner.upload_to_gpu(
                        cw_reshaped.ravel(), la + "conv1d.weight")

                # a, b projection weights (small: 48×5120)
                for wn in ("in_proj_a.weight", "in_proj_b.weight"):
                    w = self.weights.get(la + wn)
                    if w is not None:
                        w_fp32 = w.astype(np.float32) if w.dtype == np.float16 else w
                        self._fp32_cache[la + wn] = w_fp32

    def _init_gpu_kv_cache(self):
        """Pre-allocate GPU KV cache for full-attention layers."""
        runner = self.cache.runner
        n_kv = self.n_kv_heads
        HD = self.head_dim
        buf_size = self.MAX_SEQ_LEN * n_kv * HD
        self._gpu_kv_cache = {}
        for layer in FULL_ATTN_LAYERS:
            k_buf = runner.upload_to_gpu(
                np.zeros(buf_size, dtype=np.float32), f"kv_cache_K_{layer}")
            v_buf = runner.upload_to_gpu(
                np.zeros(buf_size, dtype=np.float32), f"kv_cache_V_{layer}")
            self._gpu_kv_cache[layer] = (k_buf, v_buf, 0)

    def _upload_weights_to_gpu(self):
        """Upload all weights to GPU (INT4/fp16/fp32 depending on capabilities)."""
        E = self.n_embd
        HD = self.head_dim
        IM = self.intermediate_size
        ssm_dim = self.ssm_dim  # 6144

        # Qwen3.5 uses (1+weight) style RMSNorm — add 1.0 to applicable norm weights
        # so the standard RMSNorm kernel produces correct results.
        # Applies to: input_layernorm, post_attention_layernorm, final norm,
        #             q_norm, k_norm (all initialized near 0)
        # Does NOT apply to: linear_attn.norm (initialized near 1, standard style)
        for name in list(self.weights.keys()):
            if ('layernorm.weight' in name or name == 'norm.weight'
                    or 'q_norm.weight' in name or 'k_norm.weight' in name):
                w = self.weights[name]
                self.weights[name] = (w.astype(np.float32) + 1.0).astype(w.dtype)

        # Determine upload path
        use_q4_gpu = (self._quantized
                      and getattr(self, '_has_fp16_linear', False))
        self._use_q4_gpu = use_q4_gpu
        has_fp16 = (not use_q4_gpu
                    and getattr(self, '_has_fp16_linear', False))

        if self._quantized and not use_q4_gpu:
            # Pre-dequantize when GPU INT4 kernels unavailable
            self._dequantize_all_weights()

        def _ul(name, N, K):
            if use_q4_gpu and (name + ".q4") in self.weights:
                self._upload_q4_weight(name, N, K)
            elif has_fp16:
                self._upload_linear_weight_fp16(name, N, K)
            else:
                self._upload_linear_weight(name, N, K)

        for i in range(self.n_layer):
            pfx = f"layers.{i}."
            # RMSNorm weights (all layers)
            self._upload_norm_weight(pfx + "input_layernorm.weight")
            self._upload_norm_weight(pfx + "post_attention_layernorm.weight")

            if LAYER_TYPES[i] == "full_attention":
                # Fused QKV projection
                _ul(pfx + "self_attn.qkv_proj.weight",
                    self.full_attn_qkv_dim, E)
                # O projection
                _ul(pfx + "self_attn.o_proj.weight",
                    E, self.full_attn_qo_dim)
                # QK norm weights
                self._upload_norm_weight(pfx + "self_attn.q_norm.weight")
                self._upload_norm_weight(pfx + "self_attn.k_norm.weight")
            else:
                # Linear attention (Mamba-2)
                _ul(pfx + "linear_attn.in_proj_qkv.weight",
                    self.ssm_qkv_dim, E)
                _ul(pfx + "linear_attn.in_proj_z.weight",
                    ssm_dim, E)
                _ul(pfx + "linear_attn.out_proj.weight",
                    E, ssm_dim)
                self._upload_norm_weight(
                    pfx + "linear_attn.norm.weight")

            # Fused gate_up MLP projection
            _ul(pfx + "mlp.gate_up_proj.weight", self.gate_up_out, E)
            # Down projection
            _ul(pfx + "mlp.down_proj.weight", E, IM)

        # Final norm
        self._upload_norm_weight("norm.weight")

        # LM head: upload as INT4 if quantized (saves ~2.1 GB vs fp16)
        # INT4: ~300 MB instead of 2.4 GB fp16
        _ul("lm_head.weight", self.n_vocab, E)
        self._upload_zero_bias("zero_bias_V", self.n_vocab)

        # Zero biases
        self._upload_zero_bias("zero_bias_E", E)
        self._upload_zero_bias("zero_bias_GU", self.gate_up_out)
        self._upload_zero_bias("zero_bias_KV", self.kv_dim)
        self._upload_zero_bias("zero_bias_QKV", self.full_attn_qkv_dim)
        self._upload_zero_bias("zero_bias_QO", self.full_attn_qo_dim)
        self._upload_zero_bias("zero_bias_SSM_QKV", self.ssm_qkv_dim)
        self._upload_zero_bias("zero_bias_SSM", ssm_dim)

        self._print_gpu_weight_stats()

    def _dequantize_all_weights(self):
        """Pre-dequantize all INT4 weights at startup.

        Handles both fused (qkv_proj, gate_up_proj) and unfused weight names.
        """
        import gc
        print("  Pre-dequantizing INT4 weights...")
        for layer in range(self.n_layer):
            pfx = f"layers.{layer}."
            keys = [
                pfx + "mlp.gate_up_proj.weight",
                pfx + "mlp.down_proj.weight",
            ]
            if LAYER_TYPES[layer] == "full_attention":
                keys.append(pfx + "self_attn.qkv_proj.weight")
                keys.append(pfx + "self_attn.o_proj.weight")
            else:
                keys.extend([
                    pfx + "linear_attn.in_proj_qkv.weight",
                    pfx + "linear_attn.in_proj_z.weight",
                    pfx + "linear_attn.out_proj.weight",
                ])
            for name in keys:
                q4_key = name + ".q4"
                if q4_key not in self.weights:
                    continue
                K_orig = int(self.weights[name + ".K"][0])
                self.weights[name] = dequantize_int4(
                    self.weights[q4_key],
                    self.weights[name + ".scales"],
                    self.weights[name + ".zeros"],
                    K_orig)
                del self.weights[q4_key]
                del self.weights[name + ".scales"]
                del self.weights[name + ".zeros"]
                del self.weights[name + ".K"]
        gc.collect()
        self._quantized = False
        print("  Done.")

    # ------------------------------------------------------------------
    # Full attention block (for layers 3, 7, 11, ...)
    # ------------------------------------------------------------------

    def _rms_norm_cpu(self, x, weight):
        """CPU RMSNorm for a vector: used for QK norm."""
        rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + self.rms_norm_eps)
        return (x / rms * weight).astype(np.float32)

    def _attention_cpu_from_qkv(self, qkv, layer, T, use_cache, positions):
        """CPU attention from pre-projected QKV (GPU matmul already done).

        Returns numpy attention output (T, full_attn_qo_dim).
        """
        HD = self.head_dim
        n_q_heads = self.n_q_heads
        n_head = self.n_head
        n_kv = self.n_kv_heads
        n_rep = n_q_heads // n_kv
        pfx = f"layers.{layer}.self_attn."

        # Split QKV
        q_dim = self.full_attn_q_dim
        q = qkv[:, :q_dim]
        k = qkv[:, q_dim:q_dim + self.kv_dim]
        v = qkv[:, q_dim + self.kv_dim:]

        Q = q.reshape(T, n_q_heads, HD)
        K_new = k.reshape(T, n_kv, HD)
        V_new = v.reshape(T, n_kv, HD)

        # QK-norm
        q_norm_w = self.weights[pfx + "q_norm.weight"]
        k_norm_w = self.weights[pfx + "k_norm.weight"]
        q_rms = np.sqrt(np.mean(Q * Q, axis=-1, keepdims=True) + self.rms_norm_eps)
        Q = Q / q_rms * q_norm_w
        k_rms = np.sqrt(np.mean(K_new * K_new, axis=-1, keepdims=True) + self.rms_norm_eps)
        K_new = K_new / k_rms * k_norm_w

        # Partial RoPE
        if positions is None:
            positions = np.arange(T, dtype=np.int32)
        Q = self._apply_partial_rope_fast(Q, positions)
        K_new = self._apply_partial_rope_fast(K_new, positions)

        # KV cache
        if use_cache:
            if self.kv_cache is None:
                self.kv_cache = {}
            if layer not in self.kv_cache:
                K_buf = np.zeros((self.MAX_SEQ_LEN, n_kv, HD), dtype=np.float32)
                V_buf = np.zeros((self.MAX_SEQ_LEN, n_kv, HD), dtype=np.float32)
                K_buf[:T] = K_new; V_buf[:T] = V_new
                self.kv_cache[layer] = (K_buf, V_buf, T)
            else:
                K_buf, V_buf, cur_len = self.kv_cache[layer]
                K_buf[cur_len:cur_len + T] = K_new
                V_buf[cur_len:cur_len + T] = V_new
                self.kv_cache[layer] = (K_buf, V_buf, cur_len + T)
            _, _, T_total = self.kv_cache[layer]
            K_full = K_buf[:T_total]; V_full = V_buf[:T_total]
        else:
            T_total = T; K_full = K_new; V_full = V_new

        scale = 1.0 / np.sqrt(HD)
        if T == 1 and use_cache and T_total > 1:
            K_exp = np.repeat(K_full, n_rep, axis=1)
            V_exp = np.repeat(V_full, n_rep, axis=1)
            scores = np.float32(
                (Q.transpose(1, 0, 2) @ K_exp.transpose(1, 2, 0)).squeeze(1) * scale)
            scores -= scores.max(axis=1, keepdims=True)
            exp_s = np.exp(scores)
            s = exp_s.sum(axis=1, keepdims=True)
            s = np.where(s == 0, 1.0, s)
            attn = exp_s / s
            attn_out = (attn[:, None, :] @ V_exp.transpose(1, 0, 2)).squeeze(1)
            attn_out = attn_out[None, :, :].astype(np.float32)
        else:
            attn_out = self._causal_attention_multihead(Q, K_full, V_full, n_rep)

        # Output gate: 48 Q heads → 24 output heads
        if self.attn_output_gate and n_q_heads > n_head:
            attn_out = attn_out.reshape(T, n_head, 2 * HD)
            gate = 1.0 / (1.0 + np.exp(-attn_out[:, :, :HD]))
            attn_out = gate * attn_out[:, :, HD:]

        return attn_out.reshape(T, self.full_attn_qo_dim)

    def _attention_block(self, x, layer: int,
                         use_cache: bool = False,
                         positions: np.ndarray = None,
                         **kwargs):
        """Full GQA with QK-norm, partial RoPE, and output gate.

        For T=1 decode: uses GPU KV cache + GPU attention kernel.
        For T>1 prefill: uses CPU attention with numpy KV cache.
        """
        from common.model_base import GPUBuffer

        if isinstance(x, GPUBuffer):
            T = x.shape[0] if x.shape else 1
        else:
            T = x.shape[0]
        HD = self.head_dim
        n_q_heads = self.n_q_heads  # 48 (Q heads)
        n_head = self.n_head        # 24 (output heads)
        n_kv = self.n_kv_heads
        n_rep = n_q_heads // n_kv   # 48/4 = 12 (GQA expansion)
        pfx = f"layers.{layer}.self_attn."

        # Fused QKV projection (GPU)
        if self._profiling: self._set_gpu_op(f"L{layer}/qkv")
        qkv = self._proj(
            x, pfx + "qkv_proj.weight",
            "zero_bias_QKV", self.full_attn_qkv_dim,
            K=self.n_embd)

        # Split into Q, K, V
        q_dim = self.full_attn_q_dim
        q = qkv[:, :q_dim]
        k = qkv[:, q_dim:q_dim + self.kv_dim]
        v = qkv[:, q_dim + self.kv_dim:]

        Q = q.reshape(T, n_q_heads, HD)
        K_new = k.reshape(T, n_kv, HD)
        V_new = v.reshape(T, n_kv, HD)

        # QK-norm (CPU — per-head RMSNorm, lightweight)
        q_norm_w = self.weights[pfx + "q_norm.weight"]
        k_norm_w = self.weights[pfx + "k_norm.weight"]
        q_rms = np.sqrt(np.mean(Q * Q, axis=-1, keepdims=True) + self.rms_norm_eps)
        Q = Q / q_rms * q_norm_w
        k_rms = np.sqrt(np.mean(K_new * K_new, axis=-1, keepdims=True) + self.rms_norm_eps)
        K_new = K_new / k_rms * k_norm_w

        # Partial RoPE (CPU — 25% of head_dim)
        if positions is None:
            positions = np.arange(T, dtype=np.int32)
        Q = self._apply_partial_rope_fast(Q, positions)
        K_new = self._apply_partial_rope_fast(K_new, positions)

        # --- GPU attention path for T=1 decode ---
        if T == 1 and use_cache and layer in self._gpu_kv_cache:
            runner = self.cache.runner
            K_gpu, V_gpu, cur_len = self._gpu_kv_cache[layer]

            # Upload K_new, V_new to GPU KV cache at cur_len
            kv_offset = cur_len * n_kv * HD * 4  # byte offset
            k_bytes = K_new.ravel().astype(np.float32).tobytes()
            v_bytes = V_new.ravel().astype(np.float32).tobytes()
            import ctypes
            lib = runner._lib
            lib.wgpuQueueWriteBuffer(
                runner._queue, K_gpu.handle, kv_offset,
                ctypes.c_char_p(k_bytes), len(k_bytes))
            lib.wgpuQueueWriteBuffer(
                runner._queue, V_gpu.handle, kv_offset,
                ctypes.c_char_p(v_bytes), len(v_bytes))
            T_total = cur_len + 1
            self._gpu_kv_cache[layer] = (K_gpu, V_gpu, T_total)

            # Also update CPU KV cache (for prefill fallback)
            if self.kv_cache is None:
                self.kv_cache = {}
            if layer not in self.kv_cache:
                K_buf = np.zeros((self.MAX_SEQ_LEN, n_kv, HD), dtype=np.float32)
                V_buf = np.zeros((self.MAX_SEQ_LEN, n_kv, HD), dtype=np.float32)
                self.kv_cache[layer] = (K_buf, V_buf, 0)
            K_buf, V_buf, _ = self.kv_cache[layer]
            K_buf[cur_len] = K_new[0]
            V_buf[cur_len] = V_new[0]
            self.kv_cache[layer] = (K_buf, V_buf, T_total)

            # Upload Q to GPU
            BHD = self._attn_bhd  # block size for attention kernel
            Q_flat = Q.reshape(n_q_heads, HD)
            Q_padded = np.zeros((n_q_heads, BHD), dtype=np.float32)
            Q_padded[:, :HD] = Q_flat
            Q_gpu = runner.upload_to_gpu(Q_padded.ravel(), "__attn_q_tmp")

            # GPU GQA decode attention
            if self._profiling: self._set_gpu_op(f"L{layer}/attn_gpu")
            scale = float(1.0 / np.sqrt(HD))
            attn_result = self.cache.run(
                self._gqa_attn_result, grid=(n_q_heads,),
                buffers={
                    'Q': Q_gpu,
                    'K_cache': K_gpu, 'V_cache': V_gpu,
                    'Out': np.zeros(n_q_heads * BHD, dtype=np.float32),
                },
                scalars={
                    'kv_stride': n_kv * BHD,
                    'n_rep': n_rep,
                    'T_total': T_total,
                    'scale': scale,
                    'neg_inf': float(-1e9),
                })
            attn_out = attn_result['Out'].reshape(n_q_heads, BHD)[:, :HD]
            attn_out = attn_out[None, :, :]  # (1, n_q_heads, HD)
        else:
            # --- CPU attention path for prefill ---
            if use_cache:
                if self.kv_cache is None:
                    self.kv_cache = {}
                if layer not in self.kv_cache:
                    K_buf = np.zeros((self.MAX_SEQ_LEN, n_kv, HD), dtype=np.float32)
                    V_buf = np.zeros((self.MAX_SEQ_LEN, n_kv, HD), dtype=np.float32)
                    K_buf[:T] = K_new; V_buf[:T] = V_new
                    self.kv_cache[layer] = (K_buf, V_buf, T)
                    # Also upload to GPU KV cache
                    if layer in self._gpu_kv_cache:
                        K_gpu, V_gpu, _ = self._gpu_kv_cache[layer]
                        runner = self.cache.runner
                        k_bytes = K_new.ravel().astype(np.float32).tobytes()
                        v_bytes = V_new.ravel().astype(np.float32).tobytes()
                        import ctypes
                        lib = runner._lib
                        lib.wgpuQueueWriteBuffer(
                            runner._queue, K_gpu.handle, 0,
                            ctypes.c_char_p(k_bytes), len(k_bytes))
                        lib.wgpuQueueWriteBuffer(
                            runner._queue, V_gpu.handle, 0,
                            ctypes.c_char_p(v_bytes), len(v_bytes))
                        self._gpu_kv_cache[layer] = (K_gpu, V_gpu, T)
                else:
                    K_buf, V_buf, cur_len = self.kv_cache[layer]
                    K_buf[cur_len:cur_len + T] = K_new
                    V_buf[cur_len:cur_len + T] = V_new
                    self.kv_cache[layer] = (K_buf, V_buf, cur_len + T)
                _, _, T_total = self.kv_cache[layer]
                K_full = K_buf[:T_total]; V_full = V_buf[:T_total]
            else:
                T_total = T; K_full = K_new; V_full = V_new

            attn_out = self._causal_attention_multihead(Q, K_full, V_full, n_rep)

        # Output gate: 48 Q heads → 24 output heads
        if self.attn_output_gate and n_q_heads > n_head:
            attn_out = attn_out.reshape(T, n_head, 2 * HD)
            gate = 1.0 / (1.0 + np.exp(-attn_out[:, :, :HD]))
            attn_out = gate * attn_out[:, :, HD:]

        attn_flat = attn_out.reshape(T, self.full_attn_qo_dim)

        # O projection (GPU)
        if self._profiling: self._set_gpu_op(f"L{layer}/o_proj")
        o = self._proj(
            attn_flat, pfx + "o_proj.weight",
            "zero_bias_E", self.n_embd,
            K=self.full_attn_qo_dim)
        if self._profiling: self._clear_gpu_op()
        return o

    # ------------------------------------------------------------------
    # Linear attention (Mamba-2 SSM) block
    # ------------------------------------------------------------------

    def _ssm_from_projections(self, qkv_np, z_np, layer, T, use_cache):
        """CPU SSM recurrence from pre-projected QKV and Z (GPU matmul done).

        Returns numpy output (T, ssm_dim) ready for output projection.
        """
        pfx = f"layers.{layer}.linear_attn."
        n_v = self.gdn_n_v_heads
        n_k = self.gdn_n_k_heads
        hk = self.gdn_head_k_dim
        hv = self.gdn_head_v_dim
        key_dim = self.gdn_key_dim
        v_per_k = self.gdn_v_per_k
        kernel_size = self.ssm_conv_kernel

        # a, b projections (small CPU matmul: 5120 × 48)
        x_reconstruct = None  # not available, use conv output
        # We need the original x for a,b projections — but we only
        # have the projected QKV. The a,b weights are tiny (48×5120),
        # so we'll compute from the input x that was passed to the
        # transformer block. For now, use the _ssm_decode_cpu approach
        # which gets x from the caller.
        # Actually the _ssm_decode_cpu gets x_np as input and computes
        # a = x @ a_w.T. But here we don't have x_np.
        # We must pass x through. Let the caller pass it.

        # FALLBACK: delegate to _ssm_decode_cpu for correctness
        # The optimization here is that QKV + Z projections are
        # already done on GPU (batched). But _ssm_decode_cpu also
        # does them, so we save nothing. Instead, let's just use
        # the pre-projected values.

        # For T=1 decode, use the pre-projected Q/K/V directly
        qkv_flat = qkv_np[0]  # (ssm_qkv_dim,)
        z = z_np

        # a, b projections need original x — extract from conv state
        # Actually, the a,b projection input is x_np (the norm output),
        # not the qkv output. This is a fundamental issue — we need
        # access to the norm output for a,b. For now, store it.
        x_for_ab = getattr(self, '_ssm_x_for_ab', None)
        if x_for_ab is None:
            # Can't compute a,b without access to original x
            # Return zeros as placeholder
            return np.zeros((T, self.ssm_dim), dtype=np.float32)

        a_w = self._fp32_cache.get(pfx + "in_proj_a.weight",
                                   self.weights.get(pfx + "in_proj_a.weight"))
        b_w = self._fp32_cache.get(pfx + "in_proj_b.weight",
                                   self.weights.get(pfx + "in_proj_b.weight"))
        x_flat = x_for_ab.ravel().astype(np.float32)
        a_proj = x_flat @ a_w.T
        b_proj = x_flat @ b_w.T

        # Conv1d
        conv_dim = self.ssm_qkv_dim
        if layer not in self._ssm_conv_states:
            self._ssm_conv_states[layer] = np.zeros(
                (kernel_size - 1, conv_dim), dtype=np.float32)
        conv_state = self._ssm_conv_states[layer]
        conv_w = self._fp32_cache.get(pfx + "conv1d.weight.reshaped")
        if conv_w is None:
            conv_w_raw = self.weights[pfx + "conv1d.weight"]
            if conv_w_raw.dtype == np.float16:
                conv_w_raw = conv_w_raw.astype(np.float32)
            conv_w = conv_w_raw.reshape(conv_dim, kernel_size)
        x_conv = (conv_state * conv_w[:, :kernel_size - 1].T).sum(axis=0) + \
                 qkv_flat * conv_w[:, kernel_size - 1]
        conv_state[:-1] = conv_state[1:]
        conv_state[-1] = qkv_flat
        x_conv = x_conv * (1.0 / (1.0 + np.exp(-x_conv)))  # SiLU

        # Split Q, K, V
        q = x_conv[:key_dim].reshape(n_k, hk)
        k = x_conv[key_dim:key_dim*2].reshape(n_k, hk)
        v = x_conv[key_dim*2:].reshape(n_v, hv)

        # L2 normalize Q and K
        q = q / (np.sqrt(np.sum(q ** 2, axis=-1, keepdims=True)) + 1e-6)
        k = k / (np.sqrt(np.sum(k ** 2, axis=-1, keepdims=True)) + 1e-6)
        q = q * (1.0 / np.sqrt(float(hk)))
        q = np.repeat(q, v_per_k, axis=0)
        k = np.repeat(k, v_per_k, axis=0)

        # Decay and gate
        dt_bias = self._fp32_cache.get(pfx + "dt_bias", self.weights[pfx + "dt_bias"])
        A_log = self._fp32_cache.get(pfx + "A_log", self.weights[pfx + "A_log"])
        if A_log.dtype != np.float32:
            A_log = A_log.astype(np.float32)
        g = -np.exp(A_log) * np.log1p(np.exp(a_proj.astype(np.float32) + dt_bias))
        g_exp = np.exp(g)
        beta = 1.0 / (1.0 + np.exp(-b_proj.astype(np.float32)))

        # Recurrence
        if layer not in self._ssm_states:
            self._ssm_states[layer] = np.zeros((n_v, hk, hv), dtype=np.float32)
        state = self._ssm_states[layer]
        state *= g_exp[:, None, None]
        kv_mem = np.einsum('nhv,nh->nv', state, k)
        delta = (v - kv_mem) * beta[:, None]
        state += k[:, :, None] * delta[:, None, :]
        output = np.einsum('nhv,nh->nv', state, q)

        # Gated RMSNorm
        norm_w = self._fp32_cache.get(pfx + "norm.weight",
                                       self.weights[pfx + "norm.weight"])
        rms = np.sqrt(np.mean(output ** 2, axis=-1, keepdims=True) + self.rms_norm_eps)
        y_normed = output / rms * norm_w
        z_flat = z.ravel().reshape(n_v, hv)
        z_silu = z_flat * (1.0 / (1.0 + np.exp(-z_flat)))
        return (y_normed * z_silu).reshape(1, self.ssm_dim).astype(np.float32)
        """GatedDeltaNet T=1 decode with batched GPU projections.

        GatedDeltaNet recurrence (per head):
          g_t = exp(-exp(A_log) * softplus(a_t + dt_bias))   # decay
          beta_t = sigmoid(b_t)                                # gate
          state = state * g_t
          kv_mem = (state * k_t).sum(key_dim)
          delta = (v_t - kv_mem) * beta_t
          state = state + k_t.outer(delta)
          output = (state * q_t).sum(key_dim)
        """
        E = self.n_embd
        pfx = f"layers.{layer}.linear_attn."
        n_v = self.gdn_n_v_heads      # 48
        n_k = self.gdn_n_k_heads      # 16
        hk = self.gdn_head_k_dim      # 128
        hv = self.gdn_head_v_dim      # 128
        key_dim = self.gdn_key_dim    # 2048
        val_dim = self.gdn_value_dim  # 6144
        v_per_k = self.gdn_v_per_k    # 3
        kernel_size = self.ssm_conv_kernel

        # 1. Batch QKV + Z + a + b projections
        use_gpu = getattr(self, '_use_q4_gpu', False)
        if use_gpu:
            runner = self.cache.runner
            runner.begin_batch()
            if self._profiling: self._set_gpu_op(f"L{layer}/ssm_qkv")
            qkv_gpu = self._proj(x_np, pfx + "in_proj_qkv.weight",
                                  "zero_bias_SSM_QKV", self.ssm_qkv_dim,
                                  K=E, gpu_out=True)
            if self._profiling: self._set_gpu_op(f"L{layer}/ssm_z")
            z_gpu = self._proj(x_np, pfx + "in_proj_z.weight",
                                "zero_bias_SSM", self.ssm_dim,
                                K=E, gpu_out=True)
            if self._profiling: self._clear_gpu_op()
            rb = runner.end_batch(readback_buffers=[qkv_gpu, z_gpu])
            qkv = rb[id(qkv_gpu)].reshape(1, self.ssm_qkv_dim)
            z = rb[id(z_gpu)].reshape(1, self.ssm_dim)
        else:
            qkv = self._cpu_matmul(x_np, pfx + "in_proj_qkv.weight",
                                   self.ssm_qkv_dim, E)
            z = self._cpu_matmul(x_np, pfx + "in_proj_z.weight",
                                 self.ssm_dim, E)

        # Weights are stored in contiguous Q|K|V order
        qkv_flat = qkv[0]  # (10240,)

        # 2. Compute a, b projections (for decay and gate)
        a_w = self._fp32_cache.get(pfx + "in_proj_a.weight",
                                   self.weights.get(pfx + "in_proj_a.weight"))
        b_w = self._fp32_cache.get(pfx + "in_proj_b.weight",
                                   self.weights.get(pfx + "in_proj_b.weight"))
        x_flat = x_np.ravel().astype(np.float32)
        a_proj = x_flat @ a_w.T  # (n_v,) = (48,)
        b_proj = x_flat @ b_w.T  # (n_v,) = (48,)

        # 3. Conv1d with cached state
        conv_dim = self.ssm_qkv_dim
        if layer not in self._ssm_conv_states:
            self._ssm_conv_states[layer] = np.zeros(
                (kernel_size - 1, conv_dim), dtype=np.float32)
        conv_state = self._ssm_conv_states[layer]
        conv_w = self._fp32_cache.get(pfx + "conv1d.weight.reshaped")
        if conv_w is None:
            conv_w_raw = self.weights[pfx + "conv1d.weight"]
            if conv_w_raw.dtype == np.float16:
                conv_w_raw = conv_w_raw.astype(np.float32)
            conv_w = conv_w_raw.reshape(conv_dim, kernel_size)
        x_conv = (conv_state * conv_w[:, :kernel_size - 1].T).sum(axis=0) + \
                 qkv_flat * conv_w[:, kernel_size - 1]
        conv_state[:-1] = conv_state[1:]
        conv_state[-1] = qkv_flat

        # Apply SiLU to conv output (reference: activation="silu" in conv1d)
        x_conv = x_conv * (1.0 / (1.0 + np.exp(-x_conv)))

        # 4. Split Q, K, V (contiguous layout)
        q = x_conv[:key_dim].reshape(n_k, hk)              # (16, 128)
        k = x_conv[key_dim:key_dim*2].reshape(n_k, hk)     # (16, 128)
        v = x_conv[key_dim*2:].reshape(n_v, hv)             # (48, 128)

        # L2 normalize Q and K
        q = q / (np.sqrt(np.sum(q ** 2, axis=-1, keepdims=True)) + 1e-6)
        k = k / (np.sqrt(np.sum(k ** 2, axis=-1, keepdims=True)) + 1e-6)

        # Scale Q
        scale = 1.0 / np.sqrt(float(hk))
        q = q * scale

        # Expand K heads to match V heads (GQA-style: 16 → 48)
        q = np.repeat(q, v_per_k, axis=0)  # (48, 128)
        k = np.repeat(k, v_per_k, axis=0)  # (48, 128)

        # 5. Compute decay (g) and gate (beta)
        dt_bias = self._fp32_cache.get(pfx + "dt_bias", self.weights[pfx + "dt_bias"])
        A_log = self._fp32_cache.get(pfx + "A_log", self.weights[pfx + "A_log"])
        if A_log.dtype != np.float32:
            A_log = A_log.astype(np.float32)

        # g = -exp(A_log) * softplus(a + dt_bias)
        g = -np.exp(A_log) * np.log1p(np.exp(a_proj.astype(np.float32) + dt_bias))
        g_exp = np.exp(g)  # decay factor per head (48,)
        beta = 1.0 / (1.0 + np.exp(-b_proj.astype(np.float32)))  # sigmoid (48,)

        # 6. GatedDeltaNet recurrence
        # State: (n_v, hk, hv) = (48, 128, 128)
        if layer not in self._ssm_states:
            self._ssm_states[layer] = np.zeros(
                (n_v, hk, hv), dtype=np.float32)
        state = self._ssm_states[layer]

        # Decay
        state *= g_exp[:, None, None]  # (48, 128, 128) * (48, 1, 1)

        # Read memory: kv_mem = (state * k).sum(key_dim)
        kv_mem = np.einsum('nhv,nh->nv', state, k)  # (48, 128)

        # Delta update
        delta = (v - kv_mem) * beta[:, None]  # (48, 128)

        # Write: state += k.outer(delta)
        state += k[:, :, None] * delta[:, None, :]  # (48, 128, 128)

        # Output: (state * q).sum(key_dim)
        output = np.einsum('nhv,nh->nv', state, q)  # (48, 128)

        # 7. Norm the output (per-head RMSNorm with gated activation)
        norm_w = self._fp32_cache.get(pfx + "norm.weight",
                                       self.weights[pfx + "norm.weight"])
        # Gated RMSNorm: norm(output) * silu(z)
        rms = np.sqrt(np.mean(output ** 2, axis=-1, keepdims=True) + self.rms_norm_eps)
        y_normed = (output / rms * norm_w)  # (48, 128)

        # Gate with SiLU(z) — reference uses RMSNormGated which is norm(x)*silu(z)
        z_flat = z.ravel()  # (6144,)
        z_reshaped = z_flat.reshape(n_v, hv)
        z_silu = z_reshaped * (1.0 / (1.0 + np.exp(-z_reshaped)))
        y_gated = (y_normed * z_silu).reshape(1, self.ssm_dim).astype(np.float32)

        # 8. Output projection
        if use_gpu:
            if self._profiling: self._set_gpu_op(f"L{layer}/ssm_out")
            out = self._proj(y_gated, pfx + "out_proj.weight",
                              "zero_bias_E", E, K=self.ssm_dim)
            if self._profiling: self._clear_gpu_op()
        else:
            out = self._cpu_matmul(y_gated, pfx + "out_proj.weight",
                                   E, self.ssm_dim)
        return out

    def _ssm_gpu_decode(self, x, layer):
        """GPU-resident GatedDeltaNet decode (T=1, no CPU readbacks).

        All operations run on GPU:
        1. QKV + Z + a + b projections (GPU Q4 matmul)
        2. GDN recurrence kernel (conv1d + recurrence + gated norm)
        3. Output projection (GPU Q4 matmul)
        """
        from common.model_base import GPUBuffer
        from common.gdn_kernel import GDN_BINDINGS, pack_gdn_params
        import struct

        E = self.n_embd
        pfx = f"layers.{layer}.linear_attn."
        runner = self.cache.runner

        # 1. GPU projections: QKV, Z, a, b (batched)
        runner.begin_batch()
        if self._profiling: self._set_gpu_op(f"L{layer}/ssm_qkv")
        qkv_gpu = self._proj(
            x, pfx + "in_proj_qkv.weight",
            "zero_bias_SSM_QKV", self.ssm_qkv_dim,
            K=E, gpu_out=True)
        if self._profiling: self._set_gpu_op(f"L{layer}/ssm_z")
        z_gpu = self._proj(
            x, pfx + "in_proj_z.weight",
            "zero_bias_SSM", self.ssm_dim,
            K=E, gpu_out=True)
        if self._profiling: self._clear_gpu_op()
        runner.end_batch()

        # a, b projections: small (5120→48) — use GPU Q4 if available,
        # else readback x for CPU matmul
        a_key = pfx + "in_proj_a.weight"
        b_key = pfx + "in_proj_b.weight"
        if (a_key + ".q4.gpu") in self._gpu_weights:
            a_gpu = self._proj(x, a_key, "zero_bias_SSM_A",
                               self.gdn_n_v_heads, K=E, gpu_out=True)
            b_gpu = self._proj(x, b_key, "zero_bias_SSM_B",
                               self.gdn_n_v_heads, K=E, gpu_out=True)
        else:
            # CPU path for a,b (tiny weights, not quantized)
            if isinstance(x, GPUBuffer):
                x_np = runner.readback(x).reshape(1, E)
            else:
                x_np = x.reshape(1, E)
            a_w = self._fp32_cache.get(a_key, self.weights[a_key])
            b_w = self._fp32_cache.get(b_key, self.weights[b_key])
            a_np = (x_np.astype(np.float32) @ a_w.astype(np.float32).T).ravel()
            b_np = (x_np.astype(np.float32) @ b_w.astype(np.float32).T).ravel()
            a_gpu = runner.upload_to_gpu(a_np.astype(np.float32), "__ssm_a_proj")
            b_gpu = runner.upload_to_gpu(b_np.astype(np.float32), "__ssm_b_proj")

        # 2. GDN recurrence kernel (GPU)
        if self._profiling: self._set_gpu_op(f"L{layer}/ssm_gdn")

        # Zero biases for a,b if not already created
        if "zero_bias_SSM_A" not in self._gpu_weights:
            self._upload_zero_bias("zero_bias_SSM_A", self.gdn_n_v_heads)
            self._upload_zero_bias("zero_bias_SSM_B", self.gdn_n_v_heads)

        # Pack epsilon as f32 bits
        eps_bits = struct.unpack('<I', struct.pack('<f', self.rms_norm_eps))[0]
        params = pack_gdn_params(eps_bits)
        params_gpu = runner.upload_to_gpu(params, "__gdn_params")

        ssm_out_gpu = runner.upload_to_gpu(
            np.zeros(self.ssm_dim, dtype=np.float32), "__gdn_output")

        # Dispatch GDN kernel using pre-compiled pipeline
        from common.gdn_kernel import WGSL_GDN_KERNEL
        runner.run_kernel(
            wgsl_code=WGSL_GDN_KERNEL,
            buffer_bindings=GDN_BINDINGS,
            param_fields=[],
            workgroup_size=128,
            grid=(self.gdn_n_v_heads,),  # 48 workgroups
            buffers={
                'QKV': qkv_gpu,
                'Z': z_gpu,
                'A_proj': a_gpu,
                'B_proj': b_gpu,
                'ConvState': self._ssm_gpu_conv_states[layer],
                'ConvWeight': self._gpu_weights[pfx + "conv1d.weight"],
                'SSMState': self._ssm_gpu_states[layer],
                'A_log': self._gpu_weights[pfx + "A_log"],
                'DT_bias': self._gpu_weights[pfx + "dt_bias"],
                'NormWeight': self._gpu_weights[pfx + "norm.weight"],
                'Output': ssm_out_gpu,
                '_params_': params_gpu,
            },
            gpu_outputs={'Output', 'SSMState', 'ConvState'})
        ssm_out_gpu.shape = (1, self.ssm_dim)

        # 3. Output projection (GPU)
        if self._profiling: self._set_gpu_op(f"L{layer}/ssm_out")
        out = self._proj(
            ssm_out_gpu, pfx + "out_proj.weight",
            "zero_bias_E", E, K=self.ssm_dim)
        if self._profiling: self._clear_gpu_op()
        return out

    def _linear_attention_block(self, x, layer: int,
                                use_cache: bool = False, **kwargs):
        """GatedDeltaNet linear attention block.

        For T=1 decode: dispatches GPU GDN recurrence kernel (no readbacks).
        For T>1 prefill: uses CPU sequential scan.
        """
        from common.model_base import GPUBuffer

        if isinstance(x, GPUBuffer):
            T = x.shape[0] if x.shape else 1
        else:
            T = x.shape[0]

        E = self.n_embd
        pfx = f"layers.{layer}.linear_attn."
        n_v = self.gdn_n_v_heads
        n_k = self.gdn_n_k_heads
        hk = self.gdn_head_k_dim
        hv = self.gdn_head_v_dim
        key_dim = self.gdn_key_dim
        val_dim = self.gdn_value_dim
        v_per_k = self.gdn_v_per_k

        # --- T=1 GPU decode path: all on GPU, no readbacks ---
        if T == 1 and use_cache and layer in self._ssm_gpu_states:
            return self._ssm_gpu_decode(x, layer)

        # --- T>1 prefill path: CPU sequential scan ---

        # 1. QKV projection (weights stored in contiguous Q|K|V order)
        if self._profiling: self._set_gpu_op(f"L{layer}/ssm_qkv")
        qkv = self._proj(
            x, pfx + "in_proj_qkv.weight",
            "zero_bias_SSM_QKV", self.ssm_qkv_dim,
            K=E)

        # 2. Compute a, b projections from original input
        if isinstance(x, GPUBuffer):
            x_for_ab = self.cache.runner.readback(x).reshape(T, E)
        else:
            x_for_ab = x.reshape(T, E) if x.ndim != 2 else x
        a_w = self.weights.get(pfx + "in_proj_a.weight")
        b_w = self.weights.get(pfx + "in_proj_b.weight")
        if a_w is not None and a_w.dtype == np.float16:
            a_w = a_w.astype(np.float32)
        if b_w is not None and b_w.dtype == np.float16:
            b_w = b_w.astype(np.float32)
        a_proj = x_for_ab.astype(np.float32) @ a_w.T  # (T, 48)
        b_proj = x_for_ab.astype(np.float32) @ b_w.T  # (T, 48)

        # 3. Conv1d
        conv_w = self.weights[pfx + "conv1d.weight"]
        if conv_w.dtype == np.float16:
            conv_w = conv_w.astype(np.float32)
        kernel_size = self.ssm_conv_kernel
        conv_dim = self.ssm_qkv_dim

        if use_cache:
            if layer not in self._ssm_conv_states:
                self._ssm_conv_states[layer] = np.zeros(
                    (kernel_size - 1, conv_dim), dtype=np.float32)
            conv_state = self._ssm_conv_states[layer]
            x_conv_in = np.concatenate([conv_state, qkv], axis=0)
            self._ssm_conv_states[layer] = x_conv_in[-(kernel_size - 1):].copy()
        else:
            x_conv_in = np.concatenate([
                np.zeros((kernel_size - 1, conv_dim), dtype=np.float32),
                qkv], axis=0)

        conv_w_sq = conv_w.reshape(conv_dim, kernel_size)
        from numpy.lib.stride_tricks import as_strided
        strides = x_conv_in.strides
        windows = as_strided(
            x_conv_in,
            shape=(T, kernel_size, conv_dim),
            strides=(strides[0], strides[0], strides[1]))
        x_conv = np.einsum('tkc,ck->tc', windows, conv_w_sq)

        # Apply SiLU to conv output
        x_conv = x_conv * (1.0 / (1.0 + np.exp(-x_conv)))

        # 4. Split Q, K, V (CONTIGUOUS layout after de-interleaving)
        q = x_conv[:, :key_dim].reshape(T, n_k, hk)              # (T, 16, 128)
        k = x_conv[:, key_dim:key_dim*2].reshape(T, n_k, hk)     # (T, 16, 128)
        v = x_conv[:, key_dim*2:].reshape(T, n_v, hv)             # (T, 48, 128)

        # L2 normalize Q and K
        q = q / (np.sqrt(np.sum(q ** 2, axis=-1, keepdims=True)) + 1e-6)
        k = k / (np.sqrt(np.sum(k ** 2, axis=-1, keepdims=True)) + 1e-6)
        scale = 1.0 / np.sqrt(float(hk))
        q = q * scale

        # Expand K heads to match V heads (GQA: 16 → 48)
        q = np.repeat(q, v_per_k, axis=1)  # (T, 48, 128)
        k = np.repeat(k, v_per_k, axis=1)  # (T, 48, 128)

        # 5. Compute g (decay) and beta (gate)
        A_log = self.weights[pfx + "A_log"]
        if A_log.dtype != np.float32:
            A_log = A_log.astype(np.float32)
        dt_bias = self.weights[pfx + "dt_bias"]
        if dt_bias.dtype != np.float32:
            dt_bias = dt_bias.astype(np.float32)

        # g = -exp(A_log) * softplus(a + dt_bias); shape (T, 48)
        g = -np.exp(A_log)[None, :] * np.log1p(np.exp(a_proj + dt_bias[None, :]))
        g_exp = np.exp(g)  # (T, 48)
        beta = 1.0 / (1.0 + np.exp(-b_proj))  # sigmoid (T, 48)

        # 6. GatedDeltaNet sequential scan
        if use_cache:
            if layer not in self._ssm_states:
                self._ssm_states[layer] = np.zeros(
                    (n_v, hk, hv), dtype=np.float32)
            state = self._ssm_states[layer]
        else:
            state = np.zeros((n_v, hk, hv), dtype=np.float32)

        output = np.empty((T, n_v, hv), dtype=np.float32)
        for t in range(T):
            # Decay
            state = state * g_exp[t, :, None, None]
            # Read memory
            kv_mem = np.einsum('nhv,nh->nv', state, k[t])  # (48, 128)
            # Delta update
            delta = (v[t] - kv_mem) * beta[t, :, None]  # (48, 128)
            # Write
            state = state + k[t, :, :, None] * delta[:, None, :]  # (48, 128, 128)
            # Output
            output[t] = np.einsum('nhv,nh->nv', state, q[t])

        if use_cache:
            self._ssm_states[layer] = state

        # 7. Gated RMSNorm with Z
        norm_w = self.weights[pfx + "norm.weight"]
        if norm_w.dtype == np.float16:
            norm_w = norm_w.astype(np.float32)

        rms = np.sqrt(np.mean(output ** 2, axis=-1, keepdims=True) + self.rms_norm_eps)
        y_normed = output / rms * norm_w  # (T, 48, 128)

        # Z gate with SiLU
        if self._profiling: self._set_gpu_op(f"L{layer}/ssm_z")
        z = self._proj(
            x, pfx + "in_proj_z.weight",
            "zero_bias_SSM", self.ssm_dim,
            K=E)
        z_reshaped = z.reshape(T, n_v, hv)
        z_silu = z_reshaped * (1.0 / (1.0 + np.exp(-z_reshaped)))
        y_gated = (y_normed * z_silu).reshape(T, self.ssm_dim).astype(np.float32)

        # 8. Output projection
        if self._profiling: self._set_gpu_op(f"L{layer}/ssm_out")
        out = self._proj(
            y_gated, pfx + "out_proj.weight",
            "zero_bias_E", self.n_embd,
            K=self.ssm_dim)
        if self._profiling: self._clear_gpu_op()

        return out

    # ------------------------------------------------------------------
    # MLP block (shared by all layers)
    # ------------------------------------------------------------------

    def _mlp_block(self, x, layer: int, gpu_out: bool = False):
        """SwiGLU MLP: fused gate_up → silu_mul → down_proj."""
        E = self.n_embd
        IM = self.intermediate_size
        pfx = f"layers.{layer}.mlp."

        if self._profiling: self._set_gpu_op(f"L{layer}/gate_up")
        gate_up = self._proj(
            x, pfx + "gate_up_proj.weight",
            "zero_bias_GU", self.gate_up_out,
            K=E, gpu_out=True)
        if self._profiling: self._set_gpu_op(f"L{layer}/silu_mul")
        h = self._silu_mul_fused(gate_up, IM, gpu_out=True)
        if self._profiling: self._set_gpu_op(f"L{layer}/down")
        out = self._proj(
            h, pfx + "down_proj.weight",
            "zero_bias_E", E,
            K=IM, gpu_out=gpu_out)
        if self._profiling: self._clear_gpu_op()
        return out

    # ------------------------------------------------------------------
    # Transformer block dispatcher
    # ------------------------------------------------------------------

    def _cpu_matmul(self, x, weight_name, N, K=None):
        """CPU matmul for T=1 decode.

        Handles both fp32 weights and INT4 quantized weights (on-the-fly
        dequantization for Q4 — caches result for reuse within the same
        decode step).
        """
        if weight_name in self.weights:
            W = self.weights[weight_name]
            if K is None:
                K = W.shape[1] if W.ndim == 2 else W.size // N
            W = W.reshape(N, K)
            return np.float32(x @ W.T)
        # Q4 path: dequantize on-the-fly (cached per layer)
        q4_key = weight_name + ".q4"
        if q4_key in self.weights:
            cache_key = "_cpu_deq_" + weight_name
            if not hasattr(self, cache_key):
                K_orig = int(self.weights[weight_name + ".K"][0])
                W = dequantize_int4(
                    self.weights[q4_key],
                    self.weights[weight_name + ".scales"],
                    self.weights[weight_name + ".zeros"],
                    K_orig)
                setattr(self, cache_key, W)
            W = getattr(self, cache_key)
            if K is None:
                K = W.shape[1]
            return np.float32(x @ W.reshape(N, K).T)
        raise KeyError(f"Weight not found: {weight_name}")

    def _rms_norm_vec(self, x, w_name):
        """CPU RMSNorm for 2D array."""
        w = self.weights[w_name]
        rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + self.rms_norm_eps)
        return (x / rms * w).astype(np.float32)

    def _attn_decode_cpu(self, x_np, layer, positions):
        """CPU-only full-attention decode for T=1 (GPU Q4 for projections)."""
        HD = self.head_dim
        n_q_heads = self.n_q_heads
        n_head = self.n_head
        n_kv = self.n_kv_heads
        n_rep = n_q_heads // n_kv
        pfx = f"layers.{layer}.self_attn."

        # QKV projection (GPU Q4 or CPU)
        if getattr(self, '_use_q4_gpu', False):
            if self._profiling: self._set_gpu_op(f"L{layer}/qkv")
            qkv = self._proj(x_np, pfx + "qkv_proj.weight",
                              "zero_bias_QKV", self.full_attn_qkv_dim,
                              K=self.n_embd)
            if self._profiling: self._clear_gpu_op()
        else:
            qkv = self._cpu_matmul(x_np, pfx + "qkv_proj.weight",
                                    self.full_attn_qkv_dim, self.n_embd)
        q_dim = self.full_attn_q_dim
        q = qkv[:, :q_dim]
        k = qkv[:, q_dim:q_dim + self.kv_dim]
        v = qkv[:, q_dim + self.kv_dim:]

        Q = q.reshape(1, n_q_heads, HD)
        K_new = k.reshape(1, n_kv, HD)
        V_new = v.reshape(1, n_kv, HD)

        # QK-norm
        q_norm_w = self._fp32_cache.get(pfx + "q_norm.weight",
                                        self.weights[pfx + "q_norm.weight"])
        k_norm_w = self._fp32_cache.get(pfx + "k_norm.weight",
                                        self.weights[pfx + "k_norm.weight"])
        q_rms = np.sqrt(np.mean(Q * Q, axis=-1, keepdims=True) + self.rms_norm_eps)
        Q = Q / q_rms * q_norm_w
        k_rms = np.sqrt(np.mean(K_new * K_new, axis=-1, keepdims=True) + self.rms_norm_eps)
        K_new = K_new / k_rms * k_norm_w

        # Partial RoPE
        Q = self._apply_partial_rope_fast(Q, positions)
        K_new = self._apply_partial_rope_fast(K_new, positions)

        # KV cache write-in-place
        if self.kv_cache is None:
            self.kv_cache = {}
        if layer not in self.kv_cache:
            K_buf = np.zeros((self.MAX_SEQ_LEN, n_kv, HD), dtype=np.float32)
            V_buf = np.zeros((self.MAX_SEQ_LEN, n_kv, HD), dtype=np.float32)
            K_buf[0] = K_new[0]
            V_buf[0] = V_new[0]
            self.kv_cache[layer] = (K_buf, V_buf, 1)
        else:
            K_buf, V_buf, cur_len = self.kv_cache[layer]
            K_buf[cur_len] = K_new[0]
            V_buf[cur_len] = V_new[0]
            self.kv_cache[layer] = (K_buf, V_buf, cur_len + 1)

        _, _, T_total = self.kv_cache[layer]

        # Vectorized attention with n_q_heads
        scale = 1.0 / np.sqrt(HD)
        K_exp = np.repeat(K_buf[:T_total], n_rep, axis=1)
        V_exp = np.repeat(V_buf[:T_total], n_rep, axis=1)
        Q_t = Q.transpose(1, 0, 2)
        K_t = K_exp.transpose(1, 2, 0)
        scores = np.float32((Q_t @ K_t).squeeze(1) * scale)
        scores -= scores.max(axis=1, keepdims=True)
        exp_s = np.exp(scores)
        attn = exp_s / exp_s.sum(axis=1, keepdims=True)
        V_t = V_exp.transpose(1, 0, 2)
        attn_out = (attn[:, None, :] @ V_t).squeeze(1)  # (n_q_heads, HD)

        # Output gate: 48 Q heads → 24 output heads via sigmoid gating
        if self.attn_output_gate and n_q_heads > n_head:
            attn_out = attn_out.reshape(n_head, 2 * HD)
            gate = 1.0 / (1.0 + np.exp(-attn_out[:, :HD]))
            attn_out = gate * attn_out[:, HD:]

        attn_flat = attn_out.reshape(1, self.full_attn_qo_dim)
        if getattr(self, '_use_q4_gpu', False):
            if self._profiling: self._set_gpu_op(f"L{layer}/o_proj")
            out = self._proj(attn_flat, pfx + "o_proj.weight",
                              "zero_bias_E", self.n_embd,
                              K=self.full_attn_qo_dim)
            if self._profiling: self._clear_gpu_op()
            return out
        return self._cpu_matmul(attn_flat, pfx + "o_proj.weight",
                                self.n_embd, self.full_attn_qo_dim)

    def _decode_cpu(self, x, layer, positions):
        """CPU-only decode path for T=1 (avoids GPU dispatch overhead)."""
        from common.model_base import GPUBuffer
        if isinstance(x, GPUBuffer):
            x_np = self.cache.runner.readback(x).reshape(1, self.n_embd)
        else:
            x_np = x

        pfx = f"layers.{layer}."
        E = self.n_embd
        IM = self.intermediate_size

        # RMSNorm 1
        if self._profiling: self._begin_cpu(f"L{layer}/rms_norm")
        rn1 = self._rms_norm_vec(x_np, pfx + "input_layernorm.weight")
        if self._profiling: self._end_cpu(f"L{layer}/rms_norm")

        # Attention (full CPU path for both types)
        if LAYER_TYPES[layer] == "full_attention":
            if self._profiling: self._begin_cpu(f"L{layer}/attn")
            attn = self._attn_decode_cpu(rn1, layer, positions)
            if self._profiling: self._end_cpu(f"L{layer}/attn")
        else:
            if self._profiling: self._begin_cpu(f"L{layer}/ssm")
            attn = self._ssm_decode_cpu(rn1, layer)
            if self._profiling: self._end_cpu(f"L{layer}/ssm")

        x_np = x_np + attn

        # RMSNorm 2
        if self._profiling: self._begin_cpu(f"L{layer}/rms_norm")
        rn2 = self._rms_norm_vec(x_np, pfx + "post_attention_layernorm.weight")
        if self._profiling: self._end_cpu(f"L{layer}/rms_norm")

        # MLP — use GPU Q4 if available (avoids OOM from CPU dequant)
        E = self.n_embd
        IM = self.intermediate_size
        if getattr(self, '_use_q4_gpu', False):
            if self._profiling: self._begin_cpu(f"L{layer}/mlp_gpu")
            runner = self.cache.runner
            runner.begin_batch()
            if self._profiling: self._set_gpu_op(f"L{layer}/gate_up")
            gate_up = self._proj(
                rn2, pfx + "mlp.gate_up_proj.weight",
                "zero_bias_GU", self.gate_up_out, K=E, gpu_out=True)
            if self._profiling: self._set_gpu_op(f"L{layer}/silu_mul")
            h = self._silu_mul_fused(gate_up, IM, gpu_out=True)
            if self._profiling: self._set_gpu_op(f"L{layer}/down")
            mlp_gpu = self._proj(
                h, pfx + "mlp.down_proj.weight",
                "zero_bias_E", E, K=IM, gpu_out=True)
            if self._profiling: self._clear_gpu_op()
            rb = runner.end_batch(readback_buffers=[mlp_gpu])
            mlp_out = rb[id(mlp_gpu)].reshape(1, E)
            if self._profiling: self._end_cpu(f"L{layer}/mlp_gpu")
        else:
            if self._profiling: self._begin_cpu(f"L{layer}/mlp_cpu")
            gate_up = self._cpu_matmul(
                rn2, pfx + "mlp.gate_up_proj.weight", self.gate_up_out, E)
            gate = gate_up[:, :IM]
            up = gate_up[:, IM:]
            h = gate / (1.0 + np.exp(-gate)) * up
            mlp_out = self._cpu_matmul(
                h, pfx + "mlp.down_proj.weight", E, IM)
            if self._profiling: self._end_cpu(f"L{layer}/mlp_cpu")

        return x_np + mlp_out

    def _transformer_block_prefill(self, x, layer, use_cache, positions):
        """Prefill path (T>1): uses original attention/SSM blocks."""
        from common.model_base import GPUBuffer
        pfx = f"layers.{layer}."

        if self._profiling: self._set_gpu_op(f"L{layer}/norm1")
        rn1 = self._rms_norm(
            x, self._gpu_weights[pfx + "input_layernorm.weight"],
            gpu_out=True)

        if LAYER_TYPES[layer] == "full_attention":
            attn = self._attention_block(rn1, layer, use_cache=use_cache,
                                         positions=positions)
        else:
            attn = self._linear_attention_block(rn1, layer,
                                                use_cache=use_cache)

        use_fused = (hasattr(self, '_add_rn_result') and
                     isinstance(x, GPUBuffer) and isinstance(attn, GPUBuffer))
        if use_fused:
            if self._profiling: self._set_gpu_op(f"L{layer}/res1+norm2")
            rn2 = self._add_rms_norm(
                x, attn,
                self._gpu_weights[pfx + "post_attention_layernorm.weight"],
                gpu_out=True)
        else:
            if self._profiling: self._set_gpu_op(f"L{layer}/res1")
            x = self._add(x, attn, gpu_out=True)
            if self._profiling: self._set_gpu_op(f"L{layer}/norm2")
            rn2 = self._rms_norm(
                x, self._gpu_weights[pfx + "post_attention_layernorm.weight"],
                gpu_out=True)

        mlp = self._mlp_block(rn2, layer, gpu_out=True)
        if self._profiling: self._set_gpu_op(f"L{layer}/res2")
        x = self._add(x, mlp, gpu_out=True)
        if self._profiling: self._clear_gpu_op()
        return x

    def _transformer_block(self, x, layer: int,
                           use_cache: bool = False,
                           positions: np.ndarray = None, **kwargs):
        """Pre-norm transformer block. Dispatches to full or linear attn."""
        from common.model_base import GPUBuffer
        pfx = f"layers.{layer}."

        T = (x.shape[0] if x.shape else 1) if isinstance(x, GPUBuffer) else x.shape[0]

        # All ops on GPU — use the same path for prefill and decode.
        # Matmuls run on GPU, attention/SSM readback to CPU only where needed.
        return self._transformer_block_prefill(x, layer, use_cache, positions)

        # ---------- Phase 1: Norm + QKV projection (GPU, batched) ----------
        if use_batch:
            runner.begin_batch()

        if self._profiling: self._set_gpu_op(f"L{layer}/norm1")
        rn1 = self._rms_norm(
            x, self._gpu_weights[pfx + "input_layernorm.weight"],
            gpu_out=True)

        # QKV/SSM projection stays on GPU
        if LAYER_TYPES[layer] == "full_attention":
            if self._profiling: self._set_gpu_op(f"L{layer}/qkv")
            qkv_gpu = self._proj(
                rn1, pfx + "self_attn.qkv_proj.weight",
                "zero_bias_QKV", self.full_attn_qkv_dim,
                K=self.n_embd, gpu_out=True)
            if use_batch:
                # Readback QKV for CPU attention
                rb = runner.end_batch(readback_buffers=[qkv_gpu])
                qkv = rb[id(qkv_gpu)].reshape(T, self.full_attn_qkv_dim)
            else:
                qkv = qkv_gpu if isinstance(qkv_gpu, np.ndarray) else \
                    self.cache.runner.readback(qkv_gpu).reshape(T, self.full_attn_qkv_dim)
            # CPU attention (QK-norm, RoPE, dot product, output gate)
            if self._profiling: self._set_gpu_op(f"L{layer}/attn")
            attn = self._attention_cpu_from_qkv(qkv, layer, T, use_cache, positions)
        else:
            if self._profiling: self._set_gpu_op(f"L{layer}/ssm_qkv")
            qkv_gpu = self._proj(
                rn1, pfx + "linear_attn.in_proj_qkv.weight",
                "zero_bias_SSM_QKV", self.ssm_qkv_dim,
                K=self.n_embd, gpu_out=True)
            if self._profiling: self._set_gpu_op(f"L{layer}/ssm_z")
            z_gpu = self._proj(
                rn1, pfx + "linear_attn.in_proj_z.weight",
                "zero_bias_SSM", self.ssm_dim,
                K=self.n_embd, gpu_out=True)
            if use_batch:
                rb = runner.end_batch(readback_buffers=[qkv_gpu, z_gpu])
                qkv_np = rb[id(qkv_gpu)].reshape(T, self.ssm_qkv_dim)
                z_np = rb[id(z_gpu)].reshape(T, self.ssm_dim)
            else:
                qkv_np = qkv_gpu if isinstance(qkv_gpu, np.ndarray) else \
                    self.cache.runner.readback(qkv_gpu).reshape(T, self.ssm_qkv_dim)
                z_np = z_gpu if isinstance(z_gpu, np.ndarray) else \
                    self.cache.runner.readback(z_gpu).reshape(T, self.ssm_dim)
            # Save norm output for SSM a,b projections (small CPU matmul)
            if isinstance(rn1, GPUBuffer):
                self._ssm_x_for_ab = self.cache.runner.readback(rn1).reshape(T, self.n_embd)
            else:
                self._ssm_x_for_ab = rn1
            # CPU SSM recurrence
            if self._profiling: self._set_gpu_op(f"L{layer}/ssm")
            attn = self._ssm_from_projections(qkv_np, z_np, layer, T, use_cache)

        # ---------- Phase 2: O proj + residual + norm2 + MLP (GPU, batched) ----------
        if use_batch:
            runner.begin_batch()

        # O projection (full attention) or out projection (SSM)
        if LAYER_TYPES[layer] == "full_attention":
            if self._profiling: self._set_gpu_op(f"L{layer}/o_proj")
            o = self._proj(
                attn, pfx + "self_attn.o_proj.weight",
                "zero_bias_E", self.n_embd,
                K=self.full_attn_qo_dim, gpu_out=True)
        else:
            if self._profiling: self._set_gpu_op(f"L{layer}/ssm_out")
            o = self._proj(
                attn, pfx + "linear_attn.out_proj.weight",
                "zero_bias_E", self.n_embd,
                K=self.ssm_dim, gpu_out=True)

        # Residual + Norm2 + MLP (all on GPU)
        if hasattr(self, '_add_rn_result') and isinstance(x, GPUBuffer):
            if self._profiling: self._set_gpu_op(f"L{layer}/res1+norm2")
            rn2 = self._add_rms_norm(
                x, o,
                self._gpu_weights[pfx + "post_attention_layernorm.weight"],
                gpu_out=True)
        else:
            if self._profiling: self._set_gpu_op(f"L{layer}/res1")
            x = self._add(x, o, gpu_out=True)
            if self._profiling: self._set_gpu_op(f"L{layer}/norm2")
            rn2 = self._rms_norm(
                x, self._gpu_weights[pfx + "post_attention_layernorm.weight"],
                gpu_out=True)

        mlp = self._mlp_block(rn2, layer, gpu_out=True)
        if self._profiling: self._set_gpu_op(f"L{layer}/res2")
        x = self._add(x, mlp, gpu_out=True)
        if self._profiling: self._clear_gpu_op()

        if use_batch and runner.is_batching:
            runner.end_batch()

        return x

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, token_ids: np.ndarray,
                use_cache: bool = False,
                pos_offset: int = 0) -> np.ndarray:
        """Run Qwen3.5-27B forward pass."""
        T = len(token_ids)
        wte = self.weights["embed_tokens.weight"]
        x = wte[token_ids].astype(np.float32)
        positions = np.arange(pos_offset, pos_offset + T, dtype=np.int32)

        for layer in range(self.n_layer):
            x = self._transformer_block(x, layer, use_cache=use_cache,
                                        positions=positions)

        if self._profiling: self._set_gpu_op("final_norm")
        x = self._rms_norm(x, self._gpu_weights["norm.weight"])
        if self._profiling: self._clear_gpu_op()

        # LM head on GPU (INT4 quantized, ~300 MB)
        if T > 1 and use_cache:
            x = x[-1:]  # only last token logits needed for prefill
        if self._profiling: self._set_gpu_op("lm_head")
        logits = self._proj(
            x, "lm_head.weight", "zero_bias_V", self.n_vocab,
            K=self.n_embd)
        if self._profiling: self._clear_gpu_op()
        return logits


# ---------------------------------------------------------------------------
# Weight downloading and quantization
# ---------------------------------------------------------------------------

def quantize_model_weights(weights: Dict[str, np.ndarray],
                           n_layer: int = 64) -> Dict[str, np.ndarray]:
    """Quantize Qwen3.5 weights: INT4 for linear, fp16 for others."""
    quantized = {}
    linear_keys = set()
    for i in range(n_layer):
        pfx = f"layers.{i}."
        # MLP weights (all layers)
        linear_keys.update([
            pfx + "mlp.gate_proj.weight",
            pfx + "mlp.up_proj.weight",
            pfx + "mlp.down_proj.weight",
        ])
        if LAYER_TYPES[i] == "full_attention":
            linear_keys.update([
                pfx + "self_attn.q_proj.weight",
                pfx + "self_attn.k_proj.weight",
                pfx + "self_attn.v_proj.weight",
                pfx + "self_attn.o_proj.weight",
            ])
        else:
            linear_keys.update([
                pfx + "linear_attn.in_proj_qkv.weight",
                pfx + "linear_attn.in_proj_z.weight",
                pfx + "linear_attn.out_proj.weight",
            ])

    # Also quantize lm_head to save 2.4 GB VRAM (248K×5120 fp16 → ~300 MB INT4)
    linear_keys.add("lm_head.weight")

    total_orig = 0
    total_quant = 0

    for key, val in weights.items():
        orig_bytes = val.nbytes
        total_orig += orig_bytes

        if key in linear_keys and val.ndim == 2:
            w = val.astype(np.float32)
            N, K = w.shape
            q_packed, scales, zeros = quantize_int4(w)
            quantized[key + ".q4"] = q_packed
            quantized[key + ".scales"] = scales
            quantized[key + ".zeros"] = zeros
            quantized[key + ".K"] = np.array([K], dtype=np.int32)
            q_bytes = q_packed.nbytes + scales.nbytes + zeros.nbytes
            total_quant += q_bytes
        else:
            quantized[key] = val.astype(np.float16) if val.dtype in (
                np.float32, np.float64) else val
            total_quant += quantized[key].nbytes

    print(f"  Original: {total_orig / 1024**3:.1f} GB")
    print(f"  Quantized: {total_quant / 1024**3:.1f} GB "
          f"(INT4 linear + fp16 other)")
    print(f"  Compression: {total_orig / max(total_quant, 1):.1f}x")
    return quantized


def download_qwen35_weights(model_dir: str = None) -> Tuple[str, str]:
    """Download Qwen3.5-27B weights and tokenizer from HuggingFace."""
    config = QWEN35_CONFIG
    hf_repo = config["hf_repo"]
    if model_dir is None:
        model_dir = os.path.join(_SCRIPT_DIR, "..", "..", "gitignore", "models", os.path.basename(_SCRIPT_DIR), "weights")

    def qwen35_key_transform(key, arr):
        new_key = key
        if new_key.startswith("model.language_model."):
            new_key = new_key[len("model.language_model."):]
        elif new_key.startswith("model."):
            if "visual" in new_key or "merger" in new_key:
                return None
            new_key = new_key[len("model."):]
        if new_key.startswith("mtp."):
            return None
        return new_key, arr

    shard_files = [f"model.safetensors-{i:05d}-of-00011.safetensors"
                   for i in range(1, 12)]

    npz_path, tokenizer_path = download_weights(
        hf_repo=hf_repo,
        model_dir=model_dir,
        safetensors_files=shard_files,
        key_transform=qwen35_key_transform,
        download_tokenizer=True,
    )
    return npz_path, tokenizer_path


def download_and_quantize_streaming(weights_dir: str = None):
    """Download Qwen3.5-27B shards and quantize in streaming fashion.

    Processes one shard at a time to avoid OOM on 55GB+ models.
    Each shard's tensors are extracted, transformed, quantized (if linear),
    and accumulated. Memory per shard: ~5GB peak (1 shard BF16 + converted).
    """
    import requests, gc
    from common.utils import _parse_safetensors

    config = QWEN35_CONFIG
    hf_repo = config["hf_repo"]
    if weights_dir is None:
        weights_dir = os.path.join(_SCRIPT_DIR, "..", "..", "gitignore", "models", os.path.basename(_SCRIPT_DIR), "weights")
    os.makedirs(weights_dir, exist_ok=True)

    q4_path = os.path.join(weights_dir, "weights_q4.npz")
    if os.path.exists(q4_path):
        print(f"Quantized weights already exist: {q4_path}")
        return q4_path

    base_url = f"https://huggingface.co/{hf_repo}/resolve/main"

    # Auth
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get(
        "HUGGING_FACE_HUB_TOKEN")
    if not hf_token:
        token_file = os.path.join(
            os.path.expanduser("~"), ".cache", "huggingface", "token")
        if os.path.exists(token_file):
            with open(token_file) as f:
                hf_token = f.read().strip()
    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}

    # Download tokenizer
    tok_path = os.path.join(weights_dir, "tokenizer.json")
    if not os.path.exists(tok_path):
        print("Downloading tokenizer...")
        resp = requests.get(f"{base_url}/tokenizer.json", headers=headers)
        resp.raise_for_status()
        with open(tok_path, 'wb') as f:
            f.write(resp.content)

    shard_files = [f"model.safetensors-{i:05d}-of-00011.safetensors"
                   for i in range(1, 12)]

    def key_transform(key, arr):
        new_key = key
        if new_key.startswith("model.language_model."):
            new_key = new_key[len("model.language_model."):]
        elif new_key.startswith("model."):
            if "visual" in new_key or "merger" in new_key:
                return None
            new_key = new_key[len("model."):]
        if new_key.startswith("mtp."):
            return None
        return new_key, arr

    # Determine which keys are linear weights (for INT4 quantization)
    linear_keys = set()
    for i in range(config["n_layer"]):
        pfx = f"layers.{i}."
        linear_keys.update([
            pfx + "mlp.gate_proj.weight",
            pfx + "mlp.up_proj.weight",
            pfx + "mlp.down_proj.weight",
        ])
        if LAYER_TYPES[i] == "full_attention":
            linear_keys.update([
                pfx + "self_attn.q_proj.weight",
                pfx + "self_attn.k_proj.weight",
                pfx + "self_attn.v_proj.weight",
                pfx + "self_attn.o_proj.weight",
            ])
        else:
            linear_keys.update([
                pfx + "linear_attn.in_proj_qkv.weight",
                pfx + "linear_attn.in_proj_z.weight",
                pfx + "linear_attn.out_proj.weight",
            ])

    # Also quantize lm_head to save 2.4 GB VRAM
    linear_keys.add("lm_head.weight")

    all_quantized = {}
    total_orig = 0
    total_quant = 0

    for shard_idx, sf in enumerate(shard_files):
        sf_path = os.path.join(weights_dir, sf)

        # Download shard if not cached
        if not os.path.exists(sf_path):
            st_url = f"{base_url}/{sf}"
            print(f"Downloading shard {shard_idx+1}/11: {sf}...")
            resp = requests.get(st_url, headers=headers, stream=True)
            resp.raise_for_status()
            total = int(resp.headers.get('content-length', 0))
            downloaded = 0
            with open(sf_path, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=8192 * 1024):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded * 100 // total
                        print(f"\r  Progress: {pct}% "
                              f"({downloaded // (1024*1024)}MB / "
                              f"{total // (1024*1024)}MB)",
                              end="", flush=True)
            print()
        else:
            print(f"Shard {shard_idx+1}/11: {sf} (cached)")

        # Parse this shard
        print(f"  Parsing {sf}...")
        shard_weights = _parse_safetensors(sf_path)

        # Transform + quantize each tensor in this shard
        for key, arr in shard_weights.items():
            val = arr.astype(np.float32)
            result = key_transform(key, val)
            if result is None:
                continue
            new_key, val = result

            orig_bytes = val.nbytes
            total_orig += orig_bytes

            if new_key in linear_keys and val.ndim == 2:
                N, K = val.shape
                # De-interleave SSM QKV weights to contiguous Q|K|V layout
                # before quantization (better per-group accuracy)
                if 'linear_attn.in_proj_qkv.weight' in new_key:
                    n_k_h = config["ssm_n_groups"]    # 16
                    hk_d = config["ssm_d_state"]      # 128
                    v_pk = config["ssm_n_heads"] // n_k_h  # 3
                    pg = hk_d + hk_d + v_pk * config["ssm_head_dim"]  # 640
                    g = val.reshape(n_k_h, pg, K)
                    val = np.concatenate([
                        g[:, :hk_d, :].reshape(-1, K),          # Q
                        g[:, hk_d:2*hk_d, :].reshape(-1, K),    # K
                        g[:, 2*hk_d:, :].reshape(-1, K),         # V
                    ], axis=0)
                q_packed, scales, zeros = quantize_int4(val)
                all_quantized[new_key + ".q4"] = q_packed
                all_quantized[new_key + ".scales"] = scales
                all_quantized[new_key + ".zeros"] = zeros
                all_quantized[new_key + ".K"] = np.array([K], dtype=np.int32)
                q_bytes = q_packed.nbytes + scales.nbytes + zeros.nbytes
                total_quant += q_bytes
            else:
                fp16 = val.astype(np.float16) if val.dtype in (
                    np.float32, np.float64) else val
                all_quantized[new_key] = fp16
                total_quant += fp16.nbytes

        # Free shard memory
        del shard_weights
        gc.collect()
        print(f"  Shard {shard_idx+1}/11 done. "
              f"Accumulated {len(all_quantized)} tensors "
              f"({total_quant / 1024**3:.1f} GB quantized)")

    print(f"\nQuantization complete:")
    print(f"  Original: {total_orig / 1024**3:.1f} GB")
    print(f"  Quantized: {total_quant / 1024**3:.1f} GB")
    print(f"  Compression: {total_orig / max(total_quant, 1):.1f}x")

    print(f"Saving to {q4_path}...")
    np.savez(q4_path, **all_quantized)
    print(f"Done! File: {os.path.getsize(q4_path) / 1024**3:.1f} GB")
    return q4_path


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_with_random_weights():
    """Verify Qwen3.5 pipeline with small random weights."""
    print("=" * 60)
    print("Qwen3.5 WebGPU Pipeline Verification (random weights)")
    print("=" * 60)

    # Small test config: 4 layers (3 linear + 1 full attn)
    n_layer = 4
    n_head, n_kv_heads = 4, 2
    n_embd, intermediate_size, n_vocab = 64, 128, 256
    head_dim = n_embd // n_head  # 16
    kv_dim = n_kv_heads * head_dim
    n_rep = n_head // n_kv_heads
    rope_theta = 10000.0
    eps = 1e-6
    partial_rotary_factor = 0.25
    rotary_dim = int(head_dim * partial_rotary_factor)

    # SSM params (scaled down)
    ssm_n_heads = 4
    ssm_head_dim = 8
    ssm_n_groups = 2
    ssm_d_state = 8
    ssm_conv_kernel = 4
    ssm_dim = ssm_n_heads * ssm_head_dim  # 32
    # GDN: key_dim = n_groups * d_state = 16, value_dim = n_heads * head_dim = 32
    # conv_dim = key_dim*2 + value_dim = 64
    gdn_key_dim = ssm_n_groups * ssm_d_state
    gdn_value_dim = ssm_n_heads * ssm_head_dim
    ssm_qkv_dim = gdn_key_dim * 2 + gdn_value_dim

    np.random.seed(42)
    weights = {}
    weights["embed_tokens.weight"] = np.random.randn(
        n_vocab, n_embd).astype(np.float32) * 0.02
    weights["lm_head.weight"] = np.random.randn(
        n_vocab, n_embd).astype(np.float32) * 0.02
    weights["norm.weight"] = np.ones(n_embd, dtype=np.float32)

    for i in range(n_layer):
        pfx = f"layers.{i}."
        weights[pfx + "input_layernorm.weight"] = np.ones(
            n_embd, dtype=np.float32)
        weights[pfx + "post_attention_layernorm.weight"] = np.ones(
            n_embd, dtype=np.float32)
        # MLP
        weights[pfx + "mlp.gate_proj.weight"] = np.random.randn(
            intermediate_size, n_embd).astype(np.float32) * 0.02
        weights[pfx + "mlp.up_proj.weight"] = np.random.randn(
            intermediate_size, n_embd).astype(np.float32) * 0.02
        weights[pfx + "mlp.down_proj.weight"] = np.random.randn(
            n_embd, intermediate_size).astype(np.float32) * 0.02

        # Local layer type: same pattern as full model
        if (i + 1) % 4 == 0:
            # Full attention layer
            qo_dim = n_head * head_dim
            weights[pfx + "self_attn.q_proj.weight"] = np.random.randn(
                qo_dim, n_embd).astype(np.float32) * 0.02
            weights[pfx + "self_attn.k_proj.weight"] = np.random.randn(
                kv_dim, n_embd).astype(np.float32) * 0.02
            weights[pfx + "self_attn.v_proj.weight"] = np.random.randn(
                kv_dim, n_embd).astype(np.float32) * 0.02
            weights[pfx + "self_attn.o_proj.weight"] = np.random.randn(
                n_embd, qo_dim).astype(np.float32) * 0.02
            weights[pfx + "self_attn.q_norm.weight"] = np.ones(
                head_dim, dtype=np.float32)
            weights[pfx + "self_attn.k_norm.weight"] = np.ones(
                head_dim, dtype=np.float32)
        else:
            # GatedDeltaNet linear attention layer
            weights[pfx + "linear_attn.in_proj_qkv.weight"] = np.random.randn(
                ssm_qkv_dim, n_embd).astype(np.float32) * 0.02
            weights[pfx + "linear_attn.in_proj_z.weight"] = np.random.randn(
                ssm_dim, n_embd).astype(np.float32) * 0.02
            weights[pfx + "linear_attn.out_proj.weight"] = np.random.randn(
                n_embd, ssm_dim).astype(np.float32) * 0.02
            weights[pfx + "linear_attn.conv1d.weight"] = np.random.randn(
                ssm_qkv_dim, 1, ssm_conv_kernel).astype(np.float32) * 0.02
            weights[pfx + "linear_attn.norm.weight"] = np.ones(
                ssm_head_dim, dtype=np.float32)
            weights[pfx + "linear_attn.A_log"] = np.random.randn(
                ssm_n_heads).astype(np.float32) * 0.1
            weights[pfx + "linear_attn.dt_bias"] = np.zeros(
                ssm_n_heads, dtype=np.float32)
            weights[pfx + "linear_attn.in_proj_a.weight"] = np.random.randn(
                ssm_n_heads, n_embd).astype(np.float32) * 0.02
            weights[pfx + "linear_attn.in_proj_b.weight"] = np.random.randn(
                ssm_n_heads, n_embd).astype(np.float32) * 0.02

    print(f"\nModel: {n_layer} layers (3 SSM + 1 full attn), "
          f"{n_head} Q heads, {n_kv_heads} KV heads, "
          f"{n_embd} embd, {n_vocab} vocab")
    print(f"SSM: {ssm_n_heads} heads, {ssm_head_dim} head_dim, "
          f"{ssm_n_groups} groups, {ssm_d_state} d_state")

    model = Qwen35WebGPU(
        weights, n_layer=n_layer, n_head=n_head, n_kv_heads=n_kv_heads,
        n_embd=n_embd, intermediate_size=intermediate_size,
        n_vocab=n_vocab, rope_theta=rope_theta, rms_norm_eps=eps,
        head_dim=head_dim, partial_rotary_factor=partial_rotary_factor,
        attn_output_gate=False,  # Disable for simple verification
        ssm_n_heads=ssm_n_heads, ssm_head_dim=ssm_head_dim,
        ssm_n_groups=ssm_n_groups, ssm_d_state=ssm_d_state,
        ssm_conv_kernel=ssm_conv_kernel)

    # Forward pass
    token_ids = np.array([1, 42, 100, 200], dtype=np.int32)
    T = len(token_ids)
    t0 = time.time()
    logits = model.forward(token_ids)
    t1 = time.time()

    print(f"\nForward pass: {token_ids} → shape {logits.shape} "
          f"in {(t1-t0)*1000:.0f}ms")
    print(f"Logits range: [{logits.min():.4f}, {logits.max():.4f}]")
    print(f"Predictions: {logits.argmax(axis=1)}")

    # Basic sanity checks
    ok = True
    if logits.shape != (T, n_vocab):
        print(f"FAIL: expected shape ({T}, {n_vocab}), got {logits.shape}")
        ok = False
    if np.any(np.isnan(logits)):
        print("FAIL: NaN in logits")
        ok = False
    if np.any(np.isinf(logits)):
        print("FAIL: Inf in logits")
        ok = False

    # Test with KV cache (autoregressive decode)
    print("\nTesting KV cache (autoregressive)...")
    model.kv_cache = None
    model._ssm_states = {}
    model._ssm_conv_states = {}

    # Prefill
    logits_prefill = model.forward(
        token_ids[:3], use_cache=True, pos_offset=0)
    # Decode one token
    logits_decode = model.forward(
        token_ids[3:4], use_cache=True, pos_offset=3)

    print(f"Prefill: shape {logits_prefill.shape}")
    print(f"Decode:  shape {logits_decode.shape}")

    if logits_decode.shape != (1, n_vocab):
        print(f"FAIL: decode shape should be (1, {n_vocab})")
        ok = False
    if np.any(np.isnan(logits_decode)):
        print("FAIL: NaN in decode logits")
        ok = False

    if ok:
        print("\nAll checks PASSED!")
    else:
        print("\nSome checks FAILED!")

    return ok


def bench_with_random_weights():
    """Benchmark Qwen3.5 with scaled-up random weights for profiling."""
    print("=" * 60)
    print("Qwen3.5 WebGPU Benchmark (random weights)")
    print("=" * 60)

    # Larger config for meaningful benchmarks — 8 layers
    n_layer = 8
    n_head, n_kv_heads = 8, 2
    n_embd, intermediate_size, n_vocab = 256, 512, 1024
    head_dim = n_embd // n_head  # 32
    kv_dim = n_kv_heads * head_dim

    ssm_n_heads = 8
    ssm_head_dim = 16
    ssm_n_groups = 4
    ssm_d_state = 16
    ssm_conv_kernel = 4
    ssm_dim = ssm_n_heads * ssm_head_dim
    ssm_qkv_dim = (ssm_n_groups * ssm_d_state) * 2 + ssm_dim

    np.random.seed(42)
    weights = {}
    weights["embed_tokens.weight"] = np.random.randn(
        n_vocab, n_embd).astype(np.float32) * 0.02
    weights["lm_head.weight"] = np.random.randn(
        n_vocab, n_embd).astype(np.float32) * 0.02
    weights["norm.weight"] = np.zeros(n_embd, dtype=np.float32)

    for i in range(n_layer):
        pfx = f"layers.{i}."
        weights[pfx + "input_layernorm.weight"] = np.zeros(
            n_embd, dtype=np.float32)
        weights[pfx + "post_attention_layernorm.weight"] = np.zeros(
            n_embd, dtype=np.float32)
        weights[pfx + "mlp.gate_proj.weight"] = np.random.randn(
            intermediate_size, n_embd).astype(np.float32) * 0.02
        weights[pfx + "mlp.up_proj.weight"] = np.random.randn(
            intermediate_size, n_embd).astype(np.float32) * 0.02
        weights[pfx + "mlp.down_proj.weight"] = np.random.randn(
            n_embd, intermediate_size).astype(np.float32) * 0.02

        if (i + 1) % 4 == 0:
            qo_dim = n_head * head_dim
            weights[pfx + "self_attn.q_proj.weight"] = np.random.randn(
                qo_dim, n_embd).astype(np.float32) * 0.02
            weights[pfx + "self_attn.k_proj.weight"] = np.random.randn(
                kv_dim, n_embd).astype(np.float32) * 0.02
            weights[pfx + "self_attn.v_proj.weight"] = np.random.randn(
                kv_dim, n_embd).astype(np.float32) * 0.02
            weights[pfx + "self_attn.o_proj.weight"] = np.random.randn(
                n_embd, qo_dim).astype(np.float32) * 0.02
            weights[pfx + "self_attn.q_norm.weight"] = np.zeros(
                head_dim, dtype=np.float32)
            weights[pfx + "self_attn.k_norm.weight"] = np.zeros(
                head_dim, dtype=np.float32)
        else:
            weights[pfx + "linear_attn.in_proj_qkv.weight"] = np.random.randn(
                ssm_qkv_dim, n_embd).astype(np.float32) * 0.02
            weights[pfx + "linear_attn.in_proj_z.weight"] = np.random.randn(
                ssm_dim, n_embd).astype(np.float32) * 0.02
            weights[pfx + "linear_attn.out_proj.weight"] = np.random.randn(
                n_embd, ssm_dim).astype(np.float32) * 0.02
            weights[pfx + "linear_attn.conv1d.weight"] = np.random.randn(
                ssm_qkv_dim, 1, ssm_conv_kernel).astype(np.float32) * 0.02
            weights[pfx + "linear_attn.norm.weight"] = np.ones(
                ssm_head_dim, dtype=np.float32)
            weights[pfx + "linear_attn.A_log"] = np.random.randn(
                ssm_n_heads).astype(np.float32) * 0.1
            weights[pfx + "linear_attn.dt_bias"] = np.zeros(
                ssm_n_heads, dtype=np.float32)
            weights[pfx + "linear_attn.in_proj_a.weight"] = np.random.randn(
                ssm_n_heads, n_embd).astype(np.float32) * 0.02
            weights[pfx + "linear_attn.in_proj_b.weight"] = np.random.randn(
                ssm_n_heads, n_embd).astype(np.float32) * 0.02

    model = Qwen35WebGPU(
        weights, n_layer=n_layer, n_head=n_head, n_kv_heads=n_kv_heads,
        n_embd=n_embd, intermediate_size=intermediate_size,
        n_vocab=n_vocab, rope_theta=10000.0, rms_norm_eps=1e-6,
        head_dim=head_dim, partial_rotary_factor=0.25,
        attn_output_gate=False,
        ssm_n_heads=ssm_n_heads, ssm_head_dim=ssm_head_dim,
        ssm_n_groups=ssm_n_groups, ssm_d_state=ssm_d_state,
        ssm_conv_kernel=ssm_conv_kernel)

    prompt_len = 16
    decode_steps = 20
    token_ids = np.random.randint(0, n_vocab, prompt_len, dtype=np.int32)

    print(f"\nConfig: {n_layer} layers ({n_layer*3//4} SSM + {n_layer//4} attn), "
          f"{n_embd}D, {n_vocab} vocab")
    print(f"Prompt: {prompt_len} tokens, decode: {decode_steps} tokens")

    # --- Prefill ---
    model.kv_cache = None
    model._ssm_states = {}
    model._ssm_conv_states = {}

    t0 = time.perf_counter()
    logits = model.forward(token_ids, use_cache=True, pos_offset=0)
    t_prefill = time.perf_counter() - t0
    print(f"\nPrefill: {prompt_len} tokens in {t_prefill*1000:.1f}ms "
          f"({t_prefill/prompt_len*1000:.1f}ms/tok)")

    # --- Decode ---
    times = []
    pos = prompt_len
    for step in range(decode_steps):
        next_tok = np.array([logits[-1].argmax()], dtype=np.int32)
        t0 = time.perf_counter()
        logits = model.forward(next_tok, use_cache=True, pos_offset=pos)
        dt = time.perf_counter() - t0
        times.append(dt)
        pos += 1

    times = np.array(times)
    # Skip first (warm-up)
    steady = times[1:]
    mean_ms = steady.mean() * 1000
    p50 = np.percentile(steady, 50) * 1000
    tps = 1.0 / steady.mean()
    print(f"\nDecode ({decode_steps} steps):")
    print(f"  Mean:  {mean_ms:.2f}ms/tok  ({tps:.1f} tok/s)")
    print(f"  P50:   {p50:.2f}ms/tok")
    print(f"  Min:   {steady.min()*1000:.2f}ms  Max: {steady.max()*1000:.2f}ms")

    # Per-layer type timing
    print(f"\nPer-layer breakdown (single decode step):")
    model.kv_cache = None
    model._ssm_states = {}
    model._ssm_conv_states = {}
    # Refill cache
    model.forward(token_ids, use_cache=True, pos_offset=0)

    from common.model_base import GPUBuffer
    next_tok = np.array([42], dtype=np.int32)
    x = model.weights["embed_tokens.weight"][next_tok]
    positions = np.array([prompt_len], dtype=np.int32)

    ssm_times = []
    attn_times = []
    for layer in range(n_layer):
        t0 = time.perf_counter()
        x = model._transformer_block(x, layer, use_cache=True,
                                      positions=positions)
        dt = time.perf_counter() - t0
        if LAYER_TYPES[layer] == "full_attention":
            attn_times.append(dt)
        else:
            ssm_times.append(dt)

    if ssm_times:
        print(f"  SSM layers:  mean={np.mean(ssm_times)*1000:.2f}ms  "
              f"total={sum(ssm_times)*1000:.1f}ms  ({len(ssm_times)} layers)")
    if attn_times:
        print(f"  Attn layers: mean={np.mean(attn_times)*1000:.2f}ms  "
              f"total={sum(attn_times)*1000:.1f}ms  ({len(attn_times)} layers)")

    total_layer = sum(ssm_times) + sum(attn_times)
    print(f"  Total layers: {total_layer*1000:.1f}ms")
    print(f"\n--- BENCHMARK COMPLETE ---")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Qwen3.5-27B on WebGPU via Triton")
    parser.add_argument("--verify", action="store_true",
                        help="Verify pipeline with random weights")
    parser.add_argument("--bench", action="store_true",
                        help="Benchmark with random weights")
    parser.add_argument("--quantize", action="store_true",
                        help="Download and quantize weights to INT4")
    parser.add_argument("--prompt", type=str,
                        default="The future of AI is",
                        help="Prompt for text generation")
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--weights-dir", type=str, default=None)
    parser.add_argument("--profile", action="store_true",
                        help="Enable profiling")
    args = parser.parse_args()

    if args.verify:
        success = verify_with_random_weights()
        sys.exit(0 if success else 1)

    if args.bench:
        bench_with_random_weights()
        return

    config = QWEN35_CONFIG
    weights_dir = args.weights_dir or os.path.join(_SCRIPT_DIR, "..", "..", "gitignore", "models", os.path.basename(_SCRIPT_DIR), "weights")
    q4_path = os.path.join(weights_dir, "weights_q4.npz")

    if args.quantize:
        # Streaming download + quantize (one shard at a time, low RAM)
        download_and_quantize_streaming(weights_dir)
        return

    # Load weights (prefer quantized if available)
    _t_weight_load_0 = time.perf_counter_ns()
    if os.path.exists(q4_path):
        print(f"Loading quantized weights from {q4_path}...")
        weights = {k: v for k, v in np.load(q4_path, mmap_mode='r').items()}
        quantized = True
    else:
        npz_path, _ = download_qwen35_weights(weights_dir)
        weights = load_weights(npz_path)
        quantized = False
    _t_weight_load_1 = time.perf_counter_ns()

    print(f"Loaded {len(weights)} weight tensors "
          f"({'INT4' if quantized else 'fp32'})")

    tokenizer_path = os.path.join(weights_dir, "tokenizer.json")
    tokenizer = load_tokenizer(tokenizer_path)

    _t_model_init_0 = time.perf_counter_ns()
    model = Qwen35WebGPU(
        weights,
        n_layer=config["n_layer"],
        n_head=config["n_head"],
        n_kv_heads=config["n_kv_heads"],
        n_embd=config["n_embd"],
        intermediate_size=config["intermediate_size"],
        n_vocab=config["n_vocab"],
        rope_theta=config["rope_theta"],
        rms_norm_eps=config["rms_norm_eps"],
        head_dim=config["head_dim"],
        partial_rotary_factor=config["partial_rotary_factor"],
        attn_output_gate=config["attn_output_gate"],
        ssm_n_heads=config["ssm_n_heads"],
        ssm_head_dim=config["ssm_head_dim"],
        ssm_n_groups=config["ssm_n_groups"],
        ssm_d_state=config["ssm_d_state"],
        ssm_conv_kernel=config["ssm_conv_kernel"],
        quantized=quantized,
        n_q_heads=config.get("n_q_heads"))
    _t_model_init_1 = time.perf_counter_ns()
    print(f"Model created in {(_t_model_init_1-_t_model_init_0)/1e6:.0f}ms")

    if args.profile:
        model.enable_profiling()
        print(f"Profiling enabled (GPU timestamps: {model.profiler.gpu_enabled})")
        # Inject pre-recorded init events
        init_phases = [
            ("imports", _t_script_start, _t_imports_done),
            ("weight_loading", _t_weight_load_0, _t_weight_load_1),
            ("model_init", _t_model_init_0, _t_model_init_1),
        ]
        if hasattr(model, '_init_phases'):
            init_phases.extend(model._init_phases)
        model.profiler.inject_init_events(init_phases)

    # Warmup
    import time as _wt
    _w0 = _wt.perf_counter()
    model.forward(np.array([1], dtype=np.int32), use_cache=True, pos_offset=0)
    model.kv_cache = None
    if hasattr(model, '_gpu_kv_cache'):
        for layer in model._gpu_kv_cache:
            k, v, _ = model._gpu_kv_cache[layer]
            model._gpu_kv_cache[layer] = (k, v, 0)
    model._init_ssm_state()
    print(f"Warmup: {(_wt.perf_counter()-_w0)*1000:.0f}ms")

    generate(model, args.prompt, tokenizer,
             max_tokens=args.max_tokens,
             temperature=args.temperature)

    if args.profile:
        model.save_profile(_SCRIPT_DIR, "Qwen3.5-27B")


if __name__ == "__main__":
    main()
