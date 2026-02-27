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

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_SCRIPT_DIR))

import numpy as np

from common.model_base import WebGPUModel
from common.utils import (
    load_weights, download_weights, load_tokenizer, generate,
)


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
                 max_seq_len: int = 2048,
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

        # QKV fused projection size for SSM:
        # Q (ssm_dim) + K (ssm_n_groups * ssm_d_state) + V (ssm_n_heads)
        # Actually: in_proj_qkv maps E -> ssm_dim + ssm_n_groups * ssm_d_state + ssm_n_heads
        # But looking at weights, in_proj_qkv is (ssm_dim + n_groups*d_state + n_heads, E)
        # = (6144 + 16*128 + 48, 5120) = (8240, 5120)
        # Wait, let's compute: in_proj_qkv output = ssm_dim + group_dim + dt_dim
        # ssm_dim = 48 * 128 = 6144 (BC states)
        # But Mamba-2 fused QKV = x_proj (inner_dim) + B (n_groups * d_state) + C (n_groups * d_state)
        # Actually: expand_dim = ssm_n_heads * ssm_head_dim = 6144
        # B_dim = ssm_n_groups * ssm_d_state = 16 * 128 = 2048
        # C_dim = ssm_n_groups * ssm_d_state = 16 * 128 = 2048
        # So in_proj_qkv = expand_dim + B_dim + C_dim = 6144 + 2048 + 2048 = 10240
        self.ssm_qkv_dim = self.ssm_dim + 2 * (ssm_n_groups * ssm_d_state)

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

    def _proj(self, x, weight_name, bias_key, out_features,
              K=None, gpu_out=False):
        """Linear projection using best available kernel (Q4/fp16/fp32)."""
        # INT4 GPU path
        q4_key = weight_name + ".q4.gpu"
        if q4_key in self._gpu_weights:
            return self._linear_q4(
                x, self._gpu_weights[q4_key],
                self._gpu_weights[weight_name + ".scales.gpu"],
                self._gpu_weights[weight_name + ".zeros.gpu"],
                self._gpu_weights[bias_key], out_features,
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

    def _init_gpu_kv_cache(self):
        """Pre-allocate KV cache for full-attention layers only (16 layers)."""
        # Only full attention layers need KV cache
        pass  # Use numpy-based cache in kv_cache dict

    def _upload_weights_to_gpu(self):
        """Upload all weights to GPU (INT4/fp16/fp32 depending on capabilities)."""
        E = self.n_embd
        HD = self.head_dim
        IM = self.intermediate_size
        ssm_dim = self.ssm_dim  # 6144

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

        # Embed/LM head: keep on CPU to save VRAM.
        # embed_tokens is only used for token lookup (CPU).
        # lm_head runs as CPU matmul (avoids 4.8 GB GPU upload for
        # 248K vocab × 5120 × 2 = 2.4 GB each).
        # Ensure lm_head weights are fp32 for CPU matmul.
        lm_w = self.weights.get("lm_head.weight")
        if lm_w is not None and lm_w.dtype == np.float16:
            self.weights["lm_head.weight"] = lm_w.astype(np.float32)

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

    def _attention_block(self, x, layer: int,
                         use_cache: bool = False,
                         positions: np.ndarray = None,
                         **kwargs):
        """Full GQA with QK-norm, partial RoPE, and output gate."""
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

        # Fused QKV projection (single dispatch)
        qkv = self._proj(
            x, pfx + "qkv_proj.weight",
            "zero_bias_QKV", self.full_attn_qkv_dim,
            K=self.n_embd)

        # Split into Q, K, V
        q_dim = self.full_attn_q_dim  # 48 * 256 = 12288
        q = qkv[:, :q_dim]
        k = qkv[:, q_dim:q_dim + self.kv_dim]
        v = qkv[:, q_dim + self.kv_dim:]

        # Reshape to heads
        Q = q.reshape(T, n_q_heads, HD)
        K_new = k.reshape(T, n_kv, HD)
        V_new = v.reshape(T, n_kv, HD)

        # QK-norm (RMSNorm per head) — vectorized
        q_norm_w = self.weights[pfx + "q_norm.weight"]
        k_norm_w = self.weights[pfx + "k_norm.weight"]
        q_rms = np.sqrt(np.mean(Q * Q, axis=-1, keepdims=True) + self.rms_norm_eps)
        Q = Q / q_rms * q_norm_w
        k_rms = np.sqrt(np.mean(K_new * K_new, axis=-1, keepdims=True) + self.rms_norm_eps)
        K_new = K_new / k_rms * k_norm_w

        # Partial RoPE (25% of head_dim)
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
                K_buf[:T] = K_new
                V_buf[:T] = V_new
                self.kv_cache[layer] = (K_buf, V_buf, T)
            else:
                K_buf, V_buf, cur_len = self.kv_cache[layer]
                K_buf[cur_len:cur_len + T] = K_new
                V_buf[cur_len:cur_len + T] = V_new
                self.kv_cache[layer] = (K_buf, V_buf, cur_len + T)
            _, _, T_total = self.kv_cache[layer]
            K_full = K_buf[:T_total]
            V_full = V_buf[:T_total]
        else:
            T_total = T
            K_full = K_new
            V_full = V_new

        scale = 1.0 / np.sqrt(HD)

        if T == 1 and use_cache and T_total > 1:
            # Decode: vectorized multi-head attention
            K_exp = np.repeat(K_full, n_rep, axis=1)
            V_exp = np.repeat(V_full, n_rep, axis=1)
            Q_t = Q.transpose(1, 0, 2)          # (n_head, 1, HD)
            K_t = K_exp.transpose(1, 2, 0)      # (n_head, HD, T_total)
            scores = np.float32((Q_t @ K_t).squeeze(1) * scale)
            scores -= scores.max(axis=1, keepdims=True)
            exp_s = np.exp(scores)
            s = exp_s.sum(axis=1, keepdims=True)
            s = np.where(s == 0, 1.0, s)
            attn = exp_s / s
            V_t = V_exp.transpose(1, 0, 2)
            attn_out = (attn[:, None, :] @ V_t).squeeze(1)
            attn_out = attn_out[None, :, :].astype(np.float32)
        else:
            attn_out = self._causal_attention_multihead(
                Q, K_full, V_full, n_rep)

        # Attention output: 48 Q heads → gate to 24 output heads
        # Reshape (T, 48, 256) → (T, 24, 512) then split: gate + value
        if self.attn_output_gate and n_q_heads > n_head:
            attn_out = attn_out.reshape(T, n_head, 2 * HD)
            gate_part = attn_out[:, :, :HD]
            value_part = attn_out[:, :, HD:]
            gate = 1.0 / (1.0 + np.exp(-gate_part))
            attn_out = gate * value_part  # (T, n_head, HD)

        attn_flat = attn_out.reshape(T, self.full_attn_qo_dim)

        # O projection
        o = self._proj(
            attn_flat, pfx + "o_proj.weight",
            "zero_bias_E", self.n_embd,
            K=self.full_attn_qo_dim)
        return o

    # ------------------------------------------------------------------
    # Linear attention (Mamba-2 SSM) block
    # ------------------------------------------------------------------

    def _mamba2_ssm_step(self, x_dt, B, C, A, state):
        """Single-step Mamba-2 SSM recurrence (for decode).

        h_{t} = A * h_{t-1} + B_t^T * x_t
        y_t = (C_t * h_t).sum()

        Args:
            x_dt: (n_heads,) - input scaled by dt
            B: (n_groups, d_state)
            C: (n_groups, d_state)
            A: (n_heads,) - diagonal decay per head
            state: (n_heads, d_state) - recurrent state

        Returns:
            y: (n_heads,) - output
            new_state: (n_heads, d_state)
        """
        n_heads = self.ssm_n_heads
        d_state = self.ssm_d_state
        n_groups = self.ssm_n_groups
        heads_per_group = n_heads // n_groups

        # Apply decay
        new_state = A[:, None] * state  # (n_heads, d_state)

        # Add input: B_t^T * x_t for each head in its group
        for g in range(n_groups):
            h_start = g * heads_per_group
            h_end = h_start + heads_per_group
            # B[g]: (d_state,), x_dt[h_start:h_end]: (heads_per_group,)
            new_state[h_start:h_end] += (
                x_dt[h_start:h_end, None] * B[g:g+1, :])

        # Output: y = C * h
        y = np.zeros(n_heads, dtype=np.float32)
        for g in range(n_groups):
            h_start = g * heads_per_group
            h_end = h_start + heads_per_group
            y[h_start:h_end] = (
                new_state[h_start:h_end] * C[g:g+1, :]).sum(axis=-1)

        return y, new_state

    def _mamba2_ssm_prefill(self, x_dt, B, C, A, T):
        """Prefill SSM computation (sequential scan over T tokens).

        x_dt: (T, n_heads)
        B: (T, n_groups, d_state)
        C: (T, n_groups, d_state)
        A: (n_heads,)

        Returns:
            y: (T, n_heads)
            final_state: (n_heads, d_state)
        """
        n_heads = self.ssm_n_heads
        d_state = self.ssm_d_state
        state = np.zeros((n_heads, d_state), dtype=np.float32)
        y = np.zeros((T, n_heads), dtype=np.float32)

        for t in range(T):
            y[t], state = self._mamba2_ssm_step(
                x_dt[t], B[t], C[t], A, state)

        return y, state

    def _ssm_decode_cpu(self, x_np, layer):
        """Optimized T=1 SSM decode with batched GPU projections."""
        E = self.n_embd
        pfx = f"layers.{layer}.linear_attn."
        n_heads = self.ssm_n_heads
        d_state = self.ssm_d_state
        n_groups = self.ssm_n_groups
        hd = self.ssm_head_dim
        ssm_dim = self.ssm_dim
        kernel_size = self.ssm_conv_kernel
        heads_per_group = n_heads // n_groups

        # 1. Batch QKV + Z projections (both from same input x_np)
        use_gpu = getattr(self, '_use_q4_gpu', False)
        if use_gpu:
            runner = self.cache.runner
            runner.begin_batch()
            qkv_gpu = self._proj(x_np, pfx + "in_proj_qkv.weight",
                                  "zero_bias_SSM_QKV", self.ssm_qkv_dim,
                                  K=E, gpu_out=True)
            z_gpu = self._proj(x_np, pfx + "in_proj_z.weight",
                                "zero_bias_SSM", ssm_dim,
                                K=E, gpu_out=True)
            rb = runner.end_batch(readback_buffers=[qkv_gpu, z_gpu])
            qkv = rb[id(qkv_gpu)].reshape(1, self.ssm_qkv_dim)
            z = rb[id(z_gpu)].reshape(1, ssm_dim)
        else:
            qkv = self._cpu_matmul(x_np, pfx + "in_proj_qkv.weight",
                                   self.ssm_qkv_dim, E)
            z = self._cpu_matmul(x_np, pfx + "in_proj_z.weight",
                                 ssm_dim, E)

        x_expand = qkv[0, :ssm_dim]  # (ssm_dim,)
        bc_start = ssm_dim
        b_dim = n_groups * d_state
        B_vec = qkv[0, bc_start:bc_start + b_dim].reshape(n_groups, d_state)
        C_vec = qkv[0, bc_start + b_dim:].reshape(n_groups, d_state)

        # 2. Conv1d with cached state (shift-and-add)
        # Conv1d operates on the full QKV dimension (ssm_qkv_dim)
        conv_dim = self.ssm_qkv_dim
        if layer not in self._ssm_conv_states:
            self._ssm_conv_states[layer] = np.zeros(
                (kernel_size - 1, conv_dim), dtype=np.float32)
        conv_state = self._ssm_conv_states[layer]
        conv_w_raw = self.weights[pfx + "conv1d.weight"]
        if conv_w_raw.dtype == np.float16:
            conv_w_raw = conv_w_raw.astype(np.float32)
        conv_w = conv_w_raw.reshape(conv_dim, kernel_size)
        # Full QKV vector for conv
        qkv_flat = qkv[0]  # (ssm_qkv_dim,)
        x_conv_full = (conv_state * conv_w[:, :kernel_size - 1].T).sum(axis=0) + \
                      qkv_flat * conv_w[:, kernel_size - 1]
        conv_state[:-1] = conv_state[1:]
        conv_state[-1] = qkv_flat

        # After conv, split back to x_expand, B, C
        x_expand = x_conv_full[:ssm_dim]
        B_vec = x_conv_full[bc_start:bc_start + b_dim].reshape(n_groups, d_state)
        C_vec = x_conv_full[bc_start + b_dim:].reshape(n_groups, d_state)

        # SiLU (only on x_expand)
        x_silu = x_expand / (1.0 + np.exp(-x_expand))  # (ssm_dim,)

        # 3. dt computation (uses original input x_np, not post-conv)
        A_log = self.weights[pfx + "A_log"]
        if A_log.dtype == np.float16:
            A_log = A_log.astype(np.float32)
        A = -np.exp(A_log)
        dt_bias = self.weights[pfx + "dt_bias"]
        if dt_bias.dtype == np.float16:
            dt_bias = dt_bias.astype(np.float32)
        in_proj_a = self.weights.get(pfx + "in_proj_a.weight")
        in_proj_b = self.weights.get(pfx + "in_proj_b.weight")

        if in_proj_a is not None and in_proj_b is not None:
            a = in_proj_a.astype(np.float32) if in_proj_a.dtype == np.float16 else in_proj_a
            b = in_proj_b.astype(np.float32) if in_proj_b.dtype == np.float16 else in_proj_b
            # Low-rank factored dt: dt = (x @ A^T) * (x @ B^T) + bias
            x_flat = x_np.ravel()
            dt_raw = (x_flat @ a.T) * (x_flat @ b.T) + dt_bias
        else:
            x_heads_mean = x_silu.reshape(n_heads, hd).mean(axis=-1)
            dt_raw = x_heads_mean + dt_bias
        dt = np.log1p(np.exp(dt_raw))  # (n_heads,)

        # 4. SSM step: state update + output
        A_disc = np.exp(A * dt)  # (n_heads,)
        x_heads_mean = x_silu.reshape(n_heads, hd).mean(axis=-1)
        x_dt = dt * x_heads_mean  # (n_heads,)

        if layer not in self._ssm_states:
            self._ssm_states[layer] = np.zeros(
                (n_heads, d_state), dtype=np.float32)
        state = self._ssm_states[layer]

        # Expand B,C to per-head
        B_heads = np.repeat(B_vec, heads_per_group, axis=0)  # (n_heads, d_state)
        C_heads = np.repeat(C_vec, heads_per_group, axis=0)

        state[:] = A_disc[:, None] * state + x_dt[:, None] * B_heads
        y_heads = (state * C_heads).sum(axis=-1)  # (n_heads,)

        # 5. Expand, gate with x_silu, normalize
        y_expanded = np.repeat(y_heads, hd)  # (ssm_dim,)
        y_out = y_expanded * x_silu

        norm_w = self.weights[pfx + "norm.weight"]
        if norm_w.dtype == np.float16:
            norm_w = norm_w.astype(np.float32)
        # Per-head RMSNorm: reshape to (n_heads, hd), norm per head
        y_heads_shaped = y_out.reshape(n_heads, hd)
        rms = np.sqrt(np.mean(y_heads_shaped ** 2, axis=-1, keepdims=True)
                      + self.rms_norm_eps)
        y_normed = (y_heads_shaped / rms * norm_w).reshape(ssm_dim)

        # 6. Gate with sigmoid(z) — already computed in step 1
        z_flat = z.ravel()  # (ssm_dim,)
        gate = 1.0 / (1.0 + np.exp(-z_flat))
        y_gated = y_normed * gate

        # 7. Output projection
        if getattr(self, '_use_q4_gpu', False):
            out = self._proj(y_gated[None, :], pfx + "out_proj.weight",
                              "zero_bias_E", E, K=ssm_dim)
        else:
            out = self._cpu_matmul(y_gated[None, :], pfx + "out_proj.weight",
                                   E, ssm_dim)
        return out

    def _linear_attention_block(self, x, layer: int,
                                use_cache: bool = False, **kwargs):
        """Mamba-2 SSM linear attention block.

        1. Project input: in_proj_qkv → split into x (expand), B, C
        2. Short 1D convolution on expanded x
        3. Compute dt (timestep) from x via in_proj_a / in_proj_b or dt_bias
        4. SSM scan: h = A*h + B^T*(dt*x), y = C*h
        5. Normalize output (GroupNorm-like)
        6. Gate with sigmoid(z), z = in_proj_z(input)
        7. Output projection
        """
        from common.model_base import GPUBuffer

        if isinstance(x, GPUBuffer):
            T = x.shape[0] if x.shape else 1
        else:
            T = x.shape[0]

        E = self.n_embd
        pfx = f"layers.{layer}.linear_attn."
        n_heads = self.ssm_n_heads
        d_state = self.ssm_d_state
        n_groups = self.ssm_n_groups
        hd = self.ssm_head_dim
        ssm_dim = self.ssm_dim

        # 1. Project input via in_proj_qkv: (T, E) -> (T, ssm_qkv_dim)
        # Compute z projection in parallel (both need x as input)
        qkv = self._proj(
            x, pfx + "in_proj_qkv.weight",
            "zero_bias_SSM_QKV", self.ssm_qkv_dim,
            K=E)

        # Split: x_expand (ssm_dim), B (n_groups*d_state), C (n_groups*d_state)
        x_expand = qkv[:, :ssm_dim]  # (T, 6144)
        bc_start = ssm_dim
        b_dim = n_groups * d_state  # 16*128 = 2048
        B_flat = qkv[:, bc_start:bc_start + b_dim]  # (T, 2048)
        C_flat = qkv[:, bc_start + b_dim:]           # (T, 2048)
        B_val = B_flat.reshape(T, n_groups, d_state)
        C_val = C_flat.reshape(T, n_groups, d_state)

        # 2. Short 1D convolution on full QKV output
        conv_w = self.weights[pfx + "conv1d.weight"]  # (ssm_qkv_dim, 1, kernel)
        if conv_w.dtype == np.float16:
            conv_w = conv_w.astype(np.float32)
        kernel_size = self.ssm_conv_kernel  # 4
        conv_dim = self.ssm_qkv_dim

        # Conv operates on full QKV output
        qkv_full = qkv  # (T, ssm_qkv_dim)
        if use_cache:
            if layer not in self._ssm_conv_states:
                self._ssm_conv_states[layer] = np.zeros(
                    (kernel_size - 1, conv_dim), dtype=np.float32)
            conv_state = self._ssm_conv_states[layer]
            x_conv_in = np.concatenate([conv_state, qkv_full], axis=0)
            self._ssm_conv_states[layer] = x_conv_in[-(kernel_size - 1):].copy()
        else:
            x_conv_in = np.concatenate([
                np.zeros((kernel_size - 1, conv_dim), dtype=np.float32),
                qkv_full], axis=0)

        # Depthwise conv1d: vectorized using sliding window matmul
        conv_w_sq = conv_w.reshape(conv_dim, kernel_size)
        from numpy.lib.stride_tricks import as_strided
        strides = x_conv_in.strides
        windows = as_strided(
            x_conv_in,
            shape=(T, kernel_size, conv_dim),
            strides=(strides[0], strides[0], strides[1]))
        x_conv = np.einsum('tkc,ck->tc', windows, conv_w_sq)

        # Split post-conv back to x_expand, B, C
        x_expand = x_conv[:, :ssm_dim]
        B_val = x_conv[:, bc_start:bc_start + b_dim].reshape(T, n_groups, d_state)
        C_val = x_conv[:, bc_start + b_dim:].reshape(T, n_groups, d_state)

        # Apply SiLU activation (only on x_expand)
        x_silu = x_expand / (1.0 + np.exp(-x_expand))

        # 3. Compute dt and A
        A_log = self.weights[pfx + "A_log"]  # (n_heads,)
        A = -np.exp(A_log)  # negative decay
        dt_bias = self.weights[pfx + "dt_bias"]  # (n_heads,)
        in_proj_a = self.weights.get(pfx + "in_proj_a.weight")
        in_proj_b = self.weights.get(pfx + "in_proj_b.weight")

        x_heads = x_silu.reshape(T, n_heads, hd)  # (T, 48, 128)

        if in_proj_a is not None and in_proj_b is not None:
            x_flat_for_dt = x_silu
            dt_a = x_flat_for_dt @ in_proj_a.T
            dt_raw = dt_a @ in_proj_b.T + dt_bias
        else:
            dt_raw = x_heads.mean(axis=-1) + dt_bias

        dt = np.log1p(np.exp(dt_raw))  # (T, n_heads)
        A_discrete = np.exp(A[None, :] * dt)  # (T, n_heads)

        # 4. SSM scan
        if use_cache:
            if layer not in self._ssm_states:
                self._ssm_states[layer] = np.zeros(
                    (n_heads, d_state), dtype=np.float32)
            state = self._ssm_states[layer]
        else:
            state = np.zeros((n_heads, d_state), dtype=np.float32)

        heads_per_group = n_heads // n_groups
        x_dt_all = dt * x_heads.mean(axis=-1)  # (T, n_heads)
        B_heads = np.repeat(B_val, heads_per_group, axis=1)
        C_heads = np.repeat(C_val, heads_per_group, axis=1)

        y = np.empty((T, n_heads), dtype=np.float32)
        for t in range(T):
            state = A_discrete[t, :, None] * state + (
                x_dt_all[t, :, None] * B_heads[t])
            y[t] = (state * C_heads[t]).sum(axis=-1)

        if use_cache:
            self._ssm_states[layer] = state

        # 5. Expand y back to ssm_dim and normalize
        # y: (T, n_heads) -> (T, ssm_dim) via broadcast mul
        # Instead of np.repeat, use reshape+broadcast with x_silu
        y_3d = y[:, :, None]  # (T, n_heads, 1)
        x_3d = x_silu.reshape(T, n_heads, hd)  # (T, n_heads, hd)
        y_out = (y_3d * x_3d).reshape(T, ssm_dim)  # broadcast multiply

        norm_w = self.weights[pfx + "norm.weight"]
        if norm_w.dtype == np.float16:
            norm_w = norm_w.astype(np.float32)
        # Per-head RMSNorm
        y_heads_shaped = y_out.reshape(T, n_heads, hd)
        rms = np.sqrt(
            np.mean(y_heads_shaped ** 2, axis=-1, keepdims=True)
            + self.rms_norm_eps)
        y_normed = (y_heads_shaped / rms * norm_w).reshape(T, ssm_dim)

        # 6. Gate with sigmoid(z)
        z = self._proj(
            x, pfx + "in_proj_z.weight",
            "zero_bias_SSM", ssm_dim,
            K=E)
        gate = 1.0 / (1.0 + np.exp(-z))  # sigmoid
        y_gated = y_normed * gate

        # 7. Output projection
        out = self._proj(
            y_gated, pfx + "out_proj.weight",
            "zero_bias_E", self.n_embd,
            K=ssm_dim)

        return out

    # ------------------------------------------------------------------
    # MLP block (shared by all layers)
    # ------------------------------------------------------------------

    def _mlp_block(self, x, layer: int, gpu_out: bool = False):
        """SwiGLU MLP: fused gate_up → silu_mul → down_proj."""
        E = self.n_embd
        IM = self.intermediate_size
        pfx = f"layers.{layer}.mlp."

        gate_up = self._proj(
            x, pfx + "gate_up_proj.weight",
            "zero_bias_GU", self.gate_up_out,
            K=E, gpu_out=True)
        h = self._silu_mul_fused(gate_up, IM, gpu_out=True)
        return self._proj(
            h, pfx + "down_proj.weight",
            "zero_bias_E", E,
            K=IM, gpu_out=gpu_out)

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
        qkv = self._proj(x_np, pfx + "qkv_proj.weight",
                          "zero_bias_QKV", self.full_attn_qkv_dim,
                          K=self.n_embd) if getattr(self, '_use_q4_gpu', False) \
              else self._cpu_matmul(x_np, pfx + "qkv_proj.weight",
                                    self.full_attn_qkv_dim, self.n_embd)
        q_dim = self.full_attn_q_dim
        q = qkv[:, :q_dim]
        k = qkv[:, q_dim:q_dim + self.kv_dim]
        v = qkv[:, q_dim + self.kv_dim:]

        Q = q.reshape(1, n_q_heads, HD)
        K_new = k.reshape(1, n_kv, HD)
        V_new = v.reshape(1, n_kv, HD)

        # QK-norm
        q_norm_w = self.weights[pfx + "q_norm.weight"]
        k_norm_w = self.weights[pfx + "k_norm.weight"]
        if q_norm_w.dtype == np.float16:
            q_norm_w = q_norm_w.astype(np.float32)
        if k_norm_w.dtype == np.float16:
            k_norm_w = k_norm_w.astype(np.float32)
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
            return self._proj(attn_flat, pfx + "o_proj.weight",
                              "zero_bias_E", self.n_embd,
                              K=self.full_attn_qo_dim)
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

        # RMSNorm 1
        rn1 = self._rms_norm_vec(x_np, pfx + "input_layernorm.weight")

        # Attention (full CPU path for both types)
        if LAYER_TYPES[layer] == "full_attention":
            attn = self._attn_decode_cpu(rn1, layer, positions)
        else:
            attn = self._ssm_decode_cpu(rn1, layer)

        x_np = x_np + attn

        # RMSNorm 2
        rn2 = self._rms_norm_vec(x_np, pfx + "post_attention_layernorm.weight")

        # MLP — use GPU Q4 if available (avoids OOM from CPU dequant)
        E = self.n_embd
        IM = self.intermediate_size
        if getattr(self, '_use_q4_gpu', False):
            runner = self.cache.runner
            runner.begin_batch()
            gate_up = self._proj(
                rn2, pfx + "mlp.gate_up_proj.weight",
                "zero_bias_GU", self.gate_up_out, K=E, gpu_out=True)
            h = self._silu_mul_fused(gate_up, IM, gpu_out=True)
            mlp_gpu = self._proj(
                h, pfx + "mlp.down_proj.weight",
                "zero_bias_E", E, K=IM, gpu_out=True)
            rb = runner.end_batch(readback_buffers=[mlp_gpu])
            mlp_out = rb[id(mlp_gpu)].reshape(1, E)
        else:
            gate_up = self._cpu_matmul(
                rn2, pfx + "mlp.gate_up_proj.weight", self.gate_up_out, E)
            gate = gate_up[:, :IM]
            up = gate_up[:, IM:]
            h = gate / (1.0 + np.exp(-gate)) * up
            mlp_out = self._cpu_matmul(
                h, pfx + "mlp.down_proj.weight", E, IM)

        return x_np + mlp_out

    def _transformer_block(self, x, layer: int,
                           use_cache: bool = False,
                           positions: np.ndarray = None, **kwargs):
        """Pre-norm transformer block. Dispatches to full or linear attn."""
        from common.model_base import GPUBuffer
        pfx = f"layers.{layer}."

        T = (x.shape[0] if x.shape else 1) if isinstance(x, GPUBuffer) else x.shape[0]

        # T=1 decode: CPU path avoids GPU dispatch overhead
        if T == 1 and use_cache:
            return self._decode_cpu(x, layer, positions)

        # Pre-norm
        rn1 = self._rms_norm(
            x, self._gpu_weights[pfx + "input_layernorm.weight"],
            gpu_out=True)

        # Attention (full GQA or Mamba-2 SSM)
        if LAYER_TYPES[layer] == "full_attention":
            attn = self._attention_block(rn1, layer, use_cache=use_cache,
                                         positions=positions)
        else:
            attn = self._linear_attention_block(rn1, layer,
                                                use_cache=use_cache)

        # Fused residual add + RMSNorm (saves 1 dispatch) when available
        use_fused = (hasattr(self, '_add_rn_result') and
                     isinstance(x, GPUBuffer) and isinstance(attn, GPUBuffer))
        if use_fused:
            rn2 = self._add_rms_norm(
                x, attn,
                self._gpu_weights[pfx + "post_attention_layernorm.weight"],
                gpu_out=True)
        else:
            x = self._add(x, attn, gpu_out=True)
            rn2 = self._rms_norm(
                x, self._gpu_weights[pfx + "post_attention_layernorm.weight"],
                gpu_out=True)

        mlp = self._mlp_block(rn2, layer, gpu_out=True)
        x = self._add(x, mlp, gpu_out=True)
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
        x = wte[token_ids]
        positions = np.arange(pos_offset, pos_offset + T, dtype=np.int32)

        for layer in range(self.n_layer):
            x = self._transformer_block(x, layer, use_cache=use_cache,
                                        positions=positions)

        x = self._rms_norm(x, self._gpu_weights["norm.weight"])

        # LM head on CPU (avoids uploading 2.4 GB to GPU)
        lm_w = self.weights["lm_head.weight"]
        if lm_w.dtype != np.float32:
            lm_w = lm_w.astype(np.float32)
        logits = np.float32(x @ lm_w.T)
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
        model_dir = os.path.join(_SCRIPT_DIR, "weights")

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
        weights_dir = os.path.join(_SCRIPT_DIR, "weights")
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
    ssm_qkv_dim = ssm_dim + 2 * (ssm_n_groups * ssm_d_state)  # 32+32=64

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
            # Linear attention (Mamba-2 SSM) layer
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
    ssm_qkv_dim = ssm_dim + 2 * (ssm_n_groups * ssm_d_state)

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
            weights[pfx + "self_attn.q_norm.weight"] = np.ones(
                head_dim, dtype=np.float32)
            weights[pfx + "self_attn.k_norm.weight"] = np.ones(
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
    weights_dir = args.weights_dir or os.path.join(_SCRIPT_DIR, "weights")
    q4_path = os.path.join(weights_dir, "weights_q4.npz")

    if args.quantize:
        # Streaming download + quantize (one shard at a time, low RAM)
        download_and_quantize_streaming(weights_dir)
        return

    # Load weights (prefer quantized if available)
    if os.path.exists(q4_path):
        print(f"Loading quantized weights from {q4_path}...")
        weights = {k: v for k, v in np.load(q4_path).items()}
        quantized = True
    else:
        npz_path, _ = download_qwen35_weights(weights_dir)
        weights = load_weights(npz_path)
        quantized = False

    print(f"Loaded {len(weights)} weight tensors "
          f"({'INT4' if quantized else 'fp32'})")

    tokenizer_path = os.path.join(weights_dir, "tokenizer.json")
    tokenizer = load_tokenizer(tokenizer_path)

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
    print("Model created, kernels compiled")

    if args.profile:
        model.enable_profiling()

    generate(model, args.prompt, tokenizer,
             max_tokens=args.max_tokens,
             temperature=args.temperature)


if __name__ == "__main__":
    main()
