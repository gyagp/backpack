"""
Phi-4 mini inference on WebGPU via Triton.

Demonstrates Microsoft's Phi-4-mini-instruct LLM inference using Triton
kernels compiled to WGSL and executed on WebGPU via Dawn.

Phi-4 mini uses the Phi3 architecture featuring:
  - Partial RoPE (rotary_factor=0.75, only 75% of head_dim gets rotated)
  - RMSNorm (root mean square normalization)
  - GQA (24 Q heads, 8 KV heads)
  - SwiGLU MLP (SiLU-gated linear unit)
  - Fused QKV projection and fused gate_up projection

Optimizations:
  - 4-bit weight quantization (INT4 per-group with fp16 scales)
  - fp16 storage for embeddings/norms (2x RAM reduction)
  - Per-layer weight streaming (only ~96MB GPU per layer vs 384MB fp32)
  - Pre-allocated KV cache, pre-computed RoPE tables
  - Vectorized multi-head attention (no per-head Python loop)

All matrix multiplications, normalization, attention, and activation
operations run as WebGPU compute shaders — no CUDA required.

Usage:
    python models/phi-4/model.py --verify
    python models/phi-4/model.py --quantize   # quantize weights first
    python models/phi-4/model.py --prompt "Hello"

Requirements:
    pip install requests tokenizers
    Dawn WebGPU library built at third_party/webgpu/dawn/build/
"""
import os
import sys
import time
from typing import Dict, Tuple

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_SCRIPT_DIR))

import numpy as np

from common.model_base import WebGPUModel
from common.utils import (
    load_weights, download_weights, load_tokenizer, generate,
    add_device_arg, apply_device_arg,
)


# ---------------------------------------------------------------------------
# 4-bit quantization utilities
# ---------------------------------------------------------------------------

# Group size for block-wise quantization (128 is standard for LLM INT4)
Q4_GROUP_SIZE = 128


def quantize_int4(weight: np.ndarray, group_size: int = Q4_GROUP_SIZE):
    """Quantize a 2D weight matrix to INT4 with per-group fp16 scales/zeros.

    Each group of `group_size` values is independently quantized to 4-bit
    symmetric quantization: val = (q - 8) * scale

    Args:
        weight: (N, K) float32 weight matrix
        group_size: number of elements per quantization group

    Returns:
        q_packed: uint8 array with two INT4 values packed per byte
        scales: fp16 per-group scale factors
        zeros: fp16 per-group zero points
    """
    N, K = weight.shape
    # Pad K to multiple of group_size
    K_pad = ((K + group_size - 1) // group_size) * group_size
    if K_pad != K:
        w = np.zeros((N, K_pad), dtype=np.float32)
        w[:, :K] = weight
    else:
        w = weight.copy()

    n_groups = K_pad // group_size
    w_grouped = w.reshape(N * n_groups, group_size)

    # Symmetric quantization: map [min, max] -> [0, 15]
    w_min = w_grouped.min(axis=1, keepdims=True)
    w_max = w_grouped.max(axis=1, keepdims=True)
    w_range = w_max - w_min
    w_range = np.where(w_range == 0, 1.0, w_range)

    scales = (w_range / 15.0).astype(np.float16)
    zeros = w_min.astype(np.float16)

    # Quantize to 0..15
    scales_f32 = scales.astype(np.float32)
    zeros_f32 = zeros.astype(np.float32)
    q = np.clip(np.round((w_grouped - zeros_f32) / scales_f32), 0, 15).astype(np.uint8)

    # Pack two INT4 values per byte: low nibble = even index, high = odd
    q_even = q[:, 0::2]  # (N*n_groups, group_size//2)
    q_odd = q[:, 1::2]
    q_packed = (q_odd << 4) | q_even  # pack into uint8

    return (q_packed.reshape(N, -1),
            scales.reshape(N, n_groups),
            zeros.reshape(N, n_groups))


def dequantize_int4(q_packed: np.ndarray, scales: np.ndarray,
                    zeros: np.ndarray, K_orig: int,
                    group_size: int = Q4_GROUP_SIZE,
                    dtype=np.float32) -> np.ndarray:
    """Dequantize INT4 packed weights.

    Args:
        q_packed: (N, K_pad//2) uint8 packed INT4
        scales: (N, n_groups) fp16 scale factors
        zeros: (N, n_groups) fp16 zero points
        K_orig: original (unpadded) number of columns
        group_size: quantization group size
        dtype: output dtype (np.float32 or np.float16)

    Returns:
        weight: (N, K_orig) weight matrix in requested dtype
    """
    N = q_packed.shape[0]
    # Unpack: low nibble and high nibble
    q_low = (q_packed & 0x0F).astype(np.float32)
    q_high = (q_packed >> 4).astype(np.float32)
    # Interleave back: even=low, odd=high
    K_pad_half = q_packed.shape[1]
    q = np.empty((N, K_pad_half * 2), dtype=np.float32)
    q[:, 0::2] = q_low
    q[:, 1::2] = q_high

    # Dequantize per group
    n_groups = scales.shape[1]
    q_grouped = q.reshape(N * n_groups, group_size)
    s = scales.reshape(N * n_groups, 1).astype(np.float32)
    z = zeros.reshape(N * n_groups, 1).astype(np.float32)
    w = q_grouped * s + z

    K_pad = n_groups * group_size
    w = w.reshape(N, K_pad)
    w = w[:, :K_orig]
    if dtype != np.float32:
        w = w.astype(dtype)
    return w


def quantize_model_weights(weights: Dict[str, np.ndarray],
                           n_layer: int = 32) -> Dict[str, np.ndarray]:
    """Quantize a Phi-4 model's weights: INT4 for linear, fp16 for others.

    Returns a dict with:
      - Linear weights: '{key}.q4', '{key}.scales', '{key}.zeros', '{key}.K'
      - Other weights: stored as fp16
    """
    quantized = {}
    linear_keys = set()
    for i in range(n_layer):
        pfx = f"layers.{i}."
        linear_keys.update([
            pfx + "self_attn.qkv_proj.weight",
            pfx + "self_attn.o_proj.weight",
            pfx + "mlp.gate_up_proj.weight",
            pfx + "mlp.down_proj.weight",
        ])

    total_orig = 0
    total_quant = 0

    for key, val in weights.items():
        orig_bytes = val.nbytes
        total_orig += orig_bytes

        if key in linear_keys:
            # INT4 quantization for linear projection weights
            w = val.astype(np.float32)
            if w.ndim != 2:
                w = w.reshape(w.shape[0], -1)
            N, K = w.shape
            q_packed, scales, zeros = quantize_int4(w)
            quantized[key + ".q4"] = q_packed
            quantized[key + ".scales"] = scales
            quantized[key + ".zeros"] = zeros
            quantized[key + ".K"] = np.array([K], dtype=np.int32)
            q_bytes = q_packed.nbytes + scales.nbytes + zeros.nbytes
            total_quant += q_bytes
        else:
            # fp16 for everything else (norms, embeddings)
            quantized[key] = val.astype(np.float16)
            total_quant += quantized[key].nbytes

    print(f"  Original: {total_orig / 1024 / 1024:.0f} MB (fp32)")
    print(f"  Quantized: {total_quant / 1024 / 1024:.0f} MB "
          f"(INT4 linear + fp16 other)")
    print(f"  Compression: {total_orig / total_quant:.1f}x")
    return quantized


def load_quantized_weights(path: str) -> Dict[str, np.ndarray]:
    """Load quantized weights from npz (memory-mapped for fast startup)."""
    data = np.load(path, mmap_mode='r')
    return {k: data[k] for k in data.files}


# Phi-4 mini config
PHI4_CONFIGS = {
    "mini": {
        "n_layer": 32, "n_head": 24, "n_kv_heads": 8,
        "n_embd": 3072, "intermediate_size": 8192,
        "n_vocab": 200064, "rope_theta": 10000.0,
        "rms_norm_eps": 1e-5, "head_dim": 128,
        "partial_rotary_factor": 0.75,
        "hf_repo": "microsoft/Phi-4-mini-instruct",
    },
}


class Phi4WebGPU(WebGPUModel):
    """Phi-4 mini inference on WebGPU via Triton kernels.

    Key differences from SmolLM2/LLaMA:
    - Partial RoPE: only rotary_factor (0.75) of head_dim is rotated
    - Fused QKV projection: single weight for Q+K+V
    - Fused gate_up projection: single weight for gate+up

    Optimizations:
    - 4-bit quantized weights (~4x memory reduction for linear layers)
    - fp16 storage for norms/embeddings (~2x reduction)
    - Per-layer weight streaming with on-the-fly dequantization
    - Pre-allocated KV cache (avoids np.concatenate per step)
    - Pre-computed RoPE cos/sin tables
    - Vectorized multi-head attention (no per-head Python loop)
    """

    # Max sequence length for KV cache pre-allocation
    MAX_SEQ_LEN = 2048

    def __init__(self, weights: Dict[str, np.ndarray],
                 n_layer: int = 32, n_head: int = 24,
                 n_kv_heads: int = 8, n_embd: int = 3072,
                 intermediate_size: int = 8192,
                 n_vocab: int = 200064,
                 rope_theta: float = 10000.0,
                 rms_norm_eps: float = 1e-5,
                 head_dim: int = 128,
                 partial_rotary_factor: float = 0.75,
                 max_seq_len: int = 2048,
                 quantized: bool = False,
                 decode_mode: str = 'cpu'):
        self.partial_rotary_factor = partial_rotary_factor
        self.rotary_dim = int(head_dim * partial_rotary_factor)
        self.MAX_SEQ_LEN = max_seq_len
        self._quantized = quantized
        self._decode_mode = decode_mode  # 'cpu' or 'gpu'
        # Fused projection output sizes
        self.qkv_out = n_head * head_dim + 2 * n_kv_heads * head_dim
        self.gate_up_out = 2 * intermediate_size
        super().__init__(
            weights, n_layer=n_layer, n_head=n_head, n_embd=n_embd,
            n_vocab=n_vocab,
            n_kv_heads=n_kv_heads,
            intermediate_size=intermediate_size,
            head_dim=head_dim,
            rope_theta=rope_theta,
            rms_norm_eps=rms_norm_eps,
            k_dimensions={n_embd, intermediate_size,
                          self.qkv_out, self.gate_up_out},
        )
        self._precompute_rope_tables()
        self._upload_weights_to_gpu()
        self._init_gpu_kv_cache()

    def _init_gpu_kv_cache(self):
        """Pre-allocate static GPU KV cache buffers for all layers.

        Each layer gets a pair of GPU buffers:
          K_cache: (MAX_SEQ_LEN, n_kv_heads, head_dim) fp32
          V_cache: same shape

        Total VRAM: 32 layers × 2 × (2048 × 8 × 128 × 4) = 512 MB
        """
        runner = self.cache.runner
        n_kv = self.n_kv_heads
        HD = self.head_dim
        buf_size = self.MAX_SEQ_LEN * n_kv * HD
        self._gpu_kv_cache = {}  # layer -> (K_gpu, V_gpu, cur_len)
        for i in range(self.n_layer):
            k_buf = runner.upload_to_gpu(
                np.zeros(buf_size, dtype=np.float32), f"kv_cache_K_{i}")
            v_buf = runner.upload_to_gpu(
                np.zeros(buf_size, dtype=np.float32), f"kv_cache_V_{i}")
            self._gpu_kv_cache[i] = (k_buf, v_buf, 0)

    def _precompute_rope_tables(self):
        """Pre-compute RoPE cos/sin tables for all positions up to MAX_SEQ_LEN.

        Uploads tables to GPU for GPU-side RoPE during decode.
        """
        rotary_dim = self.rotary_dim
        half_rot = rotary_dim // 2
        inv_freq = 1.0 / (self.rope_theta ** (
            np.arange(0, rotary_dim, 2, dtype=np.float32) / rotary_dim))
        positions = np.arange(self.MAX_SEQ_LEN, dtype=np.float32)
        angles = positions[:, None] * inv_freq[None, :]  # (MAX_SEQ, half_rot)
        self._rope_cos = np.cos(angles).astype(np.float32)  # (MAX_SEQ, half_rot)
        self._rope_sin = np.sin(angles).astype(np.float32)

        # Upload to GPU for GPU-side RoPE
        runner = self.cache.runner
        self._rope_cos_gpu = runner.upload_to_gpu(
            self._rope_cos.ravel(), "rope_cos_table")
        self._rope_sin_gpu = runner.upload_to_gpu(
            self._rope_sin.ravel(), "rope_sin_table")

    def _compile_model_kernels(self):
        """Compile Phi4 kernels: RMSNorm, SiLU*mul, embed gather, argmax."""
        self._compile_rms_norm()
        self._compile_silu_mul()
        self._compile_embed_gather()
        self._compile_argmax()

    def _upload_weights_to_gpu(self):
        """Upload shared weights to GPU. Linear projection weights are
        streamed per-layer during forward to save VRAM.
        """
        E = self.n_embd
        HD = self.head_dim

        # Decide weight path: INT4-on-GPU (fused dequant) or pre-dequant to fp16
        use_q4_gpu = (self._quantized
                      and getattr(self, '_has_fp16_linear', False))
        self._use_q4_gpu = use_q4_gpu

        if self._quantized and not use_q4_gpu:
            # Fallback: pre-dequantize all weights on CPU
            self._dequantize_all_weights()

        # Upload small persistent weights (norms + biases)
        for i in range(self.n_layer):
            pfx = f"layers.{i}."
            for nkey in [pfx + "input_layernorm.weight",
                         pfx + "post_attention_layernorm.weight"]:
                if self.weights[nkey].dtype == np.float16:
                    self.weights[nkey] = self.weights[nkey].astype(np.float32)
                self._upload_norm_weight(nkey)

        # Final RMSNorm
        nkey = "norm.weight"
        if self.weights[nkey].dtype == np.float16:
            self.weights[nkey] = self.weights[nkey].astype(np.float32)
        self._upload_norm_weight(nkey)

        # Embedding — ensure fp32 for CPU lookup and LM head matmul
        ekey = "embed_tokens.weight"
        if self.weights[ekey].dtype == np.float16:
            self.weights[ekey] = self.weights[ekey].astype(np.float32)

        # Upload embed_tokens as fp16 for GPU lm_head (tied weights)
        if getattr(self, '_has_fp16_linear', False):
            self._upload_linear_weight_fp16(ekey, self.n_vocab, E)

        # Upload embed_tokens as fp32 for GPU embedding gather
        runner = self.cache.runner
        wte_fp32 = self.weights[ekey].astype(np.float32).ravel()
        self._gpu_weights["embed_tokens.weight.fp32"] = \
            runner.upload_to_gpu(wte_fp32, "embed_tokens.weight.fp32")

        # Zero biases (tiny)
        self._upload_zero_bias("zero_bias_E", E)
        self._upload_zero_bias("zero_bias_V", self.n_vocab)
        self._upload_zero_bias("zero_bias_QKV", self.qkv_out)
        qo_dim = self.n_head * HD
        if qo_dim != E:
            self._upload_zero_bias("zero_bias_QO", qo_dim)
        self._upload_zero_bias("zero_bias_GU", self.gate_up_out)

        # Eagerly upload all layer weights (eliminates per-layer upload during prefill)
        import time as _t
        t_upload = _t.perf_counter()
        for layer in range(self.n_layer):
            self._upload_layer_weights(layer)
        print(f"  Uploaded all {self.n_layer} layer weights in "
              f"{(_t.perf_counter() - t_upload)*1000:.0f}ms")

        self._print_gpu_weight_stats()

    def _dequantize_all_weights(self):
        """Pre-dequantize all INT4 weights at startup.

        When native f16 is available (DXC), dequantizes to fp16 (~6GB RAM).
        Otherwise dequantizes to fp32 (~12GB RAM).
        Eliminates the per-step dequantization bottleneck.
        """
        import gc
        use_f16 = getattr(self, '_has_fp16_linear', False)
        out_dtype = np.float16 if use_f16 else np.float32
        dtype_name = 'f16' if use_f16 else 'fp32'
        print(f"  Pre-dequantizing INT4 weights to {dtype_name}...")
        for layer in range(self.n_layer):
            pfx = f"layers.{layer}."
            for suffix in ["self_attn.qkv_proj.weight",
                           "self_attn.o_proj.weight",
                           "mlp.gate_up_proj.weight",
                           "mlp.down_proj.weight"]:
                name = pfx + suffix
                q4_key = name + ".q4"
                K_orig = int(self.weights[name + ".K"][0])
                self.weights[name] = dequantize_int4(
                    self.weights[q4_key],
                    self.weights[name + ".scales"],
                    self.weights[name + ".zeros"],
                    K_orig,
                    dtype=out_dtype)
                # Free quantized copies to offset RAM
                del self.weights[q4_key]
                del self.weights[name + ".scales"]
                del self.weights[name + ".zeros"]
                del self.weights[name + ".K"]
        gc.collect()
        self._quantized = False
        print("  Done.")

    def _upload_layer_weights(self, layer: int):
        """Upload linear projection weights for a single layer to GPU.

        Uses INT4 packed weights when available (fused dequant on GPU),
        otherwise fp16 or fp32.
        """
        E = self.n_embd
        HD = self.head_dim
        IM = self.intermediate_size
        pfx = f"layers.{layer}."

        layer_keys = [
            (pfx + "self_attn.qkv_proj.weight", self.qkv_out, E),
            (pfx + "self_attn.o_proj.weight", E, self.n_head * HD),
            (pfx + "mlp.gate_up_proj.weight", self.gate_up_out, E),
            (pfx + "mlp.down_proj.weight", E, IM),
        ]

        use_q4 = getattr(self, '_use_q4_gpu', False)
        use_fp16 = getattr(self, '_has_fp16_linear', False)
        for name, N, K in layer_keys:
            if use_q4:
                q4_gpu_key = name + ".q4.gpu"
                if q4_gpu_key not in self._gpu_weights:
                    self._upload_q4_weight(name, N, K)
            elif use_fp16:
                fp16_name = name + ".fp16"
                if fp16_name not in self._gpu_weights:
                    self._upload_linear_weight_fp16(name, N, K)
            else:
                if name not in self._gpu_weights:
                    self._upload_linear_weight(name, N, K)

    def _release_layer_weights(self, layer: int):
        """Release linear projection weight GPU buffers for a layer."""
        pfx = f"layers.{layer}."
        keys = [
            pfx + "self_attn.qkv_proj.weight",
            pfx + "self_attn.o_proj.weight",
            pfx + "mlp.gate_up_proj.weight",
            pfx + "mlp.down_proj.weight",
        ]
        for k in keys:
            if k in self._gpu_weights:
                del self._gpu_weights[k]

    def _apply_partial_rope_fast(self, x, positions, rotary_dim):
        """Apply partial RoPE using pre-computed tables (no per-call trig)."""
        half_rot = rotary_dim // 2
        cos_v = self._rope_cos[positions]  # (T, half_rot)
        sin_v = self._rope_sin[positions]  # (T, half_rot)
        # Broadcast: (T, 1, half_rot) for (T, n_heads, half_rot)
        cos_v = cos_v[:, None, :]
        sin_v = sin_v[:, None, :]

        x1 = x[..., :half_rot]
        x2 = x[..., half_rot:rotary_dim]
        out = np.empty_like(x)
        out[..., :half_rot] = x1 * cos_v - x2 * sin_v
        out[..., half_rot:rotary_dim] = x2 * cos_v + x1 * sin_v
        out[..., rotary_dim:] = x[..., rotary_dim:]
        return out

    def _proj(self, x, weight_name, bias, out_features, K=None, gpu_out=False):
        """Linear projection: INT4 fused dequant, fp16, or fp32."""
        # INT4 on-GPU path
        q4_key = weight_name + ".q4.gpu"
        if q4_key in self._gpu_weights:
            # DP4A path (W4A8 with dot4I8Packed)
            if getattr(self, '_use_dp4a', False):
                return self._linear_q4_dp4a(
                    x,
                    self._gpu_weights[q4_key],
                    self._gpu_weights[weight_name + ".scales.gpu"],
                    self._gpu_weights[weight_name + ".zeros.gpu"],
                    bias, out_features, K=K, gpu_out=gpu_out)
            return self._linear_q4(
                x,
                self._gpu_weights[q4_key],
                self._gpu_weights[weight_name + ".scales.gpu"],
                self._gpu_weights[weight_name + ".zeros.gpu"],
                bias, out_features, K=K, gpu_out=gpu_out)
        # fp16 path
        if self._has_fp16_linear:
            fp16_name = weight_name + ".fp16"
            return self._linear_fp16w(
                x, self._gpu_weights[fp16_name], bias, out_features,
                K=K, gpu_out=gpu_out)
        else:
            return self._linear(
                x, self._gpu_weights[weight_name], bias, out_features,
                gpu_out=gpu_out)

    def _attention_block(self, x, layer: int,
                         use_cache: bool = False,
                         positions: np.ndarray = None,
                         **kwargs):
        """GQA with partial RoPE and fused QKV (optimized)."""
        from common.model_base import GPUBuffer

        if isinstance(x, GPUBuffer):
            T = x.shape[0] if x.shape else 1
        else:
            T = x.shape[0]
        HD = self.head_dim
        n_head = self.n_head
        n_kv = self.n_kv_heads
        n_rep = self.n_rep
        pfx = f"layers.{layer}.self_attn."

        # Fused QKV projection
        qkv = self._proj(
            x, pfx + "qkv_proj.weight",
            self._gpu_weights["zero_bias_QKV"], self.qkv_out)

        # Split Q, K, V from fused output (views, no copy)
        q_size = n_head * HD
        kv_size = n_kv * HD
        Q = qkv[:, :q_size].reshape(T, n_head, HD)
        K_new = qkv[:, q_size:q_size + kv_size].reshape(T, n_kv, HD)
        V_new = qkv[:, q_size + kv_size:].reshape(T, n_kv, HD)

        # Partial RoPE using pre-computed tables
        if positions is None:
            positions = np.arange(T, dtype=np.int32)
        Q = self._apply_partial_rope_fast(Q, positions, self.rotary_dim)
        K_new = self._apply_partial_rope_fast(K_new, positions, self.rotary_dim)

        # KV cache — pre-allocated, write-in-place
        if use_cache:
            if self.kv_cache is None:
                self.kv_cache = {}
            if layer not in self.kv_cache:
                # Pre-allocate cache buffers
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

            # Sync to GPU KV cache for subsequent GPU decode
            if hasattr(self, '_gpu_kv_cache') and layer in self._gpu_kv_cache:
                import ctypes
                K_gpu, V_gpu, _ = self._gpu_kv_cache[layer]
                runner = self.cache.runner
                k_data = np.ascontiguousarray(K_full, dtype=np.float32)
                v_data = np.ascontiguousarray(V_full, dtype=np.float32)
                runner._lib.wgpuQueueWriteBuffer(
                    runner._queue, K_gpu.handle, 0,
                    k_data.ctypes.data_as(ctypes.c_void_p), k_data.nbytes)
                runner._lib.wgpuQueueWriteBuffer(
                    runner._queue, V_gpu.handle, 0,
                    v_data.ctypes.data_as(ctypes.c_void_p), v_data.nbytes)
                self._gpu_kv_cache[layer] = (K_gpu, V_gpu, T_total)
        else:
            T_total = T
            K_full = K_new
            V_full = V_new

        scale = 1.0 / np.sqrt(HD)

        if T == 1 and use_cache and T_total > 1:
            # Decode: vectorized across all heads (no per-head loop)
            # Q: (1, n_head, HD), K_full: (T_total, n_kv, HD)
            # Expand KV heads: (T_total, n_kv, HD) -> (T_total, n_head, HD)
            K_exp = np.repeat(K_full, n_rep, axis=1)  # (T_total, n_head, HD)
            V_exp = np.repeat(V_full, n_rep, axis=1)

            # Batched scores: (n_head, 1, HD) @ (n_head, HD, T_total) -> (n_head, 1, T_total)
            Q_t = Q.transpose(1, 0, 2)          # (n_head, 1, HD)
            K_t = K_exp.transpose(1, 2, 0)      # (n_head, HD, T_total)
            scores = np.float32((Q_t @ K_t).squeeze(1) * scale)  # (n_head, T_total)

            scores -= scores.max(axis=1, keepdims=True)
            exp_s = np.exp(scores)
            s = exp_s.sum(axis=1, keepdims=True)
            s = np.where(s == 0, 1.0, s)
            attn = exp_s / s
            # (n_head, 1, T_total) @ (n_head, T_total, HD) -> (n_head, 1, HD)
            V_t = V_exp.transpose(1, 0, 2)      # (n_head, T_total, HD)
            attn_out = (attn[:, None, :] @ V_t).squeeze(1)  # (n_head, HD)
            attn_out = attn_out[None, :, :].astype(np.float32)  # (1, n_head, HD)
        else:
            # Prefill: multi-head causal attention (single GPU dispatch)
            attn_out = self._causal_attention_multihead(
                Q, K_full, V_full, n_rep)

        attn_flat = attn_out.reshape(T, n_head * HD)
        o_bias = self._gpu_weights.get("zero_bias_QO",
                                        self._gpu_weights["zero_bias_E"])
        return self._proj(
            attn_flat, pfx + "o_proj.weight",
            o_bias, self.n_embd)

    def _mlp_block(self, x, layer: int, gpu_out: bool = False):
        """SwiGLU MLP with fused gate_up projection."""
        from common.model_base import GPUBuffer
        E = self.n_embd
        IM = self.intermediate_size
        pfx = f"layers.{layer}.mlp."

        # Fused gate_up: (T, E) → (T, 2*IM)
        x_is_gpu = isinstance(x, GPUBuffer)
        T = (x.shape[0] if x.shape else 1) if x_is_gpu else x.shape[0]

        if T > 1 or x_is_gpu:
            # Fused path: keep gate_up on GPU, single dispatch for silu_mul
            gate_up = self._proj(
                x, pfx + "gate_up_proj.weight",
                self._gpu_weights["zero_bias_GU"], self.gate_up_out,
                gpu_out=True)
            h = self._silu_mul_fused(gate_up, IM, gpu_out=True)
        else:
            # T=1, numpy input: use separate gate/up path
            gate_up = self._proj(
                x, pfx + "gate_up_proj.weight",
                self._gpu_weights["zero_bias_GU"], self.gate_up_out)
            gate = np.ascontiguousarray(gate_up[:, :IM])
            up = np.ascontiguousarray(gate_up[:, IM:])
            h = self._silu_mul(gate, up, gpu_out=True)

        return self._proj(
            h, pfx + "down_proj.weight",
            self._gpu_weights["zero_bias_E"], E, K=IM,
            gpu_out=gpu_out)

    def _rms_norm_cpu(self, x, w_name):
        """CPU-side RMSNorm for T=1 — avoids GPU kernel launch overhead."""
        w = self.weights[w_name]
        rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + self.rms_norm_eps)
        return (x / rms * w).astype(np.float32)

    def _cpu_matmul(self, x, weight_name, N, K=None):
        """CPU matmul for T=1 decode: (1,K) @ (N,K).T → (1,N).

        Avoids GPU dispatch overhead (~11ms/dispatch) by doing the
        memory-bandwidth-bound vector-matrix product on CPU (~1-4ms).
        """
        W = self.weights[weight_name]
        if K is None:
            K = W.shape[1] if W.ndim == 2 else W.size // N
        W = W.reshape(N, K)
        return np.float32(x @ W.T)

    def _decode_cpu(self, x, layer, positions, pfx, p):
        """Fully CPU decode: all matmuls on CPU via numpy BLAS.

        Eliminates 4 GPU round-trips per layer. Best when GPU dispatch
        overhead is high (>5ms), e.g., D3D12 backend.
        ~7ms/layer on DDR5 at 46 GB/s.
        """
        from common.model_base import GPUBuffer
        if isinstance(x, GPUBuffer):
            x_np = self.cache.runner.readback(x).reshape(1, self.n_embd)
        else:
            x_np = x


        if self._profiling: self._begin_cpu("rms_norm_1")
        rn1 = self._rms_norm_cpu(x_np, pfx + "input_layernorm.weight")
        if self._profiling: self._end_cpu("rms_norm_1")

        if self._profiling: self._begin_cpu("qkv_linear")
        qkv = self._cpu_matmul(rn1, pfx + "self_attn.qkv_proj.weight",
                               self.qkv_out, self.n_embd)
        if self._profiling: self._end_cpu("qkv_linear")

        if self._profiling: self._begin_cpu("attention_cpu")
        attn_out = self._decode_attention(qkv, layer, positions)
        if self._profiling: self._end_cpu("attention_cpu")

        if self._profiling: self._begin_cpu("o_proj")
        o_out = self._cpu_matmul(attn_out, pfx + "self_attn.o_proj.weight",
                                 self.n_embd, self.n_head * self.head_dim)
        if self._profiling: self._end_cpu("o_proj")

        if self._profiling: self._begin_cpu("res_add_norm")
        x_np = x_np + o_out
        rn2 = self._rms_norm_cpu(x_np, pfx + "post_attention_layernorm.weight")
        if self._profiling: self._end_cpu("res_add_norm")

        if self._profiling: self._begin_cpu("gate_up")
        gate_up = self._cpu_matmul(rn2, pfx + "mlp.gate_up_proj.weight",
                                   self.gate_up_out, self.n_embd)
        if self._profiling: self._end_cpu("gate_up")

        if self._profiling: self._begin_cpu("silu_mul_cpu")
        IM = self.intermediate_size
        gate = gate_up[0, :IM]
        up = gate_up[0, IM:]
        h = (gate / (1.0 + np.exp(-gate)) * up).reshape(1, IM)
        if self._profiling: self._end_cpu("silu_mul_cpu")

        if self._profiling: self._begin_cpu("down_proj")
        mlp_out = self._cpu_matmul(h, pfx + "mlp.down_proj.weight",
                                   self.n_embd, IM)
        if self._profiling: self._end_cpu("down_proj")

        return x_np + mlp_out

    def _prefill_gpu(self, x, layer, positions, pfx, p):
        """GPU-resident prefill: all ops on GPU, no per-layer readbacks."""
        from common.model_base import GPUBuffer

        x_is_gpu = isinstance(x, GPUBuffer)
        T = (x.shape[0] if x.shape else 1) if x_is_gpu else x.shape[0]
        HD = self.head_dim
        n_head = self.n_head
        n_kv = self.n_kv_heads
        BHD = self._attn_bhd
        half_rot = self.rotary_dim // 2


        # Ensure x is on GPU (first layer gets numpy embedding)
        if not x_is_gpu:
            x = self.cache.runner.upload_to_gpu(
                x.ravel().astype(np.float32), '__prefill_residual')
            x.shape = (T, self.n_embd)

        # 1. RMSNorm
        if self._profiling: self._set_gpu_op(f"L{layer}/norm1")
        rn1 = self._rms_norm(
            x, self._gpu_weights[pfx + "input_layernorm.weight"],
            gpu_out=True)

        # 2. QKV projection (stays on GPU)
        if self._profiling: self._set_gpu_op(f"L{layer}/qkv")
        qkv = self._proj(rn1, pfx + "self_attn.qkv_proj.weight",
                         self._gpu_weights["zero_bias_QKV"], self.qkv_out,
                         gpu_out=True)

        # Pre-build per-token cos/sin tables (same for all layers)
        if not hasattr(self, '_prefill_cos_cache') or \
                self._prefill_cos_cache_key != tuple(positions):
            self._prefill_cos = np.ascontiguousarray(
                self._rope_cos[positions], dtype=np.float32).ravel()
            self._prefill_sin = np.ascontiguousarray(
                self._rope_sin[positions], dtype=np.float32).ravel()
            self._prefill_cos_cache_key = tuple(positions)
        cos_data = self._prefill_cos
        sin_data = self._prefill_sin

        # 3. GPU RoPE Q
        if self._profiling: self._set_gpu_op(f"L{layer}/rope_q")
        q_rope_out = self.cache.run(
            self._rope_prefill_result, grid=(T, n_head),
            buffers={
                'X': qkv, 'Y': np.zeros(T * n_head * BHD, dtype=np.float32),
                'Cos': cos_data, 'Sin': sin_data,
            },
            scalars={
                'x_offset': 0, 'x_stride_t': self.qkv_out,
                'y_stride_t': n_head * BHD, 'half_rot': half_rot,
            },
            gpu_outputs={'Y'})
        q_rope = q_rope_out['Y']
        q_rope.shape = (T, n_head, BHD)

        # 4. GPU RoPE K + V scatter to KV cache
        if self._profiling: self._set_gpu_op(f"L{layer}/rope_kv")
        K_cache_gpu, V_cache_gpu, _ = self._gpu_kv_cache[layer]
        self.cache.run(
            self._rope_kv_prefill_result, grid=(T, n_kv),
            buffers={
                'QKV': qkv,
                'K_cache': K_cache_gpu, 'V_cache': V_cache_gpu,
                'Cos': cos_data, 'Sin': sin_data,
            },
            scalars={
                'q_size': n_head * HD, 'kv_size': n_kv * HD,
                'qkv_stride_t': self.qkv_out,
                'cache_stride_t': n_kv * HD,
                'half_rot': half_rot,
            },
            gpu_outputs={'K_cache', 'V_cache'})
        self._gpu_kv_cache[layer] = (K_cache_gpu, V_cache_gpu, T)

        # 5. Multi-head causal attention
        if self._profiling: self._set_gpu_op(f"L{layer}/attn")
        scale = float(1.0 / np.sqrt(HD))
        attn_result = self.cache.run(
            self._mh_attn_result, grid=(T, n_head),
            buffers={
                'Q': q_rope,
                'K': K_cache_gpu, 'V': V_cache_gpu,
                'Out': np.zeros(T * n_head * BHD, dtype=np.float32),
            },
            scalars={
                'stride_q_t': n_head * BHD, 'stride_q_h': BHD,
                'stride_k_t': n_kv * BHD, 'stride_k_h': BHD,
                'stride_v_t': n_kv * BHD, 'stride_v_h': BHD,
                'stride_o_t': n_head * BHD, 'stride_o_h': BHD,
                'n_rep': self.n_rep, 'scale': scale,
                'neg_inf': float(-1e9),
            },
            gpu_outputs={'Out'})
        attn_gpu = attn_result['Out']
        attn_gpu.shape = (T, n_head * HD)

        # 6. O projection
        if self._profiling: self._set_gpu_op(f"L{layer}/o_proj")
        o_bias = self._gpu_weights.get("zero_bias_QO",
                                        self._gpu_weights["zero_bias_E"])
        o_out = self._proj(attn_gpu, pfx + "self_attn.o_proj.weight",
                           o_bias, self.n_embd, gpu_out=True)

        # 7. Residual add (in-place to avoid toggle pool aliasing)
        if self._profiling: self._set_gpu_op(f"L{layer}/res1")
        self._add_inplace(x, o_out)

        # 8. RMSNorm
        if self._profiling: self._set_gpu_op(f"L{layer}/norm2")
        rn2 = self._rms_norm(
            x, self._gpu_weights[pfx + "post_attention_layernorm.weight"],
            gpu_out=True)

        # 9. MLP (fused gate_up + silu_mul + down)
        if self._profiling: self._set_gpu_op(f"L{layer}/mlp")
        mlp = self._mlp_block(rn2, layer, gpu_out=True)

        # 10. Residual add (in-place)
        if self._profiling: self._set_gpu_op(f"L{layer}/res2")
        self._add_inplace(x, mlp)

        if self._profiling: self._clear_gpu_op()
        return x

    # ------------------------------------------------------------------
    # Fast decode path: pre-recorded bind groups & batched dispatch
    # ------------------------------------------------------------------

    def _init_fast_decode_dp4a(self):
        """Pre-allocate buffers and bind groups for DP4A fast decode.

        Same structure as _init_fast_decode but uses the WGSL Q4 kernel
        instead of Triton Q4.  The WGSL kernel has 7 bindings:
        (X, W_Q4, Scales, Zeros, Bias, Y, _params_) with workgroup_size=32.
        """
        import struct
        from common.wgsl_kernels import WGSL_Q4_DP4A_KERNEL, Q4_DP4A_BINDINGS, pack_dp4a_params, TILE_N
        runner = self.cache.runner

        # Ensure all layer weights are on GPU before building bind groups
        for layer in range(self.n_layer):
            self._upload_layer_weights(layer)

        E = self.n_embd
        HD = self.head_dim
        n_head = self.n_head
        n_kv = self.n_kv_heads
        n_rep = self.n_rep
        IM = self.intermediate_size
        q_size = n_head * HD
        kv_size = n_kv * HD
        qkv_N = self.qkv_out
        half_rot = self.rotary_dim // 2
        Q4_GS = 128

        FMTS = {'i32': '<i', 'u32': '<I', 'f32': '<f'}

        def pack(param_fields, vals):
            data = bytearray()
            for pf in param_fields:
                data.extend(struct.pack(
                    FMTS.get(pf.wgsl_type, '<i'), vals.get(pf.name, 0)))
            while len(data) < 16:
                data.extend(b'\x00')
            return bytes(data)

        def mkbuf(name, sz):
            h = runner.create_gpu_buffer(f"__fd_{name}", sz)
            return (h, sz)

        # Intermediate buffers (same as regular fast decode)
        norm_out = mkbuf('norm_out', E * 4)
        qkv_out = mkbuf('qkv_out', qkv_N * 4)
        q_rot   = mkbuf('q_rot', q_size * 4)
        attn_out = mkbuf('attn_out', q_size * 4)
        proj_out = mkbuf('proj_out', E * 4)
        gate_up_out = mkbuf('gate_up', 2 * IM * 4)
        silu_out = mkbuf('silu_out', IM * 4)
        rstd     = mkbuf('rstd', 16)
        x_buf    = mkbuf('x', E * 4)

        # Pipelines
        def get_pl(r):
            return runner.get_pipeline_info(
                r.wgsl, r.buffer_bindings, r.param_fields)

        pl_rn,   bgl_rn   = get_pl(self._rn_result)
        pl_rope, bgl_rope = get_pl(self._rope_result)
        pl_rkv,  bgl_rkv  = get_pl(self._rope_kv_result)
        pl_attn, bgl_attn = get_pl(self._gqa_attn_result)
        pl_aip,  bgl_aip  = get_pl(self._add_ip_result)
        pl_silu, bgl_silu = get_pl(self._smf_result)

        # WGSL Q4 pipeline (replaces Triton Q4)
        pl_wq4, bgl_wq4 = runner.get_pipeline_info(
            WGSL_Q4_DP4A_KERNEL, Q4_DP4A_BINDINGS, [])
        nbb_wq4 = len(Q4_DP4A_BINDINGS)

        nbb_rn   = len(self._rn_result.buffer_bindings)
        nbb_rope = len(self._rope_result.buffer_bindings)
        nbb_rkv  = len(self._rope_kv_result.buffer_bindings)
        nbb_attn = len(self._gqa_attn_result.buffer_bindings)
        nbb_aip  = len(self._add_ip_result.buffer_bindings)
        nbb_silu = len(self._smf_result.buffer_bindings)

        # Constant params
        def mk_params(name, pf, vals):
            b = pack(pf, vals)
            h = mkbuf(name, len(b))
            runner.write_buffer(h[0], b)
            return h

        rn_p = mk_params('rn_p', self._rn_result.param_fields,
                          {'stride': E, 'N': E, 'eps': self.rms_norm_eps})

        # WGSL Q4 params — one per projection shape (packed as raw bytes)
        def mk_wq4_params(name, K_val, N_val):
            data = pack_dp4a_params(K_val, K_val // 8, K_val // Q4_GS, N_val)
            h = mkbuf(name, len(data) * 4)
            runner.write_buffer(h[0], data.tobytes())
            return h

        wq4_qkv_p    = mk_wq4_params('wq4_qkv_p', E, qkv_N)
        wq4_oproj_p  = mk_wq4_params('wq4_oproj_p', q_size, E)
        wq4_gateup_p = mk_wq4_params('wq4_gateup_p', E, 2 * IM)
        wq4_down_p   = mk_wq4_params('wq4_down_p', IM, E)

        aip_p = mk_params('aip_p', self._add_ip_result.param_fields, {'N': E})
        silu_p = mk_params('silu_p', self._smf_result.param_fields, {'N': IM})

        # Dynamic params (updated per token) — same as regular fast decode
        rope_q_ph = mkbuf('ropeq_p', 16)
        rope_kv_ph = mkbuf('ropekv_p', 20)
        attn_ph = mkbuf('attn_p', 20)

        cos_h, cos_sz = self._rope_cos_gpu.handle, self._rope_cos_gpu.size
        sin_h, sin_sz = self._rope_sin_gpu.handle, self._rope_sin_gpu.size

        bias_qkv = self._gpu_weights["zero_bias_QKV"]
        bias_e   = self._gpu_weights["zero_bias_E"]
        bias_gu  = self._gpu_weights["zero_bias_GU"]
        bias_v   = self._gpu_weights["zero_bias_V"]
        o_bias   = self._gpu_weights.get("zero_bias_QO", bias_e)
        norm_final_w = self._gpu_weights["norm.weight"]
        lm_w = self._gpu_weights["embed_tokens.weight.fp16"]

        mk_bg = runner.create_bind_group

        # Shared bind groups
        bg_rope_q = mk_bg(bgl_rope, [
            (0, qkv_out[0], qkv_out[1]),
            (1, q_rot[0], q_rot[1]),
            (2, cos_h, cos_sz), (3, sin_h, sin_sz),
            (nbb_rope, rope_q_ph[0], rope_q_ph[1])])
        bg_silu = mk_bg(bgl_silu, [
            (0, gate_up_out[0], gate_up_out[1]),
            (1, silu_out[0], silu_out[1]),
            (nbb_silu, silu_p[0], silu_p[1])])
        bg_res = mk_bg(bgl_aip, [
            (0, x_buf[0], x_buf[1]),
            (1, proj_out[0], proj_out[1]),
            (nbb_aip, aip_p[0], aip_p[1])])

        aip_g = ((E + self._add_block - 1) // self._add_block,)
        silu_g = ((IM + self._smf_block - 1) // self._smf_block,)

        # Final norm + lm_head
        bg_final_rn = mk_bg(bgl_rn, [
            (0, x_buf[0], x_buf[1]),
            (1, norm_out[0], norm_out[1]),
            (2, norm_final_w.handle, norm_final_w.size),
            (3, rstd[0], rstd[1]),
            (nbb_rn, rn_p[0], rn_p[1])])

        logits_buf = mkbuf('logits', self.n_vocab * 4)
        pl_lmh, bgl_lmh = get_pl(self._linear_fp16w_wide_result)
        nbb_lmh = len(self._linear_fp16w_wide_result.buffer_bindings)
        lm_gy = self.MAX_DISPATCH_DIM
        lm_gx = (self.n_vocab + lm_gy - 1) // lm_gy
        lmh_p = mk_params('lmh_p', self._linear_fp16w_wide_result.param_fields,
                           {'K': E, 'stride_x': E, 'stride_w': E,
                            'N': self.n_vocab, 'grid_y': lm_gy})
        bg_lmh = mk_bg(bgl_lmh, [
            (0, norm_out[0], norm_out[1]),
            (1, lm_w.handle, lm_w.size),
            (2, bias_v.handle, bias_v.size),
            (3, logits_buf[0], logits_buf[1]),
            (nbb_lmh, lmh_p[0], lmh_p[1])])

        # Per-layer bind groups using WGSL Q4 kernel
        layer_dispatches = []
        for layer in range(self.n_layer):
            pfx = f"layers.{layer}."
            def w(suffix):
                return self._gpu_weights[pfx + suffix]

            qkv_wq = w("self_attn.qkv_proj.weight.q4.gpu")
            qkv_sc = w("self_attn.qkv_proj.weight.scales.gpu")
            qkv_zr = w("self_attn.qkv_proj.weight.zeros.gpu")
            op_wq  = w("self_attn.o_proj.weight.q4.gpu")
            op_sc  = w("self_attn.o_proj.weight.scales.gpu")
            op_zr  = w("self_attn.o_proj.weight.zeros.gpu")
            gu_wq  = w("mlp.gate_up_proj.weight.q4.gpu")
            gu_sc  = w("mlp.gate_up_proj.weight.scales.gpu")
            gu_zr  = w("mlp.gate_up_proj.weight.zeros.gpu")
            dn_wq  = w("mlp.down_proj.weight.q4.gpu")
            dn_sc  = w("mlp.down_proj.weight.scales.gpu")
            dn_zr  = w("mlp.down_proj.weight.zeros.gpu")
            n1w = w("input_layernorm.weight")
            n2w = w("post_attention_layernorm.weight")
            Kc, Vc, _ = self._gpu_kv_cache[layer]

            # WGSL Q4 bind groups: (X, W_Q4, Scales, Zeros, Bias, Y, _params_)
            bg_n1 = mk_bg(bgl_rn, [
                (0, x_buf[0], x_buf[1]),
                (1, norm_out[0], norm_out[1]),
                (2, n1w.handle, n1w.size),
                (3, rstd[0], rstd[1]),
                (nbb_rn, rn_p[0], rn_p[1])])
            bg_qkv = mk_bg(bgl_wq4, [
                (0, norm_out[0], norm_out[1]),      # X
                (1, qkv_wq.handle, qkv_wq.size),   # W_Q4
                (2, qkv_sc.handle, qkv_sc.size),   # Scales
                (3, qkv_zr.handle, qkv_zr.size),   # Zeros
                (4, bias_qkv.handle, bias_qkv.size), # Bias
                (5, qkv_out[0], qkv_out[1]),        # Y
                (6, wq4_qkv_p[0], wq4_qkv_p[1])])  # _params_
            bg_rkv = mk_bg(bgl_rkv, [
                (0, qkv_out[0], qkv_out[1]),
                (1, Kc.handle, Kc.size),
                (2, Vc.handle, Vc.size),
                (3, cos_h, cos_sz), (4, sin_h, sin_sz),
                (nbb_rkv, rope_kv_ph[0], rope_kv_ph[1])])
            bg_att = mk_bg(bgl_attn, [
                (0, q_rot[0], q_rot[1]),
                (1, Kc.handle, Kc.size),
                (2, Vc.handle, Vc.size),
                (3, attn_out[0], attn_out[1]),
                (nbb_attn, attn_ph[0], attn_ph[1])])
            bg_op = mk_bg(bgl_wq4, [
                (0, attn_out[0], attn_out[1]),
                (1, op_wq.handle, op_wq.size),
                (2, op_sc.handle, op_sc.size),
                (3, op_zr.handle, op_zr.size),
                (4, o_bias.handle, o_bias.size),
                (5, proj_out[0], proj_out[1]),
                (6, wq4_oproj_p[0], wq4_oproj_p[1])])
            bg_n2 = mk_bg(bgl_rn, [
                (0, x_buf[0], x_buf[1]),
                (1, norm_out[0], norm_out[1]),
                (2, n2w.handle, n2w.size),
                (3, rstd[0], rstd[1]),
                (nbb_rn, rn_p[0], rn_p[1])])
            bg_gu = mk_bg(bgl_wq4, [
                (0, norm_out[0], norm_out[1]),
                (1, gu_wq.handle, gu_wq.size),
                (2, gu_sc.handle, gu_sc.size),
                (3, gu_zr.handle, gu_zr.size),
                (4, bias_gu.handle, bias_gu.size),
                (5, gate_up_out[0], gate_up_out[1]),
                (6, wq4_gateup_p[0], wq4_gateup_p[1])])
            bg_dn = mk_bg(bgl_wq4, [
                (0, silu_out[0], silu_out[1]),
                (1, dn_wq.handle, dn_wq.size),
                (2, dn_sc.handle, dn_sc.size),
                (3, dn_zr.handle, dn_zr.size),
                (4, bias_e.handle, bias_e.size),
                (5, proj_out[0], proj_out[1]),
                (6, wq4_down_p[0], wq4_down_p[1])])

            # WGSL Q4 grid: (T=1, ceil(N/TILE_N)) with workgroup_size=256
            layer_dispatches.append([
                (pl_rn,   bg_n1,    (1,)),
                (pl_wq4,  bg_qkv,   (1, (qkv_N + TILE_N - 1) // TILE_N)),
                (pl_rope, bg_rope_q, (n_head,)),
                (pl_rkv,  bg_rkv,   (n_kv,)),
                (pl_attn, bg_att,   (n_head,)),
                (pl_wq4,  bg_op,    (1, (E + TILE_N - 1) // TILE_N)),
                (pl_aip,  bg_res,   aip_g),
                (pl_rn,   bg_n2,    (1,)),
                (pl_wq4,  bg_gu,    (1, (2 * IM + TILE_N - 1) // TILE_N)),
                (pl_silu, bg_silu,  silu_g),
                (pl_wq4,  bg_dn,    (1, (E + TILE_N - 1) // TILE_N)),
                (pl_aip,  bg_res,   aip_g),
            ])

        final_dispatches = [
            (pl_rn,  bg_final_rn, (1,)),
            (pl_lmh, bg_lmh,     (lm_gx, lm_gy)),
        ]

        self._fd_x_h = x_buf[0]
        self._fd_logits_h = logits_buf[0]
        self._fd_logits_sz = logits_buf[1]
        self._fd_rope_q_ph = rope_q_ph[0]
        self._fd_rope_kv_ph = rope_kv_ph[0]
        self._fd_attn_ph = attn_ph[0]
        self._fd_all_batches = layer_dispatches + [final_dispatches]

        self._fd_rq_buf = bytearray(16)
        struct.pack_into('<i', self._fd_rq_buf, 0, 0)
        struct.pack_into('<i', self._fd_rq_buf, 8, half_rot)
        self._fd_rkv_buf = bytearray(20)
        struct.pack_into('<ii', self._fd_rkv_buf, 0, q_size, kv_size)
        struct.pack_into('<i', self._fd_rkv_buf, 12, half_rot)
        self._fd_attn_buf = bytearray(20)
        struct.pack_into('<ii', self._fd_attn_buf, 0, n_kv * HD, n_rep)
        struct.pack_into('<ff', self._fd_attn_buf, 12,
            float(np.float32(1.0 / np.sqrt(HD))),
            float(np.float32(-1e9)))

        self._fast_decode_ready = True

    def _free_cpu_weights(self):
        """Free CPU copies of weights after GPU upload.

        All layer weights and embedding are on GPU; CPU copies waste RAM.
        Keeps only tokenizer-related data.
        """
        import gc
        freed = 0
        keys_to_del = []
        for name in list(self.weights.keys()):
            if name in ("embed_tokens.weight",):
                continue  # Keep for CPU fallback paths
            keys_to_del.append(name)
        for name in keys_to_del:
            freed += self.weights[name].nbytes
            del self.weights[name]
        gc.collect()
        if freed > 0:
            print(f"  Freed {freed / 1024 / 1024:.0f} MB CPU weight copies")

    def _warmup_fast_decode(self):
        """Eagerly initialize fast decode pipeline so that init cost
        is excluded from the decode timer in generate()."""
        if getattr(self, '_fast_decode_ready', False):
            return
        if self._decode_mode != 'gpu' or not getattr(self, '_use_q4_gpu', False):
            return
        if getattr(self, '_use_dp4a', False):
            self._init_fast_decode_dp4a()
        else:
            self._init_fast_decode()
        # Free CPU weight copies — all weights are now on GPU
        self._free_cpu_weights()
        print("  Fast decode pipeline ready")

    def _init_fast_decode(self):
        """Pre-allocate buffers, bind groups, and params for fast decode.

        Called once on the first decode token.  Pre-creates everything
        needed for 32 layers × 12 ops so that per-token dispatch is
        just: update 3 params buffers → submit pre-recorded dispatches.
        """
        import struct
        runner = self.cache.runner

        # Ensure all layer weights are on GPU before building bind groups
        for layer in range(self.n_layer):
            self._upload_layer_weights(layer)

        E = self.n_embd           # 3072
        HD = self.head_dim         # 128
        n_head = self.n_head       # 24
        n_kv = self.n_kv_heads     # 8
        n_rep = self.n_rep         # 3
        IM = self.intermediate_size  # 8192
        q_size = n_head * HD       # 3072
        kv_size = n_kv * HD        # 1024
        qkv_N = self.qkv_out      # 5120
        half_rot = self.rotary_dim // 2
        Q4_GS = 128               # quantization group size

        FMTS = {'i32': '<i', 'u32': '<I', 'f32': '<f'}

        def pack(param_fields, vals):
            data = bytearray()
            for pf in param_fields:
                data.extend(struct.pack(
                    FMTS.get(pf.wgsl_type, '<i'), vals.get(pf.name, 0)))
            while len(data) < 16:
                data.extend(b'\x00')
            return bytes(data)

        def mkbuf(name, sz):
            h = runner.create_gpu_buffer(f"__fd_{name}", sz)
            return (h, sz)

        # --- 1. Intermediate buffers (shared across layers) ---
        norm_out = mkbuf('norm_out', E * 4)
        qkv_out = mkbuf('qkv_out', qkv_N * 4)
        q_rot   = mkbuf('q_rot',   q_size * 4)
        attn_out = mkbuf('attn_out', q_size * 4)
        proj_out = mkbuf('proj_out', E * 4)       # o_proj & down reuse
        gate_up_out = mkbuf('gate_up', 2 * IM * 4)
        silu_out = mkbuf('silu_out', IM * 4)
        rstd     = mkbuf('rstd', 16)
        x_buf    = mkbuf('x', E * 4)

        # --- 2. Pipelines and layouts ---
        def get_pl(r):
            return runner.get_pipeline_info(
                r.wgsl, r.buffer_bindings, r.param_fields)

        pl_rn,   bgl_rn   = get_pl(self._rn_result)
        pl_rope, bgl_rope = get_pl(self._rope_result)
        pl_rkv,  bgl_rkv  = get_pl(self._rope_kv_result)
        pl_attn, bgl_attn = get_pl(self._gqa_attn_result)
        pl_aip,  bgl_aip  = get_pl(self._add_ip_result)
        pl_silu, bgl_silu = get_pl(self._smf_result)

        # Use WGSL Q4 kernel for matmul (faster than Triton Q4:
        # TILE_N=8 multi-output, native subgroupAdd, fewer workgroups)
        from common.wgsl_kernels import WGSL_Q4_DP4A_KERNEL, Q4_DP4A_BINDINGS, pack_dp4a_params, TILE_N
        pl_q4, bgl_q4 = runner.get_pipeline_info(
            WGSL_Q4_DP4A_KERNEL, Q4_DP4A_BINDINGS, [])

        # Fused residual+RMSNorm kernel: x += residual; y = rms_norm(x)
        # Saves 1 dispatch per layer (res1 + norm2 → single kernel)
        has_add_rn = True
        pl_arn, bgl_arn = get_pl(self._add_rn_result)
        nbb_arn = len(self._add_rn_result.buffer_bindings)

        # Fused RoPE Q + K scatter + V copy (2 dispatches → 1)
        pl_fused_rope, bgl_fused_rope = get_pl(self._fused_rope_result)
        nbb_fused_rope = len(self._fused_rope_result.buffer_bindings)

        nbb_rn   = len(self._rn_result.buffer_bindings)
        nbb_rope = len(self._rope_result.buffer_bindings)
        nbb_rkv  = len(self._rope_kv_result.buffer_bindings)
        nbb_attn = len(self._gqa_attn_result.buffer_bindings)
        nbb_aip  = len(self._add_ip_result.buffer_bindings)
        nbb_silu = len(self._smf_result.buffer_bindings)

        # --- 3. Constant params buffers ---
        def mk_params(name, pf, vals):
            b = pack(pf, vals)
            h = mkbuf(name, len(b))
            runner.write_buffer(h[0], b)
            return h

        rn_p = mk_params('rn_p', self._rn_result.param_fields,
                          {'stride': E, 'N': E, 'eps': self.rms_norm_eps})

        # Fused add+norm params 
        arn_p = mk_params('arn_p', self._add_rn_result.param_fields,
                          {'stride': E, 'N': E, 'eps': self.rms_norm_eps})

        # WGSL Q4 params — one per projection shape
        def mk_wq4_params(name, K_val, N_val):
            data = pack_dp4a_params(K_val, K_val // 8, K_val // Q4_GS, N_val)
            h = mkbuf(name, len(data) * 4)
            runner.write_buffer(h[0], data.tobytes())
            return h

        qkv_p    = mk_wq4_params('wq4_qkv_p', E, qkv_N)
        oproj_p  = mk_wq4_params('wq4_oproj_p', q_size, E)
        gateup_p = mk_wq4_params('wq4_gateup_p', E, 2 * IM)
        down_p   = mk_wq4_params('wq4_down_p', IM, E)
        aip_p = mk_params('aip_p', self._add_ip_result.param_fields,
                           {'N': E})
        silu_p = mk_params('silu_p', self._smf_result.param_fields,
                           {'N': IM})

        # Dynamic params (updated per token)
        rope_q_ph = mkbuf('ropeq_p', 16)
        rope_kv_ph = mkbuf('ropekv_p', 20)
        attn_ph = mkbuf('attn_p', 20)
        fused_rope_ph = mkbuf('fused_rope_p', 24)

        # --- 4. Global buffer handles ---
        cos_h, cos_sz = self._rope_cos_gpu.handle, self._rope_cos_gpu.size
        sin_h, sin_sz = self._rope_sin_gpu.handle, self._rope_sin_gpu.size

        bias_qkv = self._gpu_weights["zero_bias_QKV"]
        bias_e   = self._gpu_weights["zero_bias_E"]
        bias_gu  = self._gpu_weights["zero_bias_GU"]
        bias_v   = self._gpu_weights["zero_bias_V"]
        o_bias   = self._gpu_weights.get("zero_bias_QO", bias_e)
        norm_final_w = self._gpu_weights["norm.weight"]
        lm_w = self._gpu_weights["embed_tokens.weight.fp16"]

        # --- 5. Bind groups ---
        mk_bg = runner.create_bind_group

        # Shared (same for all layers)
        bg_rope_q = mk_bg(bgl_rope, [
            (0, qkv_out[0], qkv_out[1]),
            (1, q_rot[0], q_rot[1]),
            (2, cos_h, cos_sz), (3, sin_h, sin_sz),
            (nbb_rope, rope_q_ph[0], rope_q_ph[1])])
        bg_silu = mk_bg(bgl_silu, [
            (0, gate_up_out[0], gate_up_out[1]),
            (1, silu_out[0], silu_out[1]),
            (nbb_silu, silu_p[0], silu_p[1])])
        bg_res = mk_bg(bgl_aip, [
            (0, x_buf[0], x_buf[1]),
            (1, proj_out[0], proj_out[1]),
            (nbb_aip, aip_p[0], aip_p[1])])

        # Grids
        aip_g = ((E + self._add_block - 1) // self._add_block,)
        silu_g = ((IM + self._smf_block - 1) // self._smf_block,)

        # Final norm + lm_head bind groups
        bg_final_rn = mk_bg(bgl_rn, [
            (0, x_buf[0], x_buf[1]),
            (1, norm_out[0], norm_out[1]),
            (2, norm_final_w.handle, norm_final_w.size),
            (3, rstd[0], rstd[1]),
            (nbb_rn, rn_p[0], rn_p[1])])

        logits_buf = mkbuf('logits', self.n_vocab * 4)

        pl_lmh,  bgl_lmh  = get_pl(self._linear_fp16w_wide_result)
        nbb_lmh  = len(self._linear_fp16w_wide_result.buffer_bindings)
        lm_gy = self.MAX_DISPATCH_DIM
        lm_gx = (self.n_vocab + lm_gy - 1) // lm_gy
        lmh_p = mk_params('lmh_p', self._linear_fp16w_wide_result.param_fields,
                           {'K': E, 'stride_x': E, 'stride_w': E,
                            'N': self.n_vocab, 'grid_y': lm_gy})
        bg_lmh = mk_bg(bgl_lmh, [
            (0, norm_out[0], norm_out[1]),
            (1, lm_w.handle, lm_w.size),
            (2, bias_v.handle, bias_v.size),
            (3, logits_buf[0], logits_buf[1]),
            (nbb_lmh, lmh_p[0], lmh_p[1])])

        # Per-layer bind groups
        layer_dispatches = []
        for layer in range(self.n_layer):
            pfx = f"layers.{layer}."

            # Weight handles
            def w(suffix):
                return self._gpu_weights[pfx + suffix]

            qkv_wq = w("self_attn.qkv_proj.weight.q4.gpu")
            qkv_sc = w("self_attn.qkv_proj.weight.scales.gpu")
            qkv_zr = w("self_attn.qkv_proj.weight.zeros.gpu")
            op_wq  = w("self_attn.o_proj.weight.q4.gpu")
            op_sc  = w("self_attn.o_proj.weight.scales.gpu")
            op_zr  = w("self_attn.o_proj.weight.zeros.gpu")
            gu_wq  = w("mlp.gate_up_proj.weight.q4.gpu")
            gu_sc  = w("mlp.gate_up_proj.weight.scales.gpu")
            gu_zr  = w("mlp.gate_up_proj.weight.zeros.gpu")
            dn_wq  = w("mlp.down_proj.weight.q4.gpu")
            dn_sc  = w("mlp.down_proj.weight.scales.gpu")
            dn_zr  = w("mlp.down_proj.weight.zeros.gpu")
            n1w = w("input_layernorm.weight")
            n2w = w("post_attention_layernorm.weight")
            Kc, Vc, _ = self._gpu_kv_cache[layer]

            # norm1
            bg_n1 = mk_bg(bgl_rn, [
                (0, x_buf[0], x_buf[1]),
                (1, norm_out[0], norm_out[1]),
                (2, n1w.handle, n1w.size),
                (3, rstd[0], rstd[1]),
                (nbb_rn, rn_p[0], rn_p[1])])
            # qkv (WGSL Q4: _params_ at binding 6)
            bg_qkv = mk_bg(bgl_q4, [
                (0, norm_out[0], norm_out[1]),
                (1, qkv_wq.handle, qkv_wq.size),
                (2, qkv_sc.handle, qkv_sc.size),
                (3, qkv_zr.handle, qkv_zr.size),
                (4, bias_qkv.handle, bias_qkv.size),
                (5, qkv_out[0], qkv_out[1]),
                (6, qkv_p[0], qkv_p[1])])
            # rope_kv
            bg_rkv = mk_bg(bgl_rkv, [
                (0, qkv_out[0], qkv_out[1]),
                (1, Kc.handle, Kc.size),
                (2, Vc.handle, Vc.size),
                (3, cos_h, cos_sz), (4, sin_h, sin_sz),
                (nbb_rkv, rope_kv_ph[0], rope_kv_ph[1])])
            # attn
            bg_att = mk_bg(bgl_attn, [
                (0, q_rot[0], q_rot[1]),
                (1, Kc.handle, Kc.size),
                (2, Vc.handle, Vc.size),
                (3, attn_out[0], attn_out[1]),
                (nbb_attn, attn_ph[0], attn_ph[1])])
            # o_proj (WGSL Q4: _params_ at binding 6)
            bg_op = mk_bg(bgl_q4, [
                (0, attn_out[0], attn_out[1]),
                (1, op_wq.handle, op_wq.size),
                (2, op_sc.handle, op_sc.size),
                (3, op_zr.handle, op_zr.size),
                (4, o_bias.handle, o_bias.size),
                (5, proj_out[0], proj_out[1]),
                (6, oproj_p[0], oproj_p[1])])

            # Fused RoPE Q + K scatter + V copy (replaces separate rope_q + rope_kv)
            bg_fused_rope = mk_bg(bgl_fused_rope, [
                (0, qkv_out[0], qkv_out[1]),
                (1, q_rot[0], q_rot[1]),
                (2, Kc.handle, Kc.size),
                (3, Vc.handle, Vc.size),
                (4, cos_h, cos_sz), (5, sin_h, sin_sz),
                (nbb_fused_rope, fused_rope_ph[0], fused_rope_ph[1])])

            if has_add_rn:
                # Fused res1+norm2: x += proj_out; norm_out = rms_norm(x)
                bg_arn2 = mk_bg(bgl_arn, [
                    (0, x_buf[0], x_buf[1]),
                    (1, proj_out[0], proj_out[1]),
                    (2, norm_out[0], norm_out[1]),
                    (3, n2w.handle, n2w.size),
                    (4, rstd[0], rstd[1]),
                    (nbb_arn, arn_p[0], arn_p[1])])
            else:
                # norm2
                bg_n2 = mk_bg(bgl_rn, [
                    (0, x_buf[0], x_buf[1]),
                    (1, norm_out[0], norm_out[1]),
                    (2, n2w.handle, n2w.size),
                    (3, rstd[0], rstd[1]),
                    (nbb_rn, rn_p[0], rn_p[1])])

            # gate_up (WGSL Q4: _params_ at binding 6)
            bg_gu = mk_bg(bgl_q4, [
                (0, norm_out[0], norm_out[1]),
                (1, gu_wq.handle, gu_wq.size),
                (2, gu_sc.handle, gu_sc.size),
                (3, gu_zr.handle, gu_zr.size),
                (4, bias_gu.handle, bias_gu.size),
                (5, gate_up_out[0], gate_up_out[1]),
                (6, gateup_p[0], gateup_p[1])])
            # down (WGSL Q4: _params_ at binding 6)
            bg_dn = mk_bg(bgl_q4, [
                (0, silu_out[0], silu_out[1]),
                (1, dn_wq.handle, dn_wq.size),
                (2, dn_sc.handle, dn_sc.size),
                (3, dn_zr.handle, dn_zr.size),
                (4, bias_e.handle, bias_e.size),
                (5, proj_out[0], proj_out[1]),
                (6, down_p[0], down_p[1])])

            # Fused dispatch list: 10 dispatches per layer (was 12)
            # Fusions: rope_q+rope_kv→fused_rope, res1+norm2→add_rn
            if has_add_rn:
                layer_dispatches.append([
                    (pl_rn,         bg_n1,          (1,)),         # norm1
                    (pl_q4,         bg_qkv,         (1, (qkv_N + TILE_N - 1) // TILE_N)),  # qkv
                    (pl_fused_rope, bg_fused_rope,  (n_head + n_kv,)),  # fused rope_q+kv (n_head+n_kv workgroups)
                    (pl_attn,       bg_att,         (n_head,)),    # attn
                    (pl_q4,         bg_op,          (1, (E + TILE_N - 1) // TILE_N)),       # o_proj
                    (pl_arn,        bg_arn2,         (1,)),         # fused res1+norm2
                    (pl_q4,         bg_gu,          (1, (2 * IM + TILE_N - 1) // TILE_N)),  # gate_up
                    (pl_silu,       bg_silu,         silu_g),       # silu_mul
                    (pl_q4,         bg_dn,          (1, (E + TILE_N - 1) // TILE_N)),       # down
                    (pl_aip,        bg_res,          aip_g),        # res2
                ])
        final_dispatches = [
            (pl_rn,  bg_final_rn, (1,)),
            (pl_lmh, bg_lmh,     (lm_gx, lm_gy)),
        ]

        # --- GPU embed gather + argmax for fully GPU-resident decode ---
        # Token ID buffer (single i32, fed back between iterations)
        token_id_buf = mkbuf('token_id', 4)

        # Embed gather: token_id → x_buf
        pl_eg, bgl_eg = get_pl(self._embed_gather_result)
        nbb_eg = len(self._embed_gather_result.buffer_bindings)
        embed_fp32 = self._gpu_weights["embed_tokens.weight.fp32"]
        eg_p = mk_params('eg_p', self._embed_gather_result.param_fields,
                          {'stride_e': E})
        bg_eg = mk_bg(bgl_eg, [
            (0, token_id_buf[0], token_id_buf[1]),
            (1, embed_fp32.handle, embed_fp32.size),
            (2, x_buf[0], x_buf[1]),
            (nbb_eg, eg_p[0], eg_p[1])])

        embed_dispatch = [(pl_eg, bg_eg, (1,))]

        # Argmax: logits_buf → token_id_buf
        pl_am, bgl_am = get_pl(self._argmax_result)
        nbb_am = len(self._argmax_result.buffer_bindings)
        am_p = mk_params('am_p', self._argmax_result.param_fields,
                          {'N': self.n_vocab})
        bg_am = mk_bg(bgl_am, [
            (0, logits_buf[0], logits_buf[1]),
            (1, token_id_buf[0], token_id_buf[1]),
            (nbb_am, am_p[0], am_p[1])])

        argmax_dispatch = [(pl_am, bg_am, (1,))]

        # Store for per-token use
        self._fd_x_h = x_buf[0]
        self._fd_token_id_h = token_id_buf[0]
        self._fd_token_id_sz = token_id_buf[1]
        self._fd_logits_h = logits_buf[0]
        self._fd_logits_sz = logits_buf[1]
        self._fd_fused_rope_ph = fused_rope_ph[0]
        self._fd_attn_ph = attn_ph[0]

        # Full pipeline with GPU embed + argmax
        self._fd_all_batches = (
            [embed_dispatch] +   # GPU embed gather
            layer_dispatches +   # 32 transformer layers
            [final_dispatches] + # final norm + lm_head
            [argmax_dispatch]    # GPU argmax → token_id
        )

        # Legacy: dispatch list without embed/argmax for fallback
        self._fd_batches_no_embed = (
            layer_dispatches +
            [final_dispatches]
        )

        # Pre-allocated bytearrays for dynamic params (avoid per-token alloc)
        # fused_rope: n_head(i32), q_size(i32), kv_size(i32), pos(i32), half_rot(i32), cache_offset(i32)
        self._fd_fused_rope_buf = bytearray(24)
        struct.pack_into('<iii', self._fd_fused_rope_buf, 0, n_head, q_size, kv_size)
        struct.pack_into('<i', self._fd_fused_rope_buf, 16, half_rot)
        # attn: kv_stride(i32), n_rep(i32), T_total(i32), scale(f32), neg_inf(f32)
        self._fd_attn_buf = bytearray(20)
        struct.pack_into('<ii', self._fd_attn_buf, 0, n_kv * HD, n_rep)
        struct.pack_into('<ff', self._fd_attn_buf, 12,
            float(np.float32(1.0 / np.sqrt(HD))),
            float(np.float32(-1e9)))

        self._fast_decode_ready = True

    def _decode_fast(self, token_ids, pos_offset):
        """Fast decode: fully GPU-resident with embed gather + argmax.

        Flow: write token_id → GPU embed → 32 layers → norm → lm_head → argmax
        Only reads back 4 bytes (token ID) instead of 512KB (logits).
        """
        runner = self.cache.runner
        import struct

        # Write token ID to GPU buffer (4 bytes)
        runner.write_buffer(self._fd_token_id_h,
                           struct.pack('<i', int(token_ids[0])))

        # Dynamic params
        pos = pos_offset
        _, _, cur_len = self._gpu_kv_cache[0]
        cache_offset = cur_len * self.n_kv_heads * self.head_dim
        T_total = cur_len + 1

        # Fused RoPE params: n_head, q_size, kv_size already packed; update pos + cache_offset
        struct.pack_into('<i', self._fd_fused_rope_buf, 12, pos)
        struct.pack_into('<i', self._fd_fused_rope_buf, 20, cache_offset)
        runner.write_buffer(self._fd_fused_rope_ph, bytes(self._fd_fused_rope_buf))

        struct.pack_into('<i', self._fd_attn_buf, 8, T_total)
        runner.write_buffer(self._fd_attn_ph, bytes(self._fd_attn_buf))

        # Update KV cache counters
        for layer in range(self.n_layer):
            K, V, c = self._gpu_kv_cache[layer]
            self._gpu_kv_cache[layer] = (K, V, c + 1)

        # Submit full pipeline: embed → layers → norm → lm_head → argmax
        # Read back token ID (4 bytes) instead of full logits (512KB)
        token_id_np = runner.submit_dispatches_pipelined(
            self._fd_all_batches,
            readback=(self._fd_token_id_h, self._fd_token_id_sz, np.int32))
        next_token = int(token_id_np[0])

        # Build logits with just the argmax token set (for generate() compatibility)
        logits = np.full((1, self.n_vocab), -1e9, dtype=np.float32)
        logits[0, next_token] = 1.0
        return logits

    def _decode_gpu(self, x, layer, positions, pfx, p):
        """Fully GPU-resident decode: all operations stay on GPU.

        Uses GPU RoPE, GPU residual add, and GPU RMSNorm to eliminate
        all per-layer CPU readbacks. Only the final output (after all
        32 layers) is read back for the LM head.

        Data flow (all on GPU):
          GPU: RMSNorm → QKV proj → RoPE Q → RoPE+scatter KV →
               attention → O proj → residual add → RMSNorm →
               gate_up → SiLU·mul → down proj → residual add
        """
        from common.model_base import GPUBuffer
        runner = self.cache.runner

        # Ensure layer weights are on GPU
        self._upload_layer_weights(layer)

        HD = self.head_dim
        E = self.n_embd
        n_head = self.n_head
        n_kv = self.n_kv_heads
        n_rep = self.n_rep
        half_rot = self.rotary_dim // 2
        q_size = n_head * HD
        kv_size = n_kv * HD
        pos = int(positions[0])
        IM = self.intermediate_size

        # Ensure x is on GPU
        if not isinstance(x, GPUBuffer):
            x_np = x.ravel().astype(np.float32)
            x_gpu = runner.upload_to_gpu(x_np, f"__decode_x_{layer}")
            x_gpu.shape = (1, E)
        else:
            x_gpu = x

        # 1. GPU RMSNorm (input layernorm)
        if self._profiling: self._set_gpu_op(f"L{layer}/norm1")
        rn1 = self._rms_norm(
            x_gpu, self._gpu_weights[pfx + "input_layernorm.weight"],
            gpu_out=True)

        # 2. QKV projection → stays on GPU
        if self._profiling: self._set_gpu_op(f"L{layer}/qkv")
        qkv_gpu = self._proj(rn1, pfx + "self_attn.qkv_proj.weight",
                             self._gpu_weights["zero_bias_QKV"],
                             self.qkv_out, gpu_out=True)

        # 3. GPU RoPE for Q heads
        if self._profiling: self._set_gpu_op(f"L{layer}/rope_q")
        q_rot_out = self.cache.run(
            self._rope_result, grid=(n_head,),
            buffers={
                'X': qkv_gpu,
                'Y': np.zeros(q_size, dtype=np.float32),
                'CosTable': self._rope_cos_gpu,
                'SinTable': self._rope_sin_gpu,
            },
            scalars={
                'src_offset': 0,
                'pos': pos,
                'half_rot': half_rot,
            },
            gpu_outputs={'Y'})
        q_rot_gpu = q_rot_out['Y']

        # 4. GPU RoPE + scatter K,V into KV cache
        K_cache_gpu, V_cache_gpu, cur_len = self._gpu_kv_cache[layer]
        cache_offset = cur_len * n_kv * HD

        if self._profiling: self._set_gpu_op(f"L{layer}/rope_kv")
        self.cache.run(
            self._rope_kv_result, grid=(n_kv,),
            buffers={
                'QKV': qkv_gpu,
                'K_cache': K_cache_gpu,
                'V_cache': V_cache_gpu,
                'CosTable': self._rope_cos_gpu,
                'SinTable': self._rope_sin_gpu,
            },
            scalars={
                'q_size': q_size,
                'kv_size': kv_size,
                'pos': pos,
                'half_rot': half_rot,
                'cache_offset': cache_offset,
            },
            gpu_outputs={'K_cache', 'V_cache'})

        T_total = cur_len + 1
        self._gpu_kv_cache[layer] = (K_cache_gpu, V_cache_gpu, T_total)

        # 5. GPU GQA decode attention
        if self._profiling: self._set_gpu_op(f"L{layer}/attn")
        scale = np.float32(1.0 / np.sqrt(HD))
        kv_stride = n_kv * HD
        attn_out = self.cache.run(
            self._gqa_attn_result, grid=(n_head,),
            buffers={
                'Q': q_rot_gpu,
                'K_cache': K_cache_gpu,
                'V_cache': V_cache_gpu,
                'Out': np.zeros(n_head * HD, dtype=np.float32),
            },
            scalars={
                'kv_stride': kv_stride,
                'n_rep': n_rep,
                'T_total': T_total,
                'scale': scale,
                'neg_inf': np.float32(-1e9),
            },
            gpu_outputs={'Out'})
        attn_gpu = attn_out['Out']
        attn_gpu.shape = (1, n_head * HD)

        # 6. O projection → stays on GPU
        o_bias = self._gpu_weights.get("zero_bias_QO",
                                       self._gpu_weights["zero_bias_E"])
        if self._profiling: self._set_gpu_op(f"L{layer}/o_proj")
        o_gpu = self._proj(attn_gpu, pfx + "self_attn.o_proj.weight",
                           o_bias, E, gpu_out=True)

        # 7. Residual add: x += o_proj (in-place)
        if self._profiling: self._set_gpu_op(f"L{layer}/res1")
        self._add_inplace(x_gpu, o_gpu)

        # 8. GPU RMSNorm (post-attention layernorm)
        if self._profiling: self._set_gpu_op(f"L{layer}/norm2")
        rn2 = self._rms_norm(
            x_gpu, self._gpu_weights[pfx + "post_attention_layernorm.weight"],
            gpu_out=True)

        # 9. Gate_up projection → stays on GPU
        if self._profiling: self._set_gpu_op(f"L{layer}/gate_up")
        gate_up_gpu = self._proj(rn2, pfx + "mlp.gate_up_proj.weight",
                                 self._gpu_weights["zero_bias_GU"],
                                 self.gate_up_out, gpu_out=True)

        # 10. SiLU*mul → stays on GPU
        if self._profiling: self._set_gpu_op(f"L{layer}/silu_mul")
        h_gpu = self._silu_mul_fused(gate_up_gpu, IM, gpu_out=True)

        # 11. Down projection → stays on GPU
        if self._profiling: self._set_gpu_op(f"L{layer}/down")
        down_gpu = self._proj(h_gpu, pfx + "mlp.down_proj.weight",
                              self._gpu_weights["zero_bias_E"],
                              E, K=IM, gpu_out=True)

        # 12. Residual add: x += down (in-place)
        if self._profiling: self._set_gpu_op(f"L{layer}/res2")
        self._add_inplace(x_gpu, down_gpu)

        x_gpu.shape = (1, E)
        return x_gpu

    def _transformer_block(self, x, layer: int,
                           use_cache: bool = False,
                           positions: np.ndarray = None, **kwargs):
        """Pre-norm transformer block.

        For decode (T=1): selects CPU or GPU decode based on self._decode_mode.
        CPU mode: ~7ms/layer via numpy BLAS (best for high-overhead backends).
        GPU mode: ~1.3ms/layer via GPU dispatch (best for Vulkan/low-overhead).
        """
        pfx = f"layers.{layer}."
        p = self.profiler

        # Determine if we're in decode mode (T=1)
        from common.model_base import GPUBuffer
        if isinstance(x, GPUBuffer):
            T = x.shape[0] if x.shape else 1
        else:
            T = x.shape[0]

        if T == 1 and use_cache:
            if self._decode_mode == 'gpu':
                return self._decode_gpu(x, layer, positions, pfx, p)
            else:
                return self._decode_cpu(x, layer, positions, pfx, p)
        elif use_cache and self._decode_mode == 'gpu':
            # GPU-resident prefill: all ops on GPU, no readbacks
            self._upload_layer_weights(layer)
            return self._prefill_gpu(x, layer, positions, pfx, p)
        else:
            # === PREFILL PATH (original) ===
            # Upload weights to GPU (needed for GPU dispatch path)
            self._upload_layer_weights(layer)
            rn1 = self._rms_norm(
                x, self._gpu_weights[pfx + "input_layernorm.weight"],
                gpu_out=True)
            attn = self._attention_block(rn1, layer, use_cache=use_cache,
                                         positions=positions)
            x = self._add(x, attn, gpu_out=True)

            rn2 = self._rms_norm(
                x, self._gpu_weights[pfx + "post_attention_layernorm.weight"],
                gpu_out=True)
            mlp = self._mlp_block(rn2, layer, gpu_out=True)
            x = self._add(x, mlp, gpu_out=True)
            return x

    def _decode_attention(self, qkv, layer, positions):
        """Fast CPU-side GQA attention for T=1 decode.

        Inlined for speed — no function call overhead per head.
        """
        HD = self.head_dim
        n_head = self.n_head
        n_kv = self.n_kv_heads
        n_rep = self.n_rep

        # Split QKV
        q_size = n_head * HD
        kv_size = n_kv * HD
        Q = qkv[0, :q_size].reshape(n_head, HD)
        K_new = qkv[0, q_size:q_size + kv_size].reshape(n_kv, HD)
        V_new = qkv[0, q_size + kv_size:].reshape(n_kv, HD)

        # Partial RoPE — inlined for speed
        half_rot = self.rotary_dim // 2
        pos = positions[0]
        cos_v = self._rope_cos[pos]  # (half_rot,)
        sin_v = self._rope_sin[pos]

        # RoPE on Q
        q1 = Q[:, :half_rot]; q2 = Q[:, half_rot:self.rotary_dim]
        Q_rot = np.empty_like(Q)
        Q_rot[:, :half_rot] = q1 * cos_v - q2 * sin_v
        Q_rot[:, half_rot:self.rotary_dim] = q2 * cos_v + q1 * sin_v
        Q_rot[:, self.rotary_dim:] = Q[:, self.rotary_dim:]

        # RoPE on K
        k1 = K_new[:, :half_rot]; k2 = K_new[:, half_rot:self.rotary_dim]
        K_rot = np.empty_like(K_new)
        K_rot[:, :half_rot] = k1 * cos_v - k2 * sin_v
        K_rot[:, half_rot:self.rotary_dim] = k2 * cos_v + k1 * sin_v
        K_rot[:, self.rotary_dim:] = K_new[:, self.rotary_dim:]

        # KV cache update
        if self.kv_cache is None:
            self.kv_cache = {}
        if layer not in self.kv_cache:
            K_buf = np.zeros((self.MAX_SEQ_LEN, n_kv, HD), dtype=np.float32)
            V_buf = np.zeros((self.MAX_SEQ_LEN, n_kv, HD), dtype=np.float32)
            cur_len = 0
        else:
            K_buf, V_buf, cur_len = self.kv_cache[layer]
        K_buf[cur_len] = K_rot
        V_buf[cur_len] = V_new
        T_total = cur_len + 1
        self.kv_cache[layer] = (K_buf, V_buf, T_total)

        K_full = K_buf[:T_total]  # (T_total, n_kv, HD)
        V_full = V_buf[:T_total]

        # GQA attention — expand KV heads and batch matmul
        scale = np.float32(1.0 / np.sqrt(HD))
        K_exp = np.repeat(K_full, n_rep, axis=1)  # (T_total, n_head, HD)
        V_exp = np.repeat(V_full, n_rep, axis=1)

        # scores: (n_head, HD) @ (n_head, HD, T_total) -> (n_head, T_total)
        scores = np.float32(np.einsum('nh,tnh->nt', Q_rot, K_exp) * scale)
        scores -= scores.max(axis=1, keepdims=True)
        exp_s = np.exp(scores)
        denom = exp_s.sum(axis=1, keepdims=True)
        denom = np.where(denom == 0, 1.0, denom)
        attn = exp_s / denom

        # output: (n_head, T_total) @ (T_total, n_head, HD) -> (n_head, HD)
        attn_out = np.float32(np.einsum('nt,tnh->nh', attn, V_exp))
        return attn_out.reshape(1, n_head * HD)

    def forward(self, token_ids: np.ndarray,
                use_cache: bool = False,
                pos_offset: int = 0) -> np.ndarray:
        """Run Phi-4 mini forward pass."""
        from common.model_base import GPUBuffer

        T = len(token_ids)

        # Fast decode path: pre-recorded bind groups, 3 ctypes calls/dispatch
        if (T == 1 and use_cache and self._decode_mode == 'gpu'
                and getattr(self, '_use_q4_gpu', False)):
            if getattr(self, '_use_dp4a', False):
                # DP4A fast decode: WGSL Q4 kernel with pre-compiled pipeline
                if not getattr(self, '_fast_decode_ready', False):
                    self._init_fast_decode_dp4a()
                if self._fast_decode_ready:
                    return self._decode_fast(token_ids, pos_offset)
            else:
                # Triton fast decode: pre-compiled Triton Q4 kernel
                if not getattr(self, '_fast_decode_ready', False):
                    self._init_fast_decode()
                if self._fast_decode_ready:
                    return self._decode_fast(token_ids, pos_offset)

        wte = self.weights["embed_tokens.weight"]

        import time as _t
        _times = {}
        _t0 = _t.perf_counter()

        if self._profiling: self._begin_cpu("embed")
        x = wte[token_ids]
        if self._profiling: self._end_cpu("embed")
        _times["embed"] = (_t.perf_counter() - _t0) * 1000

        positions = np.arange(pos_offset, pos_offset + T, dtype=np.int32)

        # Progressive command batching: group GPU dispatches into batches
        # of N layers, submitting each batch as soon as it's recorded.
        # This gets the GPU started early (after the first batch) while CPU
        # records the next batch — pipelining CPU recording and GPU execution.
        #   batch_layers=0 : no batching (individual submits, default)
        #   batch_layers=1 : per-layer batching (32 submits of ~12 dispatches)
        #   batch_layers=4 : every 4 layers (8 submits of ~48 dispatches)
        #   batch_layers=32: single batch (1 submit, no overlap)
        runner = self.cache.runner
        batch_layers = getattr(self, '_batch_layers', 4)
        use_batch = (batch_layers > 0
                     and use_cache and self._decode_mode == 'gpu')
        if use_batch:
            runner.begin_batch()

        _t1 = _t.perf_counter()
        for layer in range(self.n_layer):
            x = self._transformer_block(x, layer, use_cache=use_cache,
                                        positions=positions)
            # Flush batch every N layers: submit to GPU and start next batch
            if use_batch and runner.is_batching and (layer + 1) % batch_layers == 0:
                runner.end_batch()
                if layer + 1 < self.n_layer:
                    runner.begin_batch()

        # Flush any remaining dispatches
        if use_batch and runner.is_batching:
            runner.end_batch()
        _times["layers (32)"] = (_t.perf_counter() - _t1) * 1000

        # Final RMSNorm
        if self._profiling:
            self._set_gpu_op("final_norm")
        # Check if GPU lm_head is available (fp16 embed weights uploaded)
        gpu_lm_head = ("embed_tokens.weight.fp16" in self._gpu_weights
                       and isinstance(x, GPUBuffer))
        if isinstance(x, GPUBuffer):
            # GPU-resident path: keep on GPU if we'll do GPU lm_head
            x = self._rms_norm(x, self._gpu_weights["norm.weight"],
                               gpu_out=gpu_lm_head)
        elif T == 1 and use_cache and self._decode_mode == 'cpu':
            x = self._rms_norm_cpu(x, "norm.weight")
        else:
            x = self._rms_norm(x, self._gpu_weights["norm.weight"])
        if self._profiling:
            self._clear_gpu_op()

        # LM head (tied with embed_tokens)
        if gpu_lm_head:
            if self._profiling:
                self._set_gpu_op("lm_head")
            logits = self._linear_fp16w(
                x, self._gpu_weights["embed_tokens.weight.fp16"],
                self._gpu_weights["zero_bias_V"], self.n_vocab,
                K=self.n_embd)
            if self._profiling:
                self._clear_gpu_op()
        else:
            if self._profiling: self._begin_cpu("lm_head")
            logits = np.float32(x @ wte.T)
            if self._profiling: self._end_cpu("lm_head")

        _times["norm + lm_head"] = (_t.perf_counter() - _t1 - _times["layers (32)"] / 1000) * 1000
        self._prefill_times = _times
        return logits


# ---------------------------------------------------------------------------
# Weight downloading
# ---------------------------------------------------------------------------

def download_phi4_weights(model_size: str = "mini",
                          model_dir: str = None) -> Tuple[str, str]:
    """Download Phi-4 mini weights and tokenizer from HuggingFace."""
    config = PHI4_CONFIGS[model_size]
    hf_repo = config["hf_repo"]
    if model_dir is None:
        model_dir = os.path.join(_SCRIPT_DIR, "..", "..", "gitignore", "models", os.path.basename(_SCRIPT_DIR), "weights")

    def phi4_key_transform(key, arr):
        new_key = key.replace("model.", "")
        if new_key == "lm_head.weight":
            return None  # tied
        return new_key, arr

    npz_path, tokenizer_path = download_weights(
        hf_repo=hf_repo,
        model_dir=model_dir,
        safetensors_files=[
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ],
        key_transform=phi4_key_transform,
        download_tokenizer=True,
    )
    return npz_path, tokenizer_path


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_with_random_weights():
    """Verify Phi-4 mini pipeline with small random weights."""
    print("=" * 60)
    print("Phi-4 mini WebGPU Pipeline Verification (random weights)")
    print("=" * 60)

    n_layer, n_head, n_kv_heads = 2, 6, 2
    n_embd = 96
    head_dim = n_embd // n_head  # 16
    intermediate_size = 128
    n_vocab = 256
    n_rep = n_head // n_kv_heads
    rope_theta = 10000.0
    eps = 1e-5
    partial_rotary_factor = 0.75
    rotary_dim = int(head_dim * partial_rotary_factor)  # 12
    kv_dim = n_kv_heads * head_dim
    qkv_out = n_head * head_dim + 2 * kv_dim
    gate_up_out = 2 * intermediate_size
    np.random.seed(42)

    weights = {}
    weights["embed_tokens.weight"] = np.random.randn(
        n_vocab, n_embd).astype(np.float32) * 0.02
    weights["norm.weight"] = np.ones(n_embd, dtype=np.float32)

    for i in range(n_layer):
        pfx = f"layers.{i}."
        weights[pfx + "input_layernorm.weight"] = np.ones(
            n_embd, dtype=np.float32)
        weights[pfx + "post_attention_layernorm.weight"] = np.ones(
            n_embd, dtype=np.float32)
        # Fused QKV
        weights[pfx + "self_attn.qkv_proj.weight"] = np.random.randn(
            qkv_out, n_embd).astype(np.float32) * 0.02
        weights[pfx + "self_attn.o_proj.weight"] = np.random.randn(
            n_embd, n_head * head_dim).astype(np.float32) * 0.02
        # Fused gate_up
        weights[pfx + "mlp.gate_up_proj.weight"] = np.random.randn(
            gate_up_out, n_embd).astype(np.float32) * 0.02
        weights[pfx + "mlp.down_proj.weight"] = np.random.randn(
            n_embd, intermediate_size).astype(np.float32) * 0.02

    print(f"\nModel: {n_layer} layers, {n_head} Q heads, {n_kv_heads} KV heads, "
          f"{n_embd} embd, {intermediate_size} intermediate, {n_vocab} vocab")
    print(f"  Partial RoPE: rotary_dim={rotary_dim}/{head_dim}")

    model = Phi4WebGPU(
        weights, n_layer=n_layer, n_head=n_head, n_kv_heads=n_kv_heads,
        n_embd=n_embd, intermediate_size=intermediate_size,
        n_vocab=n_vocab, rope_theta=rope_theta, rms_norm_eps=eps,
        head_dim=head_dim, partial_rotary_factor=partial_rotary_factor)

    # Forward pass
    token_ids = np.array([1, 42, 100, 200], dtype=np.int32)
    T = len(token_ids)
    t0 = time.time()
    logits = model.forward(token_ids)
    t1 = time.time()

    print(f"\nForward pass: {token_ids} → shape {logits.shape} "
          f"in {(t1-t0)*1000:.0f}ms")

    # --- NumPy reference ---
    def _partial_rope_numpy(x, positions, theta, hd, rot_dim):
        half_rot = rot_dim // 2
        inv_freq = 1.0 / (theta ** (
            np.arange(0, rot_dim, 2, dtype=np.float32) / rot_dim))
        angles = positions[:, None].astype(np.float32) * inv_freq[None, :]
        cos_v = np.cos(angles)[:, None, :]
        sin_v = np.sin(angles)[:, None, :]
        x_rot = x[..., :rot_dim]
        x_pass = x[..., rot_dim:]
        x1 = x_rot[..., :half_rot]
        x2 = x_rot[..., half_rot:]
        out = np.empty_like(x)
        out[..., :half_rot] = x1 * cos_v - x2 * sin_v
        out[..., half_rot:rot_dim] = x2 * cos_v + x1 * sin_v
        out[..., rot_dim:] = x_pass
        return out

    positions = np.arange(T, dtype=np.int32)
    x = weights["embed_tokens.weight"][token_ids]

    for layer in range(n_layer):
        pfx = f"layers.{layer}."
        rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
        ln1 = x / rms * weights[pfx + "input_layernorm.weight"]

        # Fused QKV
        qkv = ln1 @ weights[pfx + "self_attn.qkv_proj.weight"].T
        q_size = n_head * head_dim
        q = qkv[:, :q_size]
        k = qkv[:, q_size:q_size + kv_dim]
        v = qkv[:, q_size + kv_dim:]

        Q = q.reshape(T, n_head, head_dim)
        K_ = k.reshape(T, n_kv_heads, head_dim)
        V_ = v.reshape(T, n_kv_heads, head_dim)

        Q = _partial_rope_numpy(Q, positions, rope_theta, head_dim, rotary_dim)
        K_ = _partial_rope_numpy(K_, positions, rope_theta, head_dim, rotary_dim)

        scale = 1.0 / np.sqrt(head_dim)
        attn_out = np.zeros_like(Q)
        for h in range(n_head):
            kv_h = h // n_rep
            scores = Q[:, h, :] @ K_[:, kv_h, :].T * scale
            mask = np.triu(np.ones((T, T), dtype=bool), k=1)
            scores[mask] = -1e9
            exp_s = np.exp(scores - scores.max(axis=-1, keepdims=True))
            attn = exp_s / exp_s.sum(axis=-1, keepdims=True)
            attn_out[:, h, :] = attn @ V_[:, kv_h, :]

        attn_flat = attn_out.reshape(T, n_head * head_dim)
        proj = attn_flat @ weights[pfx + "self_attn.o_proj.weight"].T
        x = x + proj

        rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
        ln2 = x / rms * weights[pfx + "post_attention_layernorm.weight"]

        # Fused gate_up
        gate_up = ln2 @ weights[pfx + "mlp.gate_up_proj.weight"].T
        gate = gate_up[:, :intermediate_size]
        up_val = gate_up[:, intermediate_size:]
        silu_gate = gate / (1.0 + np.exp(-gate))
        mlp_out = (silu_gate * up_val) @ \
                  weights[pfx + "mlp.down_proj.weight"].T
        x = x + mlp_out

    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    x = x / rms * weights["norm.weight"]
    logits_ref = x @ weights["embed_tokens.weight"].T

    max_diff = np.abs(logits - logits_ref).max()
    gpu_preds = logits.argmax(axis=1)
    ref_preds = logits_ref.argmax(axis=1)
    argmax_match = np.array_equal(gpu_preds, ref_preds)

    print(f"Max diff vs NumPy: {max_diff:.6f}")
    print(f"Predictions match: {argmax_match}")
    print(f"  GPU: {gpu_preds}  Ref: {ref_preds}")

    return max_diff < 0.1 and argmax_match


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Phi-4 mini on WebGPU via Triton")
    parser.add_argument("--verify", action="store_true",
                        help="Verify pipeline with random weights")
    parser.add_argument("--quantize", action="store_true",
                        help="Quantize weights to INT4 + fp16 and save")
    parser.add_argument("--prompt", type=str,
                        default="The future of AI is",
                        help="Prompt for text generation")
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--weights-dir", type=str, default=None)
    parser.add_argument("--decode-mode", type=str, default="gpu",
                        choices=["cpu", "gpu"],
                        help="Decode mode: 'gpu' (GPU dispatch) or 'cpu' (numpy BLAS)")
    parser.add_argument("--profile", action="store_true",
                        help="Enable CPU+GPU profiling")
    parser.add_argument("--batch-layers", type=int, default=1, metavar="N",
                        help="Progressive batching: submit every N layers "
                             "(0=off, 1=per-layer, 4=every 4 layers, 32=single batch)")
    parser.add_argument("--use-dp4a", action="store_true",
                        help="Use DP4A (dot4I8Packed) for INT4 matmul")
    add_device_arg(parser)
    args = parser.parse_args()
    apply_device_arg(args)

    if args.verify:
        success = verify_with_random_weights()
        sys.exit(0 if success else 1)

    config = PHI4_CONFIGS["mini"]
    weights_dir = args.weights_dir or os.path.join(_SCRIPT_DIR, "..", "..", "gitignore", "models", os.path.basename(_SCRIPT_DIR), "weights")

    # Download if needed
    npz_path, tokenizer_path = download_phi4_weights("mini", weights_dir)
    q4_path = os.path.join(weights_dir, "weights_q4.npz")

    # Quantize mode: convert fp32 → INT4 + fp16
    if args.quantize:
        print("=" * 60)
        print("Quantizing Phi-4 mini weights to INT4 + fp16")
        print("=" * 60)
        weights = load_weights(npz_path)
        print(f"Loaded {len(weights)} weight tensors")
        q_weights = quantize_model_weights(weights, config["n_layer"])
        print(f"Saving quantized weights to {q4_path}...")
        np.savez(q4_path, **q_weights)
        q4_size = os.path.getsize(q4_path) / 1024 / 1024
        print(f"Saved: {q4_size:.0f} MB")
        print("Done! Run without --quantize to use quantized weights.")
        return

    # Load weights — prefer quantized if available
    quantized = False
    if os.path.exists(q4_path):
        print(f"Loading quantized weights from {q4_path}...")
        t0 = time.time()
        weights = load_quantized_weights(q4_path)
        t1 = time.time()
        quantized = True
        ram_mb = sum(v.nbytes for v in weights.values()) / 1024 / 1024
        print(f"Loaded {len(weights)} tensors ({ram_mb:.0f} MB RAM) "
              f"in {t1-t0:.1f}s [INT4 quantized]")
    else:
        print(f"Loading fp32 weights from {npz_path}...")
        print("  TIP: Run with --quantize first to reduce memory 4x")
        weights = load_weights(npz_path)
        print(f"Loaded {len(weights)} weight tensors")

    tokenizer = load_tokenizer(tokenizer_path)

    model = Phi4WebGPU(
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
        quantized=quantized,
        decode_mode=args.decode_mode)
    model._batch_layers = args.batch_layers
    model._use_dp4a = getattr(args, 'use_dp4a', False)

    # Print adapter/GPU info
    adapter = model.cache.runner.adapter_info
    gpu_name = adapter.get('device', adapter.get('description', 'unknown'))
    print(f"GPU: {gpu_name}")
    print(f"Backend: {adapter.get('backend', 'unknown')}")
    features = []
    if model.cache.runner.has_subgroups:
        features.append("Subgroups")
    if model.cache.runner.has_f16:
        features.append("ShaderF16")
    if model.cache.runner.has_timestamp_query:
        features.append("TimestampQuery")
    print(f"Features: {', '.join(features) if features else 'none'}")
    print(f"Decode mode: {args.decode_mode}")
    if model._use_dp4a:
        print("DP4A: enabled (dot4I8Packed for INT4 matmul)")

    if args.profile:
        model.enable_profiling()
        print(f"Profiling enabled (GPU timestamps: "
              f"{model.profiler.gpu_enabled})")

        # Profile a short generation
        from common.profiler import InferenceProfiler
        tokenizer_loaded = load_tokenizer(tokenizer_path)
        tokens = tokenizer_loaded.encode(args.prompt)
        token_ids = np.array(tokens, dtype=np.int32)
        model.kv_cache = None
        if hasattr(model, '_gpu_kv_cache'):
            for layer in model._gpu_kv_cache:
                k, v, _ = model._gpu_kv_cache[layer]
                model._gpu_kv_cache[layer] = (k, v, 0)

        # Prefill with detailed timing
        import time as _time
        print("\n--- Prefill Breakdown ---")

        t0 = _time.perf_counter()
        with model.profiler.step("prefill"):
            logits = model.forward(token_ids, use_cache=True, pos_offset=0)
        t1 = _time.perf_counter()

        # Print per-phase timing from the forward pass
        if hasattr(model, '_prefill_times'):
            for name, ms in model._prefill_times.items():
                print(f"  {name:22s} {ms:.1f}ms")
        else:
            print(f"  Forward pass:       {(t1 - t0)*1000:.1f}ms")

        # Warmup fast decode (uploads all layer weights + creates bind groups)
        t2 = _time.perf_counter()
        if hasattr(model, '_warmup_fast_decode'):
            model._warmup_fast_decode()
        t3 = _time.perf_counter()
        print(f"  Fast decode init:   {(t3 - t2)*1000:.1f}ms")
        print(f"  Total TTFT:         {(t3 - t0)*1000:.1f}ms")

        # Decode N tokens with profiling
        n_profile = min(args.max_tokens, 10)
        generated = list(tokens)
        next_logits = logits[-1, :].copy()
        for step in range(n_profile):
            if args.temperature > 0:
                next_logits = next_logits / args.temperature
                next_logits -= next_logits.max()
                probs = np.exp(next_logits)
                probs /= probs.sum()
                next_token = int(np.random.choice(len(probs), p=probs))
            else:
                next_token = int(next_logits.argmax())
            generated.append(next_token)

            with model.profiler.step(f"decode_{step}"):
                with model.profiler.scope(f"forward"):
                    logits = model.forward(
                        np.array([next_token], dtype=np.int32),
                        use_cache=True,
                        pos_offset=len(generated) - 1)
                with model.profiler.cpu("sampling"):
                    next_logits = logits[-1, :].copy()

        model.save_profile(_SCRIPT_DIR, "Phi-4 mini")
        return

    generate(model, args.prompt, tokenizer,
             max_tokens=args.max_tokens,
             temperature=args.temperature)


if __name__ == "__main__":
    main()
