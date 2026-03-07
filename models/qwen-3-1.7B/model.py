"""
Qwen3-1.7B inference on WebGPU via Triton.

Qwen3 is a LLaMA-family model featuring:
  - RoPE (rotary position embeddings)
  - RMSNorm (root mean square normalization)
  - GQA (grouped query attention)
  - SwiGLU MLP (SiLU-gated linear unit)
  - No attention biases (unlike Qwen2.5)
  - Tied word embeddings (embed_tokens == lm_head)

Optimizations:
  - Q8_0 GGUF weights kept packed on GPU (W8A32, ~1 byte/param)
  - Fused QKV projection (3→1 dispatch)
  - Fused gate+up MLP projection (2→1 dispatch)
  - Pre-computed RoPE tables
  - GPU-resident KV cache with fused RoPE+scatter
  - Fast decode pipeline (pre-recorded dispatches, 160+ tok/s)

Usage:
    python models/qwen-3-1.7B/model.py --verify
    python models/qwen-3-1.7B/model.py --gguf-file path/to/Qwen3-1.7B-Q8_0.gguf --prompt "Hello"
    python models/qwen-3-1.7B/model.py --gguf-file path/to/Qwen3-1.7B-Q8_0.gguf --use-q8-gpu --prompt "Hello"
    python models/qwen-3-1.7B/model.py --gguf-file path/to/Qwen3-1.7B-Q8_0.gguf --use-q8-gpu --profile

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
    add_common_args, apply_device_arg, run_inference,
)


# ---------------------------------------------------------------------------
# INT4 quantization utilities
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
    q = np.clip(np.round(
        (w_grouped - zeros.astype(np.float32)) / scales.astype(np.float32)
    ), 0, 15).astype(np.uint8)
    q_packed = (q[:, 1::2] << 4) | q[:, 0::2]
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
    return w.astype(dtype)


# Qwen3 model configs
QWEN_CONFIGS = {
    "1.7B": {
        "n_layer": 28, "n_head": 16, "n_kv_heads": 8,
        "n_embd": 2048, "intermediate_size": 6144,
        "n_vocab": 151936, "rope_theta": 1000000.0,
        "rms_norm_eps": 1e-6, "head_dim": 128,
        "hf_repo": "Qwen/Qwen3-1.7B",
        "attention_bias": False,
        "tie_word_embeddings": True,
    },
}


class Qwen3WebGPU(WebGPUModel):
    """Qwen3 inference on WebGPU via Triton kernels.

    Supports Qwen3-0.6B with:
      - Fused QKV projection (3→1 dispatch, no biases)
      - Fused gate+up MLP projection (2→1 dispatch)
      - INT4 per-group quantization with fp16 scales
      - GPU-resident KV cache with fused RoPE+scatter
      - Fast decode pipeline (pre-recorded dispatches)
    """

    MAX_SEQ_LEN = 2048

    def __init__(self, weights: Dict[str, np.ndarray],
                 n_layer: int = 24, n_head: int = 14,
                 n_kv_heads: int = 2, n_embd: int = 896,
                 intermediate_size: int = 4864,
                 n_vocab: int = 151936,
                 rope_theta: float = 1000000.0,
                 rms_norm_eps: float = 1e-6,
                 head_dim: int = 64,
                 attention_bias: bool = True,
                 tie_word_embeddings: bool = True,
                 quantized: bool = False,
                 decode_mode: str = 'cpu',
                 q8_mode: bool = False):
        self.attention_bias = attention_bias
        self.tie_word_embeddings = tie_word_embeddings
        self.q_dim = n_head * head_dim
        self.kv_dim = n_kv_heads * head_dim
        self.qkv_out = self.q_dim + 2 * self.kv_dim
        self.gate_up_out = 2 * intermediate_size
        self._quantized = quantized
        self._decode_mode = decode_mode
        self._q8_mode = q8_mode
        super().__init__(
            weights, n_layer=n_layer, n_head=n_head, n_embd=n_embd,
            n_vocab=n_vocab,
            n_kv_heads=n_kv_heads,
            intermediate_size=intermediate_size,
            head_dim=head_dim,
            rope_theta=rope_theta,
            rms_norm_eps=rms_norm_eps,
            k_dimensions={n_embd, intermediate_size, self.q_dim},
        )
        self._fuse_weights()
        if self._quantized and not getattr(self, '_has_fp16_linear', False):
            self._dequantize_all_weights()
        self._precompute_rope_tables()
        self._upload_weights_to_gpu()
        self._has_qk_norm = any("q_norm" in k for k in self._gpu_weights)
        if self._decode_mode == 'gpu':
            self._init_gpu_kv_cache()

    def _compile_model_kernels(self):
        """Compile Qwen-specific kernels: RMSNorm, SiLU*mul, argmax."""
        self._compile_rms_norm()
        self._compile_silu_mul()
        self._compile_argmax()

    def _fuse_weights(self):
        """Fuse Q/K/V weights+biases and gate/up weights into single matrices."""
        E = self.n_embd
        HD = self.head_dim
        for i in range(self.n_layer):
            pfx = f"layers.{i}."
            # Fuse QKV weights
            q_key = pfx + "self_attn.q_proj.weight"
            if q_key in self.weights:
                k_key = pfx + "self_attn.k_proj.weight"
                v_key = pfx + "self_attn.v_proj.weight"
                self.weights[pfx + "self_attn.qkv_proj.weight"] = \
                    np.concatenate([
                        self.weights[q_key],
                        self.weights[k_key],
                        self.weights[v_key],
                    ], axis=0).astype(np.float32)
                del self.weights[q_key], self.weights[k_key], self.weights[v_key]
                # Fuse QKV biases
                if self.attention_bias:
                    qb = pfx + "self_attn.q_proj.bias"
                    kb = pfx + "self_attn.k_proj.bias"
                    vb = pfx + "self_attn.v_proj.bias"
                    if qb in self.weights:
                        self.weights[pfx + "self_attn.qkv_proj.bias"] = \
                            np.concatenate([
                                self.weights[qb].ravel(),
                                self.weights[kb].ravel(),
                                self.weights[vb].ravel(),
                            ]).astype(np.float32)
                        del self.weights[qb], self.weights[kb], self.weights[vb]
            # Fuse gate+up
            gate_key = pfx + "mlp.gate_proj.weight"
            if gate_key in self.weights:
                up_key = pfx + "mlp.up_proj.weight"
                self.weights[pfx + "mlp.gate_up_proj.weight"] = \
                    np.concatenate([
                        self.weights[gate_key],
                        self.weights[up_key],
                    ], axis=0).astype(np.float32)
                del self.weights[gate_key], self.weights[up_key]

    def _dequantize_all_weights(self):
        """Pre-dequantize INT4 weights when GPU INT4 kernels unavailable."""
        import gc
        print("  Pre-dequantizing INT4 weights to fp32...")
        for layer in range(self.n_layer):
            pfx = f"layers.{layer}."
            for suffix in ["self_attn.qkv_proj.weight",
                           "self_attn.o_proj.weight",
                           "mlp.gate_up_proj.weight",
                           "mlp.down_proj.weight"]:
                name = pfx + suffix
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

    def _precompute_rope_tables(self):
        """Pre-compute RoPE cos/sin tables up to MAX_SEQ_LEN."""
        HD = self.head_dim
        half = HD // 2
        inv_freq = 1.0 / (self.rope_theta ** (
            np.arange(0, HD, 2, dtype=np.float32) / HD))
        positions = np.arange(self.MAX_SEQ_LEN, dtype=np.float32)
        angles = positions[:, None] * inv_freq[None, :]
        self._rope_cos = np.cos(angles).astype(np.float32)
        self._rope_sin = np.sin(angles).astype(np.float32)
        if self._decode_mode == 'gpu':
            runner = self.cache.runner
            self._rope_cos_gpu = runner.upload_to_gpu(
                self._rope_cos.ravel(), "rope_cos_table")
            self._rope_sin_gpu = runner.upload_to_gpu(
                self._rope_sin.ravel(), "rope_sin_table")

    def _apply_rope_fast(self, x, positions):
        """Apply RoPE using pre-computed tables."""
        half = self.head_dim // 2
        cos_v = self._rope_cos[positions][:, None, :]
        sin_v = self._rope_sin[positions][:, None, :]
        x1, x2 = x[..., :half], x[..., half:]
        out = np.empty_like(x)
        out[..., :half] = x1 * cos_v - x2 * sin_v
        out[..., half:] = x2 * cos_v + x1 * sin_v
        return out

    def _init_gpu_kv_cache(self):
        """Pre-allocate static GPU KV cache buffers for all layers."""
        runner = self.cache.runner
        n_kv = self.n_kv_heads
        HD = self.head_dim
        buf_size = self.MAX_SEQ_LEN * n_kv * HD
        self._gpu_kv_cache = {}
        for i in range(self.n_layer):
            k_buf = runner.upload_to_gpu(
                np.zeros(buf_size, dtype=np.float32), f"kv_cache_K_{i}")
            v_buf = runner.upload_to_gpu(
                np.zeros(buf_size, dtype=np.float32), f"kv_cache_V_{i}")
            self._gpu_kv_cache[i] = (k_buf, v_buf, 0)
        total_mb = 2 * self.n_layer * buf_size * 4 / 1024 / 1024
        print(f"  GPU KV cache: {total_mb:.0f} MB "
              f"({self.n_layer} layers × {self.MAX_SEQ_LEN} seq)")

    def _upload_weights_to_gpu(self):
        """Upload weights to GPU (Q8/INT4/fp16/fp32 depending on capabilities)."""
        E = self.n_embd
        IM = self.intermediate_size

        use_q8_gpu = self._q8_mode
        use_q4_gpu = (not use_q8_gpu and self._quantized
                      and getattr(self, '_has_fp16_linear', False))
        self._use_q4_gpu = use_q4_gpu
        self._use_q8_gpu = use_q8_gpu
        use_fp16 = (not use_q4_gpu and not use_q8_gpu
                    and getattr(self, '_has_fp16_linear', False))

        for i in range(self.n_layer):
            pfx = f"layers.{i}."
            # RMSNorm
            for nkey in [pfx + "input_layernorm.weight",
                         pfx + "post_attention_layernorm.weight"]:
                if self.weights[nkey].dtype == np.float16:
                    self.weights[nkey] = self.weights[nkey].astype(np.float32)
                self._upload_norm_weight(nkey)

            # QK-norm weights (Qwen3 uses per-head RMSNorm on Q and K)
            for qk_nkey in [pfx + "self_attn.q_norm.weight",
                            pfx + "self_attn.k_norm.weight"]:
                if qk_nkey in self.weights:
                    if self.weights[qk_nkey].dtype == np.float16:
                        self.weights[qk_nkey] = self.weights[qk_nkey].astype(np.float32)
                    self._upload_norm_weight(qk_nkey)

            # Linear projections
            layer_projs = [
                (pfx + "self_attn.qkv_proj.weight", self.qkv_out, E),
                (pfx + "self_attn.o_proj.weight", E, self.q_dim),
                (pfx + "mlp.gate_up_proj.weight", self.gate_up_out, E),
                (pfx + "mlp.down_proj.weight", E, IM),
            ]
            for name, N, K in layer_projs:
                if use_q8_gpu:
                    self._upload_q8_weight(name, N, K)
                elif use_q4_gpu:
                    self._upload_q4_weight(name, N, K)
                elif use_fp16:
                    self._upload_linear_weight_fp16(name, N, K)
                else:
                    self._upload_linear_weight(name, N, K)

            # QKV bias (fused)
            if self.attention_bias:
                bkey = pfx + "self_attn.qkv_proj.bias"
                if bkey in self.weights:
                    b = self.weights[bkey]
                    if b.dtype != np.float32:
                        b = b.astype(np.float32)
                    self._gpu_weights[bkey] = \
                        self.cache.runner.upload_to_gpu(b, bkey)

        # Final RMSNorm
        nkey = "norm.weight"
        if self.weights[nkey].dtype == np.float16:
            self.weights[nkey] = self.weights[nkey].astype(np.float32)
        self._upload_norm_weight(nkey)

        # Embedding / LM head
        ekey = "embed_tokens.weight"
        if self.weights[ekey].dtype == np.float16:
            self.weights[ekey] = self.weights[ekey].astype(np.float32)
        if use_fp16 or use_q4_gpu or use_q8_gpu:
            self._upload_linear_weight_fp16(ekey, self.n_vocab, E)
        self._upload_embedding_weight(ekey, self.n_vocab, E)

        if not self.tie_word_embeddings:
            lm_key = "lm_head.weight"
            if self.weights[lm_key].dtype == np.float16:
                self.weights[lm_key] = self.weights[lm_key].astype(np.float32)
            if use_fp16 or use_q4_gpu or use_q8_gpu:
                self._upload_linear_weight_fp16(lm_key, self.n_vocab, E)
            self._upload_embedding_weight(lm_key, self.n_vocab, E)

        # Zero biases
        self._upload_zero_bias("zero_bias_E", E)
        self._upload_zero_bias("zero_bias_QKV", self.qkv_out)
        self._upload_zero_bias("zero_bias_QO", self.q_dim)
        self._upload_zero_bias("zero_bias_GU", self.gate_up_out)
        self._upload_zero_bias("zero_bias_V", self.n_vocab)

        self._print_gpu_weight_stats()

    # -- Projection helper --

    def _proj(self, x, name, bias_key, N, K=None, gpu_out=False,
              prefer_subgroup_matrix: bool = False):
        """Linear projection using best available kernel (Q8/Q4/fp16/fp32)."""
        if self._use_q8_gpu:
            return self._linear_q8(
                x, self._gpu_weights[name + ".q8.gpu"],
                self._gpu_weights[name + ".q8_scales.gpu"],
                self._gpu_weights[bias_key], N, K=K, gpu_out=gpu_out,
                prefer_subgroup_matrix=prefer_subgroup_matrix)
        if self._use_q4_gpu:
            return self._linear_q4(
                x, self._gpu_weights[name + ".q4.gpu"],
                self._gpu_weights[name + ".scales.gpu"],
                self._gpu_weights[name + ".zeros.gpu"],
                self._gpu_weights[bias_key], N, K=K, gpu_out=gpu_out)
        fp16_name = name + ".fp16"
        if fp16_name in self._gpu_weights:
            return self._linear_fp16w(
                x, self._gpu_weights[fp16_name],
                self._gpu_weights[bias_key], N, K=K, gpu_out=gpu_out)
        return self._linear(
            x, self._gpu_weights[name],
            self._gpu_weights[bias_key], N, gpu_out=gpu_out)

    # -- Attention --

    def _attention_block(self, x, layer: int,
                         use_cache: bool = False,
                         positions: np.ndarray = None, **kwargs):
        """GQA with fused QKV, pre-computed RoPE, vectorized attention."""
        from common.model_base import GPUBuffer

        T = (x.shape[0] if x.shape else 1) if isinstance(x, GPUBuffer) else x.shape[0]
        E, HD = self.n_embd, self.head_dim
        n_head, n_kv, n_rep = self.n_head, self.n_kv_heads, self.n_rep
        pfx = f"layers.{layer}.self_attn."

        # Fused QKV (single dispatch)
        bias_key = (pfx + "qkv_proj.bias") if self.attention_bias else "zero_bias_QKV"

        q_dim = n_head * HD

        # GPU prefill path: keep QKV on GPU, do QK-norm+RoPE on GPU
        if (T > 1 and self._has_qk_norm
                and hasattr(self, '_qknorm_rope_prefill_result')):
            qkv_gpu = self._proj(x, pfx + "qkv_proj.weight", bias_key,
                                 self.qkv_out, K=E, gpu_out=True,
                                 prefer_subgroup_matrix=True)

            if positions is None:
                positions = np.arange(T, dtype=np.int32)
            half = HD // 2
            cos = self._rope_cos[positions]
            sin = self._rope_sin[positions]

            q_norm_w = self.weights.get(pfx + "q_norm.weight")
            k_norm_w = self.weights.get(pfx + "k_norm.weight")

            q_stride_t = n_head * HD
            kv_stride_t = n_kv * HD
            qkv_stride_t = self.qkv_out

            out = self.cache.run(
                self._qknorm_rope_prefill_result,
                grid=(T, n_head + n_kv),
                buffers={
                    'QKV': qkv_gpu,
                    'Q_out': np.zeros(T * n_head * HD, dtype=np.float32),
                    'K_out': np.zeros(T * n_kv * HD, dtype=np.float32),
                    'V_out': np.zeros(T * n_kv * HD, dtype=np.float32),
                    'CosTable': cos.astype(np.float32).ravel(),
                    'SinTable': sin.astype(np.float32).ravel(),
                    'NormQ': q_norm_w.astype(np.float32),
                    'NormK': k_norm_w.astype(np.float32),
                },
                scalars={
                    'n_head': n_head, 'q_size': q_dim,
                    'kv_size': self.kv_dim,
                    'qkv_stride_t': qkv_stride_t,
                    'q_stride_t': q_stride_t,
                    'kv_stride_t': kv_stride_t,
                    'half_rot': half, 'eps': float(self.rms_norm_eps),
                },
                gpu_outputs={'Q_out', 'K_out', 'V_out'})

            Q_gpu = out['Q_out']
            K_gpu = out['K_out']
            V_gpu = out['V_out']
            Q_gpu.shape = (T, n_head, HD)
            K_gpu.shape = (T, n_kv, HD)
            V_gpu.shape = (T, n_kv, HD)

            # GPU causal attention
            attn_out = self._causal_attention_multihead(
                Q_gpu, K_gpu, V_gpu, n_rep, gpu_out=True)

            # Write K/V directly to GPU KV cache via GPU-to-GPU copy
            if use_cache:
                runner = self.cache.runner
                if hasattr(self, '_gpu_kv_cache'):
                    K_cache_gpu, V_cache_gpu, _ = self._gpu_kv_cache[layer]
                    kv_bytes = T * n_kv * HD * 4
                    if runner.is_batching:
                        # GPU-to-GPU copy within batch (no fence)
                        runner.copy_buffer_in_batch(
                            K_gpu.handle, 0, K_cache_gpu.handle, 0, kv_bytes)
                        runner.copy_buffer_in_batch(
                            V_gpu.handle, 0, V_cache_gpu.handle, 0, kv_bytes)
                    else:
                        K_np = runner.readback(K_gpu).reshape(T, n_kv, HD)
                        V_np = runner.readback(V_gpu).reshape(T, n_kv, HD)
                        runner.write_buffer(K_cache_gpu.handle,
                                            K_np.ravel().astype(np.float32).tobytes())
                        runner.write_buffer(V_cache_gpu.handle,
                                            V_np.ravel().astype(np.float32).tobytes())
                        if self.kv_cache is None:
                            self.kv_cache = {}
                        self.kv_cache[layer] = (K_np, V_np)
                    self._gpu_kv_cache[layer] = (K_cache_gpu, V_cache_gpu, T)
                else:
                    K_np = runner.readback(K_gpu).reshape(T, n_kv, HD)
                    V_np = runner.readback(V_gpu).reshape(T, n_kv, HD)
                    if self.kv_cache is None:
                        self.kv_cache = {}
                    self.kv_cache[layer] = (K_np, V_np)

            attn_out.shape = (T, q_dim)
            attn_flat = attn_out
            return self._proj(attn_flat, pfx + "o_proj.weight",
                              "zero_bias_E", E, K=q_dim, gpu_out=True,
                              prefer_subgroup_matrix=True)

        # Standard CPU path (T=1 decode or no QK-norm)
        qkv = self._proj(x, pfx + "qkv_proj.weight", bias_key,
                         self.qkv_out, K=E)
        q = qkv[:, :q_dim]
        k = qkv[:, q_dim:q_dim + self.kv_dim]
        v = qkv[:, q_dim + self.kv_dim:]

        Q = q.reshape(T, n_head, HD)
        K_new = k.reshape(T, n_kv, HD)
        V_new = v.reshape(T, n_kv, HD)

        # QK-norm (per-head RMSNorm on Q and K)
        q_norm_w = self.weights.get(pfx + "q_norm.weight")
        k_norm_w = self.weights.get(pfx + "k_norm.weight")
        if q_norm_w is not None:
            q_rms = np.sqrt(np.mean(Q * Q, axis=-1, keepdims=True) + self.rms_norm_eps)
            Q = Q / q_rms * q_norm_w.astype(np.float32)
            k_rms = np.sqrt(np.mean(K_new * K_new, axis=-1, keepdims=True) + self.rms_norm_eps)
            K_new = K_new / k_rms * k_norm_w.astype(np.float32)

        # RoPE (pre-computed tables)
        if positions is None:
            positions = np.arange(T, dtype=np.int32)
        Q = self._apply_rope_fast(Q, positions)
        K_new = self._apply_rope_fast(K_new, positions)

        # KV cache
        if use_cache:
            if self.kv_cache is not None and layer in self.kv_cache:
                K_prev, V_prev = self.kv_cache[layer]
                K_full = np.concatenate([K_prev, K_new], axis=0)
                V_full = np.concatenate([V_prev, V_new], axis=0)
            else:
                K_full, V_full = K_new, V_new
            if self.kv_cache is None:
                self.kv_cache = {}
            self.kv_cache[layer] = (K_full, V_full)
        else:
            K_full, V_full = K_new, V_new

        T_total = K_full.shape[0]

        if T == 1 and use_cache and T_total > 1:
            # Decode: vectorized multi-head attention
            scale = 1.0 / np.sqrt(HD)
            K_exp = np.repeat(K_full, n_rep, axis=1)
            V_exp = np.repeat(V_full, n_rep, axis=1)
            Q_t = Q.transpose(1, 0, 2)
            K_t = K_exp.transpose(1, 2, 0)
            scores = np.float32((Q_t @ K_t).squeeze(1) * scale)
            scores -= scores.max(axis=1, keepdims=True)
            exp_s = np.exp(scores)
            attn = exp_s / exp_s.sum(axis=1, keepdims=True)
            V_t = V_exp.transpose(1, 0, 2)
            attn_out = (attn[:, None, :] @ V_t).squeeze(1)
            attn_out = attn_out[None, :, :].astype(np.float32)
        else:
            attn_out = self._causal_attention_multihead(
                Q, K_full, V_full, n_rep)

        attn_flat = attn_out.reshape(T, q_dim)
        return self._proj(attn_flat, pfx + "o_proj.weight",
                          "zero_bias_E", E, K=q_dim, gpu_out=True)

    # -- MLP --

    def _mlp_block(self, x, layer: int, gpu_out: bool = False):
        """SwiGLU MLP with fused gate+up projection."""
        from common.model_base import GPUBuffer
        E, IM = self.n_embd, self.intermediate_size
        pfx = f"layers.{layer}.mlp."
        use_prefill_subgroup = isinstance(x, GPUBuffer) and x.shape[0] > 1
        gate_up = self._proj(x, pfx + "gate_up_proj.weight",
                             "zero_bias_GU", self.gate_up_out, K=E,
                             gpu_out=True,
                             prefer_subgroup_matrix=use_prefill_subgroup)
        h = self._silu_mul_fused(gate_up, IM, gpu_out=True)
        return self._proj(h, pfx + "down_proj.weight",
                          "zero_bias_E", E, K=IM, gpu_out=gpu_out,
                          prefer_subgroup_matrix=use_prefill_subgroup)

    # -- Transformer block --

    def _transformer_block(self, x, layer: int,
                           use_cache: bool = False,
                           positions: np.ndarray = None, **kwargs):
        """Pre-norm transformer block with fused residual+norm."""
        from common.model_base import GPUBuffer
        pfx = f"layers.{layer}."

        rn1 = self._rms_norm(
            x, self._gpu_weights[pfx + "input_layernorm.weight"],
            gpu_out=True)
        attn = self._attention_block(rn1, layer, use_cache=use_cache,
                                     positions=positions)

        # Fused residual + norm when available
        use_fused = (hasattr(self, '_add_rn_result')
                     and isinstance(x, GPUBuffer)
                     and isinstance(attn, GPUBuffer))
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

    # -- Fast decode pipeline --

    def _warmup_fast_decode(self):
        """Initialize fast decode pipeline eagerly."""
        if getattr(self, '_fast_decode_ready', False):
            return
        if self._decode_mode != 'gpu':
            return
        if not (self._use_q4_gpu or self._use_q8_gpu):
            return
        self._init_fast_decode()

    def _init_fast_decode(self):
        """Pre-allocate buffers, bind groups, and dispatch lists for fast decode."""
        import struct
        runner = self.cache.runner

        E, HD = self.n_embd, self.head_dim
        n_head, n_kv = self.n_head, self.n_kv_heads
        n_rep = self.n_rep
        IM = self.intermediate_size
        q_size = n_head * HD
        kv_size = n_kv * HD
        qkv_N = self.qkv_out
        half_rot = HD // 2

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

        # Intermediate buffers
        norm_out = mkbuf('norm_out', E * 4)
        qkv_buf = mkbuf('qkv_out', qkv_N * 4)
        q_rot = mkbuf('q_rot', q_size * 4)
        attn_out_buf = mkbuf('attn_out', q_size * 4)
        proj_out = mkbuf('proj_out', E * 4)
        gate_up_buf = mkbuf('gate_up', 2 * IM * 4)
        silu_out = mkbuf('silu_out', IM * 4)
        rstd = mkbuf('rstd', 16)
        x_buf = mkbuf('x', E * 4)

        def get_pl(r):
            return runner.get_pipeline_info(
                r.wgsl, r.buffer_bindings, r.param_fields)

        pl_rn,    bgl_rn    = get_pl(self._rn_result)
        pl_arn,   bgl_arn   = get_pl(self._add_rn_result)
        if self._use_q8_gpu:
            from common.wgsl_kernels import (
                WSGL_Q8_0_KERNEL, Q8_DP4A_BINDINGS, pack_q8_params, Q8_TILE_N,
                WSGL_Q8_0_ADD_KERNEL, Q8_ADD_BINDINGS,
            )
            pl_matmul, bgl_matmul = runner.get_pipeline_info(
                WSGL_Q8_0_KERNEL, Q8_DP4A_BINDINGS, [])
            pl_matmul_add, bgl_matmul_add = runner.get_pipeline_info(
                WSGL_Q8_0_ADD_KERNEL, Q8_ADD_BINDINGS, [])
            nbb_matmul = len(Q8_DP4A_BINDINGS)
            MATMUL_TILE = Q8_TILE_N
        else:
            pl_q4a,   bgl_q4a   = get_pl(self._linear_q4_add_result)
        if self._has_qk_norm:
            pl_fused, bgl_fused = get_pl(self._fused_qknorm_rope_result)
            nbb_fused = len(self._fused_qknorm_rope_result.buffer_bindings)
        else:
            pl_fused, bgl_fused = get_pl(self._fused_rope_result)
            nbb_fused = len(self._fused_rope_result.buffer_bindings)
        pl_attn,  bgl_attn  = get_pl(self._gqa_attn_result)

        # Chunked GQA attention (two-pass for high GPU occupancy)
        from common.wgsl_kernels import (
            WGSL_GQA_CHUNKED_PASS1, WGSL_GQA_CHUNKED_PASS2,
            GQA_CHUNKED_PASS1_BINDINGS, GQA_CHUNKED_PASS2_BINDINGS,
            GQA_CHUNK_SIZE,
        )
        pl_attn_p1, bgl_attn_p1 = runner.get_pipeline_info(
            WGSL_GQA_CHUNKED_PASS1, GQA_CHUNKED_PASS1_BINDINGS, [])
        pl_attn_p2, bgl_attn_p2 = runner.get_pipeline_info(
            WGSL_GQA_CHUNKED_PASS2, GQA_CHUNKED_PASS2_BINDINGS, [])
        max_n_chunks = (self.MAX_SEQ_LEN + GQA_CHUNK_SIZE - 1) // GQA_CHUNK_SIZE
        self._gqa_chunk_size = GQA_CHUNK_SIZE
        partial_stride = HD + 2
        partials_buf = mkbuf('attn_partials',
                             n_head * max_n_chunks * partial_stride * 4)

        pl_aip,   bgl_aip   = get_pl(self._add_ip_result)
        pl_silu,  bgl_silu  = get_pl(self._smf_result)
        if not self._use_q8_gpu:
            pl_q4,    bgl_q4    = get_pl(self._linear_q4_result)

        nbb_rn = len(self._rn_result.buffer_bindings)
        nbb_arn = len(self._add_rn_result.buffer_bindings)
        nbb_attn = len(self._gqa_attn_result.buffer_bindings)
        nbb_aip = len(self._add_ip_result.buffer_bindings)
        nbb_silu = len(self._smf_result.buffer_bindings)
        if not self._use_q8_gpu:
            nbb_q4 = len(self._linear_q4_result.buffer_bindings)

        def mk_params(name, pf, vals):
            b = pack(pf, vals)
            h = mkbuf(name, len(b))
            runner.write_buffer(h[0], b)
            return h

        rn_p = mk_params('rn_p', self._rn_result.param_fields,
                          {'stride': E, 'N': E, 'eps': self.rms_norm_eps})
        arn_p = mk_params('arn_p', self._add_rn_result.param_fields,
                          {'stride': E, 'N': E, 'eps': self.rms_norm_eps})
        aip_p = mk_params('aip_p', self._add_ip_result.param_fields,
                          {'N': E})
        silu_p = mk_params('silu_p', self._smf_result.param_fields,
                           {'N': IM})

        if self._use_q8_gpu:
            # Q8 params: just (K, N) as 2 u32s — 8 bytes, padded to 16
            def mk_q8_params(name, K_val, N_val):
                data = struct.pack('<II', K_val, N_val)
                while len(data) < 16:
                    data += b'\x00'
                h = mkbuf(name, len(data))
                runner.write_buffer(h[0], data)
                return h

            matmul_qkv_p  = mk_q8_params('q8_qkv_p', E, qkv_N)
            matmul_oproj_p = mk_q8_params('q8_oproj_p', q_size, E)
            matmul_gateup_p = mk_q8_params('q8_gateup_p', E, 2 * IM)
            matmul_down_p  = mk_q8_params('q8_down_p', IM, E)
        else:
            Q4_GS = 128

            def mk_q4_params(name, K_val, N_val):
                pf = self._linear_q4_result.param_fields
                return mk_params(name, pf, {
                    'K': K_val, 'stride_x': K_val,
                    'stride_w_q4': K_val // 8,
                    'n_groups': K_val // Q4_GS, 'N': N_val,
                })

            matmul_qkv_p   = mk_q4_params('q4_qkv_p', E, qkv_N)
            matmul_oproj_p = mk_q4_params('q4_oproj_p', q_size, E)
            matmul_gateup_p = mk_q4_params('q4_gateup_p', E, 2 * IM)
            matmul_down_p  = mk_q4_params('q4_down_p', IM, E)

        # Dynamic params (updated per token)
        if self._has_qk_norm:
            fused_rope_ph = mkbuf('fused_rope_p', 32)  # extra eps field
        else:
            fused_rope_ph = mkbuf('fused_rope_p', 28)
        attn_ph = mkbuf('attn_p', 24)
        # Chunked attention params: kv_stride, n_rep, T_total, chunk_size,
        # n_chunks, scale_bits, neg_inf_bits (7 u32 = 28 bytes, pad to 32)
        chunked_attn_ph = mkbuf('chunked_attn_p', 32)

        cos_h = self._rope_cos_gpu.handle
        cos_sz = self._rope_cos_gpu.size
        sin_h = self._rope_sin_gpu.handle
        sin_sz = self._rope_sin_gpu.size

        # Bias: for QKV, use actual bias if attention_bias, else zero
        if self.attention_bias:
            bias_qkv = self._gpu_weights[
                f"layers.0.self_attn.qkv_proj.bias"]
        else:
            bias_qkv = self._gpu_weights["zero_bias_QKV"]
        bias_e = self._gpu_weights["zero_bias_E"]
        bias_gu = self._gpu_weights["zero_bias_GU"]
        bias_v = self._gpu_weights["zero_bias_V"]
        norm_final_w = self._gpu_weights["norm.weight"]

        mk_bg = runner.create_bind_group

        bg_silu = mk_bg(bgl_silu, [
            (0, gate_up_buf[0], gate_up_buf[1]),
            (1, silu_out[0], silu_out[1]),
            (nbb_silu, silu_p[0], silu_p[1])])

        aip_g = ((E + self._add_block - 1) // self._add_block,)
        silu_g = ((IM + self._smf_block - 1) // self._smf_block,)

        layer_dispatches = []
        for i in range(self.n_layer):
            pfx = f"layers.{i}."
            norm1_w = self._gpu_weights[pfx + "input_layernorm.weight"]
            norm2_w = self._gpu_weights[pfx + "post_attention_layernorm.weight"]
            K_cache, V_cache, _ = self._gpu_kv_cache[i]

            # Per-layer QKV bias
            if self.attention_bias:
                layer_bias_qkv = self._gpu_weights[pfx + "self_attn.qkv_proj.bias"]
            else:
                layer_bias_qkv = bias_qkv

            # --- Build matmul bind groups (Q8 or Q4) ---
            if self._use_q8_gpu:
                def _mk_q8_bg(x_buf_pair, w_name, bias_gpu, out_buf_pair,
                              params_pair):
                    wq = self._gpu_weights[w_name + ".q8.gpu"]
                    sc = self._gpu_weights[w_name + ".q8_scales.gpu"]
                    # Q8 bindings: X(0), W_Q8(1), Scales(2), Bias(3), Y(4), _params_(5)
                    return mk_bg(bgl_matmul, [
                        (0, x_buf_pair[0], x_buf_pair[1]),
                        (1, wq.handle, wq.size),
                        (2, sc.handle, sc.size),
                        (3, bias_gpu.handle, bias_gpu.size),
                        (4, out_buf_pair[0], out_buf_pair[1]),
                        (5, params_pair[0], params_pair[1])])

                bg_qkv = _mk_q8_bg(
                    norm_out,
                    pfx + "self_attn.qkv_proj.weight",
                    layer_bias_qkv, qkv_buf, matmul_qkv_p)
                bg_op = _mk_q8_bg(
                    attn_out_buf,
                    pfx + "self_attn.o_proj.weight",
                    bias_e, proj_out, matmul_oproj_p)
                bg_gu = _mk_q8_bg(
                    norm_out,
                    pfx + "mlp.gate_up_proj.weight",
                    bias_gu, gate_up_buf, matmul_gateup_p)
                bg_dn = _mk_q8_bg(
                    silu_out,
                    pfx + "mlp.down_proj.weight",
                    bias_e, proj_out, matmul_down_p)

                # Fused down_proj + residual add: writes directly to x_buf
                dn_w = self._gpu_weights[pfx + "mlp.down_proj.weight.q8.gpu"]
                dn_s = self._gpu_weights[pfx + "mlp.down_proj.weight.q8_scales.gpu"]
                bg_dn_add = mk_bg(bgl_matmul_add, [
                    (0, silu_out[0], silu_out[1]),
                    (1, dn_w.handle, dn_w.size),
                    (2, dn_s.handle, dn_s.size),
                    (3, bias_e.handle, bias_e.size),
                    (4, x_buf[0], x_buf[1]),
                    (5, matmul_down_p[0], matmul_down_p[1])])

                qkv_grid = (1, (qkv_N + MATMUL_TILE - 1) // MATMUL_TILE)
                op_grid  = (1, (E + MATMUL_TILE - 1) // MATMUL_TILE)
                gu_grid  = (1, (2 * IM + MATMUL_TILE - 1) // MATMUL_TILE)
                dn_grid  = (1, (E + MATMUL_TILE - 1) // MATMUL_TILE)
            else:
                qkv_wq = self._gpu_weights[pfx + "self_attn.qkv_proj.weight.q4.gpu"]
                qkv_sc = self._gpu_weights[pfx + "self_attn.qkv_proj.weight.scales.gpu"]
                qkv_zr = self._gpu_weights[pfx + "self_attn.qkv_proj.weight.zeros.gpu"]
                op_wq = self._gpu_weights[pfx + "self_attn.o_proj.weight.q4.gpu"]
                op_sc = self._gpu_weights[pfx + "self_attn.o_proj.weight.scales.gpu"]
                op_zr = self._gpu_weights[pfx + "self_attn.o_proj.weight.zeros.gpu"]
                gu_wq = self._gpu_weights[pfx + "mlp.gate_up_proj.weight.q4.gpu"]
                gu_sc = self._gpu_weights[pfx + "mlp.gate_up_proj.weight.scales.gpu"]
                gu_zr = self._gpu_weights[pfx + "mlp.gate_up_proj.weight.zeros.gpu"]
                dn_wq = self._gpu_weights[pfx + "mlp.down_proj.weight.q4.gpu"]
                dn_sc = self._gpu_weights[pfx + "mlp.down_proj.weight.scales.gpu"]
                dn_zr = self._gpu_weights[pfx + "mlp.down_proj.weight.zeros.gpu"]

                bg_qkv = mk_bg(bgl_q4, [
                    (0, norm_out[0], norm_out[1]),
                    (1, qkv_wq.handle, qkv_wq.size),
                    (2, qkv_sc.handle, qkv_sc.size),
                    (3, qkv_zr.handle, qkv_zr.size),
                    (4, layer_bias_qkv.handle, layer_bias_qkv.size),
                    (5, qkv_buf[0], qkv_buf[1]),
                    (nbb_q4, matmul_qkv_p[0], matmul_qkv_p[1])])

                bg_op = mk_bg(bgl_q4, [
                    (0, attn_out_buf[0], attn_out_buf[1]),
                    (1, op_wq.handle, op_wq.size),
                    (2, op_sc.handle, op_sc.size),
                    (3, op_zr.handle, op_zr.size),
                    (4, bias_e.handle, bias_e.size),
                    (5, proj_out[0], proj_out[1]),
                    (nbb_q4, matmul_oproj_p[0], matmul_oproj_p[1])])

                bg_gu = mk_bg(bgl_q4, [
                    (0, norm_out[0], norm_out[1]),
                    (1, gu_wq.handle, gu_wq.size),
                    (2, gu_sc.handle, gu_sc.size),
                    (3, gu_zr.handle, gu_zr.size),
                    (4, bias_gu.handle, bias_gu.size),
                    (5, gate_up_buf[0], gate_up_buf[1]),
                    (nbb_q4, matmul_gateup_p[0], matmul_gateup_p[1])])

                bg_dn = mk_bg(bgl_q4, [
                    (0, silu_out[0], silu_out[1]),
                    (1, dn_wq.handle, dn_wq.size),
                    (2, dn_sc.handle, dn_sc.size),
                    (3, dn_zr.handle, dn_zr.size),
                    (4, bias_e.handle, bias_e.size),
                    (5, proj_out[0], proj_out[1]),
                    (nbb_q4, matmul_down_p[0], matmul_down_p[1])])

                qkv_grid = (1, qkv_N)
                op_grid  = (1, E)
                gu_grid  = (1, 2 * IM)
                dn_grid  = (1, E)

            bg_n1 = mk_bg(bgl_rn, [
                (0, x_buf[0], x_buf[1]),
                (1, norm_out[0], norm_out[1]),
                (2, norm1_w.handle, norm1_w.size),
                (3, rstd[0], rstd[1]),
                (nbb_rn, rn_p[0], rn_p[1])])

            if self._has_qk_norm:
                norm_q_w = self._gpu_weights[pfx + "self_attn.q_norm.weight"]
                norm_k_w = self._gpu_weights[pfx + "self_attn.k_norm.weight"]
                bg_frope = mk_bg(bgl_fused, [
                    (0, qkv_buf[0], qkv_buf[1]),
                    (1, q_rot[0], q_rot[1]),
                    (2, K_cache.handle, K_cache.size),
                    (3, V_cache.handle, V_cache.size),
                    (4, cos_h, cos_sz), (5, sin_h, sin_sz),
                    (6, norm_q_w.handle, norm_q_w.size),
                    (7, norm_k_w.handle, norm_k_w.size),
                    (nbb_fused, fused_rope_ph[0], fused_rope_ph[1])])
            else:
                bg_frope = mk_bg(bgl_fused, [
                    (0, qkv_buf[0], qkv_buf[1]),
                    (1, q_rot[0], q_rot[1]),
                    (2, K_cache.handle, K_cache.size),
                    (3, V_cache.handle, V_cache.size),
                    (4, cos_h, cos_sz), (5, sin_h, sin_sz),
                    (nbb_fused, fused_rope_ph[0], fused_rope_ph[1])])

            bg_att = mk_bg(bgl_attn, [
                (0, q_rot[0], q_rot[1]),
                (1, K_cache.handle, K_cache.size),
                (2, V_cache.handle, V_cache.size),
                (3, attn_out_buf[0], attn_out_buf[1]),
                (nbb_attn, attn_ph[0], attn_ph[1])])

            # Chunked attention pass 1: Q, K_cache, V_cache → Partials
            bg_attn_p1 = mk_bg(bgl_attn_p1, [
                (0, q_rot[0], q_rot[1]),
                (1, K_cache.handle, K_cache.size),
                (2, V_cache.handle, V_cache.size),
                (3, partials_buf[0], partials_buf[1]),
                (4, chunked_attn_ph[0], chunked_attn_ph[1])])

            # Chunked attention pass 2: Partials → attn_out
            bg_attn_p2 = mk_bg(bgl_attn_p2, [
                (0, partials_buf[0], partials_buf[1]),
                (1, attn_out_buf[0], attn_out_buf[1]),
                (2, chunked_attn_ph[0], chunked_attn_ph[1])])

            bg_arn2 = mk_bg(bgl_arn, [
                (0, x_buf[0], x_buf[1]),
                (1, proj_out[0], proj_out[1]),
                (2, norm_out[0], norm_out[1]),
                (3, norm2_w.handle, norm2_w.size),
                (4, rstd[0], rstd[1]),
                (nbb_arn, arn_p[0], arn_p[1])])

            bg_res = mk_bg(bgl_aip, [
                (0, x_buf[0], x_buf[1]),
                (1, proj_out[0], proj_out[1]),
                (nbb_aip, aip_p[0], aip_p[1])])

            # Select pipeline and grid for matmul dispatches
            if self._use_q8_gpu:
                pl_mm = pl_matmul
            else:
                pl_mm = pl_q4

            if i < self.n_layer - 1:
                next_norm1_w = self._gpu_weights[
                    f"layers.{i+1}.input_layernorm.weight"]
                bg_rn_next = mk_bg(bgl_rn, [
                    (0, x_buf[0], x_buf[1]),
                    (1, norm_out[0], norm_out[1]),
                    (2, next_norm1_w.handle, next_norm1_w.size),
                    (3, rstd[0], rstd[1]),
                    (nbb_rn, rn_p[0], rn_p[1])])
                dispatches = []
                if i == 0:
                    dispatches.append((pl_rn, bg_n1, (1,)))
                dispatches.extend([
                    (pl_mm,    bg_qkv,     qkv_grid),
                    (pl_fused, bg_frope,    (n_head + n_kv,)),
                    (pl_attn_p1, bg_attn_p1, (n_head, max_n_chunks)),
                    (pl_attn_p2, bg_attn_p2, (n_head,)),
                    (pl_mm,    bg_op,       op_grid),
                    (pl_arn,   bg_arn2,     (1,)),
                    (pl_mm,    bg_gu,       gu_grid),
                    (pl_silu,  bg_silu,     silu_g),
                    (pl_matmul_add if self._use_q8_gpu else pl_mm,
                     bg_dn_add if self._use_q8_gpu else bg_dn,
                     dn_grid),
                    (pl_rn,    bg_rn_next,  (1,)),
                ])
            else:
                dispatches = [
                    (pl_mm,    bg_qkv,    qkv_grid),
                    (pl_fused, bg_frope,   (n_head + n_kv,)),
                    (pl_attn_p1, bg_attn_p1, (n_head, max_n_chunks)),
                    (pl_attn_p2, bg_attn_p2, (n_head,)),
                    (pl_mm,    bg_op,      op_grid),
                    (pl_arn,   bg_arn2,    (1,)),
                    (pl_mm,    bg_gu,      gu_grid),
                    (pl_silu,  bg_silu,    silu_g),
                    (pl_matmul_add if self._use_q8_gpu else pl_mm,
                     bg_dn_add if self._use_q8_gpu else bg_dn,
                     dn_grid),
                ]
            layer_dispatches.append(dispatches)

        # Final norm + LM head
        bg_final_rn = mk_bg(bgl_rn, [
            (0, x_buf[0], x_buf[1]),
            (1, norm_out[0], norm_out[1]),
            (2, norm_final_w.handle, norm_final_w.size),
            (3, rstd[0], rstd[1]),
            (nbb_rn, rn_p[0], rn_p[1])])

        logits_buf = mkbuf('logits', self.n_vocab * 4)
        ekey = ("embed_tokens.weight" if self.tie_word_embeddings
                else "lm_head.weight")

        # Always use fp16 for LM head (INT4 degrades quality for small models)
        lm_w_fp16 = self._gpu_weights[ekey + ".fp16"]

        pl_lmh, bgl_lmh = get_pl(self._linear_fp16w_wide_result)
        nbb_lmh = len(self._linear_fp16w_wide_result.buffer_bindings)
        lm_gy = self.MAX_DISPATCH_DIM
        lm_gx = (self.n_vocab + lm_gy - 1) // lm_gy
        lmh_p = mk_params('lmh_p',
                           self._linear_fp16w_wide_result.param_fields,
                           {'K': E, 'stride_x': E, 'stride_w': E,
                            'N': self.n_vocab, 'grid_y': lm_gy})
        bg_lmh = mk_bg(bgl_lmh, [
            (0, norm_out[0], norm_out[1]),
            (1, lm_w_fp16.handle, lm_w_fp16.size),
            (2, bias_v.handle, bias_v.size),
            (3, logits_buf[0], logits_buf[1]),
            (nbb_lmh, lmh_p[0], lmh_p[1])])

        # Use N-tiled matvec for LM head on iGPUs (8× fewer workgroups)
        # Discrete GPUs prefer the wide kernel (larger scheduler)
        use_matvec = (hasattr(self, '_linear_fp16w_matvec_result')
                      and self._gpu_vendor != 'nvidia')
        if use_matvec:
            pl_lmh_mv, bgl_lmh_mv = get_pl(self._linear_fp16w_matvec_result)
            nbb_lmh_mv = len(self._linear_fp16w_matvec_result.buffer_bindings)
            BN = self._matvec_block_n
            lm_grid_n = (self.n_vocab + BN - 1) // BN
            lmh_mv_p = mk_params('lmh_mv_p',
                                  self._linear_fp16w_matvec_result.param_fields,
                                  {'K': E, 'stride_w': E, 'N': self.n_vocab})
            bg_lmh_mv = mk_bg(bgl_lmh_mv, [
                (0, norm_out[0], norm_out[1]),
                (1, lm_w_fp16.handle, lm_w_fp16.size),
                (2, bias_v.handle, bias_v.size),
                (3, logits_buf[0], logits_buf[1]),
                (nbb_lmh_mv, lmh_mv_p[0], lmh_mv_p[1])])
            final_dispatches = [
                (pl_rn,  bg_final_rn, (1,)),
                (pl_lmh_mv, bg_lmh_mv, (lm_grid_n,)),
            ]
        else:
            final_dispatches = [
                (pl_rn,  bg_final_rn, (1,)),
                (pl_lmh, bg_lmh,      (lm_gx, lm_gy)),
            ]

        self._fd_x_h = x_buf[0]
        self._fd_logits_h = logits_buf[0]
        self._fd_logits_sz = logits_buf[1]
        self._fd_fused_rope_ph = fused_rope_ph[0]
        self._fd_attn_ph = attn_ph[0]
        self._fd_chunked_attn_ph = chunked_attn_ph[0]

        # GPU argmax: readback 4 bytes instead of 592KB.
        # On discrete NVIDIA GPUs, buffer mapping has fixed PCIe latency
        # regardless of size, so the extra dispatch hurts more than the
        # smaller readback helps. On integrated GPUs the mapping cost
        # scales with buffer size, so GPU argmax is beneficial there.
        self._fd_use_gpu_argmax = (self._gpu_vendor != 'nvidia')
        if self._fd_use_gpu_argmax:
            token_id_buf = mkbuf('token_id', 4)
            pl_am, bgl_am = get_pl(self._argmax_result)
            nbb_am = len(self._argmax_result.buffer_bindings)
            am_p = mk_params('am_p', self._argmax_result.param_fields,
                              {'N': self.n_vocab})
            bg_am = mk_bg(bgl_am, [
                (0, logits_buf[0], logits_buf[1]),
                (1, token_id_buf[0], token_id_buf[1]),
                (nbb_am, am_p[0], am_p[1])])
            argmax_dispatch = [(pl_am, bg_am, (1,))]
            self._fd_token_id_h = token_id_buf[0]
            self._fd_token_id_sz = token_id_buf[1]

        # Full pipeline
        if self._fd_use_gpu_argmax:
            self._fd_all_batches = (
                layer_dispatches +
                [final_dispatches] +
                [argmax_dispatch]
            )
        else:
            self._fd_all_batches = layer_dispatches + [final_dispatches]

        # Set up persistently-mapped readback buffer (Dawn-specific).
        # With skip_validation enabled, the buffer stays mapped while being
        # used as COPY_DST, eliminating the BufferMapAsync/WaitAny/Unmap
        # cycle (~4ms per token).
        try:
            if self._fd_use_gpu_argmax:
                rb_size = 4
                rb_dtype = np.int32
            else:
                rb_size = self.n_vocab * 4
                rb_dtype = np.float32
            prb_buf, prb_view = runner.create_persistently_mapped_readback(
                "__persistent_rb__", rb_size, rb_dtype)
            runner._persistent_rb = (prb_buf, prb_view, rb_size)
        except Exception:
            runner._persistent_rb = None

        # Per-dispatch names for GPU timestamp profiling
        mm_label = "q8_matmul" if self._use_q8_gpu else "q4_matmul"
        self._fd_dispatch_names = []
        for i in range(self.n_layer):
            pfx = f"L{i}/"
            names = []
            if i == 0:
                names.append(pfx + "rms_norm")
            names.extend([
                pfx + mm_label + "_qkv",
                pfx + "fused_rope",
                pfx + "attn_chunk",
                pfx + "attn_reduce",
                pfx + mm_label + "_oproj",
                pfx + "add_rms_norm",
                pfx + mm_label + "_gateup",
                pfx + "silu_mul",
                pfx + mm_label + "_down_add",
            ])
            if i < self.n_layer - 1:
                names.append(pfx + "rms_norm_next")
            self._fd_dispatch_names.extend(names)
        self._fd_dispatch_names.append("final_rms_norm")
        self._fd_dispatch_names.append("lm_head")
        if self._fd_use_gpu_argmax:
            self._fd_dispatch_names.append("argmax")

        self._fd_all_flat = []
        for batch in self._fd_all_batches:
            self._fd_all_flat.extend(batch)

        MERGE = 6
        self._fd_merged_batches = []
        for i in range(0, len(self._fd_all_batches), MERGE):
            merged = []
            for batch in self._fd_all_batches[i:i+MERGE]:
                merged.extend(batch)
            self._fd_merged_batches.append(merged)

        if self._has_qk_norm:
            self._fd_frope_buf = bytearray(32)
            struct.pack_into('<iii', self._fd_frope_buf, 0,
                             n_head, q_size, kv_size)
            struct.pack_into('<i', self._fd_frope_buf, 16, half_rot)
            # eps at offset 24
            struct.pack_into('<f', self._fd_frope_buf, 24,
                             float(np.float32(self.rms_norm_eps)))
        else:
            self._fd_frope_buf = bytearray(28)
            struct.pack_into('<iii', self._fd_frope_buf, 0,
                             n_head, q_size, kv_size)
            struct.pack_into('<i', self._fd_frope_buf, 16, half_rot)
        self._fd_attn_buf = bytearray(24)
        struct.pack_into('<ii', self._fd_attn_buf, 0, n_kv * HD, n_rep)
        struct.pack_into('<ff', self._fd_attn_buf, 12,
                         float(np.float32(1.0 / np.sqrt(HD))),
                         float(np.float32(-1e9)))

        # Chunked attention params: [kv_stride, n_rep, T_total, chunk_size,
        #   n_chunks, scale_bits, neg_inf_bits] = 7 u32s = 28 bytes → pad to 32
        self._fd_chunked_attn_buf = bytearray(32)
        struct.pack_into('<II', self._fd_chunked_attn_buf, 0,
                         n_kv * HD, n_rep)
        # T_total at offset 8 (updated per token)
        struct.pack_into('<I', self._fd_chunked_attn_buf, 12,
                         GQA_CHUNK_SIZE)
        # n_chunks at offset 16 (updated per token)
        struct.pack_into('<i', self._fd_chunked_attn_buf, 20,
                         struct.unpack('<i', struct.pack('<f',
                             float(np.float32(1.0 / np.sqrt(HD)))))[0])
        struct.pack_into('<i', self._fd_chunked_attn_buf, 24,
                         struct.unpack('<i', struct.pack('<f',
                             float(np.float32(-1e9))))[0])

        self._fast_decode_ready = True
        print(f"  Fast decode initialized "
              f"({sum(len(b) for b in self._fd_all_batches)} "
              f"pre-recorded dispatches)")

    def _decode_fast(self, token_ids, pos_offset):
        """Fast decode: submit pre-recorded dispatches with minimal overhead."""
        import time as _time, struct
        runner = self.cache.runner
        wte = self.weights["embed_tokens.weight"]
        p = self.profiler
        _p = p and p.enabled

        if _p: p._cpu.begin("fast_decode/embed")
        x = wte[token_ids].ravel().astype(np.float32)
        runner.write_buffer(self._fd_x_h, x.tobytes())
        if _p: p._cpu.end("fast_decode/embed")

        if _p: p._cpu.begin("fast_decode/params")
        pos = pos_offset
        _, _, cur_len = self._gpu_kv_cache[0]
        cache_offset = cur_len * self.n_kv_heads * self.head_dim
        T_total = cur_len + 1

        struct.pack_into('<i', self._fd_frope_buf, 12, pos)
        struct.pack_into('<i', self._fd_frope_buf, 20, cache_offset)
        runner.write_buffer(self._fd_fused_rope_ph,
                            bytes(self._fd_frope_buf))
        struct.pack_into('<i', self._fd_attn_buf, 8, T_total)
        runner.write_buffer(self._fd_attn_ph, bytes(self._fd_attn_buf))

        n_chunks = (T_total + self._gqa_chunk_size - 1) // self._gqa_chunk_size
        struct.pack_into('<I', self._fd_chunked_attn_buf, 8, T_total)
        struct.pack_into('<I', self._fd_chunked_attn_buf, 16, n_chunks)
        runner.write_buffer(self._fd_chunked_attn_ph,
                            bytes(self._fd_chunked_attn_buf))

        if _p: p._cpu.end("fast_decode/params")

        for layer in range(self.n_layer):
            K, V, c = self._gpu_kv_cache[layer]
            self._gpu_kv_cache[layer] = (K, V, c + 1)

        if self._fd_use_gpu_argmax:
            readback = (self._fd_token_id_h, self._fd_token_id_sz,
                        np.int32)
        else:
            readback = (self._fd_logits_h, self._fd_logits_sz,
                        np.float32)

        if _p:
            from common.profiler import GPUDispatchEvent

            p._cpu.begin("fast_decode/gpu")
            t0 = _time.perf_counter_ns()

            result = runner.submit_dispatches_pipelined(
                self._fd_all_batches,
                readback=readback,
                profiler=p,
                dispatch_names=self._fd_dispatch_names)

            t1 = _time.perf_counter_ns()
            p._cpu.end("fast_decode/gpu")

            n_disp = sum(len(b) for b in self._fd_all_batches)
            link = f"fd_{pos}"
            p._cpu._events[-1].link_id = link
            p._dispatch_events.append(GPUDispatchEvent(
                name=f"fast_decode({n_disp}disp)",
                begin_ns=t0, end_ns=t1, link_id=link))
        else:
            result = runner.submit_dispatches_pipelined(
                self._fd_all_batches, readback=readback)

        if self._fd_use_gpu_argmax:
            logits = np.full(self.n_vocab, -1e9, dtype=np.float32)
            logits[int(result[0])] = 1.0
        else:
            logits = result

        return logits.reshape(1, self.n_vocab)

    # -- Forward pass --

    def forward(self, token_ids: np.ndarray,
                use_cache: bool = False,
                pos_offset: int = 0) -> np.ndarray:
        """Run Qwen3 forward pass."""
        T = len(token_ids)

        # Fast decode path
        if (T == 1 and use_cache and self._decode_mode == 'gpu'
                and (self._use_q4_gpu or self._use_q8_gpu)):
            if not getattr(self, '_fast_decode_ready', False):
                self._init_fast_decode()
            return self._decode_fast(token_ids, pos_offset)

        wte = self.weights["embed_tokens.weight"]
        x = wte[token_ids]
        positions = np.arange(pos_offset, pos_offset + T, dtype=np.int32)

        for layer in range(self.n_layer):
            x = self._transformer_block(x, layer, use_cache=use_cache,
                                        positions=positions)

        # Sync CPU KV cache to GPU for subsequent fast decode steps
        if (use_cache and self._decode_mode == 'gpu'
                and hasattr(self, '_gpu_kv_cache') and self.kv_cache):
            runner = self.cache.runner
            HD = self.head_dim
            n_kv = self.n_kv_heads
            for layer_idx in range(self.n_layer):
                if layer_idx not in self.kv_cache:
                    continue
                K_gpu, V_gpu, cached_len = self._gpu_kv_cache[layer_idx]
                # Skip if already synced by GPU prefill path
                if cached_len > 0:
                    continue
                K_cpu, V_cpu = self.kv_cache[layer_idx]
                T_cached = K_cpu.shape[0]
                k_bytes = K_cpu.ravel().astype(np.float32).tobytes()
                v_bytes = V_cpu.ravel().astype(np.float32).tobytes()
                runner.write_buffer(K_gpu.handle, k_bytes)
                runner.write_buffer(V_gpu.handle, v_bytes)
                self._gpu_kv_cache[layer_idx] = (K_gpu, V_gpu, T_cached)

        x = self._rms_norm(x, self._gpu_weights["norm.weight"])

        lm_key = ("embed_tokens.weight" if self.tie_word_embeddings
                   else "lm_head.weight")
        # Always use fp16 weight for LM head (INT4 quantization degrades
        # quality for small models; fp16 is lossless from BF16 originals)
        fp16_key = lm_key + ".fp16"
        if fp16_key in self._gpu_weights:
            logits = self._linear_fp16w(
                x, self._gpu_weights[fp16_key],
                self._gpu_weights["zero_bias_V"], self.n_vocab,
                K=self.n_embd)
        else:
            logits = self._linear(
                x, self._gpu_weights[lm_key],
                self._gpu_weights["zero_bias_V"], self.n_vocab)
        return logits


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------

def quantize_qwen_weights(weights: Dict[str, np.ndarray],
                          n_layer: int) -> Dict[str, np.ndarray]:
    """Quantize Qwen3 fused weights to INT4 with per-group fp16 scales."""
    linear_keys = set()
    for i in range(n_layer):
        pfx = f"layers.{i}."
        linear_keys.update([
            pfx + "self_attn.qkv_proj.weight",
            pfx + "self_attn.o_proj.weight",
            pfx + "mlp.gate_up_proj.weight",
            pfx + "mlp.down_proj.weight",
        ])

    quantized = {}
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
            total_quant += q_packed.nbytes + scales.nbytes + zeros.nbytes
        else:
            fp16 = val.astype(np.float16) if val.dtype in (
                np.float32, np.float64) else val
            quantized[key] = fp16
            total_quant += fp16.nbytes

    print(f"  Original: {total_orig / 1024**3:.1f} GB")
    print(f"  Quantized: {total_quant / 1024**3:.1f} GB")
    print(f"  Compression: {total_orig / max(total_quant, 1):.1f}×")
    return quantized


# ---------------------------------------------------------------------------
# GGUF loading
# ---------------------------------------------------------------------------

def load_gguf_qwen3(gguf_path: str, config: dict) -> Dict[str, np.ndarray]:
    """Load Qwen3 weights from GGUF file with fast mmap parsing.

    Dequantizes Q8_0 tensors to fp16 using pure numpy (no torch needed).
    Maps GGUF tensor names to Backpack internal names.
    """
    import gc
    from common.gguf_utils import GGUFFile, dequantize_q8_0

    n_layer = config["n_layer"]

    t_start = time.time()
    print(f"Loading GGUF: {gguf_path}")
    gf = GGUFFile(gguf_path)
    print(f"  Parsed {len(gf.tensors)} tensor headers in "
          f"{(time.time() - t_start)*1000:.0f}ms")

    out = {}

    def _load_tensor(backpack_name, gguf_name, out_dtype=np.float16):
        """Load and dequantize a GGUF tensor."""
        info = gf.tensors[gguf_name]
        if info.qtype == 0:  # F32
            arr = gf.tensor_data_f32(gguf_name)
            if len(info.shape) == 2:
                K, N = info.shape
                arr = arr.reshape(N, K)
            out[backpack_name] = arr.astype(out_dtype, copy=True)
        elif info.qtype == 1:  # F16
            arr = gf.tensor_data_f16(gguf_name)
            if len(info.shape) == 2:
                K, N = info.shape
                arr = arr.reshape(N, K)
            out[backpack_name] = np.array(arr, dtype=out_dtype)
        elif info.qtype == 8:  # Q8_0
            raw = gf.tensor_data(gguf_name)
            out[backpack_name] = dequantize_q8_0(raw, info.shape,
                                                  out_dtype=out_dtype)
        else:
            raise ValueError(
                f"Unsupported quant type {info.qtype_name} for {gguf_name}")

    # Global tensors
    _load_tensor("embed_tokens.weight", "token_embd.weight")
    _load_tensor("norm.weight", "output_norm.weight", out_dtype=np.float32)

    # lm_head (Qwen3 ties embeddings — no output.weight in GGUF)
    has_lm_head = "output.weight" in gf.tensors
    if has_lm_head:
        _load_tensor("lm_head.weight", "output.weight")

    # Per-layer tensors
    for i in range(n_layer):
        src = f"blk.{i}."
        dst = f"layers.{i}."

        # Norms (F32 → keep as f32)
        _load_tensor(dst + "input_layernorm.weight",
                     src + "attn_norm.weight", out_dtype=np.float32)
        _load_tensor(dst + "post_attention_layernorm.weight",
                     src + "ffn_norm.weight", out_dtype=np.float32)

        # QK-norm
        if src + "attn_q_norm.weight" in gf.tensors:
            _load_tensor(dst + "self_attn.q_norm.weight",
                         src + "attn_q_norm.weight", out_dtype=np.float32)
            _load_tensor(dst + "self_attn.k_norm.weight",
                         src + "attn_k_norm.weight", out_dtype=np.float32)

        # Attention projections
        _load_tensor(dst + "self_attn.q_proj.weight",
                     src + "attn_q.weight")
        _load_tensor(dst + "self_attn.k_proj.weight",
                     src + "attn_k.weight")
        _load_tensor(dst + "self_attn.v_proj.weight",
                     src + "attn_v.weight")
        _load_tensor(dst + "self_attn.o_proj.weight",
                     src + "attn_output.weight")

        # MLP
        _load_tensor(dst + "mlp.gate_proj.weight",
                     src + "ffn_gate.weight")
        _load_tensor(dst + "mlp.up_proj.weight",
                     src + "ffn_up.weight")
        _load_tensor(dst + "mlp.down_proj.weight",
                     src + "ffn_down.weight")

        if (i + 1) % 7 == 0:
            print(f"  loaded layer {i+1}/{n_layer}")
        gc.collect()

    t_done = time.time()
    print(f"  Loaded {len(out)} tensors in {(t_done - t_start)*1000:.0f}ms")
    gf.close()
    return out


def load_gguf_qwen3_q8(gguf_path: str, config: dict) -> Dict[str, np.ndarray]:
    """Load Qwen3 GGUF keeping Q8_0 weights as packed GPU format.

    Q8_0 weight tensors are stored as:
      name + '.q8':        (N, K/4) uint32 — 4 int8 values per u32
      name + '.q8_scales': (N, K/32) float16 — per-block scale

    QKV and gate_up projections are fused by concatenating along N axis.
    Embedding is dequantized to fp16 (needed for gather lookup).
    """
    import gc
    from common.gguf_utils import GGUFFile, dequantize_q8_0, repack_q8_0_for_gpu

    n_layer = config["n_layer"]

    t_start = time.time()
    print(f"Loading GGUF (Q8 GPU path): {gguf_path}")
    gf = GGUFFile(gguf_path)
    print(f"  Parsed {len(gf.tensors)} tensor headers in "
          f"{(time.time() - t_start)*1000:.0f}ms")

    out = {}

    def _load_q8(backpack_name, gguf_name):
        """Load Q8_0 tensor as repacked GPU format (no dequant)."""
        info = gf.tensors[gguf_name]
        if info.qtype == 8:  # Q8_0
            raw = gf.tensor_data(gguf_name)
            w_u32, s_fp16 = repack_q8_0_for_gpu(raw, info.shape)
            out[backpack_name + ".q8"] = w_u32
            out[backpack_name + ".q8_scales"] = s_fp16
        elif info.qtype in (0, 1):  # F32/F16 — dequant and store as q8 anyway
            raw = gf.tensor_data(gguf_name)
            fp = dequantize_q8_0(raw, info.shape) if info.qtype == 8 else None
            if info.qtype == 0:
                arr = gf.tensor_data_f32(gguf_name)
                if len(info.shape) == 2:
                    arr = arr.reshape(info.shape[1], info.shape[0])
                out[backpack_name] = arr.astype(np.float16, copy=True)
            else:
                arr = gf.tensor_data_f16(gguf_name)
                if len(info.shape) == 2:
                    arr = arr.reshape(info.shape[1], info.shape[0])
                out[backpack_name] = np.array(arr, dtype=np.float16)
        else:
            raise ValueError(
                f"Unsupported quant type {info.qtype_name} for {gguf_name}")

    def _load_scalar(backpack_name, gguf_name, out_dtype=np.float32):
        """Load small tensor (norms, etc.) as fp32."""
        info = gf.tensors[gguf_name]
        if info.qtype == 0:
            arr = gf.tensor_data_f32(gguf_name)
            out[backpack_name] = arr.astype(out_dtype, copy=True)
        elif info.qtype == 1:
            arr = gf.tensor_data_f16(gguf_name)
            out[backpack_name] = np.array(arr, dtype=out_dtype)
        else:
            raw = gf.tensor_data(gguf_name)
            out[backpack_name] = dequantize_q8_0(
                raw, info.shape, out_dtype=out_dtype)

    # Embedding — must dequantize for gather lookup
    info_emb = gf.tensors["token_embd.weight"]
    if info_emb.qtype == 8:
        raw = gf.tensor_data("token_embd.weight")
        out["embed_tokens.weight"] = dequantize_q8_0(
            raw, info_emb.shape, out_dtype=np.float16)
    else:
        _load_scalar("embed_tokens.weight", "token_embd.weight",
                     out_dtype=np.float16)

    _load_scalar("norm.weight", "output_norm.weight", out_dtype=np.float32)

    if "output.weight" in gf.tensors:
        _load_q8("lm_head.weight", "output.weight")

    # Per-layer tensors
    for i in range(n_layer):
        src = f"blk.{i}."
        dst = f"layers.{i}."

        _load_scalar(dst + "input_layernorm.weight",
                     src + "attn_norm.weight")
        _load_scalar(dst + "post_attention_layernorm.weight",
                     src + "ffn_norm.weight")

        if src + "attn_q_norm.weight" in gf.tensors:
            _load_scalar(dst + "self_attn.q_norm.weight",
                         src + "attn_q_norm.weight")
            _load_scalar(dst + "self_attn.k_norm.weight",
                         src + "attn_k_norm.weight")

        # Load Q/K/V as q8, then fuse into qkv_proj
        _load_q8(dst + "self_attn.q_proj.weight", src + "attn_q.weight")
        _load_q8(dst + "self_attn.k_proj.weight", src + "attn_k.weight")
        _load_q8(dst + "self_attn.v_proj.weight", src + "attn_v.weight")
        _load_q8(dst + "self_attn.o_proj.weight", src + "attn_output.weight")

        # Fuse QKV: concat q8 packed weights along N axis
        qkv_name = dst + "self_attn.qkv_proj.weight"
        q_q8 = out.pop(dst + "self_attn.q_proj.weight.q8")
        k_q8 = out.pop(dst + "self_attn.k_proj.weight.q8")
        v_q8 = out.pop(dst + "self_attn.v_proj.weight.q8")
        out[qkv_name + ".q8"] = np.concatenate([q_q8, k_q8, v_q8], axis=0)
        q_sc = out.pop(dst + "self_attn.q_proj.weight.q8_scales")
        k_sc = out.pop(dst + "self_attn.k_proj.weight.q8_scales")
        v_sc = out.pop(dst + "self_attn.v_proj.weight.q8_scales")
        out[qkv_name + ".q8_scales"] = np.concatenate(
            [q_sc, k_sc, v_sc], axis=0)

        # Load gate/up as q8, then fuse
        _load_q8(dst + "mlp.gate_proj.weight", src + "ffn_gate.weight")
        _load_q8(dst + "mlp.up_proj.weight", src + "ffn_up.weight")
        _load_q8(dst + "mlp.down_proj.weight", src + "ffn_down.weight")

        gu_name = dst + "mlp.gate_up_proj.weight"
        g_q8 = out.pop(dst + "mlp.gate_proj.weight.q8")
        u_q8 = out.pop(dst + "mlp.up_proj.weight.q8")
        out[gu_name + ".q8"] = np.concatenate([g_q8, u_q8], axis=0)
        g_sc = out.pop(dst + "mlp.gate_proj.weight.q8_scales")
        u_sc = out.pop(dst + "mlp.up_proj.weight.q8_scales")
        out[gu_name + ".q8_scales"] = np.concatenate([g_sc, u_sc], axis=0)

        if (i + 1) % 7 == 0:
            print(f"  loaded layer {i+1}/{n_layer}")
        gc.collect()

    t_done = time.time()
    total_bytes = sum(v.nbytes for v in out.values())
    print(f"  Loaded {len(out)} tensors ({total_bytes / 1024**3:.2f} GB) "
          f"in {(t_done - t_start)*1000:.0f}ms")
    # Don't close GGUFFile — repacked arrays are already copies, but
    # embeddings may still reference the mmap. GC will reclaim later.
    return out


# ---------------------------------------------------------------------------
# Weight downloading
# ---------------------------------------------------------------------------

def download_qwen_weights(model_size: str = "0.6B",
                          model_dir: str = None) -> Tuple[str, str]:
    """Download Qwen3 weights and tokenizer from HuggingFace."""
    config = QWEN_CONFIGS[model_size]
    hf_repo = config["hf_repo"]
    if model_dir is None:
        model_dir = os.path.join(_SCRIPT_DIR, "..", "..", "gitignore", "models", os.path.basename(_SCRIPT_DIR), "weights")

    def qwen_key_transform(key, arr):
        new_key = key.replace("model.", "")
        if new_key == "lm_head.weight" and config.get("tie_word_embeddings"):
            return None
        return new_key, arr

    npz_path, tokenizer_path = download_weights(
        hf_repo=hf_repo,
        model_dir=model_dir,
        key_transform=qwen_key_transform,
        download_tokenizer=True,
    )
    return npz_path, tokenizer_path


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_with_random_weights():
    """Verify Qwen3 pipeline with small random weights."""
    print("=" * 60)
    print("Qwen3 WebGPU Pipeline Verification (random weights)")
    print("=" * 60)

    n_layer, n_head, n_kv_heads = 2, 4, 2
    n_embd, intermediate_size, n_vocab = 64, 128, 256
    head_dim = n_embd // n_head  # 16
    kv_dim = n_kv_heads * head_dim
    n_rep = n_head // n_kv_heads
    rope_theta = 10000.0
    eps = 1e-6
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
        weights[pfx + "self_attn.q_proj.weight"] = np.random.randn(
            n_embd, n_embd).astype(np.float32) * 0.02
        weights[pfx + "self_attn.k_proj.weight"] = np.random.randn(
            kv_dim, n_embd).astype(np.float32) * 0.02
        weights[pfx + "self_attn.v_proj.weight"] = np.random.randn(
            kv_dim, n_embd).astype(np.float32) * 0.02
        weights[pfx + "self_attn.o_proj.weight"] = np.random.randn(
            n_embd, n_embd).astype(np.float32) * 0.02
        # MLP
        weights[pfx + "mlp.gate_proj.weight"] = np.random.randn(
            intermediate_size, n_embd).astype(np.float32) * 0.02
        weights[pfx + "mlp.up_proj.weight"] = np.random.randn(
            intermediate_size, n_embd).astype(np.float32) * 0.02
        weights[pfx + "mlp.down_proj.weight"] = np.random.randn(
            n_embd, intermediate_size).astype(np.float32) * 0.02

    print(f"\nModel: {n_layer} layers, {n_head} Q heads, {n_kv_heads} KV heads, "
          f"{n_embd} embd, {intermediate_size} intermediate, {n_vocab} vocab")

    # --- NumPy reference (before model creation which fuses weights) ---
    token_ids = np.array([1, 42, 100, 200], dtype=np.int32)
    T = len(token_ids)

    def _rope_numpy(x, positions, theta, hd):
        half = hd // 2
        inv_freq = 1.0 / (theta ** (
            np.arange(0, hd, 2, dtype=np.float32) / hd))
        angles = positions[:, None].astype(np.float32) * inv_freq[None, :]
        cos_v = np.cos(angles)[:, None, :]
        sin_v = np.sin(angles)[:, None, :]
        x1, x2 = x[..., :half], x[..., half:]
        out = np.empty_like(x)
        out[..., :half] = x1 * cos_v - x2 * sin_v
        out[..., half:] = x2 * cos_v + x1 * sin_v
        return out

    positions = np.arange(T, dtype=np.int32)
    x = weights["embed_tokens.weight"][token_ids]

    for layer in range(n_layer):
        pfx = f"layers.{layer}."
        rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
        ln1 = x / rms * weights[pfx + "input_layernorm.weight"]

        q = ln1 @ weights[pfx + "self_attn.q_proj.weight"].T
        k = ln1 @ weights[pfx + "self_attn.k_proj.weight"].T
        v = ln1 @ weights[pfx + "self_attn.v_proj.weight"].T

        Q = q.reshape(T, n_head, head_dim)
        K_ = k.reshape(T, n_kv_heads, head_dim)
        V_ = v.reshape(T, n_kv_heads, head_dim)

        Q = _rope_numpy(Q, positions, rope_theta, head_dim)
        K_ = _rope_numpy(K_, positions, rope_theta, head_dim)

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

        attn_flat = attn_out.reshape(T, n_embd)
        proj = attn_flat @ weights[pfx + "self_attn.o_proj.weight"].T
        x = x + proj

        rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
        ln2 = x / rms * weights[pfx + "post_attention_layernorm.weight"]

        gate = ln2 @ weights[pfx + "mlp.gate_proj.weight"].T
        up_val = ln2 @ weights[pfx + "mlp.up_proj.weight"].T
        silu_gate = gate / (1.0 + np.exp(-gate))
        mlp_out = (silu_gate * up_val) @ \
                  weights[pfx + "mlp.down_proj.weight"].T
        x = x + mlp_out

    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    x = x / rms * weights["norm.weight"]
    logits_ref = x @ weights["embed_tokens.weight"].T

    # --- GPU forward pass (fuses weights internally) ---
    model = Qwen3WebGPU(
        weights, n_layer=n_layer, n_head=n_head, n_kv_heads=n_kv_heads,
        n_embd=n_embd, intermediate_size=intermediate_size,
        n_vocab=n_vocab, rope_theta=rope_theta, rms_norm_eps=eps,
        head_dim=head_dim, attention_bias=False, tie_word_embeddings=True)

    t0 = time.time()
    logits = model.forward(token_ids)
    t1 = time.time()

    print(f"\nForward pass: {token_ids} → shape {logits.shape} "
          f"in {(t1-t0)*1000:.0f}ms")

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
        description="Qwen3-1.7B on WebGPU via Triton")
    add_common_args(parser)
    parser.add_argument("--model", type=str, default="1.7B",
                        choices=["1.7B"],
                        help="Model size")
    parser.add_argument("--gguf-file", type=str, default=None,
                        help="Path to GGUF file (loads directly, no safetensors conversion)")
    parser.add_argument("--quantize", action="store_true",
                        help="Quantize weights to INT4 and save")
    parser.add_argument("--use-q4", action="store_true",
                        help="Use INT4 weights_q4.npz for faster inference (lower quality)")
    parser.add_argument("--use-q8-gpu", action="store_true",
                        help="Keep Q8_0 weights packed on GPU (W8A32, ~1 byte/param)")
    parser.add_argument("--decode-mode", type=str, default="gpu",
                        choices=["cpu", "gpu"],
                        help="Decode mode: cpu or gpu (fast decode)")
    args = parser.parse_args()
    apply_device_arg(args)

    if args.verify:
        success = verify_with_random_weights()
        sys.exit(0 if success else 1)

    config = QWEN_CONFIGS[args.model]
    weights_dir = args.weights_dir or os.path.join(_SCRIPT_DIR, "..", "..", "gitignore", "models", os.path.basename(_SCRIPT_DIR), "weights")

    # GGUF loading path (preferred for Q8_0 / quantized models)
    if args.gguf_file:
        if not os.path.exists(args.gguf_file):
            print(f"Error: GGUF file not found: {args.gguf_file}")
            sys.exit(1)

        q8_mode = getattr(args, 'use_q8_gpu', False)
        if q8_mode:
            weights = load_gguf_qwen3_q8(args.gguf_file, config)
        else:
            weights = load_gguf_qwen3(args.gguf_file, config)
        quantized = False

        # Quantize from GGUF if requested
        if args.quantize:
            q4_path = os.path.join(weights_dir, "weights_q4.npz")
            print(f"Quantizing Qwen3-{args.model} weights from GGUF...")
            n_layer = config["n_layer"]
            # Fuse before quantizing
            for i in range(n_layer):
                pfx = f"layers.{i}."
                q_key = pfx + "self_attn.q_proj.weight"
                if q_key in weights:
                    weights[pfx + "self_attn.qkv_proj.weight"] = np.concatenate([
                        weights.pop(q_key).astype(np.float32),
                        weights.pop(pfx + "self_attn.k_proj.weight").astype(np.float32),
                        weights.pop(pfx + "self_attn.v_proj.weight").astype(np.float32),
                    ], axis=0)
                gate_key = pfx + "mlp.gate_proj.weight"
                if gate_key in weights:
                    weights[pfx + "mlp.gate_up_proj.weight"] = np.concatenate([
                        weights.pop(gate_key).astype(np.float32),
                        weights.pop(pfx + "mlp.up_proj.weight").astype(np.float32),
                    ], axis=0)
            quantized_w = quantize_qwen_weights(weights, n_layer)
            os.makedirs(weights_dir, exist_ok=True)
            print(f"Saving to {q4_path}...")
            np.savez(q4_path, **quantized_w)
            print(f"Done! File: {os.path.getsize(q4_path) / 1024**2:.0f} MB")
            return

        # Find tokenizer
        tokenizer_path = os.path.join(
            os.path.dirname(args.gguf_file), "tokenizer.json")
        if not os.path.exists(tokenizer_path):
            # Check in gitignore weights dir
            tokenizer_path = os.path.join(weights_dir, "tokenizer.json")
        if not os.path.exists(tokenizer_path):
            # Download tokenizer only from HuggingFace
            import requests
            hf_repo = config["hf_repo"]
            url = f"https://huggingface.co/{hf_repo}/resolve/main/tokenizer.json"
            print(f"Downloading tokenizer from {hf_repo}...")
            os.makedirs(weights_dir, exist_ok=True)
            tokenizer_path = os.path.join(weights_dir, "tokenizer.json")
            resp = requests.get(url)
            resp.raise_for_status()
            with open(tokenizer_path, "wb") as f:
                f.write(resp.content)
            print(f"  Saved tokenizer to {tokenizer_path}")
        tokenizer = load_tokenizer(tokenizer_path)
    else:
        npz_path, tokenizer_path = download_qwen_weights(
            args.model, weights_dir)
        q4_path = os.path.join(weights_dir, "weights_q4.npz")

        if args.quantize:
            print(f"Quantizing Qwen3-{args.model} weights...")
            weights = load_weights(npz_path)
            # Fuse before quantizing
            n_layer = config["n_layer"]
            E = config["n_embd"]
            for i in range(n_layer):
                pfx = f"layers.{i}."
                q_key = pfx + "self_attn.q_proj.weight"
                if q_key in weights:
                    weights[pfx + "self_attn.qkv_proj.weight"] = np.concatenate([
                        weights.pop(q_key),
                        weights.pop(pfx + "self_attn.k_proj.weight"),
                        weights.pop(pfx + "self_attn.v_proj.weight"),
                    ], axis=0)
                    if config.get("attention_bias"):
                        weights[pfx + "self_attn.qkv_proj.bias"] = np.concatenate([
                            weights.pop(pfx + "self_attn.q_proj.bias").ravel(),
                            weights.pop(pfx + "self_attn.k_proj.bias").ravel(),
                            weights.pop(pfx + "self_attn.v_proj.bias").ravel(),
                        ])
                gate_key = pfx + "mlp.gate_proj.weight"
                if gate_key in weights:
                    weights[pfx + "mlp.gate_up_proj.weight"] = np.concatenate([
                        weights.pop(gate_key),
                        weights.pop(pfx + "mlp.up_proj.weight"),
                    ], axis=0)
            quantized_w = quantize_qwen_weights(weights, n_layer)
            print(f"Saving to {q4_path}...")
            np.savez(q4_path, **quantized_w)
            print(f"Done! File: {os.path.getsize(q4_path) / 1024**2:.0f} MB")
            return

        # Load weights (default: full precision for better output quality)
        quantized = False
        if args.use_q4:
            if os.path.exists(q4_path):
                print(f"Loading quantized weights from {q4_path}")
                data = np.load(q4_path, mmap_mode='r')
                weights = {k: data[k] for k in data.files}
                quantized = True
            else:
                print("INT4 file not found; falling back to full-precision weights.")
                weights = load_weights(npz_path)
        else:
            weights = load_weights(npz_path)
        tokenizer = load_tokenizer(tokenizer_path)

    print(f"Loaded {len(weights)} weight tensors"
          f"{' (INT4 quantized)' if quantized else ''}")

    model = Qwen3WebGPU(
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
        attention_bias=config.get("attention_bias", False),
        tie_word_embeddings=config.get("tie_word_embeddings", True),
        quantized=quantized,
        decode_mode=args.decode_mode,
        q8_mode=getattr(args, 'use_q8_gpu', False))
    print(f"Model created, kernels compiled (decode={args.decode_mode})")

    run_inference(model, args, tokenizer,
                  model_name="Qwen3-1.7B", script_dir=_SCRIPT_DIR)


if __name__ == "__main__":
    main()
