"""
SmolLM2 inference on WebGPU via Triton.

Demonstrates LLaMA-architecture LLM inference using Triton kernels compiled
to WGSL and executed on the GPU through Dawn's D3D12/Vulkan/Metal backend.

SmolLM2 is a LLaMA-family model featuring:
  - RoPE (rotary position embeddings)
  - RMSNorm (root mean square normalization)
  - GQA (grouped query attention)
  - SwiGLU MLP (SiLU-gated linear unit)

All matrix multiplications, normalization, attention, and activation
operations run as WebGPU compute shaders — no CUDA required.

Usage:
    python models/smollm-2/model.py
    python models/smollm-2/model.py --model 360M --prompt "Hello"

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
    _parse_safetensors, load_weights, download_weights,
    load_tokenizer, generate,
)


# SmolLM2 model configs
SMOLLM2_CONFIGS = {
    "135M": {
        "n_layer": 30, "n_head": 9, "n_kv_heads": 3,
        "n_embd": 576, "intermediate_size": 1536,
        "n_vocab": 49152, "rope_theta": 100000.0,
        "rms_norm_eps": 1e-5,
        "hf_repo": "HuggingFaceTB/SmolLM2-135M",
    },
    "360M": {
        "n_layer": 32, "n_head": 15, "n_kv_heads": 5,
        "n_embd": 960, "intermediate_size": 2560,
        "n_vocab": 49152, "rope_theta": 100000.0,
        "rms_norm_eps": 1e-5,
        "hf_repo": "HuggingFaceTB/SmolLM2-360M",
    },
    "1.7B": {
        "n_layer": 24, "n_head": 32, "n_kv_heads": 32,
        "n_embd": 2048, "intermediate_size": 8192,
        "n_vocab": 49152, "rope_theta": 100000.0,
        "rms_norm_eps": 1e-5,
        "hf_repo": "HuggingFaceTB/SmolLM2-1.7B",
    },
}


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
    scales_f32 = scales.astype(np.float32)
    zeros_f32 = zeros.astype(np.float32)
    q = np.clip(np.round((w_grouped - zeros_f32) / scales_f32), 0, 15).astype(np.uint8)
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


def quantize_smollm2_weights(weights: Dict[str, np.ndarray],
                              n_layer: int) -> Dict[str, np.ndarray]:
    """Quantize SmolLM2 weights: INT4 for linear, fp16 for others."""
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
            w = val.astype(np.float32)
            if w.ndim != 2:
                w = w.reshape(w.shape[0], -1)
            N, K = w.shape
            q_packed, scales, zeros = quantize_int4(w)
            quantized[key + ".q4"] = q_packed
            quantized[key + ".scales"] = scales
            quantized[key + ".zeros"] = zeros
            quantized[key + ".K"] = np.array([K], dtype=np.int32)
            total_quant += q_packed.nbytes + scales.nbytes + zeros.nbytes
        else:
            quantized[key] = val.astype(np.float16)
            total_quant += quantized[key].nbytes

    print(f"  Original: {total_orig / 1024 / 1024:.0f} MB (fp32)")
    print(f"  Quantized: {total_quant / 1024 / 1024:.0f} MB "
          f"(INT4 linear + fp16 other)")
    print(f"  Compression: {total_orig / total_quant:.1f}x")
    return quantized


def load_quantized_weights(path: str) -> Dict[str, np.ndarray]:
    """Load quantized weights from npz."""
    data = np.load(path)
    return {k: data[k] for k in data.files}


class SmolLM2WebGPU(WebGPUModel):
    """SmolLM2 inference on WebGPU via Triton kernels.

    Supports SmolLM2-135M, 360M, and 1.7B (LLaMA architecture).
    Features: RoPE, RMSNorm, GQA (grouped query attention), SwiGLU MLP.

    Optimizations:
      - Fused QKV projection (3→1 dispatch)
      - Fused gate_up projection (2→1 dispatch)
      - Pre-computed RoPE cos/sin tables (no trig per call)
      - Pre-allocated KV cache (no np.concatenate)
      - Vectorized multi-head attention (no per-head Python loop)
      - Multi-head GPU causal attention for prefill (n_head→1 dispatch)
      - CPU decode path for small models (avoids GPU dispatch overhead)
      - GPU-resident decode for 1.7B+ (INT4 fused dequant, GPU attention)
      - INT4 weight quantization (4× memory reduction)
    """

    MAX_SEQ_LEN = 2048

    def __init__(self, weights: Dict[str, np.ndarray],
                 n_layer: int = 30, n_head: int = 9,
                 n_kv_heads: int = 3, n_embd: int = 576,
                 intermediate_size: int = 1536,
                 n_vocab: int = 49152,
                 rope_theta: float = 100000.0,
                 rms_norm_eps: float = 1e-5,
                 quantized: bool = False,
                 decode_mode: str = 'cpu'):
        head_dim = n_embd // n_head
        self.qkv_out = n_head * head_dim + 2 * n_kv_heads * head_dim
        self.gate_up_out = 2 * intermediate_size
        self._quantized = quantized
        self._decode_mode = decode_mode  # 'cpu' or 'gpu'
        super().__init__(
            weights, n_layer=n_layer, n_head=n_head, n_embd=n_embd,
            n_vocab=n_vocab,
            n_kv_heads=n_kv_heads,
            intermediate_size=intermediate_size,
            rope_theta=rope_theta,
            rms_norm_eps=rms_norm_eps,
            k_dimensions={n_embd, intermediate_size},
        )
        self._fuse_weights()
        if self._quantized and not getattr(self, '_has_fp16_linear', False):
            self._dequantize_all_weights()
        self._precompute_rope_tables()
        self._upload_weights_to_gpu()
        if self._decode_mode == 'gpu':
            self._init_gpu_kv_cache()

    def _fuse_weights(self):
        """Fuse separate Q/K/V and gate/up weight matrices.

        QKV fusion: 3 separate projections → 1 fused (saves 2 dispatches/layer).
        gate_up fusion: 2 separate projections → 1 fused (saves 1 dispatch/layer).
        Skips if weights are already fused (e.g. loaded from quantized npz).
        """
        for i in range(self.n_layer):
            pfx = f"layers.{i}."
            # Fuse Q, K, V → QKV
            q_key = pfx + "self_attn.q_proj.weight"
            k_key = pfx + "self_attn.k_proj.weight"
            v_key = pfx + "self_attn.v_proj.weight"
            if q_key in self.weights:
                self.weights[pfx + "self_attn.qkv_proj.weight"] = \
                    np.concatenate([
                        self.weights[q_key],
                        self.weights[k_key],
                        self.weights[v_key],
                    ], axis=0).astype(np.float32)
                del self.weights[q_key]
                del self.weights[k_key]
                del self.weights[v_key]
            # Fuse gate, up → gate_up
            gate_key = pfx + "mlp.gate_proj.weight"
            up_key = pfx + "mlp.up_proj.weight"
            if gate_key in self.weights:
                self.weights[pfx + "mlp.gate_up_proj.weight"] = \
                    np.concatenate([
                        self.weights[gate_key],
                        self.weights[up_key],
                    ], axis=0).astype(np.float32)
                del self.weights[gate_key]
                del self.weights[up_key]

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

    def _precompute_rope_tables(self):
        """Pre-compute RoPE cos/sin tables for all positions up to MAX_SEQ_LEN.

        Eliminates per-call trigonometric computation during inference.
        Uploads tables to GPU for GPU-side RoPE during decode.
        """
        HD = self.head_dim
        half = HD // 2
        inv_freq = 1.0 / (self.rope_theta ** (
            np.arange(0, HD, 2, dtype=np.float32) / HD))
        positions = np.arange(self.MAX_SEQ_LEN, dtype=np.float32)
        angles = positions[:, None] * inv_freq[None, :]  # (MAX_SEQ, half)
        self._rope_cos = np.cos(angles).astype(np.float32)
        self._rope_sin = np.sin(angles).astype(np.float32)

        # Upload to GPU for GPU-side RoPE
        if self._decode_mode == 'gpu':
            runner = self.cache.runner
            self._rope_cos_gpu = runner.upload_to_gpu(
                self._rope_cos.ravel(), "rope_cos_table")
            self._rope_sin_gpu = runner.upload_to_gpu(
                self._rope_sin.ravel(), "rope_sin_table")

    def _apply_rope_fast(self, x, positions):
        """Apply RoPE using pre-computed tables (no trig per call).

        x: (T, n_heads, head_dim)
        positions: (T,) integer position indices
        """
        half = self.head_dim // 2
        cos_v = self._rope_cos[positions][:, None, :]  # (T, 1, half)
        sin_v = self._rope_sin[positions][:, None, :]
        x1 = x[..., :half]
        x2 = x[..., half:]
        out = np.empty_like(x)
        out[..., :half] = x1 * cos_v - x2 * sin_v
        out[..., half:] = x2 * cos_v + x1 * sin_v
        return out

    def _compile_model_kernels(self):
        """Compile SmolLM2-specific kernels: RMSNorm, SiLU*mul."""
        self._compile_rms_norm()
        self._compile_silu_mul()

    def _upload_weights_to_gpu(self):
        """Upload weights to GPU memory.

        INT4 quantized: uploads packed INT4 + fp16 scales/zeros for GPU dequant.
        fp16 mode: uploads weights as fp16 (2× bandwidth reduction).
        fp32 fallback: uploads as fp32. 
        """
        E = self.n_embd
        IM = self.intermediate_size

        use_q4_gpu = (self._quantized
                      and getattr(self, '_has_fp16_linear', False))
        self._use_q4_gpu = use_q4_gpu
        use_fp16 = (not use_q4_gpu
                    and getattr(self, '_has_fp16_linear', False))

        # Per-layer weights
        for i in range(self.n_layer):
            pfx = f"layers.{i}."
            # RMSNorm weights (always fp32)
            for nkey in [pfx + "input_layernorm.weight",
                         pfx + "post_attention_layernorm.weight"]:
                if self.weights[nkey].dtype == np.float16:
                    self.weights[nkey] = self.weights[nkey].astype(np.float32)
                self._upload_norm_weight(nkey)

            # Linear projections
            layer_projs = [
                (pfx + "self_attn.qkv_proj.weight", self.qkv_out, E),
                (pfx + "self_attn.o_proj.weight", E, E),
                (pfx + "mlp.gate_up_proj.weight", self.gate_up_out, E),
                (pfx + "mlp.down_proj.weight", E, IM),
            ]
            for name, N, K in layer_projs:
                if use_q4_gpu:
                    self._upload_q4_weight(name, N, K)
                elif use_fp16:
                    self._upload_linear_weight_fp16(name, N, K)
                else:
                    self._upload_linear_weight(name, N, K)

        # Final RMSNorm
        nkey = "norm.weight"
        if self.weights[nkey].dtype == np.float16:
            self.weights[nkey] = self.weights[nkey].astype(np.float32)
        self._upload_norm_weight(nkey)

        # Embedding / LM head (tied weights)
        ekey = "embed_tokens.weight"
        if self.weights[ekey].dtype == np.float16:
            self.weights[ekey] = self.weights[ekey].astype(np.float32)
        if use_fp16 or use_q4_gpu:
            self._upload_linear_weight_fp16(ekey, self.n_vocab, E)
        self._upload_embedding_weight(ekey, self.n_vocab, E)

        # Zero bias buffers (LLaMA has no biases)
        self._upload_zero_bias("zero_bias_E", E)
        self._upload_zero_bias("zero_bias_QKV", self.qkv_out)
        self._upload_zero_bias("zero_bias_GU", self.gate_up_out)
        self._upload_zero_bias("zero_bias_V", self.n_vocab)

        self._print_gpu_weight_stats()

    def _proj(self, x, weight_name, bias, out_features, K=None,
              gpu_out=False):
        """Linear projection: routes to INT4, fp16, or fp32 kernel."""
        q4_key = weight_name + ".q4.gpu"
        if q4_key in self._gpu_weights:
            return self._linear_q4(
                x,
                self._gpu_weights[q4_key],
                self._gpu_weights[weight_name + ".scales.gpu"],
                self._gpu_weights[weight_name + ".zeros.gpu"],
                bias, out_features, K=K, gpu_out=gpu_out)
        fp16_name = weight_name + ".fp16"
        if fp16_name in self._gpu_weights:
            return self._linear_fp16w(
                x, self._gpu_weights[fp16_name], bias, out_features,
                K=K, gpu_out=gpu_out)
        return self._linear(
            x, self._gpu_weights[weight_name], bias, out_features,
            gpu_out=gpu_out)

    # -- CPU decode helpers --

    def _rms_norm_cpu(self, x, w_name):
        """CPU-side RMSNorm for T=1 — avoids GPU kernel launch overhead."""
        w = self.weights[w_name]
        rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True)
                      + self.rms_norm_eps)
        return (x / rms * w).astype(np.float32)

    def _cpu_matmul(self, x, weight_name, N, K=None):
        """CPU matmul for T=1 decode: (1,K) @ (N,K).T → (1,N).

        For small models like SmolLM2-135M, CPU BLAS is faster than
        GPU dispatch overhead (~0.7ms/dispatch).
        """
        W = self.weights[weight_name]
        if K is None:
            K = W.shape[1] if W.ndim == 2 else W.size // N
        W = W.reshape(N, K)
        return np.float32(x @ W.T)

    def _decode_attention(self, qkv, layer, positions):
        """CPU decode attention: vectorized multi-head with KV cache.

        Replaces per-head Python loop with batched numpy matmul.
        """
        HD = self.head_dim
        n_head = self.n_head
        n_kv = self.n_kv_heads
        n_rep = self.n_rep
        q_size = n_head * HD
        kv_size = n_kv * HD

        Q = qkv[0, :q_size].reshape(1, n_head, HD)
        K_new = qkv[0, q_size:q_size + kv_size].reshape(1, n_kv, HD)
        V_new = qkv[0, q_size + kv_size:].reshape(1, n_kv, HD)

        Q = self._apply_rope_fast(Q, positions)
        K_new = self._apply_rope_fast(K_new, positions)

        # KV cache: pre-allocated, write-in-place (no np.concatenate)
        K_buf, V_buf, cur_len = self.kv_cache[layer]
        K_buf[cur_len] = K_new[0]
        V_buf[cur_len] = V_new[0]
        T_total = cur_len + 1
        self.kv_cache[layer] = (K_buf, V_buf, T_total)
        K_full = K_buf[:T_total]
        V_full = V_buf[:T_total]

        # Vectorized multi-head attention (no per-head loop)
        scale = 1.0 / np.sqrt(HD)
        K_exp = np.repeat(K_full, n_rep, axis=1)  # (T_total, n_head, HD)
        V_exp = np.repeat(V_full, n_rep, axis=1)
        Q_t = Q.transpose(1, 0, 2)          # (n_head, 1, HD)
        K_t = K_exp.transpose(1, 2, 0)      # (n_head, HD, T_total)
        scores = np.float32((Q_t @ K_t).squeeze(1) * scale)  # (n_head, T_total)
        scores -= scores.max(axis=1, keepdims=True)
        exp_s = np.exp(scores)
        s = exp_s.sum(axis=1, keepdims=True)
        s = np.where(s == 0, 1.0, s)
        attn = exp_s / s
        V_t = V_exp.transpose(1, 0, 2)  # (n_head, T_total, HD)
        attn_out = (attn[:, None, :] @ V_t).squeeze(1)  # (n_head, HD)
        return attn_out.reshape(1, n_head * HD).astype(np.float32)

    def _decode_cpu(self, x, layer, positions):
        """Fully CPU decode: all matmuls on CPU via numpy BLAS.

        Eliminates all GPU dispatches per layer during decode.
        For SmolLM2-135M (576 embd), CPU BLAS is ~10x faster than
        paying GPU dispatch overhead for each small matmul.
        """
        from common.model_base import GPUBuffer
        if isinstance(x, GPUBuffer):
            x_np = self.cache.runner.readback(x).reshape(1, self.n_embd)
        else:
            x_np = x

        pfx = f"layers.{layer}."
        E = self.n_embd
        IM = self.intermediate_size
        p = self.profiler
        _p = p and p.enabled

        # RMSNorm 1 (CPU)
        if _p: p._cpu.begin(f"L{layer}/rms_norm_1")
        rn1 = self._rms_norm_cpu(x_np, pfx + "input_layernorm.weight")
        if _p: p._cpu.end(f"L{layer}/rms_norm_1")

        # Fused QKV linear (CPU)
        if _p: p._cpu.begin(f"L{layer}/qkv_linear")
        qkv = self._cpu_matmul(rn1, pfx + "self_attn.qkv_proj.weight",
                               self.qkv_out, E)
        if _p: p._cpu.end(f"L{layer}/qkv_linear")

        # Attention (CPU, vectorized multi-head)
        if _p: p._cpu.begin(f"L{layer}/attention")
        attn_out = self._decode_attention(qkv, layer, positions)
        if _p: p._cpu.end(f"L{layer}/attention")

        # O projection (CPU)
        if _p: p._cpu.begin(f"L{layer}/o_proj")
        o_out = self._cpu_matmul(attn_out, pfx + "self_attn.o_proj.weight",
                                 E, self.n_head * self.head_dim)
        if _p: p._cpu.end(f"L{layer}/o_proj")

        # Residual + RMSNorm 2 (CPU)
        if _p: p._cpu.begin(f"L{layer}/res_norm")
        x_np = x_np + o_out
        rn2 = self._rms_norm_cpu(x_np, pfx + "post_attention_layernorm.weight")
        if _p: p._cpu.end(f"L{layer}/res_norm")

        # Fused gate+up linear (CPU)
        if _p: p._cpu.begin(f"L{layer}/gate_up")
        gate_up = self._cpu_matmul(rn2, pfx + "mlp.gate_up_proj.weight",
                                   self.gate_up_out, E)
        if _p: p._cpu.end(f"L{layer}/gate_up")

        # SiLU*mul (CPU)
        if _p: p._cpu.begin(f"L{layer}/silu_mul")
        gate = gate_up[0, :IM]
        up = gate_up[0, IM:]
        h = (gate / (1.0 + np.exp(-gate)) * up).reshape(1, IM)
        if _p: p._cpu.end(f"L{layer}/silu_mul")

        # Down projection (CPU)
        if _p: p._cpu.begin(f"L{layer}/down_proj")
        mlp_out = self._cpu_matmul(h, pfx + "mlp.down_proj.weight", E, IM)
        if _p: p._cpu.end(f"L{layer}/down_proj")

        return x_np + mlp_out

    # -- GPU-resident decode (for 1.7B+) --

    def _decode_gpu(self, x, layer, positions):
        """Fully GPU-resident decode with fused kernels.

        Optimizations vs naive GPU decode:
          - Fused RoPE: rope_q + rope_kv_scatter → 1 dispatch (was 2)
          - Fused residual add + RMSNorm: res + norm → 1 dispatch (was 2)
          - Total: 8 dispatches/layer (was 12), saves 96 dispatches/token

        Data flow (all on GPU):
          GPU: RMSNorm → QKV proj → Fused RoPE+KVscatter →
               attention → O proj → Fused(res_add+RMSNorm) →
               gate_up → SiLU·mul → down proj → res_add
        """
        from common.model_base import GPUBuffer
        runner = self.cache.runner
        p = self.profiler
        _p = p and p.enabled

        HD = self.head_dim
        E = self.n_embd
        n_head = self.n_head
        n_kv = self.n_kv_heads
        n_rep = self.n_rep
        half_rot = HD // 2  # full RoPE
        q_size = n_head * HD
        kv_size = n_kv * HD
        pos = int(positions[0])
        IM = self.intermediate_size
        pfx = f"layers.{layer}."

        # Ensure x is on GPU
        if not isinstance(x, GPUBuffer):
            x_np = x.ravel().astype(np.float32)
            x_gpu = runner.upload_to_gpu(x_np, f"__decode_x_{layer}")
            x_gpu.shape = (1, E)
        else:
            x_gpu = x

        # 1. GPU RMSNorm (only for first layer; subsequent layers
        #    get norm from the fused res+norm at end of previous layer)
        if not hasattr(self, '_decode_norm_out') or layer == 0:
            if _p: self.cache._gpu_op_name = f"L{layer}/norm1"
            rn1 = self._rms_norm(
                x_gpu, self._gpu_weights[pfx + "input_layernorm.weight"],
                gpu_out=True)
        else:
            rn1 = self._decode_norm_out

        # 2. QKV projection → stays on GPU
        if _p: self.cache._gpu_op_name = f"L{layer}/qkv"
        qkv_gpu = self._proj(rn1, pfx + "self_attn.qkv_proj.weight",
                             self._gpu_weights["zero_bias_QKV"],
                             self.qkv_out, gpu_out=True)

        # 3. Fused RoPE Q + K scatter + V copy (1 dispatch instead of 2)
        K_cache_gpu, V_cache_gpu, cur_len = self._gpu_kv_cache[layer]
        cache_offset = cur_len * n_kv * HD

        if _p: self.cache._gpu_op_name = f"L{layer}/fused_rope"
        fused_out = self.cache.run(
            self._fused_rope_result, grid=(n_head + n_kv,),
            buffers={
                'QKV': qkv_gpu,
                'Q_out': np.zeros(q_size, dtype=np.float32),
                'K_cache': K_cache_gpu,
                'V_cache': V_cache_gpu,
                'CosTable': self._rope_cos_gpu,
                'SinTable': self._rope_sin_gpu,
            },
            scalars={
                'n_head': n_head,
                'q_size': q_size,
                'kv_size': kv_size,
                'pos': pos,
                'half_rot': half_rot,
                'cache_offset': cache_offset,
            },
            gpu_outputs={'Q_out', 'K_cache', 'V_cache'})
        q_rot_gpu = fused_out['Q_out']

        T_total = cur_len + 1
        self._gpu_kv_cache[layer] = (K_cache_gpu, V_cache_gpu, T_total)

        # 4. GPU GQA decode attention
        if _p: self.cache._gpu_op_name = f"L{layer}/attn"
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

        # 5. O projection → stays on GPU
        if _p: self.cache._gpu_op_name = f"L{layer}/o_proj"
        o_gpu = self._proj(attn_gpu, pfx + "self_attn.o_proj.weight",
                           self._gpu_weights["zero_bias_E"],
                           E, gpu_out=True)

        # 6. Fused residual add + RMSNorm (1 dispatch instead of 2)
        #    x += o_proj; norm2 = rms_norm(x)
        if _p: self.cache._gpu_op_name = f"L{layer}/res_norm2"
        rn2 = self._add_rms_norm(
            x_gpu, o_gpu,
            self._gpu_weights[pfx + "post_attention_layernorm.weight"],
            gpu_out=True)

        # 7. Gate_up projection → stays on GPU
        if _p: self.cache._gpu_op_name = f"L{layer}/gate_up"
        gate_up_gpu = self._proj(rn2, pfx + "mlp.gate_up_proj.weight",
                                 self._gpu_weights["zero_bias_GU"],
                                 self.gate_up_out, gpu_out=True)

        # 8. SiLU*mul → stays on GPU
        if _p: self.cache._gpu_op_name = f"L{layer}/silu_mul"
        h_gpu = self._silu_mul_fused(gate_up_gpu, IM, gpu_out=True)

        # 9. Down projection → stays on GPU
        if _p: self.cache._gpu_op_name = f"L{layer}/down"
        down_gpu = self._proj(h_gpu, pfx + "mlp.down_proj.weight",
                              self._gpu_weights["zero_bias_E"],
                              E, K=IM, gpu_out=True)

        # 10. Fused residual add + RMSNorm for next layer's input_layernorm
        #     (saves 1 dispatch by combining res2 + next layer's norm1)
        if layer < self.n_layer - 1:
            next_pfx = f"layers.{layer + 1}."
            if _p: self.cache._gpu_op_name = f"L{layer}/res_norm_next"
            self._decode_norm_out = self._add_rms_norm(
                x_gpu, down_gpu,
                self._gpu_weights[next_pfx + "input_layernorm.weight"],
                gpu_out=True)
        else:
            # Last layer: just residual add (final norm done in forward())
            if _p: self.cache._gpu_op_name = f"L{layer}/res2"
            self._add_inplace(x_gpu, down_gpu)
            self._decode_norm_out = None

        if _p: self.cache._gpu_op_name = None
        x_gpu.shape = (1, E)
        return x_gpu

    # -- Fast decode: pre-recorded bind groups & batched dispatch --

    def _init_fast_decode(self):
        """Pre-allocate buffers, bind groups, and dispatch lists for fast decode.

        Called once on the first decode token. Pre-creates everything needed
        for all layers so that per-token dispatch is just:
          update 3 param buffers → submit pre-recorded dispatches → readback

        This eliminates per-dispatch Python overhead (buffer alloc, bind group
        creation, compute pass begin/end) — the main bottleneck identified
        in profiling (52% of decode time was dispatch overhead, not GPU compute).
        """
        import struct
        runner = self.cache.runner

        E = self.n_embd
        HD = self.head_dim
        n_head = self.n_head
        n_kv = self.n_kv_heads
        n_rep = self.n_rep
        IM = self.intermediate_size
        q_size = n_head * HD
        kv_size = n_kv * HD
        qkv_N = self.qkv_out
        half_rot = HD // 2
        LB = self.LOOP_BLOCK

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

        # Intermediate buffers (reused every token)
        norm_out = mkbuf('norm_out', E * 4)
        qkv_buf = mkbuf('qkv_out', qkv_N * 4)
        q_rot = mkbuf('q_rot', q_size * 4)
        attn_out_buf = mkbuf('attn_out', q_size * 4)
        proj_out = mkbuf('proj_out', E * 4)
        gate_up_buf = mkbuf('gate_up', 2 * IM * 4)
        silu_out = mkbuf('silu_out', IM * 4)
        rstd = mkbuf('rstd', 16)
        x_buf = mkbuf('x', E * 4)

        # Get pipelines + bind group layouts
        def get_pl(r):
            return runner.get_pipeline_info(
                r.wgsl, r.buffer_bindings, r.param_fields)

        pl_rn,    bgl_rn    = get_pl(self._rn_result)
        pl_arn,   bgl_arn   = get_pl(self._add_rn_result)
        pl_fused, bgl_fused = get_pl(self._fused_rope_result)
        pl_attn,  bgl_attn  = get_pl(self._gqa_attn_result)
        pl_aip,   bgl_aip   = get_pl(self._add_ip_result)
        pl_silu,  bgl_silu  = get_pl(self._smf_result)
        pl_q4,    bgl_q4    = get_pl(self._linear_q4_result)

        nbb_rn    = len(self._rn_result.buffer_bindings)
        nbb_arn   = len(self._add_rn_result.buffer_bindings)
        nbb_fused = len(self._fused_rope_result.buffer_bindings)
        nbb_attn  = len(self._gqa_attn_result.buffer_bindings)
        nbb_aip   = len(self._add_ip_result.buffer_bindings)
        nbb_silu  = len(self._smf_result.buffer_bindings)
        nbb_q4    = len(self._linear_q4_result.buffer_bindings)

        # Constant params
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

        # INT4 Q4 params per projection shape
        Q4_GS = 128

        def mk_q4_params(name, K_val, N_val):
            pf = self._linear_q4_result.param_fields
            vals = {
                'K': K_val, 'stride_x': K_val,
                'stride_w_q4': K_val // 8,
                'n_groups': K_val // Q4_GS, 'N': N_val,
            }
            return mk_params(name, pf, vals)

        q4_qkv_p = mk_q4_params('q4_qkv_p', E, qkv_N)
        q4_oproj_p = mk_q4_params('q4_oproj_p', E, E)
        q4_gateup_p = mk_q4_params('q4_gateup_p', E, 2 * IM)
        q4_down_p = mk_q4_params('q4_down_p', IM, E)

        # Dynamic params (updated per token)
        fused_rope_ph = mkbuf('fused_rope_p', 28)
        attn_ph = mkbuf('attn_p', 24)

        cos_h = self._rope_cos_gpu.handle
        cos_sz = self._rope_cos_gpu.size
        sin_h = self._rope_sin_gpu.handle
        sin_sz = self._rope_sin_gpu.size

        bias_e = self._gpu_weights["zero_bias_E"]
        bias_qkv = self._gpu_weights["zero_bias_QKV"]
        bias_gu = self._gpu_weights["zero_bias_GU"]
        bias_v = self._gpu_weights["zero_bias_V"]
        norm_final_w = self._gpu_weights["norm.weight"]

        mk_bg = runner.create_bind_group

        # Shared bind groups
        bg_silu = mk_bg(bgl_silu, [
            (0, gate_up_buf[0], gate_up_buf[1]),
            (1, silu_out[0], silu_out[1]),
            (nbb_silu, silu_p[0], silu_p[1])])

        aip_g = ((E + self._add_block - 1) // self._add_block,)
        silu_g = ((IM + self._smf_block - 1) // self._smf_block,)

        # Per-layer bind groups and dispatch lists
        layer_dispatches = []
        for i in range(self.n_layer):
            pfx = f"layers.{i}."
            norm1_w = self._gpu_weights[pfx + "input_layernorm.weight"]
            norm2_w = self._gpu_weights[pfx + "post_attention_layernorm.weight"]
            K_cache, V_cache, _ = self._gpu_kv_cache[i]

            # Weight buffers (INT4)
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

            # norm1 (only for layer 0; fused into res_norm_next for others)
            bg_n1 = mk_bg(bgl_rn, [
                (0, x_buf[0], x_buf[1]),
                (1, norm_out[0], norm_out[1]),
                (2, norm1_w.handle, norm1_w.size),
                (3, rstd[0], rstd[1]),
                (nbb_rn, rn_p[0], rn_p[1])])

            # qkv (INT4)
            bg_qkv = mk_bg(bgl_q4, [
                (0, norm_out[0], norm_out[1]),
                (1, qkv_wq.handle, qkv_wq.size),
                (2, qkv_sc.handle, qkv_sc.size),
                (3, qkv_zr.handle, qkv_zr.size),
                (4, bias_qkv.handle, bias_qkv.size),
                (5, qkv_buf[0], qkv_buf[1]),
                (nbb_q4, q4_qkv_p[0], q4_qkv_p[1])])

            # fused rope
            bg_frope = mk_bg(bgl_fused, [
                (0, qkv_buf[0], qkv_buf[1]),
                (1, q_rot[0], q_rot[1]),
                (2, K_cache.handle, K_cache.size),
                (3, V_cache.handle, V_cache.size),
                (4, cos_h, cos_sz),
                (5, sin_h, sin_sz),
                (nbb_fused, fused_rope_ph[0], fused_rope_ph[1])])

            # attention
            bg_att = mk_bg(bgl_attn, [
                (0, q_rot[0], q_rot[1]),
                (1, K_cache.handle, K_cache.size),
                (2, V_cache.handle, V_cache.size),
                (3, attn_out_buf[0], attn_out_buf[1]),
                (nbb_attn, attn_ph[0], attn_ph[1])])

            # o_proj (INT4)
            bg_op = mk_bg(bgl_q4, [
                (0, attn_out_buf[0], attn_out_buf[1]),
                (1, op_wq.handle, op_wq.size),
                (2, op_sc.handle, op_sc.size),
                (3, op_zr.handle, op_zr.size),
                (4, bias_e.handle, bias_e.size),
                (5, proj_out[0], proj_out[1]),
                (nbb_q4, q4_oproj_p[0], q4_oproj_p[1])])

            # fused res1 + norm2 (add_rms_norm)
            bg_arn2 = mk_bg(bgl_arn, [
                (0, x_buf[0], x_buf[1]),
                (1, proj_out[0], proj_out[1]),
                (2, norm_out[0], norm_out[1]),
                (3, norm2_w.handle, norm2_w.size),
                (4, rstd[0], rstd[1]),
                (nbb_arn, arn_p[0], arn_p[1])])

            # gate_up (INT4)
            bg_gu = mk_bg(bgl_q4, [
                (0, norm_out[0], norm_out[1]),
                (1, gu_wq.handle, gu_wq.size),
                (2, gu_sc.handle, gu_sc.size),
                (3, gu_zr.handle, gu_zr.size),
                (4, bias_gu.handle, bias_gu.size),
                (5, gate_up_buf[0], gate_up_buf[1]),
                (nbb_q4, q4_gateup_p[0], q4_gateup_p[1])])

            # down (INT4)
            bg_dn = mk_bg(bgl_q4, [
                (0, silu_out[0], silu_out[1]),
                (1, dn_wq.handle, dn_wq.size),
                (2, dn_sc.handle, dn_sc.size),
                (3, dn_zr.handle, dn_zr.size),
                (4, bias_e.handle, bias_e.size),
                (5, proj_out[0], proj_out[1]),
                (nbb_q4, q4_down_p[0], q4_down_p[1])])

            # Residual add (x += proj_out)
            bg_res = mk_bg(bgl_aip, [
                (0, x_buf[0], x_buf[1]),
                (1, proj_out[0], proj_out[1]),
                (nbb_aip, aip_p[0], aip_p[1])])

            # Build dispatch list for this layer
            if i < self.n_layer - 1:
                # Fused res2 + next_layer_norm1 (add_rms_norm)
                next_norm1_w = self._gpu_weights[
                    f"layers.{i+1}.input_layernorm.weight"]
                bg_arn_next = mk_bg(bgl_arn, [
                    (0, x_buf[0], x_buf[1]),
                    (1, proj_out[0], proj_out[1]),
                    (2, norm_out[0], norm_out[1]),
                    (3, next_norm1_w.handle, next_norm1_w.size),
                    (4, rstd[0], rstd[1]),
                    (nbb_arn, arn_p[0], arn_p[1])])

                dispatches = []
                if i == 0:
                    dispatches.append((pl_rn, bg_n1, (1,)))  # norm1 only L0
                dispatches.extend([
                    (pl_q4,    bg_qkv,      (1, qkv_N)),       # qkv
                    (pl_fused, bg_frope,     (n_head + n_kv,)), # fused rope
                    (pl_attn,  bg_att,       (n_head,)),        # attn
                    (pl_q4,    bg_op,        (1, E)),           # o_proj
                    (pl_arn,   bg_arn2,      (1,)),             # fused res1+norm2
                    (pl_q4,    bg_gu,        (1, 2 * IM)),      # gate_up
                    (pl_silu,  bg_silu,      silu_g),           # silu_mul
                    (pl_q4,    bg_dn,        (1, E)),           # down
                    (pl_arn,   bg_arn_next,  (1,)),             # fused res2+norm_next
                ])
            else:
                # Last layer: res2 is just add_inplace
                dispatches = [
                    (pl_q4,    bg_qkv,    (1, qkv_N)),
                    (pl_fused, bg_frope,   (n_head + n_kv,)),
                    (pl_attn,  bg_att,     (n_head,)),
                    (pl_q4,    bg_op,      (1, E)),
                    (pl_arn,   bg_arn2,    (1,)),
                    (pl_q4,    bg_gu,      (1, 2 * IM)),
                    (pl_silu,  bg_silu,    silu_g),
                    (pl_q4,    bg_dn,      (1, E)),
                    (pl_aip,   bg_res,     aip_g),
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
        lm_w_fp16 = self._gpu_weights["embed_tokens.weight.fp16"]

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

        final_dispatches = [
            (pl_rn,  bg_final_rn, (1,)),
            (pl_lmh, bg_lmh,     (lm_gx, lm_gy)),
        ]

        # Store for per-token use
        self._fd_x_h = x_buf[0]
        self._fd_logits_h = logits_buf[0]
        self._fd_logits_sz = logits_buf[1]
        self._fd_fused_rope_ph = fused_rope_ph[0]
        self._fd_attn_ph = attn_ph[0]
        self._fd_all_batches = layer_dispatches + [final_dispatches]

        # Flatten all dispatches into a single list for single-submit path
        self._fd_all_flat = []
        for batch in self._fd_all_batches:
            self._fd_all_flat.extend(batch)

        # Also create merged batches: group 6 layers per batch for pipelined
        # submit (fewer submits = less overhead, but still some CPU/GPU overlap)
        MERGE = 6
        self._fd_merged_batches = []
        for i in range(0, len(self._fd_all_batches), MERGE):
            merged = []
            for batch in self._fd_all_batches[i:i+MERGE]:
                merged.extend(batch)
            self._fd_merged_batches.append(merged)

        # Pre-allocated bytearrays for dynamic params
        # fused_rope: n_head(i32), q_size(i32), kv_size(i32),
        #             pos(i32), half_rot(i32), cache_offset(i32)
        self._fd_frope_buf = bytearray(28)
        struct.pack_into('<iii', self._fd_frope_buf, 0,
                         n_head, q_size, kv_size)
        struct.pack_into('<i', self._fd_frope_buf, 16, half_rot)
        # attn: kv_stride(i32), n_rep(i32), T_total(i32),
        #       scale(f32), neg_inf(f32)
        self._fd_attn_buf = bytearray(24)
        struct.pack_into('<ii', self._fd_attn_buf, 0, n_kv * HD, n_rep)
        struct.pack_into('<ff', self._fd_attn_buf, 12,
                         float(np.float32(1.0 / np.sqrt(HD))),
                         float(np.float32(-1e9)))

        self._fast_decode_ready = True
        print(f"  Fast decode initialized "
              f"({sum(len(b) for b in self._fd_all_batches)} "
              f"pre-recorded dispatches)")

    def _warmup_fast_decode(self):
        """Initialize fast decode pipeline eagerly (before decode timer)."""
        if getattr(self, '_fast_decode_ready', False):
            return
        if self._decode_mode != 'gpu' or not self._use_q4_gpu:
            return
        self._init_fast_decode()

    def _decode_fast(self, token_ids, pos_offset):
        """Fast decode: submit pre-recorded dispatches with minimal overhead.

        Only updates 3 GPU buffers per token (embed, rope params, attn params),
        then submits the entire pre-recorded command list.

        When profiling is enabled, submits per-group with CPU+GPU timing
        so the timeline shows CPU recording time and GPU execution per group.
        """
        import time as _time
        runner = self.cache.runner
        wte = self.weights["embed_tokens.weight"]
        import struct
        p = self.profiler
        _p = p and p.enabled

        # Embed + upload
        if _p: p._cpu.begin("fast_decode/embed")
        x = wte[token_ids].ravel().astype(np.float32)
        runner.write_buffer(self._fd_x_h, x.tobytes())
        if _p: p._cpu.end("fast_decode/embed")

        # Dynamic params
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
        if _p: p._cpu.end("fast_decode/params")

        # Update KV cache counters
        for layer in range(self.n_layer):
            K, V, c = self._gpu_kv_cache[layer]
            self._gpu_kv_cache[layer] = (K, V, c + 1)

        if _p:
            # Profiling path: per-group submit with CPU+GPU timeline events
            # and link_ids connecting CPU recording to GPU execution
            n_batches = len(self._fd_all_batches)
            GROUP_SIZE = 4
            group_start = 0
            group_idx = 0
            while group_start < n_batches:
                group_end = min(group_start + GROUP_SIZE, n_batches)
                is_last = (group_end == n_batches)
                n_disp = sum(len(self._fd_all_batches[i])
                             for i in range(group_start, group_end))

                # Layer range label
                if is_last and group_end == n_batches:
                    label = f"fast/L{group_start}-{group_end-2}+lmh"
                else:
                    label = f"fast/L{group_start}-{group_end-1}"

                # Unique link_id connecting CPU record → GPU execution
                link = f"fd_{pos}_{group_idx}"

                # CPU: record time (encode + submit)
                p._cpu.begin(label)
                t0 = _time.perf_counter_ns()

                # Collect dispatches for this group
                group_dispatches = []
                for bi in range(group_start, group_end):
                    group_dispatches.extend(self._fd_all_batches[bi])

                readback = None
                if is_last:
                    readback = (self._fd_logits_h, self._fd_logits_sz,
                                np.float32)

                result = runner.submit_dispatches(
                    group_dispatches, readback=readback)

                t1 = _time.perf_counter_ns()
                p._cpu.end(label)

                # Set link_id on the CPU event we just ended
                p._cpu._events[-1].link_id = link

                # GPU dispatch event with matching link_id
                from common.profiler import GPUDispatchEvent
                p._dispatch_events.append(GPUDispatchEvent(
                    name=f"{label}({n_disp}disp)",
                    begin_ns=t0, end_ns=t1, link_id=link))

                if is_last:
                    logits = result

                group_start = group_end
                group_idx += 1
        else:
            # Fast path: single pipelined submit (no per-group overhead)
            logits = runner.submit_dispatches_pipelined(
                self._fd_all_batches,
                readback=(self._fd_logits_h, self._fd_logits_sz,
                          np.float32))

        return logits.reshape(1, self.n_vocab)

    # -- Transformer blocks (GPU prefill path) --

    def _attention_block(self, x, layer: int,
                         use_cache: bool = False,
                         positions: np.ndarray = None,
                         **kwargs):
        """GQA with fused QKV, pre-computed RoPE, multi-head attention."""
        from common.model_base import GPUBuffer

        if isinstance(x, GPUBuffer):
            T = x.shape[0] if x.shape else 1
        else:
            T = x.shape[0]
        E = self.n_embd
        HD = self.head_dim
        n_head = self.n_head
        n_kv = self.n_kv_heads
        n_rep = self.n_rep
        pfx = f"layers.{layer}.self_attn."

        # Fused QKV projection (1 dispatch instead of 3)
        qkv = self._proj(
            x, pfx + "qkv_proj.weight",
            self._gpu_weights["zero_bias_QKV"], self.qkv_out)

        # Split Q, K, V from fused output
        q_size = n_head * HD
        kv_size = n_kv * HD
        Q = qkv[:, :q_size].reshape(T, n_head, HD)
        K_new = qkv[:, q_size:q_size + kv_size].reshape(T, n_kv, HD)
        V_new = qkv[:, q_size + kv_size:].reshape(T, n_kv, HD)

        # RoPE with pre-computed tables
        if positions is None:
            positions = np.arange(T, dtype=np.int32)
        Q = self._apply_rope_fast(Q, positions)
        K_new = self._apply_rope_fast(K_new, positions)

        # KV cache: pre-allocated, in-place write
        if use_cache:
            if self.kv_cache is None:
                self.kv_cache = {}
            if layer not in self.kv_cache:
                K_buf = np.zeros((self.MAX_SEQ_LEN, n_kv, HD),
                                dtype=np.float32)
                V_buf = np.zeros((self.MAX_SEQ_LEN, n_kv, HD),
                                dtype=np.float32)
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

        if T == 1 and use_cache and T_total > 1:
            # Decode: vectorized multi-head attention (no per-head loop)
            scale = 1.0 / np.sqrt(HD)
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
            # Prefill: multi-head causal attention (1 dispatch, not n_head)
            attn_out = self._causal_attention_multihead(
                Q, K_full, V_full, n_rep)

        # O projection
        attn_flat = attn_out.reshape(T, E)
        return self._proj(
            attn_flat, pfx + "o_proj.weight",
            self._gpu_weights["zero_bias_E"], E)

    def _mlp_block(self, x, layer: int,
                   gpu_out: bool = False):
        """SwiGLU MLP with fused gate_up projection.

        1 dispatch for gate+up (was 2), fused SiLU*mul from concat buffer.
        """
        from common.model_base import GPUBuffer
        E = self.n_embd
        IM = self.intermediate_size
        pfx = f"layers.{layer}.mlp."

        x_is_gpu = isinstance(x, GPUBuffer)
        T = (x.shape[0] if x.shape else 1) if x_is_gpu else x.shape[0]

        # Fused gate+up: (T, E) → (T, 2*IM)  — 1 dispatch instead of 2
        if T > 1 or x_is_gpu:
            gate_up = self._proj(
                x, pfx + "gate_up_proj.weight",
                self._gpu_weights["zero_bias_GU"], self.gate_up_out,
                gpu_out=True)
            h = self._silu_mul_fused(gate_up, IM, gpu_out=True)
        else:
            gate_up = self._proj(
                x, pfx + "gate_up_proj.weight",
                self._gpu_weights["zero_bias_GU"], self.gate_up_out)
            gate = np.ascontiguousarray(gate_up[:, :IM])
            up = np.ascontiguousarray(gate_up[:, IM:])
            h = self._silu_mul(gate, up, gpu_out=True)

        # down_proj: (T, IM) → (T, E)
        return self._proj(
            h, pfx + "down_proj.weight",
            self._gpu_weights["zero_bias_E"], E, K=IM,
            gpu_out=gpu_out)

    def _transformer_block(self, x, layer: int,
                           use_cache: bool = False,
                           positions: np.ndarray = None,
                           **kwargs):
        """Pre-norm transformer block with full GPU chain."""
        pfx = f"layers.{layer}."

        # Attention sub-block
        rn1 = self._rms_norm(
            x, self._gpu_weights[pfx + "input_layernorm.weight"],
            gpu_out=True)
        attn = self._attention_block(rn1, layer, use_cache=use_cache,
                                     positions=positions)
        x = self._add(x, attn, gpu_out=True)

        # MLP sub-block
        rn2 = self._rms_norm(
            x, self._gpu_weights[pfx + "post_attention_layernorm.weight"],
            gpu_out=True)
        mlp = self._mlp_block(rn2, layer, gpu_out=True)
        x = self._add(x, mlp, gpu_out=True)
        return x

    def forward(self, token_ids: np.ndarray,
                use_cache: bool = False,
                pos_offset: int = 0) -> np.ndarray:
        """SmolLM2 forward pass with CPU or GPU decode.

        Args:
            token_ids: (T,) int32 token IDs
            use_cache: if True, use/update KV cache
            pos_offset: position offset for RoPE

        Returns:
            logits: (T, n_vocab) float32
        """
        from common.model_base import GPUBuffer
        T = len(token_ids)

        # Fast decode path: pre-recorded bind groups, minimal per-token overhead
        if (T == 1 and use_cache and self._decode_mode == 'gpu'
                and self._use_q4_gpu):
            if not getattr(self, '_fast_decode_ready', False):
                self._init_fast_decode()
            if self._fast_decode_ready:
                return self._decode_fast(token_ids, pos_offset)

        # Token embeddings (no position embeddings — RoPE applied later)
        wte = self.weights["embed_tokens.weight"]
        x = wte[token_ids]

        # Position indices for RoPE
        positions = np.arange(pos_offset, pos_offset + T, dtype=np.int32)

        if T == 1 and use_cache and self._decode_mode == 'gpu' \
                and hasattr(self, '_gpu_kv_cache'):
            # GPU-resident decode: all ops on GPU, single readback at end
            for layer in range(self.n_layer):
                x = self._decode_gpu(x, layer, positions)

            # Final RMSNorm (on GPU)
            if isinstance(x, GPUBuffer):
                x = self._rms_norm(
                    x, self._gpu_weights["norm.weight"],
                    gpu_out=True)
            else:
                x = self._rms_norm_cpu(x, "norm.weight")

            # LM head — use GPU fp16 weight if available
            fp16_key = "embed_tokens.weight.fp16"
            if fp16_key in self._gpu_weights and isinstance(x, GPUBuffer):
                logits = self._linear_fp16w(
                    x, self._gpu_weights[fp16_key],
                    self._gpu_weights["zero_bias_V"], self.n_vocab,
                    K=self.n_embd)
            elif isinstance(x, GPUBuffer):
                logits = self._linear(
                    x, self._gpu_weights["embed_tokens.weight"],
                    self._gpu_weights["zero_bias_V"], self.n_vocab)
            else:
                logits = np.float32(x @ wte.T)
            return logits

        elif T == 1 and use_cache and self.kv_cache is not None:
            # CPU decode: all matmuls on CPU (for small models)
            for layer in range(self.n_layer):
                x = self._decode_cpu(x, layer, positions)

            # Final RMSNorm (CPU)
            x = self._rms_norm_cpu(x, "norm.weight")

            # LM head (CPU matmul — tied embed_tokens)
            logits = self._cpu_matmul(x, "embed_tokens.weight",
                                      self.n_vocab, self.n_embd)
            return logits
        else:
            # GPU prefill path
            for layer in range(self.n_layer):
                x = self._transformer_block(x, layer, use_cache=use_cache,
                                            positions=positions)

            # Final RMSNorm
            x = self._rms_norm(x, self._gpu_weights["norm.weight"])

            # LM head (tied embed_tokens)
            logits = self._linear(
                x, self._gpu_weights["embed_tokens.weight"],
                self._gpu_weights["zero_bias_V"],
                self.n_vocab)
            return logits


# ---------------------------------------------------------------------------
# Weight downloading
# ---------------------------------------------------------------------------

def download_smollm2_weights(model_size: str = "135M",
                              model_dir: str = None) -> Tuple[str, str]:
    """Download SmolLM2 weights and tokenizer from HuggingFace."""
    config = SMOLLM2_CONFIGS[model_size]
    hf_repo = config["hf_repo"]
    if model_dir is None:
        model_dir = os.path.join(_SCRIPT_DIR, "weights", model_size)

    def smollm2_key_transform(key, arr):
        """Strip 'model.' prefix, handle tied weights."""
        new_key = key.replace("model.", "")
        # lm_head is tied with embed_tokens — skip duplicate
        if new_key == "lm_head.weight":
            return None
        return new_key, arr

    npz_path, tokenizer_path = download_weights(
        hf_repo=hf_repo,
        model_dir=model_dir,
        key_transform=smollm2_key_transform,
        download_tokenizer=True,
    )
    return npz_path, tokenizer_path


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_with_random_weights():
    """Verify SmolLM2 pipeline with small random weights (no download)."""
    print("=" * 60)
    print("SmolLM2 WebGPU Pipeline Verification (random weights)")
    print("=" * 60)

    n_layer, n_head, n_kv_heads = 2, 6, 2
    n_embd, intermediate_size, n_vocab = 96, 256, 512
    head_dim = n_embd // n_head
    kv_dim = n_kv_heads * head_dim
    n_rep = n_head // n_kv_heads
    rope_theta = 10000.0
    eps = 1e-5
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
        weights[pfx + "mlp.gate_proj.weight"] = np.random.randn(
            intermediate_size, n_embd).astype(np.float32) * 0.02
        weights[pfx + "mlp.up_proj.weight"] = np.random.randn(
            intermediate_size, n_embd).astype(np.float32) * 0.02
        weights[pfx + "mlp.down_proj.weight"] = np.random.randn(
            n_embd, intermediate_size).astype(np.float32) * 0.02

    print(f"\nModel: {n_layer} layers, {n_head} Q heads, {n_kv_heads} KV heads, "
          f"{n_embd} embd, {intermediate_size} intermediate, {n_vocab} vocab")

    model = SmolLM2WebGPU(
        weights, n_layer=n_layer, n_head=n_head, n_kv_heads=n_kv_heads,
        n_embd=n_embd, intermediate_size=intermediate_size,
        n_vocab=n_vocab, rope_theta=rope_theta, rms_norm_eps=eps)

    # Forward pass
    token_ids = np.array([1, 42, 100, 200], dtype=np.int32)
    T = len(token_ids)
    t0 = time.time()
    logits = model.forward(token_ids)
    t1 = time.time()

    print(f"\nForward pass: {token_ids} → shape {logits.shape} "
          f"in {(t1-t0)*1000:.0f}ms")

    # --- NumPy reference ---
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
    # Use model.weights which has fused QKV and gate_up from _fuse_weights()
    ref_w = model.weights
    x = ref_w["embed_tokens.weight"][token_ids]

    q_size = n_head * head_dim
    kv_size = n_kv_heads * head_dim

    for layer in range(n_layer):
        pfx = f"layers.{layer}."
        rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
        ln1 = x / rms * ref_w[pfx + "input_layernorm.weight"]

        # Fused QKV — split for reference computation
        qkv_w = ref_w[pfx + "self_attn.qkv_proj.weight"]
        q = ln1 @ qkv_w[:q_size].T
        k = ln1 @ qkv_w[q_size:q_size + kv_size].T
        v = ln1 @ qkv_w[q_size + kv_size:].T

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
        proj = attn_flat @ ref_w[pfx + "self_attn.o_proj.weight"].T
        x = x + proj

        rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
        ln2 = x / rms * ref_w[pfx + "post_attention_layernorm.weight"]

        # Fused gate+up — split for reference computation
        gate_up_w = ref_w[pfx + "mlp.gate_up_proj.weight"]
        gate = ln2 @ gate_up_w[:intermediate_size].T
        up_val = ln2 @ gate_up_w[intermediate_size:].T
        silu_gate = gate / (1.0 + np.exp(-gate))
        mlp_h = silu_gate * up_val
        mlp_out = mlp_h @ ref_w[pfx + "mlp.down_proj.weight"].T
        x = x + mlp_out

    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    x = x / rms * ref_w["norm.weight"]
    logits_ref = x @ ref_w["embed_tokens.weight"].T

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
        description="SmolLM2 on WebGPU via Triton")
    parser.add_argument("--verify", action="store_true",
                        help="Verify pipeline with random weights")
    parser.add_argument("--model", type=str, default="1.7B",
                        choices=["135M", "360M", "1.7B"],
                        help="Model size: 135M, 360M, or 1.7B")
    parser.add_argument("--prompt", type=str,
                        default="The future of AI is",
                        help="Prompt for text generation")
    parser.add_argument("--max-tokens", type=int, default=50,
                        help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature")
    parser.add_argument("--weights-dir", type=str, default=None,
                        help="Directory for cached weights")
    parser.add_argument("--profile", action="store_true",
                        help="Profile inference and generate HTML report")
    parser.add_argument("--quantize", action="store_true",
                        help="Quantize weights to INT4 and save")
    parser.add_argument("--decode-mode", type=str, default='auto',
                        choices=['auto', 'cpu', 'gpu'],
                        help="Decode mode: cpu, gpu, or auto (gpu for 1.7B+)")
    args = parser.parse_args()

    if args.verify:
        success = verify_with_random_weights()
        sys.exit(0 if success else 1)

    # Download and load real weights
    config = SMOLLM2_CONFIGS[args.model]
    npz_path, tokenizer_path = download_smollm2_weights(
        args.model, args.weights_dir)

    # Determine weight and quantization paths
    model_dir = os.path.dirname(npz_path)
    q4_path = os.path.join(model_dir, "weights_q4.npz")

    if args.quantize:
        # Quantize: load fp32 weights, fuse, quantize, save
        print(f"Quantizing SmolLM2-{args.model} weights...")
        weights = load_weights(npz_path)
        # Fuse Q/K/V and gate/up before quantizing
        n_layer = config["n_layer"]
        n_head = config["n_head"]
        n_kv_heads = config["n_kv_heads"]
        head_dim = config["n_embd"] // n_head
        for i in range(n_layer):
            pfx = f"layers.{i}."
            q_key = pfx + "self_attn.q_proj.weight"
            if q_key in weights:
                weights[pfx + "self_attn.qkv_proj.weight"] = \
                    np.concatenate([weights[q_key],
                                    weights[pfx + "self_attn.k_proj.weight"],
                                    weights[pfx + "self_attn.v_proj.weight"]],
                                   axis=0).astype(np.float32)
                del weights[q_key]
                del weights[pfx + "self_attn.k_proj.weight"]
                del weights[pfx + "self_attn.v_proj.weight"]
            gate_key = pfx + "mlp.gate_proj.weight"
            if gate_key in weights:
                weights[pfx + "mlp.gate_up_proj.weight"] = \
                    np.concatenate([weights[gate_key],
                                    weights[pfx + "mlp.up_proj.weight"]],
                                   axis=0).astype(np.float32)
                del weights[gate_key]
                del weights[pfx + "mlp.up_proj.weight"]
        quantized = quantize_smollm2_weights(weights, n_layer)
        np.savez(q4_path, **quantized)
        print(f"  Saved to {q4_path}")
        return

    # Load weights: prefer quantized if available
    quantized = False
    if os.path.exists(q4_path):
        print(f"Loading quantized weights from {q4_path}")
        weights = load_quantized_weights(q4_path)
        quantized = True
    else:
        weights = load_weights(npz_path)
    print(f"Loaded {len(weights)} weight tensors"
          + (" (INT4 quantized)" if quantized else ""))

    tokenizer = load_tokenizer(tokenizer_path)

    # Auto-select decode mode
    decode_mode = args.decode_mode
    if decode_mode == 'auto':
        decode_mode = 'gpu' if config["n_embd"] >= 2048 else 'cpu'

    # Create model
    model = SmolLM2WebGPU(
        weights,
        n_layer=config["n_layer"],
        n_head=config["n_head"],
        n_kv_heads=config["n_kv_heads"],
        n_embd=config["n_embd"],
        intermediate_size=config["intermediate_size"],
        n_vocab=config["n_vocab"],
        rope_theta=config["rope_theta"],
        rms_norm_eps=config["rms_norm_eps"],
        quantized=quantized,
        decode_mode=decode_mode)
    print(f"Model created, kernels compiled (decode={decode_mode})")

    # Print GPU info
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

    if args.profile:
        model.enable_profiling()
        print(f"Profiling enabled (GPU timestamps: "
              f"{model.profiler.gpu_enabled})")

        from common.profiler_html import generate_html_report
        tokens = tokenizer.encode(args.prompt)
        token_ids = np.array(tokens, dtype=np.int32)
        model.kv_cache = None

        # Prefill
        with model.profiler.step("prefill"):
            logits = model.forward(token_ids, use_cache=True, pos_offset=0)

        # Decode N tokens with profiling
        n_profile = min(args.max_tokens, 10)
        generated = list(tokens)
        next_logits = logits[-1, :].copy()
        for step in range(n_profile):
            next_logits = next_logits / args.temperature
            next_logits -= next_logits.max()
            probs = np.exp(next_logits)
            probs /= probs.sum()
            next_token = int(np.random.choice(len(probs), p=probs))
            generated.append(next_token)

            with model.profiler.step(f"decode_{step}"):
                with model.profiler.scope("forward"):
                    logits = model.forward(
                        np.array([next_token], dtype=np.int32),
                        use_cache=True,
                        pos_offset=len(generated) - 1)
                with model.profiler.cpu("sampling"):
                    next_logits = logits[-1, :].copy()

        model.profiler.finish()
        model.profiler.report()

        # Generate HTML timeline report
        profile_path = os.path.join(_SCRIPT_DIR, "profile.html")
        generate_html_report(
            model.profiler,
            output_path=profile_path,
            title=f"SmolLM2-{args.model} — {adapter.get('description', 'GPU')} ({adapter.get('backend', '')})",
            adapter_info=adapter,
            memory_info=model.get_memory_info())
        print(f"\nHTML profile: {profile_path}")

        model.disable_profiling()
        return

    # Generate
    generate(model, args.prompt, tokenizer,
             max_tokens=args.max_tokens,
             temperature=args.temperature)


if __name__ == "__main__":
    main()
