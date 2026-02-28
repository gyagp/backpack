"""
Qwen2.5 inference on WebGPU via Triton.

Qwen2.5 is a LLaMA-family model featuring:
  - RoPE (rotary position embeddings)
  - RMSNorm (root mean square normalization)
  - GQA (grouped query attention)
  - SwiGLU MLP (SiLU-gated linear unit)
  - Attention biases on Q, K, V projections (unlike SmolLM2/LLaMA)
  - Separate (untied) lm_head

Optimizations:
  - INT4 per-group weight quantization (4× memory reduction)
  - Fused QKV projection (3→1 dispatch) with bias concatenation
  - Fused gate+up MLP projection (2→1 dispatch)
  - fp16 weight storage (2× bandwidth reduction)
  - Pre-computed RoPE tables
  - GPU-resident KV cache with fused RoPE+scatter
  - Fast decode pipeline (pre-recorded dispatches)

Usage:
    python models/qwen-2.5/model.py --verify
    python models/qwen-2.5/model.py --prompt "Hello"
    python models/qwen-2.5/model.py --quantize
    python models/qwen-2.5/model.py --prompt "Hello" --decode-mode gpu

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


# Qwen2.5 model configs
QWEN_CONFIGS = {
    "0.5B": {
        "n_layer": 24, "n_head": 14, "n_kv_heads": 2,
        "n_embd": 896, "intermediate_size": 4864,
        "n_vocab": 151936, "rope_theta": 1000000.0,
        "rms_norm_eps": 1e-6, "head_dim": 64,
        "hf_repo": "Qwen/Qwen2.5-0.5B",
        "attention_bias": True,
        "tie_word_embeddings": True,
    },
    "1.5B": {
        "n_layer": 28, "n_head": 12, "n_kv_heads": 2,
        "n_embd": 1536, "intermediate_size": 8960,
        "n_vocab": 151936, "rope_theta": 1000000.0,
        "rms_norm_eps": 1e-6, "head_dim": 128,
        "hf_repo": "Qwen/Qwen2.5-1.5B",
        "attention_bias": True,
        "tie_word_embeddings": True,
    },
}


class QwenWebGPU(WebGPUModel):
    """Qwen2.5 inference on WebGPU via Triton kernels.

    Supports Qwen2.5-0.5B and Qwen2.5-1.5B with:
      - Fused QKV projection (3→1 dispatch) with concatenated biases
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
                 decode_mode: str = 'cpu'):
        self.attention_bias = attention_bias
        self.tie_word_embeddings = tie_word_embeddings
        self.qkv_out = n_head * head_dim + 2 * n_kv_heads * head_dim
        self.gate_up_out = 2 * intermediate_size
        self._quantized = quantized
        self._decode_mode = decode_mode
        super().__init__(
            weights, n_layer=n_layer, n_head=n_head, n_embd=n_embd,
            n_vocab=n_vocab,
            n_kv_heads=n_kv_heads,
            intermediate_size=intermediate_size,
            head_dim=head_dim,
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

    def _compile_model_kernels(self):
        """Compile Qwen-specific kernels: RMSNorm, SiLU*mul."""
        self._compile_rms_norm()
        self._compile_silu_mul()

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
        """Upload weights to GPU (INT4/fp16/fp32 depending on capabilities)."""
        E = self.n_embd
        IM = self.intermediate_size

        use_q4_gpu = (self._quantized
                      and getattr(self, '_has_fp16_linear', False))
        self._use_q4_gpu = use_q4_gpu
        use_fp16 = (not use_q4_gpu
                    and getattr(self, '_has_fp16_linear', False))

        for i in range(self.n_layer):
            pfx = f"layers.{i}."
            # RMSNorm
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
        if use_fp16 or use_q4_gpu:
            self._upload_linear_weight_fp16(ekey, self.n_vocab, E)
        self._upload_embedding_weight(ekey, self.n_vocab, E)

        if not self.tie_word_embeddings:
            lm_key = "lm_head.weight"
            if self.weights[lm_key].dtype == np.float16:
                self.weights[lm_key] = self.weights[lm_key].astype(np.float32)
            if use_fp16 or use_q4_gpu:
                self._upload_linear_weight_fp16(lm_key, self.n_vocab, E)
            self._upload_embedding_weight(lm_key, self.n_vocab, E)

        # Zero biases
        self._upload_zero_bias("zero_bias_E", E)
        self._upload_zero_bias("zero_bias_QKV", self.qkv_out)
        self._upload_zero_bias("zero_bias_GU", self.gate_up_out)
        self._upload_zero_bias("zero_bias_V", self.n_vocab)

        self._print_gpu_weight_stats()

    # -- Projection helper --

    def _proj(self, x, name, bias_key, N, K=None, gpu_out=False):
        """Linear projection using best available kernel (Q4/fp16/fp32)."""
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
        qkv = self._proj(x, pfx + "qkv_proj.weight", bias_key, self.qkv_out, K=E)

        # Split
        q_dim = n_head * HD
        q = qkv[:, :q_dim]
        k = qkv[:, q_dim:q_dim + self.kv_dim]
        v = qkv[:, q_dim + self.kv_dim:]

        Q = q.reshape(T, n_head, HD)
        K_new = k.reshape(T, n_kv, HD)
        V_new = v.reshape(T, n_kv, HD)

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

        attn_flat = attn_out.reshape(T, n_head * HD)
        return self._proj(attn_flat, pfx + "o_proj.weight",
                          "zero_bias_E", E, K=n_head * HD)

    # -- MLP --

    def _mlp_block(self, x, layer: int, gpu_out: bool = False):
        """SwiGLU MLP with fused gate+up projection."""
        E, IM = self.n_embd, self.intermediate_size
        pfx = f"layers.{layer}.mlp."
        gate_up = self._proj(x, pfx + "gate_up_proj.weight",
                             "zero_bias_GU", self.gate_up_out, K=E, gpu_out=True)
        h = self._silu_mul_fused(gate_up, IM, gpu_out=True)
        return self._proj(h, pfx + "down_proj.weight",
                          "zero_bias_E", E, K=IM, gpu_out=gpu_out)

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
        if self._decode_mode != 'gpu' or not self._use_q4_gpu:
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
        pl_fused, bgl_fused = get_pl(self._fused_rope_result)
        pl_attn,  bgl_attn  = get_pl(self._gqa_attn_result)
        pl_aip,   bgl_aip   = get_pl(self._add_ip_result)
        pl_silu,  bgl_silu  = get_pl(self._smf_result)
        pl_q4,    bgl_q4    = get_pl(self._linear_q4_result)

        nbb_rn = len(self._rn_result.buffer_bindings)
        nbb_arn = len(self._add_rn_result.buffer_bindings)
        nbb_fused = len(self._fused_rope_result.buffer_bindings)
        nbb_attn = len(self._gqa_attn_result.buffer_bindings)
        nbb_aip = len(self._add_ip_result.buffer_bindings)
        nbb_silu = len(self._smf_result.buffer_bindings)
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

        Q4_GS = 128

        def mk_q4_params(name, K_val, N_val):
            pf = self._linear_q4_result.param_fields
            return mk_params(name, pf, {
                'K': K_val, 'stride_x': K_val,
                'stride_w_q4': K_val // 8,
                'n_groups': K_val // Q4_GS, 'N': N_val,
            })

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

            bg_n1 = mk_bg(bgl_rn, [
                (0, x_buf[0], x_buf[1]),
                (1, norm_out[0], norm_out[1]),
                (2, norm1_w.handle, norm1_w.size),
                (3, rstd[0], rstd[1]),
                (nbb_rn, rn_p[0], rn_p[1])])

            bg_qkv = mk_bg(bgl_q4, [
                (0, norm_out[0], norm_out[1]),
                (1, qkv_wq.handle, qkv_wq.size),
                (2, qkv_sc.handle, qkv_sc.size),
                (3, qkv_zr.handle, qkv_zr.size),
                (4, layer_bias_qkv.handle, layer_bias_qkv.size),
                (5, qkv_buf[0], qkv_buf[1]),
                (nbb_q4, q4_qkv_p[0], q4_qkv_p[1])])

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

            bg_op = mk_bg(bgl_q4, [
                (0, attn_out_buf[0], attn_out_buf[1]),
                (1, op_wq.handle, op_wq.size),
                (2, op_sc.handle, op_sc.size),
                (3, op_zr.handle, op_zr.size),
                (4, bias_e.handle, bias_e.size),
                (5, proj_out[0], proj_out[1]),
                (nbb_q4, q4_oproj_p[0], q4_oproj_p[1])])

            bg_arn2 = mk_bg(bgl_arn, [
                (0, x_buf[0], x_buf[1]),
                (1, proj_out[0], proj_out[1]),
                (2, norm_out[0], norm_out[1]),
                (3, norm2_w.handle, norm2_w.size),
                (4, rstd[0], rstd[1]),
                (nbb_arn, arn_p[0], arn_p[1])])

            bg_gu = mk_bg(bgl_q4, [
                (0, norm_out[0], norm_out[1]),
                (1, gu_wq.handle, gu_wq.size),
                (2, gu_sc.handle, gu_sc.size),
                (3, gu_zr.handle, gu_zr.size),
                (4, bias_gu.handle, bias_gu.size),
                (5, gate_up_buf[0], gate_up_buf[1]),
                (nbb_q4, q4_gateup_p[0], q4_gateup_p[1])])

            bg_dn = mk_bg(bgl_q4, [
                (0, silu_out[0], silu_out[1]),
                (1, dn_wq.handle, dn_wq.size),
                (2, dn_sc.handle, dn_sc.size),
                (3, dn_zr.handle, dn_zr.size),
                (4, bias_e.handle, bias_e.size),
                (5, proj_out[0], proj_out[1]),
                (nbb_q4, q4_down_p[0], q4_down_p[1])])

            bg_res = mk_bg(bgl_aip, [
                (0, x_buf[0], x_buf[1]),
                (1, proj_out[0], proj_out[1]),
                (nbb_aip, aip_p[0], aip_p[1])])

            if i < self.n_layer - 1:
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
                    dispatches.append((pl_rn, bg_n1, (1,)))
                dispatches.extend([
                    (pl_q4,    bg_qkv,     (1, qkv_N)),
                    (pl_fused, bg_frope,    (n_head + n_kv,)),
                    (pl_attn,  bg_att,      (n_head,)),
                    (pl_q4,    bg_op,       (1, E)),
                    (pl_arn,   bg_arn2,     (1,)),
                    (pl_q4,    bg_gu,       (1, 2 * IM)),
                    (pl_silu,  bg_silu,     silu_g),
                    (pl_q4,    bg_dn,       (1, E)),
                    (pl_arn,   bg_arn_next, (1,)),
                ])
            else:
                dispatches = [
                    (pl_q4,    bg_qkv,  (1, qkv_N)),
                    (pl_fused, bg_frope, (n_head + n_kv,)),
                    (pl_attn,  bg_att,   (n_head,)),
                    (pl_q4,    bg_op,    (1, E)),
                    (pl_arn,   bg_arn2,  (1,)),
                    (pl_q4,    bg_gu,    (1, 2 * IM)),
                    (pl_silu,  bg_silu,  silu_g),
                    (pl_q4,    bg_dn,    (1, E)),
                    (pl_aip,   bg_res,   aip_g),
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

        final_dispatches = [
            (pl_rn,  bg_final_rn, (1,)),
            (pl_lmh, bg_lmh,      (lm_gx, lm_gy)),
        ]

        self._fd_x_h = x_buf[0]
        self._fd_logits_h = logits_buf[0]
        self._fd_logits_sz = logits_buf[1]
        self._fd_fused_rope_ph = fused_rope_ph[0]
        self._fd_attn_ph = attn_ph[0]
        self._fd_all_batches = layer_dispatches + [final_dispatches]

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

        self._fd_frope_buf = bytearray(28)
        struct.pack_into('<iii', self._fd_frope_buf, 0,
                         n_head, q_size, kv_size)
        struct.pack_into('<i', self._fd_frope_buf, 16, half_rot)
        self._fd_attn_buf = bytearray(24)
        struct.pack_into('<ii', self._fd_attn_buf, 0, n_kv * HD, n_rep)
        struct.pack_into('<ff', self._fd_attn_buf, 12,
                         float(np.float32(1.0 / np.sqrt(HD))),
                         float(np.float32(-1e9)))

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
        if _p: p._cpu.end("fast_decode/params")

        for layer in range(self.n_layer):
            K, V, c = self._gpu_kv_cache[layer]
            self._gpu_kv_cache[layer] = (K, V, c + 1)

        if _p:
            from common.profiler import GPUDispatchEvent
            n_batches = len(self._fd_all_batches)
            GROUP_SIZE = 4
            group_start = 0
            group_idx = 0
            while group_start < n_batches:
                group_end = min(group_start + GROUP_SIZE, n_batches)
                is_last = (group_end == n_batches)
                n_disp = sum(len(self._fd_all_batches[i])
                             for i in range(group_start, group_end))
                label = (f"fast/L{group_start}-{group_end-2}+lmh"
                         if is_last else f"fast/L{group_start}-{group_end-1}")
                link = f"fd_{pos}_{group_idx}"
                p._cpu.begin(label)
                t0 = _time.perf_counter_ns()
                group_dispatches = []
                for bi in range(group_start, group_end):
                    group_dispatches.extend(self._fd_all_batches[bi])
                readback = ((self._fd_logits_h, self._fd_logits_sz,
                             np.float32) if is_last else None)
                result = runner.submit_dispatches(
                    group_dispatches, readback=readback)
                t1 = _time.perf_counter_ns()
                p._cpu.end(label)
                p._cpu._events[-1].link_id = link
                p._dispatch_events.append(GPUDispatchEvent(
                    name=f"{label}({n_disp}disp)",
                    begin_ns=t0, end_ns=t1, link_id=link))
                if is_last:
                    logits = result
                group_start = group_end
                group_idx += 1
        else:
            logits = runner.submit_dispatches_pipelined(
                self._fd_all_batches,
                readback=(self._fd_logits_h, self._fd_logits_sz,
                          np.float32))

        return logits.reshape(1, self.n_vocab)

    # -- Forward pass --

    def forward(self, token_ids: np.ndarray,
                use_cache: bool = False,
                pos_offset: int = 0) -> np.ndarray:
        """Run Qwen2.5 forward pass."""
        T = len(token_ids)

        # Fast decode path
        if (T == 1 and use_cache and self._decode_mode == 'gpu'
                and self._use_q4_gpu):
            if not getattr(self, '_fast_decode_ready', False):
                self._init_fast_decode()
            return self._decode_fast(token_ids, pos_offset)

        wte = self.weights["embed_tokens.weight"]
        x = wte[token_ids]
        positions = np.arange(pos_offset, pos_offset + T, dtype=np.int32)

        for layer in range(self.n_layer):
            x = self._transformer_block(x, layer, use_cache=use_cache,
                                        positions=positions)

        x = self._rms_norm(x, self._gpu_weights["norm.weight"])

        lm_key = ("embed_tokens.weight" if self.tie_word_embeddings
                   else "lm_head.weight")
        logits = self._linear(
            x, self._gpu_weights[lm_key],
            self._gpu_weights["zero_bias_V"], self.n_vocab)
        return logits


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------

def quantize_qwen_weights(weights: Dict[str, np.ndarray],
                          n_layer: int) -> Dict[str, np.ndarray]:
    """Quantize Qwen2.5 fused weights to INT4 with per-group fp16 scales."""
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
# Weight downloading
# ---------------------------------------------------------------------------

def download_qwen_weights(model_size: str = "0.5B",
                          model_dir: str = None) -> Tuple[str, str]:
    """Download Qwen2.5 weights and tokenizer from HuggingFace."""
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
    """Verify Qwen2.5 pipeline with small random weights."""
    print("=" * 60)
    print("Qwen2.5 WebGPU Pipeline Verification (random weights)")
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
        # Attention biases
        weights[pfx + "self_attn.q_proj.bias"] = np.random.randn(
            n_embd).astype(np.float32) * 0.01
        weights[pfx + "self_attn.k_proj.bias"] = np.random.randn(
            kv_dim).astype(np.float32) * 0.01
        weights[pfx + "self_attn.v_proj.bias"] = np.random.randn(
            kv_dim).astype(np.float32) * 0.01
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

        q = ln1 @ weights[pfx + "self_attn.q_proj.weight"].T + \
            weights[pfx + "self_attn.q_proj.bias"]
        k = ln1 @ weights[pfx + "self_attn.k_proj.weight"].T + \
            weights[pfx + "self_attn.k_proj.bias"]
        v = ln1 @ weights[pfx + "self_attn.v_proj.weight"].T + \
            weights[pfx + "self_attn.v_proj.bias"]

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
    model = QwenWebGPU(
        weights, n_layer=n_layer, n_head=n_head, n_kv_heads=n_kv_heads,
        n_embd=n_embd, intermediate_size=intermediate_size,
        n_vocab=n_vocab, rope_theta=rope_theta, rms_norm_eps=eps,
        head_dim=head_dim, attention_bias=True, tie_word_embeddings=True)

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
        description="Qwen2.5 on WebGPU via Triton")
    parser.add_argument("--verify", action="store_true",
                        help="Verify pipeline with random weights")
    parser.add_argument("--model", type=str, default="1.5B",
                        choices=["0.5B", "1.5B"],
                        help="Model size")
    parser.add_argument("--prompt", type=str,
                        default="The future of AI is",
                        help="Prompt for text generation")
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--weights-dir", type=str, default=None)
    parser.add_argument("--quantize", action="store_true",
                        help="Quantize weights to INT4 and save")
    parser.add_argument("--decode-mode", type=str, default="gpu",
                        choices=["cpu", "gpu"],
                        help="Decode mode: cpu or gpu (fast decode)")
    parser.add_argument("--profile", action="store_true",
                        help="Enable profiling")
    add_device_arg(parser)
    args = parser.parse_args()
    apply_device_arg(args)

    if args.verify:
        success = verify_with_random_weights()
        sys.exit(0 if success else 1)

    config = QWEN_CONFIGS[args.model]
    weights_dir = args.weights_dir or os.path.join(_SCRIPT_DIR, "..", "..", "gitignore", "models", os.path.basename(_SCRIPT_DIR), "weights")
    npz_path, tokenizer_path = download_qwen_weights(
        args.model, weights_dir)
    q4_path = os.path.join(weights_dir, "weights_q4.npz")

    if args.quantize:
        print(f"Quantizing Qwen2.5-{args.model} weights...")
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
        quantized = quantize_qwen_weights(weights, n_layer)
        print(f"Saving to {q4_path}...")
        np.savez(q4_path, **quantized)
        print(f"Done! File: {os.path.getsize(q4_path) / 1024**2:.0f} MB")
        return

    # Load weights (prefer quantized)
    quantized = False
    if os.path.exists(q4_path):
        print(f"Loading quantized weights from {q4_path}")
        data = np.load(q4_path, mmap_mode='r')
        weights = {k: data[k] for k in data.files}
        quantized = True
    else:
        weights = load_weights(npz_path)
    print(f"Loaded {len(weights)} weight tensors"
          f"{' (INT4 quantized)' if quantized else ''}")

    tokenizer = load_tokenizer(tokenizer_path)

    model = QwenWebGPU(
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
        attention_bias=config.get("attention_bias", True),
        tie_word_embeddings=config.get("tie_word_embeddings", True),
        quantized=quantized,
        decode_mode=args.decode_mode)
    print(f"Model created, kernels compiled (decode={args.decode_mode})")

    if args.profile:
        model.enable_profiling()
        print(f"Profiling enabled (GPU timestamps: {model.profiler.gpu_enabled})")

    # Warmup fast decode before generating
    model._warmup_fast_decode()

    generate(model, args.prompt, tokenizer,
             max_tokens=args.max_tokens,
             temperature=args.temperature)

    if args.profile:
        model.profiler.report()
        from common.profiler_html import generate_html_report
        profile_path = os.path.join(_SCRIPT_DIR, "..", "..", "gitignore", "models", os.path.basename(_SCRIPT_DIR), "profile.html")
        runner = model.cache.runner
        generate_html_report(
            model.profiler,
            output_path=profile_path,
            title=f"Qwen2.5-{args.model} — {runner.adapter_info.get('description', 'GPU')}",
            adapter_info=runner.adapter_info,
            memory_info=model.get_memory_info())
        print(f"\nHTML profile: {profile_path}")


if __name__ == "__main__":
    main()
