#!/usr/bin/env python3
"""
Model-agnostic GGUF inference engine for WebGPU.

Reads model architecture, weights, and tokenizer directly from any
llama.cpp-compatible GGUF file. No model-specific Python code needed.

Supports:
  - All llama-family models (LLaMA, Qwen, Mistral, Gemma, Yi, Phi-3, ...)
  - Q8_0 quantization (Q4_K, Q6_K planned)
  - GPU fast decode with pre-recorded dispatches
  - BPE tokenizer from GGUF metadata
  - Hardware GPU profiling via --profile
  - Streaming text output

Usage:
    python -m runtimes.python.gguf_engine --model path/to/model.gguf \\
        --prompt "Hello" --max-tokens 50 [--profile]
"""

import argparse
import gc
import struct
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add runtimes/python/ and project root to path
import os
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(os.path.dirname(_here))
for p in [_here, _root]:
    if p not in sys.path:
        sys.path.insert(0, p)

from common.gguf_utils import (
    GGUFFile, dequantize_q8_0, repack_q8_0_for_gpu,
)
from common.model_base import WebGPUModel


# ─── Model config extraction from GGUF metadata ──────────────────────────────

def extract_config(gf: GGUFFile) -> dict:
    """Extract model config from GGUF metadata, like the C++ engine."""
    md = gf.metadata
    arch = md.get("general.architecture", "llama")
    a = arch

    n_layer = md.get(f"{a}.block_count", 0)
    n_embd = md.get(f"{a}.embedding_length", 0)
    intermediate_size = md.get(f"{a}.feed_forward_length", 0)
    n_head = md.get(f"{a}.attention.head_count", 0)
    n_kv_heads = md.get(f"{a}.attention.head_count_kv", n_head)
    rms_norm_eps = md.get(f"{a}.attention.layer_norm_rms_epsilon", 1e-6)
    rope_theta = md.get(f"{a}.rope.freq_base", 1000000.0)
    rope_dim = md.get(f"{a}.rope.dimension_count", 0)
    head_dim = rope_dim if rope_dim > 0 else n_embd // n_head

    # n_vocab from embedding tensor shape
    emb_info = gf.tensors.get("token_embd.weight")
    n_vocab = emb_info.shape[1] if emb_info and len(emb_info.shape) >= 2 else 0

    tie_word_embeddings = "output.weight" not in gf.tensors
    has_qk_norm = "blk.0.attn_q_norm.weight" in gf.tensors

    return {
        "arch": arch,
        "n_layer": n_layer,
        "n_head": n_head,
        "n_kv_heads": n_kv_heads,
        "n_embd": n_embd,
        "intermediate_size": intermediate_size,
        "n_vocab": n_vocab,
        "head_dim": head_dim,
        "rope_theta": float(rope_theta),
        "rms_norm_eps": float(rms_norm_eps),
        "tie_word_embeddings": tie_word_embeddings,
        "has_qk_norm": has_qk_norm,
    }


# ─── Weight loading from GGUF (model-agnostic) ───────────────────────────────

def load_gguf_weights(gf: GGUFFile, cfg: dict) -> Dict[str, np.ndarray]:
    """Load weights from any llama-family GGUF in Q8_0 GPU-packed format."""
    n_layer = cfg["n_layer"]
    out = {}

    def _load_q8(backpack_name: str, gguf_name: str):
        info = gf.tensors[gguf_name]
        if info.qtype == 8:  # Q8_0
            raw = gf.tensor_data(gguf_name)
            w_u32, s_fp16 = repack_q8_0_for_gpu(raw, info.shape)
            out[backpack_name + ".q8"] = w_u32
            out[backpack_name + ".q8_scales"] = s_fp16
        elif info.qtype in (0, 1):  # F32/F16
            if info.qtype == 0:
                arr = gf.tensor_data_f32(gguf_name)
            else:
                arr = np.array(gf.tensor_data_f16(gguf_name), dtype=np.float16)
            if len(info.shape) == 2:
                arr = arr.reshape(info.shape[1], info.shape[0])
            out[backpack_name] = arr.astype(np.float16, copy=True)
        else:
            raise ValueError(
                f"Unsupported quant type {info.qtype_name} for {gguf_name}")

    def _load_scalar(backpack_name: str, gguf_name: str,
                     out_dtype=np.float32):
        if gguf_name not in gf.tensors:
            return
        info = gf.tensors[gguf_name]
        if info.qtype == 0:
            arr = gf.tensor_data_f32(gguf_name)
            out[backpack_name] = arr.astype(out_dtype, copy=True)
        elif info.qtype == 1:
            arr = gf.tensor_data_f16(gguf_name)
            out[backpack_name] = np.array(arr, dtype=out_dtype)
        elif info.qtype == 8:
            raw = gf.tensor_data(gguf_name)
            out[backpack_name] = dequantize_q8_0(
                raw, info.shape, out_dtype=out_dtype)
        else:
            raise ValueError(
                f"Unsupported type {info.qtype_name} for {gguf_name}")

    # Embedding
    emb_info = gf.tensors["token_embd.weight"]
    if emb_info.qtype == 8:
        raw = gf.tensor_data("token_embd.weight")
        out["embed_tokens.weight"] = dequantize_q8_0(
            raw, emb_info.shape, out_dtype=np.float16)
    else:
        _load_scalar("embed_tokens.weight", "token_embd.weight",
                     out_dtype=np.float16)

    # Final norm
    _load_scalar("norm.weight", "output_norm.weight")

    # LM head (if not tied)
    if "output.weight" in gf.tensors:
        _load_q8("lm_head.weight", "output.weight")

    # Per-layer weights
    for i in range(n_layer):
        src = f"blk.{i}."
        dst = f"layers.{i}."

        _load_scalar(dst + "input_layernorm.weight",
                     src + "attn_norm.weight")
        _load_scalar(dst + "post_attention_layernorm.weight",
                     src + "ffn_norm.weight")

        # QK norm (Qwen3)
        _load_scalar(dst + "self_attn.q_norm.weight",
                     src + "attn_q_norm.weight")
        _load_scalar(dst + "self_attn.k_norm.weight",
                     src + "attn_k_norm.weight")

        # Q/K/V → load individually then fuse
        _load_q8(dst + "self_attn.q_proj.weight", src + "attn_q.weight")
        _load_q8(dst + "self_attn.k_proj.weight", src + "attn_k.weight")
        _load_q8(dst + "self_attn.v_proj.weight", src + "attn_v.weight")
        _load_q8(dst + "self_attn.o_proj.weight", src + "attn_output.weight")

        # Fuse QKV
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

        # Gate + Up → load individually then fuse
        if src + "ffn_gate.weight" in gf.tensors:
            _load_q8(dst + "mlp.gate_proj.weight", src + "ffn_gate.weight")
            _load_q8(dst + "mlp.up_proj.weight", src + "ffn_up.weight")
            gu_name = dst + "mlp.gate_up_proj.weight"
            g_q8 = out.pop(dst + "mlp.gate_proj.weight.q8")
            u_q8 = out.pop(dst + "mlp.up_proj.weight.q8")
            out[gu_name + ".q8"] = np.concatenate([g_q8, u_q8], axis=0)
            g_sc = out.pop(dst + "mlp.gate_proj.weight.q8_scales")
            u_sc = out.pop(dst + "mlp.up_proj.weight.q8_scales")
            out[gu_name + ".q8_scales"] = np.concatenate(
                [g_sc, u_sc], axis=0)
        else:
            # Non-gated MLP (GPT-2 style): just up
            _load_q8(dst + "mlp.up_proj.weight", src + "ffn_up.weight")

        _load_q8(dst + "mlp.down_proj.weight", src + "ffn_down.weight")

        if (i + 1) % 7 == 0 or i == n_layer - 1:
            print(f"  loaded layer {i+1}/{n_layer}")
        gc.collect()

    return out


# ─── BPE Tokenizer from GGUF metadata ────────────────────────────────────────

def _build_gpt2_byte_tables():
    """GPT-2 byte-level BPE: map each byte 0-255 to a printable character."""
    byte_to_unicode = {}
    n = 0
    for b in range(256):
        if (33 <= b <= 126) or (161 <= b <= 172) or (174 <= b <= 255):
            byte_to_unicode[b] = chr(b)
        else:
            byte_to_unicode[b] = chr(256 + n)
            n += 1
    unicode_to_byte = {v: k for k, v in byte_to_unicode.items()}
    return byte_to_unicode, unicode_to_byte


class GGUFTokenizer:
    """BPE tokenizer loaded from GGUF metadata."""

    def __init__(self, gf: GGUFFile):
        md = gf.metadata
        self.vocab: List[str] = md.get("tokenizer.ggml.tokens", [])
        self.token_to_id = {t: i for i, t in enumerate(self.vocab)}

        merges = md.get("tokenizer.ggml.merges", [])
        self.merge_rank = {m: i for i, m in enumerate(merges)}

        self.eos_token_id = md.get("tokenizer.ggml.eos_token_id", 151645)
        self.bos_token_id = md.get("tokenizer.ggml.bos_token_id", 151643)

        self._b2u, self._u2b = _build_gpt2_byte_tables()
        print(f"  Tokenizer: {len(self.vocab)} tokens, "
              f"{len(self.merge_rank)} merges, EOS={self.eos_token_id}")

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs via BPE."""
        # Convert bytes to GPT-2 unicode representation
        bpe_chars = [self._b2u[b] for b in text.encode('utf-8')]

        # Iteratively merge best pair
        pieces = list(bpe_chars)
        while len(pieces) > 1:
            best_idx, best_rank = -1, float('inf')
            for i in range(len(pieces) - 1):
                pair = pieces[i] + " " + pieces[i + 1]
                rank = self.merge_rank.get(pair, float('inf'))
                if rank < best_rank:
                    best_rank = rank
                    best_idx = i
            if best_idx < 0:
                break
            pieces[best_idx] = pieces[best_idx] + pieces[best_idx + 1]
            del pieces[best_idx + 1]

        # Map pieces to token IDs
        ids = []
        for p in pieces:
            tid = self.token_to_id.get(p)
            if tid is not None:
                ids.append(tid)
            else:
                # Unknown — encode each byte individually
                for ch in p:
                    single_id = self.token_to_id.get(ch)
                    if single_id is not None:
                        ids.append(single_id)
        return ids

    def decode_token(self, token_id: int) -> str:
        """Decode a single token ID to text."""
        if token_id < 0 or token_id >= len(self.vocab):
            return ""
        tok = self.vocab[token_id]
        # Special tokens
        if len(tok) >= 2 and tok[0] == '<' and tok[-1] == '>':
            return tok
        # Convert GPT-2 unicode chars back to bytes
        result = bytearray()
        for ch in tok:
            if ch in self._u2b:
                result.append(self._u2b[ch])
            else:
                result.extend(ch.encode('utf-8'))
        return result.decode('utf-8', errors='replace')

    def decode(self, token_ids: List[int]) -> str:
        return "".join(self.decode_token(t) for t in token_ids)


# ─── Model-agnostic GGUF Model ───────────────────────────────────────────────

class GGUFModel(WebGPUModel):
    """Model-agnostic WebGPU inference from any llama-family GGUF.

    Like the C++ engine, reads architecture from GGUF metadata and
    builds the decode pipeline at runtime. No model-specific code.
    """

    MAX_SEQ_LEN = 2048

    def __init__(self, weights: Dict[str, np.ndarray], cfg: dict):
        self.tie_word_embeddings = cfg["tie_word_embeddings"]
        self._has_qk_norm = cfg["has_qk_norm"]
        self._decode_mode = 'gpu'

        # Detect quantization mode from available weights
        has_q8 = any(k.endswith('.q8') for k in weights)
        self._q8_mode = has_q8

        q_dim = cfg["n_head"] * cfg["head_dim"]
        kv_dim = cfg["n_kv_heads"] * cfg["head_dim"]
        self.q_dim = q_dim
        self.kv_dim = kv_dim
        self.qkv_out = q_dim + 2 * kv_dim
        self.gate_up_out = 2 * cfg["intermediate_size"]

        super().__init__(
            weights,
            n_layer=cfg["n_layer"],
            n_head=cfg["n_head"],
            n_embd=cfg["n_embd"],
            n_vocab=cfg["n_vocab"],
            n_kv_heads=cfg["n_kv_heads"],
            intermediate_size=cfg["intermediate_size"],
            head_dim=cfg["head_dim"],
            rope_theta=cfg["rope_theta"],
            rms_norm_eps=cfg["rms_norm_eps"],
            k_dimensions={cfg["n_embd"], cfg["intermediate_size"], q_dim},
        )

        self._precompute_rope_tables()
        self._upload_weights_to_gpu()
        self._init_gpu_kv_cache()

    # ─── GPU KV cache ─────────────────────────────────────────────────────

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
              f"({self.n_layer} layers x {self.MAX_SEQ_LEN} seq)")

    # ─── RoPE ─────────────────────────────────────────────────────────────

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

    # ─── Projection helper ────────────────────────────────────────────────

    def _proj(self, x, name, bias_key, N, K=None, gpu_out=False,
              prefer_subgroup_matrix: bool = False):
        """Linear projection using best available kernel (Q8 or fp16)."""
        q8_key = name + ".q8.gpu"
        if q8_key in self._gpu_weights:
            return self._linear_q8(
                x, self._gpu_weights[q8_key],
                self._gpu_weights[name + ".q8_scales.gpu"],
                self._gpu_weights[bias_key], N, K=K, gpu_out=gpu_out,
                prefer_subgroup_matrix=prefer_subgroup_matrix)
        fp16_key = name + ".fp16"
        if fp16_key in self._gpu_weights:
            return self._linear_fp16w(
                x, self._gpu_weights[fp16_key],
                self._gpu_weights[bias_key], N, K=K, gpu_out=gpu_out)
        raise KeyError(f"No GPU weights for {name}")

    # ─── Compile model-specific kernels ───────────────────────────────────

    def _compile_model_kernels(self):
        """Compile RMSNorm + SiLU*mul kernels."""
        self._compile_rms_norm()
        self._compile_silu_mul()

    # ─── Weight upload (same as Qwen3 but without model-specific names) ───

    def _upload_weights_to_gpu(self):
        """Upload all weights to GPU in best available format."""
        runner = self.cache.runner
        self._use_q8_gpu = True

        # Embedding as fp16
        fp16_key = "embed_tokens.weight.fp16"
        emb = self.weights["embed_tokens.weight"]
        if emb.dtype != np.float16:
            emb = emb.astype(np.float16)
        self._gpu_weights[fp16_key] = runner.upload_to_gpu(
            emb.ravel(), fp16_key)
        self._gpu_weights[fp16_key].shape = emb.shape

        # Final norm
        nw = self.weights["norm.weight"]
        if nw.dtype != np.float32:
            nw = nw.astype(np.float32)
        self._gpu_weights["norm.weight"] = runner.upload_to_gpu(
            nw.ravel(), "norm.weight")

        # Zero biases
        max_bias_n = max(self.n_embd, self.qkv_out,
                         self.gate_up_out, self.n_vocab)
        zero = np.zeros(max_bias_n, dtype=np.float32)
        for name, sz in [("zero_bias_E", self.n_embd),
                         ("zero_bias_QKV", self.qkv_out),
                         ("zero_bias_GU", self.gate_up_out),
                         ("zero_bias_V", self.n_vocab)]:
            self._gpu_weights[name] = runner.upload_to_gpu(
                zero[:sz], name)

        # Per-layer weights
        for i in range(self.n_layer):
            dst = f"layers.{i}."

            # Linear projection weights (Q8 or fp16)
            for proj_name in ["self_attn.qkv_proj.weight",
                              "self_attn.o_proj.weight",
                              "mlp.gate_up_proj.weight",
                              "mlp.down_proj.weight"]:
                full = dst + proj_name
                q8_key = full + ".q8"
                if q8_key in self.weights:
                    q8 = self.weights[q8_key]
                    N = q8.shape[0]
                    K = q8.shape[1] * 4
                    self._upload_q8_weight(full, N, K)
                elif full in self.weights:
                    # fp16 weight (from dequantized ONNX Q4)
                    w = self.weights[full]
                    if w.dtype != np.float16:
                        w = w.astype(np.float16)
                    fp16_name = full + ".fp16"
                    self._gpu_weights[fp16_name] = runner.upload_to_gpu(
                        w.ravel(), fp16_name)
                    self._gpu_weights[fp16_name].shape = w.shape

            # Norm weights (fp32)
            for norm_name in ["input_layernorm.weight",
                              "post_attention_layernorm.weight"]:
                full = dst + norm_name
                if full in self.weights:
                    w = self.weights[full]
                    if w.dtype != np.float32:
                        w = w.astype(np.float32)
                    self._gpu_weights[full] = runner.upload_to_gpu(
                        w.ravel(), full)

            # QK norm weights (optional)
            for qk_name in ["self_attn.q_norm.weight",
                            "self_attn.k_norm.weight"]:
                full = dst + qk_name
                if full in self.weights:
                    w = self.weights[full].astype(np.float32)
                    self._gpu_weights[full] = runner.upload_to_gpu(
                        w.ravel(), full)

        print(f"  Uploaded {len(self._gpu_weights)} weight tensors to GPU")

    # ─── RoPE tables ──────────────────────────────────────────────────────

    def _precompute_rope_tables(self):
        """Pre-compute cos/sin RoPE tables."""
        HD = self.head_dim
        half = HD // 2
        positions = np.arange(self.MAX_SEQ_LEN, dtype=np.float32)
        freqs = 1.0 / (self.rope_theta ** (
            np.arange(0, HD, 2, dtype=np.float32) / HD))
        angles = np.outer(positions, freqs)
        self._rope_cos = np.cos(angles).astype(np.float32)
        self._rope_sin = np.sin(angles).astype(np.float32)

        runner = self.cache.runner
        self._gpu_rope_cos = runner.upload_to_gpu(
            self._rope_cos.ravel(), "rope_cos")
        self._gpu_rope_sin = runner.upload_to_gpu(
            self._rope_sin.ravel(), "rope_sin")

    # ─── Transformer block (model-agnostic) ─────────────────────────────

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

    def _attention_block(self, x, layer: int,
                         use_cache: bool = False,
                         positions: np.ndarray = None, **kwargs):
        """GQA with fused QKV and RoPE."""
        from common.model_base import GPUBuffer
        T = (x.shape[0] if x.shape else 1) if isinstance(x, GPUBuffer) else x.shape[0]
        E, HD = self.n_embd, self.head_dim
        n_head, n_kv, n_rep = self.n_head, self.n_kv_heads, self.n_rep
        pfx = f"layers.{layer}.self_attn."
        q_dim = n_head * HD

        qkv = self._proj(x, pfx + "qkv_proj.weight", "zero_bias_QKV",
                         self.qkv_out, K=E)
        q = qkv[:, :q_dim]
        k = qkv[:, q_dim:q_dim + self.kv_dim]
        v = qkv[:, q_dim + self.kv_dim:]

        Q = q.reshape(T, n_head, HD)
        K_new = k.reshape(T, n_kv, HD)
        V_new = v.reshape(T, n_kv, HD)

        # QK-norm (optional)
        q_norm_w = self.weights.get(pfx + "q_norm.weight")
        k_norm_w = self.weights.get(pfx + "k_norm.weight")
        if q_norm_w is not None:
            q_rms = np.sqrt(np.mean(Q * Q, axis=-1, keepdims=True)
                            + self.rms_norm_eps)
            Q = Q / q_rms * q_norm_w.astype(np.float32)
            k_rms = np.sqrt(np.mean(K_new * K_new, axis=-1, keepdims=True)
                            + self.rms_norm_eps)
            K_new = K_new / k_rms * k_norm_w.astype(np.float32)

        # RoPE
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

    def _mlp_block(self, x, layer: int, gpu_out: bool = False):
        """SwiGLU MLP with fused gate+up projection."""
        E, IM = self.n_embd, self.intermediate_size
        pfx = f"layers.{layer}.mlp."
        gate_up = self._proj(x, pfx + "gate_up_proj.weight",
                             "zero_bias_GU", self.gate_up_out, K=E,
                             gpu_out=True)
        h = self._silu_mul_fused(gate_up, IM, gpu_out=True)
        return self._proj(h, pfx + "down_proj.weight",
                          "zero_bias_E", E, K=IM, gpu_out=gpu_out)

    # ─── Forward pass ─────────────────────────────────────────────────────

    def forward(self, token_ids: np.ndarray,
                use_cache: bool = False,
                pos_offset: int = 0) -> np.ndarray:
        """Run forward pass. Uses fast GPU decode for T=1 with cache."""
        T = len(token_ids)

        if (T == 1 and use_cache and self._decode_mode == 'gpu'
                and self._use_q8_gpu
                and getattr(self, '_fast_decode_ready', False)):
            return self._decode_fast(token_ids, pos_offset)

        # Standard prefill path
        wte = self.weights["embed_tokens.weight"]
        x = wte[token_ids].astype(np.float32)
        positions = np.arange(pos_offset, pos_offset + T, dtype=np.int32)

        for layer in range(self.n_layer):
            x = self._transformer_block(x, layer, use_cache=use_cache,
                                        positions=positions)

        # Sync KV cache to GPU after prefill
        if use_cache and hasattr(self, '_gpu_kv_cache') and self.kv_cache:
            runner = self.cache.runner
            HD = self.head_dim
            n_kv = self.n_kv_heads
            for li in range(self.n_layer):
                if li not in self.kv_cache:
                    continue
                K_gpu, V_gpu, cached = self._gpu_kv_cache[li]
                if cached > 0:
                    continue
                K_cpu, V_cpu = self.kv_cache[li]
                T_cached = K_cpu.shape[0]
                runner.write_buffer(K_gpu.handle,
                                    K_cpu.ravel().astype(np.float32).tobytes())
                runner.write_buffer(V_gpu.handle,
                                    V_cpu.ravel().astype(np.float32).tobytes())
                self._gpu_kv_cache[li] = (K_gpu, V_gpu, T_cached)

        x = self._rms_norm(x, self._gpu_weights["norm.weight"])

        lm_key = ("embed_tokens.weight" if self.tie_word_embeddings
                   else "lm_head.weight")
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


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Model-agnostic GGUF inference engine (WebGPU)")
    parser.add_argument("--model", required=True,
                        help="Path to GGUF model file")
    parser.add_argument("--prompt", default="Hello",
                        help="Input prompt text")
    parser.add_argument("--max-tokens", type=int, default=50,
                        help="Maximum tokens to generate")
    parser.add_argument("--profile", action="store_true",
                        help="Enable GPU hardware profiling")
    parser.add_argument("--backend", default=None,
                        help="WebGPU backend (vulkan/d3d12)")
    args = parser.parse_args()

    if args.backend:
        os.environ["DAWN_BACKEND"] = args.backend

    t0 = time.time()

    # 1. Resolve model path (file or directory)
    model_path = args.model
    if os.path.isdir(model_path):
        # Find GGUF inside directory
        gguf_files = [os.path.join(r, f)
                      for r, _, files in os.walk(model_path)
                      for f in files if f.endswith('.gguf')]
        if not gguf_files:
            print(f"No GGUF file found in: {model_path}")
            sys.exit(1)
        # Prefer Q8_0
        model_path = next((g for g in gguf_files if 'Q8_0' in g), gguf_files[0])

    print(f"Loading GGUF: {model_path}")
    gf = GGUFFile(model_path)
    cfg = extract_config(gf)
    print(f"Model: {cfg['arch']} ({cfg['n_layer']} layers, "
          f"E={cfg['n_embd']}, HD={cfg['head_dim']}, "
          f"V={cfg['n_vocab']}, KV={cfg['n_kv_heads']})")

    # 2. Load tokenizer
    tokenizer = GGUFTokenizer(gf)

    # 3. Load weights
    weights = load_gguf_weights(gf, cfg)

    # 4. Create model
    model = GGUFModel(weights, cfg)
    t1 = time.time()
    print(f"Model loaded in {(t1-t0)*1000:.0f}ms\n")

    # 5. Enable profiling
    if args.profile:
        model.enable_profiling()
        print("GPU profiling enabled")

    # 6. Tokenize prompt
    prompt_tokens = tokenizer.encode(args.prompt)
    print(f'Prompt: "{args.prompt}"')
    print(f"Tokens ({len(prompt_tokens)}): {prompt_tokens}")

    # 7. Prefill
    print("Prefilling...")
    t_prefill = time.time()
    logits = model.forward(np.array(prompt_tokens),
                           use_cache=True, pos_offset=0)
    t_prefill_done = time.time()
    prefill_ms = (t_prefill_done - t_prefill) * 1000

    # Use last position's logits for next token prediction
    last_logits = logits[-1] if logits.ndim > 1 else logits.ravel()
    first_token = int(np.argmax(last_logits))
    print(f"Prefill: {prefill_ms:.0f}ms")

    # 8. Decode with streaming output
    print(f"\n--- Output ---")
    print(args.prompt, end="", flush=True)

    t_decode = time.time()
    generated = []
    next_token = first_token

    for step in range(args.max_tokens):
        if next_token == tokenizer.eos_token_id:
            break

        generated.append(next_token)
        text = tokenizer.decode_token(next_token)
        print(text, end="", flush=True)

        pos = len(prompt_tokens) + step
        logits = model.forward(np.array([next_token]),
                               use_cache=True, pos_offset=pos)
        last_logits = logits[-1] if logits.ndim > 1 else logits.ravel()
        next_token = int(np.argmax(last_logits))

    t_decode_done = time.time()
    decode_ms = (t_decode_done - t_decode) * 1000
    n_decode = len(generated)

    print(f"\n\n--- Performance ---")
    print(f"  Prefill: {prefill_ms:.0f}ms ({len(prompt_tokens)} tokens)")
    tps = n_decode * 1000.0 / decode_ms if decode_ms > 0 else 0
    print(f"  Decode:  {n_decode} tokens in {decode_ms:.0f}ms "
          f"({tps:.1f} tok/s)")

    # 9. Print profile report
    if args.profile and model.profiler:
        model.profiler.finish()
        model.profiler.report()


if __name__ == "__main__":
    main()
