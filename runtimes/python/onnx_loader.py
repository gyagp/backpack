#!/usr/bin/env python3
"""
ONNX model loader for the Python runtime.

Loads ONNX GenAI models (like phi-4-mini) directly for inference.
Extracts model config from genai_config.json, tokenizer from
tokenizer.json, and Q4 weights from model.onnx + model.onnx.data.

Supports:
  - MatMulNBits (4-bit per-group quantized matmul)
  - GatherBlockQuantized (quantized embedding)
  - GroupQueryAttention (fused GQA + RoPE)
  - SkipSimplifiedLayerNormalization (fused residual + RMSNorm)
"""

import json
import os
import struct
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np


def extract_onnx_config(model_dir: str) -> dict:
    """Extract model config from genai_config.json."""
    config_path = os.path.join(model_dir, "genai_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"No genai_config.json in {model_dir}")

    with open(config_path) as f:
        cfg = json.load(f)

    mc = cfg["model"]
    dec = mc["decoder"]

    n_head = dec["num_attention_heads"]
    n_kv_heads = dec.get("num_key_value_heads", n_head)
    head_dim = dec["head_size"]
    n_embd = dec["hidden_size"]
    n_layer = dec["num_hidden_layers"]
    n_vocab = mc["vocab_size"]

    # EOS can be int or list
    eos = mc.get("eos_token_id", 0)
    if isinstance(eos, list):
        eos = eos[0]

    return {
        "arch": mc.get("type", "phi3"),
        "n_layer": n_layer,
        "n_head": n_head,
        "n_kv_heads": n_kv_heads,
        "n_embd": n_embd,
        "intermediate_size": 0,  # inferred from weights
        "n_vocab": n_vocab,
        "head_dim": head_dim,
        "rope_theta": 10000.0,  # default, may override from model
        "rms_norm_eps": 1e-5,
        "tie_word_embeddings": True,
        "has_qk_norm": False,
        "eos_token_id": eos,
        "bos_token_id": mc.get("bos_token_id", 0),
        "model_format": "onnx",
    }


def load_onnx_tokenizer(model_dir: str) -> 'OnnxTokenizer':
    """Load tokenizer from tokenizer.json (HuggingFace format)."""
    return OnnxTokenizer(model_dir)


class OnnxTokenizer:
    """HuggingFace tokenizer.json loader (BPE)."""

    def __init__(self, model_dir: str):
        tok_path = os.path.join(model_dir, "tokenizer.json")
        with open(tok_path, encoding='utf-8') as f:
            tok_data = json.load(f)

        # Extract vocab
        vocab = tok_data["model"]["vocab"]
        self.vocab = [""] * len(vocab)
        self.token_to_id = {}
        for token, idx in vocab.items():
            if idx < len(self.vocab):
                self.vocab[idx] = token
                self.token_to_id[token] = idx

        # Extract merges (may be strings "a b" or lists [a, b])
        merges = tok_data["model"].get("merges", [])
        self.merge_rank = {}
        for i, m in enumerate(merges):
            key = m if isinstance(m, str) else f"{m[0]} {m[1]}"
            self.merge_rank[key] = i

        # Special tokens
        config_path = os.path.join(model_dir, "genai_config.json")
        with open(config_path) as f:
            cfg = json.load(f)
        eos = cfg["model"].get("eos_token_id", 0)
        self.eos_token_id = eos[0] if isinstance(eos, list) else eos
        self.bos_token_id = cfg["model"].get("bos_token_id", 0)

        # GPT-2 byte encoding
        self._b2u = {}
        self._u2b = {}
        n = 0
        for b in range(256):
            if (33 <= b <= 126) or (161 <= b <= 172) or (174 <= b <= 255):
                self._b2u[b] = chr(b)
            else:
                self._b2u[b] = chr(256 + n)
                n += 1
        self._u2b = {v: k for k, v in self._b2u.items()}

        print(f"  Tokenizer: {len(self.vocab)} tokens, "
              f"{len(self.merge_rank)} merges, EOS={self.eos_token_id}")

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs via BPE."""
        bpe_chars = [self._b2u[b] for b in text.encode('utf-8')]
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

        ids = []
        for p in pieces:
            tid = self.token_to_id.get(p)
            if tid is not None:
                ids.append(tid)
            else:
                for ch in p:
                    sid = self.token_to_id.get(ch)
                    if sid is not None:
                        ids.append(sid)
        return ids

    def decode_token(self, token_id: int) -> str:
        if token_id < 0 or token_id >= len(self.vocab):
            return ""
        tok = self.vocab[token_id]
        if tok.startswith('<') and tok.endswith('>'):
            return tok
        result = bytearray()
        for ch in tok:
            if ch in self._u2b:
                result.append(self._u2b[ch])
            else:
                result.extend(ch.encode('utf-8'))
        return result.decode('utf-8', errors='replace')

    def decode(self, ids: List[int]) -> str:
        return "".join(self.decode_token(t) for t in ids)


def load_onnx_weights(model_dir: str, cfg: dict) -> Dict[str, np.ndarray]:
    """Load ONNX model weights for inference.

    Extracts Q4 quantized weights from ONNX and converts to our runtime format.
    Returns weights dict compatible with the GGUF engine's GGUFModel.
    """
    import onnx
    from onnx import numpy_helper

    model_path = os.path.join(model_dir, "model.onnx")
    print(f"  Loading ONNX: {model_path}")

    # Load model with external data
    model = onnx.load(model_path, load_external_data=True)
    g = model.graph

    n_layer = cfg["n_layer"]
    n_head = cfg["n_head"]
    n_kv_heads = cfg["n_kv_heads"]
    head_dim = cfg["head_dim"]
    n_embd = cfg["n_embd"]
    q_dim = n_head * head_dim
    kv_dim = n_kv_heads * head_dim
    qkv_out = q_dim + 2 * kv_dim

    out = {}

    # Build initializer lookup
    init_map = {init.name: init for init in g.initializer}

    # Extract per-layer weights from the graph
    for node in g.node:
        if node.op_type == "MatMulNBits":
            # Q4 quantized matmul
            out_name = node.output[0]
            w_name = node.input[1]
            s_name = node.input[2]

            K = next(a.i for a in node.attribute if a.name == "K")
            N = next(a.i for a in node.attribute if a.name == "N")
            bits = next(a.i for a in node.attribute if a.name == "bits")
            block_size = next(a.i for a in node.attribute if a.name == "block_size")

            # Convert weight tensor
            w_init = init_map[w_name]
            s_init = init_map[s_name]
            w_data = numpy_helper.to_array(w_init)  # [N, K/block_size, block_size/2]
            s_data = numpy_helper.to_array(s_init)  # [N, K/block_size] fp16

            # Determine the backpack weight name from ONNX naming
            bp_name = _onnx_name_to_backpack(w_name, n_embd, q_dim, kv_dim, qkv_out)
            if bp_name:
                out[bp_name + ".q4"] = w_data
                out[bp_name + ".q4_scales"] = s_data
                out[bp_name + ".q4_K"] = K
                out[bp_name + ".q4_N"] = N
                out[bp_name + ".q4_block_size"] = block_size

                # Infer intermediate_size from gate/up projections
                if "gate_up_proj" in bp_name or "gate_proj" in bp_name:
                    if cfg["intermediate_size"] == 0:
                        cfg["intermediate_size"] = N

        elif node.op_type == "SimplifiedLayerNormalization":
            w_name = node.input[1]
            if w_name in init_map:
                w_data = numpy_helper.to_array(init_map[w_name])
                bp_name = _onnx_norm_to_backpack(w_name)
                if bp_name:
                    out[bp_name] = w_data.astype(np.float32)

        elif node.op_type == "SkipSimplifiedLayerNormalization":
            w_name = node.input[2] if len(node.input) > 2 else None
            if w_name and w_name in init_map:
                w_data = numpy_helper.to_array(init_map[w_name])
                bp_name = _onnx_norm_to_backpack(w_name)
                if bp_name:
                    out[bp_name] = w_data.astype(np.float32)

        elif node.op_type == "GatherBlockQuantized":
            # Quantized embedding — find the actual weight from the input chain
            emb_input = node.input[0]
            scale_name = node.input[2] if len(node.input) > 2 else None

            # The embedding may come through a Reshape; trace back
            emb_init_name = emb_input
            for n2 in g.node:
                if n2.op_type == "Reshape" and n2.output[0] == emb_input:
                    emb_init_name = n2.input[0]
                    break

            if emb_init_name in init_map:
                emb_data = numpy_helper.to_array(init_map[emb_init_name])
                out["embed_tokens.weight._raw"] = emb_data
                block_size = next((a.i for a in node.attribute
                                   if a.name == "block_size"), 32)
                out["embed_tokens.weight._block_size"] = block_size
                if scale_name and scale_name in init_map:
                    scale_data = numpy_helper.to_array(init_map[scale_name])
                    out["embed_tokens.weight._scales"] = scale_data

    # Dequantize embedding for CPU lookup
    if "embed_tokens.weight._raw" in out:
        raw = out.pop("embed_tokens.weight._raw")     # [V, n_groups, block_size] uint8
        scales = out.pop("embed_tokens.weight._scales", None)  # [V, n_groups] fp16
        block_size = out.pop("embed_tokens.weight._block_size", 32)

        n_vocab_actual = raw.shape[0]
        if scales is not None:
            n_groups = raw.shape[1]
            # INT8 dequant: raw is unsigned uint8, interpret as int8
            raw_i8 = raw.view(np.int8).astype(np.float32)
            raw_flat = raw_i8.reshape(n_vocab_actual, -1)  # [V, K]
            K = raw_flat.shape[1]
            scales_f32 = scales.astype(np.float32)  # [V, n_groups]
            # Expand scales to match each block
            scales_exp = np.repeat(scales_f32, block_size, axis=1)[:, :K]
            emb = raw_flat * scales_exp
            out["embed_tokens.weight"] = emb.astype(np.float16)
        else:
            out["embed_tokens.weight"] = raw.astype(np.float16)

    # RoPE tables from ONNX initializers
    if "cos_cache" in init_map:
        cos = numpy_helper.to_array(init_map["cos_cache"]).astype(np.float32)
        sin = numpy_helper.to_array(init_map["sin_cache"]).astype(np.float32)
        out["_rope_cos"] = cos
        out["_rope_sin"] = sin

    # Also scan initializers directly for final norm
    for init in g.initializer:
        name = init.name
        if "final_norm" in name or name in ("model.norm.weight", "norm.weight"):
            if "norm.weight" not in out:
                out["norm.weight"] = numpy_helper.to_array(init).astype(np.float32)

    # Fuse gate + up projections per layer
    for i in range(n_layer):
        gate_key = f"layers.{i}.mlp.gate_proj.weight"
        up_key = f"layers.{i}.mlp.up_proj.weight"
        gu_key = f"layers.{i}.mlp.gate_up_proj.weight"

        if gate_key + ".q4" in out and up_key + ".q4" in out:
            g_q4 = out.pop(gate_key + ".q4")
            u_q4 = out.pop(up_key + ".q4")
            out[gu_key + ".q4"] = np.concatenate([g_q4, u_q4], axis=0)
            g_sc = out.pop(gate_key + ".q4_scales")
            u_sc = out.pop(up_key + ".q4_scales")
            out[gu_key + ".q4_scales"] = np.concatenate([g_sc, u_sc], axis=0)

            K = out.pop(gate_key + ".q4_K")
            N_gate = out.pop(gate_key + ".q4_N")
            N_up = out.pop(up_key + ".q4_N")
            out.pop(up_key + ".q4_K", None)
            out.pop(gate_key + ".q4_block_size", None)
            out.pop(up_key + ".q4_block_size", None)
            out[gu_key + ".q4_K"] = K
            out[gu_key + ".q4_N"] = N_gate + N_up
            out[gu_key + ".q4_block_size"] = 32

            if cfg["intermediate_size"] == 0:
                cfg["intermediate_size"] = N_gate

    # Dequantize Q4 weights to fp16 for the runtime
    # (Q4 GPU kernels can be added later for better perf)
    q4_keys = [k.replace(".q4", "") for k in list(out.keys()) if k.endswith(".q4")]
    for base_key in q4_keys:
        q4_data = out.pop(base_key + ".q4")       # [N, n_groups, block_size/2] uint8
        scales = out.pop(base_key + ".q4_scales")  # [N, n_groups] fp16
        K = out.pop(base_key + ".q4_K", None)
        N = out.pop(base_key + ".q4_N", None)
        out.pop(base_key + ".q4_block_size", None)

        # Unpack 4-bit: each byte has 2 nibbles (low nibble = first element)
        n_rows = q4_data.shape[0]
        n_groups = q4_data.shape[1]
        block_half = q4_data.shape[2]  # block_size / 2

        low = (q4_data & 0x0F).astype(np.int8) - 8   # [-8, 7]
        high = (q4_data >> 4).astype(np.int8) - 8
        # Interleave per-byte: byte[i] → [value[2i], value[2i+1]]
        unpacked = np.empty((n_rows, n_groups, block_half * 2), dtype=np.int8)
        unpacked[:, :, 0::2] = low
        unpacked[:, :, 1::2] = high

        # Dequantize
        scales_f32 = scales.astype(np.float32)[:, :, None]  # [N, n_groups, 1]
        dequant = unpacked.astype(np.float32) * scales_f32
        fp16 = dequant.reshape(n_rows, -1).astype(np.float16)  # [N, K]
        out[base_key] = fp16

    print(f"  Loaded {len(out)} weight tensors")
    return out


def _onnx_name_to_backpack(onnx_name: str, n_embd: int, q_dim: int,
                            kv_dim: int, qkv_out: int) -> Optional[str]:
    """Map ONNX weight names to backpack naming convention."""
    # model.layers.0.attn.qkv_proj.MatMul.weight_Q4G32 -> layers.0.self_attn.qkv_proj.weight
    parts = onnx_name.replace("MatMul.", "").replace(".weight_Q4G32", "").replace(".weight_scale", "")
    parts = parts.replace("model.", "")

    # Map ONNX names to our convention
    name_map = {
        "attn.qkv_proj": "self_attn.qkv_proj.weight",
        "attn.o_proj": "self_attn.o_proj.weight",
        "mlp.gate_up_proj": "mlp.gate_up_proj.weight",
        "mlp.gate_proj": "mlp.gate_proj.weight",
        "mlp.up_proj": "mlp.up_proj.weight",
        "mlp.down_proj": "mlp.down_proj.weight",
        "lm_head": "lm_head.weight",
    }

    for onnx_suffix, bp_suffix in name_map.items():
        if onnx_suffix in parts:
            # Extract layer number
            layer_match = None
            for p in parts.split('.'):
                if p.isdigit():
                    layer_match = p
                    break
            if layer_match is not None:
                return f"layers.{layer_match}.{bp_suffix}"
            if "lm_head" in onnx_suffix:
                return bp_suffix

    return None


def _onnx_norm_to_backpack(onnx_name: str) -> Optional[str]:
    """Map ONNX norm weight names to backpack convention."""
    name = onnx_name.replace("model.", "")
    if "input_layernorm" in name:
        parts = name.split('.')
        for i, p in enumerate(parts):
            if p.startswith("layers") and i + 1 < len(parts) and parts[i+1].isdigit():
                return f"layers.{parts[i+1]}.input_layernorm.weight"
    if "post_attention_layernorm" in name:
        parts = name.split('.')
        for i, p in enumerate(parts):
            if p.startswith("layers") and i + 1 < len(parts) and parts[i+1].isdigit():
                return f"layers.{parts[i+1]}.post_attention_layernorm.weight"
    if "norm.weight" in name and "layers" not in name:
        return "norm.weight"
    return None
