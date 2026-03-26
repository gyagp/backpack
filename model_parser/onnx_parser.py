"""
ONNX model parser — reads config and tokenizer from ONNX GenAI models.

This is the shared parsing layer used by both compiler and runtime stages.
For GPU weight loading, see runtimes/python/onnx_loader.py.
"""

import json
import os
from typing import Dict, List

import numpy as np

from model_parser.gguf_parser import _build_gpt2_byte_tables


def extract_onnx_config(model_dir: str) -> dict:
    """Extract model config from genai_config.json or config.json."""
    config_path = os.path.join(model_dir, "genai_config.json")
    use_genai = os.path.exists(config_path)
    if not use_genai:
        config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"No genai_config.json or config.json in {model_dir}")

    with open(config_path) as f:
        cfg = json.load(f)

    if use_genai:
        mc = cfg["model"]
        dec = mc["decoder"]
        n_head = dec["num_attention_heads"]
        n_kv_heads = dec.get("num_key_value_heads", n_head)
        head_dim = dec["head_size"]
        n_embd = dec["hidden_size"]
        n_layer = dec["num_hidden_layers"]
        n_vocab = mc["vocab_size"]
        eos = mc.get("eos_token_id", 0)
        arch = mc.get("type", "phi3")
        bos = mc.get("bos_token_id", 0)
    else:
        n_head = cfg.get("num_attention_heads", 32)
        n_kv_heads = cfg.get("num_key_value_heads", n_head)
        n_embd = cfg.get("hidden_size", 2048)
        head_dim = n_embd // n_head
        n_layer = cfg.get("num_hidden_layers", 24)
        n_vocab = cfg.get("vocab_size", 32000)
        eos = cfg.get("eos_token_id", 0)
        arch = cfg.get("model_type", "llama")
        bos = cfg.get("bos_token_id", 0)

    if isinstance(eos, list):
        eos = eos[0]

    return {
        "arch": arch,
        "n_layer": n_layer,
        "n_head": n_head,
        "n_kv_heads": n_kv_heads,
        "n_embd": n_embd,
        "intermediate_size": 0,
        "n_vocab": n_vocab,
        "head_dim": head_dim,
        "rope_theta": 10000.0,
        "rms_norm_eps": 1e-5,
        "tie_word_embeddings": True,
        "has_qk_norm": False,
        "eos_token_id": eos,
        "bos_token_id": bos,
        "model_format": "onnx",
    }


class OnnxTokenizer:
    """HuggingFace tokenizer.json loader (BPE)."""

    def __init__(self, model_dir: str):
        tok_path = os.path.join(model_dir, "tokenizer.json")
        with open(tok_path, encoding='utf-8') as f:
            tok_data = json.load(f)

        vocab = tok_data["model"]["vocab"]
        self.vocab = [""] * len(vocab)
        self.token_to_id = {}
        for token, idx in vocab.items():
            if idx < len(self.vocab):
                self.vocab[idx] = token
                self.token_to_id[token] = idx

        merges = tok_data["model"].get("merges", [])
        self.merge_rank = {}
        for i, m in enumerate(merges):
            key = m if isinstance(m, str) else f"{m[0]} {m[1]}"
            self.merge_rank[key] = i

        config_path = os.path.join(model_dir, "genai_config.json")
        with open(config_path) as f:
            cfg = json.load(f)
        eos = cfg["model"].get("eos_token_id", 0)
        self.eos_token_id = eos[0] if isinstance(eos, list) else eos
        self.bos_token_id = cfg["model"].get("bos_token_id", 0)

        self._b2u, self._u2b = _build_gpt2_byte_tables()

    def encode(self, text: str) -> List[int]:
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
