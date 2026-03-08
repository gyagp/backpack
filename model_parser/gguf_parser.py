"""
GGUF file parser — reads metadata, config, and tokenizer from GGUF files.

This is the shared parsing layer used by both compiler and runtime stages.
For GPU weight loading (Q8 repacking etc.), see runtimes/python/common/gguf_utils.py.
"""

import struct
import mmap
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

# Re-export the low-level GGUF reader from the runtime common
# (it already has the full mmap-based parser with metadata support)
import sys
import os
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_root, 'runtimes', 'python'))
from common.gguf_utils import (
    GGUFFile, GGUFTensorInfo, dequantize_q8_0, repack_q8_0_for_gpu,
    GGUF_MAGIC, GGML_TYPE_NAMES, GGML_BLOCK_SIZES,
)


def extract_gguf_config(gf: GGUFFile) -> dict:
    """Extract model config from GGUF metadata.

    Returns a dict with standardized keys matching the C++ runtime's
    extractModelConfig().
    """
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
    head_dim = rope_dim if rope_dim > 0 else n_embd // n_head if n_head else 0

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
        "model_format": "gguf",
    }


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
        if len(tok) >= 2 and tok[0] == '<' and tok[-1] == '>':
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


def _build_gpt2_byte_tables():
    """GPT-2 byte-level BPE: map each byte 0-255 to a printable character."""
    b2u = {}
    n = 0
    for b in range(256):
        if (33 <= b <= 126) or (161 <= b <= 172) or (174 <= b <= 255):
            b2u[b] = chr(b)
        else:
            b2u[b] = chr(256 + n)
            n += 1
    u2b = {v: k for k, v in b2u.items()}
    return b2u, u2b
