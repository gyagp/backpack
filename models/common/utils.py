"""
Shared utilities for WebGPU model inference.

Provides:
  - _parse_safetensors(): parse safetensors files into numpy arrays
  - load_weights(): load weights from npz files
  - download_weights(): generic HuggingFace weight downloader
  - load_tokenizer(): load tokenizer from tokenizer.json
  - generate(): unified text generation with prefill/decode timing
"""
import os
import sys
import time
import json
import struct
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Safetensors parsing
# ---------------------------------------------------------------------------

def _parse_safetensors(path: str) -> Dict[str, np.ndarray]:
    """Parse a safetensors file into a dict of numpy arrays."""
    DTYPE_MAP = {
        "F32": (np.float32, 4),
        "F16": (np.float16, 2),
        "BF16": (np.float16, 2),  # will be converted
        "I32": (np.int32, 4),
        "I64": (np.int64, 8),
        "U8": (np.uint8, 1),
    }

    with open(path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header_json = f.read(header_size).decode("utf-8")
        header = json.loads(header_json)
        data_start = 8 + header_size

        tensors = {}
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            dtype_str = meta["dtype"]
            shape = meta["shape"]
            offsets = meta["data_offsets"]
            start, end = offsets

            np_dtype, elem_size = DTYPE_MAP.get(dtype_str, (np.float32, 4))
            f.seek(data_start + start)
            raw = f.read(end - start)

            if dtype_str == "BF16":
                bf16 = np.frombuffer(raw, dtype=np.uint16)
                f32 = np.zeros(len(bf16), dtype=np.float32)
                f32_view = f32.view(np.uint32)
                f32_view[:] = bf16.astype(np.uint32) << 16
                tensors[name] = f32.reshape(shape)
            else:
                tensors[name] = np.frombuffer(raw, dtype=np_dtype).reshape(
                    shape).copy()

        return tensors


def load_weights(path: str) -> Dict[str, np.ndarray]:
    """Load weights from npz file (memory-mapped for fast startup)."""
    data = np.load(path, mmap_mode='r')
    return {k: data[k].astype(np.float32) for k in data.files}


def load_weights_mmap(path: str) -> Dict[str, np.ndarray]:
    """Load weights from npz via memory-mapping for fast startup.

    Memory-mapping avoids reading the entire file into RAM upfront.
    Instead, pages are loaded on-demand as weights are accessed (e.g.,
    during GPU upload).  This can reduce weight loading time from
    seconds to near-instant for large models (10–20 GB).

    The returned arrays are read-only memory-mapped views.  They must
    be copied before modification (e.g., quantization, dtype cast).
    GPU upload (wgpuQueueWriteBuffer) reads directly from the mapped
    pages, so the OS page cache provides the I/O.

    Usage:
        weights = load_weights_mmap("weights_q4.npz")
        # weights[k] is a read-only mmap view — fast to iterate
        model = Model(weights)  # GPU upload reads from mmap on demand

    Falls back to regular np.load if mmap is not supported (e.g.,
    compressed npz).
    """
    try:
        data = np.load(path, mmap_mode='r')
        return {k: data[k] for k in data.files}
    except ValueError:
        # Compressed npz doesn't support mmap — fall back to regular load
        data = np.load(path)
        return {k: data[k] for k in data.files}


# ---------------------------------------------------------------------------
# Weight downloading
# ---------------------------------------------------------------------------

def download_weights(hf_repo: str, model_dir: str,
                     safetensors_files: list = None,
                     key_transform=None,
                     download_tokenizer: bool = True) -> Tuple[str, Optional[str]]:
    """Download model weights from HuggingFace and convert to npz.

    Args:
        hf_repo: HuggingFace repo ID (e.g. "openai-community/gpt2")
        model_dir: local directory for caching files
        safetensors_files: list of safetensors filenames to download
                          (default: ["model.safetensors"])
        key_transform: function(key, arr) -> (new_key, new_arr) or None
                       to rename/transform weights. Return None to skip a key.
        download_tokenizer: whether to download tokenizer.json

    Returns:
        (npz_path, tokenizer_path) — tokenizer_path is None if not downloaded
    """
    import requests

    os.makedirs(model_dir, exist_ok=True)
    npz_path = os.path.join(model_dir, "weights.npz")
    tokenizer_path = os.path.join(model_dir, "tokenizer.json") \
        if download_tokenizer else None

    base_url = f"https://huggingface.co/{hf_repo}/resolve/main"

    # Build auth headers if HF token is available
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get(
        "HUGGING_FACE_HUB_TOKEN")
    if not hf_token:
        # Check cached token from huggingface-cli login
        token_file = os.path.join(
            os.path.expanduser("~"), ".cache", "huggingface", "token")
        if os.path.exists(token_file):
            with open(token_file) as f:
                hf_token = f.read().strip()
    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}

    # Download tokenizer
    if download_tokenizer:
        if os.path.exists(tokenizer_path):
            print(f"Tokenizer already cached at {tokenizer_path}")
        else:
            tok_url = f"{base_url}/tokenizer.json"
            print(f"Downloading tokenizer from {hf_repo}...")
            resp = requests.get(tok_url, headers=headers)
            resp.raise_for_status()
            with open(tokenizer_path, 'wb') as f:
                f.write(resp.content)
            print(f"  Saved tokenizer to {tokenizer_path}")

    if os.path.exists(npz_path):
        print(f"Weights already cached at {npz_path}")
        return npz_path, tokenizer_path

    if safetensors_files is None:
        safetensors_files = ["model.safetensors"]

    print(f"Downloading weights from {hf_repo}...")

    # Download safetensors files
    all_weights = {}
    for st_file in safetensors_files:
        st_path = os.path.join(model_dir, st_file)
        if os.path.exists(st_path):
            # Validate file size via HEAD request to detect partial downloads
            try:
                head_resp = requests.head(
                    f"{base_url}/{st_file}", headers=headers,
                    allow_redirects=True, timeout=10)
                expected_size = int(head_resp.headers.get('content-length', 0))
                actual_size = os.path.getsize(st_path)
                if expected_size and actual_size < expected_size:
                    print(f"  {st_file} incomplete ({actual_size // (1024*1024)}MB / "
                          f"{expected_size // (1024*1024)}MB), re-downloading...")
                    os.remove(st_path)
                else:
                    print(f"  {st_file} already cached")
            except Exception:
                print(f"  {st_file} already cached (size check skipped)")
        if not os.path.exists(st_path):
            st_url = f"{base_url}/{st_file}"
            print(f"  Downloading {st_file}...")
            resp = requests.get(st_url, headers=headers, stream=True)
            resp.raise_for_status()
            total = int(resp.headers.get('content-length', 0))
            downloaded = 0
            with open(st_path, 'wb') as f:
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

        print(f"  Parsing {st_file}...")
        weights = _parse_safetensors(st_path)
        all_weights.update(weights)

    # Apply key transformation
    renamed = {}
    for key, arr in all_weights.items():
        val = arr.astype(np.float32)
        if key_transform is not None:
            result = key_transform(key, val)
            if result is None:
                continue
            new_key, val = result
        else:
            new_key = key
        renamed[new_key] = val

    print(f"  Loaded {len(renamed)} tensors")
    np.savez(npz_path, **renamed)
    print(f"  Saved to {npz_path}")
    return npz_path, tokenizer_path


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def load_tokenizer(tokenizer_path: str):
    """Load tokenizer from tokenizer.json.

    Tries the `tokenizers` library first, then `transformers`.
    Returns an object with encode() and decode() methods.
    """
    try:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(tokenizer_path)

        class TokenizerWrapper:
            def __init__(self, tok):
                self._tok = tok

            def encode(self, text):
                return self._tok.encode(text).ids

            def decode(self, ids):
                return self._tok.decode(ids)

        return TokenizerWrapper(tok)
    except ImportError:
        pass

    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(
            os.path.dirname(tokenizer_path))
        return tok
    except ImportError:
        pass

    print("WARNING: No tokenizer library available. "
          "Install 'tokenizers' or 'transformers'.")
    return None


# ---------------------------------------------------------------------------
# Text generation
# ---------------------------------------------------------------------------

def generate(model, prompt: str, tokenizer=None,
             max_tokens: int = 50,
             temperature: float = 0.8, top_k: int = 40) -> str:
    """Generate text from a prompt using a WebGPU model.

    Uses KV-cache for incremental decoding with separate
    prefill/decode performance reporting.

    Args:
        model: WebGPUModel subclass with forward() method
        prompt: input text prompt
        tokenizer: tokenizer with encode()/decode() methods
                   If None, uses tiktoken gpt2 encoding (GPT-2 compat)
        max_tokens: number of tokens to generate
        temperature: sampling temperature
        top_k: top-k sampling parameter

    Returns:
        generated text string
    """
    if tokenizer is not None:
        tokens = tokenizer.encode(prompt)
        dec_fn = lambda ids: tokenizer.decode(ids)
        dec_one = lambda t: tokenizer.decode([t])
    else:
        # Fallback: try tiktoken for GPT-2
        try:
            import tiktoken
            enc = tiktoken.get_encoding("gpt2")
            tokens = enc.encode(prompt)
            dec_fn = lambda ids: enc.decode(ids)
            dec_one = lambda t: enc.decode([t])
        except ImportError:
            tokens = [ord(c) for c in prompt]
            dec_fn = lambda ids: "".join(
                chr(t) if t < 128 else "?" for t in ids)
            dec_one = lambda t: chr(t) if t < 128 else "?"

    token_ids = np.array(tokens, dtype=np.int32)
    generated = list(tokens)

    print(f"\nPrompt: {prompt}")
    print(f"Generating {max_tokens} tokens...\n")
    print(prompt, end="", flush=True)

    # Profiler hooks (if available)
    _p = hasattr(model, 'profiler') and model.profiler and model.profiler.enabled
    _cpu = model.profiler._cpu if _p else None

    # Reset KV cache
    model.kv_cache = None
    if hasattr(model, '_gpu_kv_cache'):
        for layer in model._gpu_kv_cache:
            k, v, _ = model._gpu_kv_cache[layer]
            model._gpu_kv_cache[layer] = (k, v, 0)

    # Prefill timer starts here (TTFT = time to first token)
    prefill_start = time.perf_counter()

    # Prefill: process full prompt, populate KV cache
    if _p: _cpu.begin("prefill")
    if _p: _cpu.begin("prefill/forward")
    logits = model.forward(token_ids, use_cache=True, pos_offset=0)
    if _p: _cpu.end("prefill/forward")
    if _p: _cpu.end("prefill")
    next_logits = logits[-1, :]

    # Warm up fast decode pipeline (if available)
    if hasattr(model, '_warmup_fast_decode'):
        model._warmup_fast_decode()

    _token_buf = np.empty(1, dtype=np.int32)  # reusable buffer
    decode_tokens = 0
    decode_forward_ns = 0  # accumulated forward() time only
    decode_start = None
    for step in range(max_tokens):
        if _p: _cpu.begin(f"decode_{step}")
        if step > 0:
            if _p: _cpu.begin(f"decode_{step}/forward")
            if _p: _cpu.begin(f"decode_{step}/forward/embed")
            _token_buf[0] = generated[-1]
            if _p: _cpu.end(f"decode_{step}/forward/embed")
            _fwd_t0 = time.perf_counter_ns()
            logits = model.forward(
                _token_buf, use_cache=True,
                pos_offset=len(generated) - 1)
            _fwd_t1 = time.perf_counter_ns()
            next_logits = logits[-1, :]
            if _p: _cpu.end(f"decode_{step}/forward")

        # Top-k sampling: only compute softmax on top-k logits
        if _p: _cpu.begin(f"decode_{step}/sampling")
        if top_k > 0 and temperature > 0:
            top_k_idx = np.argpartition(next_logits, -top_k)[-top_k:]
            top_k_vals = next_logits[top_k_idx] / temperature
            top_k_vals -= top_k_vals.max()
            top_k_probs = np.exp(top_k_vals)
            top_k_probs /= top_k_probs.sum()
            next_token = top_k_idx[np.random.choice(top_k, p=top_k_probs)]
        elif temperature > 0:
            next_logits = next_logits / temperature
            next_logits -= next_logits.max()
            probs = np.exp(next_logits)
            probs /= probs.sum()
            next_token = np.random.choice(len(probs), p=probs)
        else:
            next_token = next_logits.argmax()
        generated.append(int(next_token))
        if _p: _cpu.end(f"decode_{step}/sampling")

        # Decode and print
        print(dec_one(int(next_token)), end="", flush=True)
        if _p: _cpu.end(f"decode_{step}")

        # After the 1st token is output, mark end of prefill / start of decode
        if step == 0:
            prefill_end = time.perf_counter()
            decode_start = time.perf_counter()
        else:
            decode_tokens += 1
            decode_forward_ns += (_fwd_t1 - _fwd_t0)

    decode_end = time.perf_counter()

    prefill_ms = (prefill_end - prefill_start) * 1000
    decode_ms = (decode_end - decode_start) * 1000 if decode_start else 0
    decode_fwd_ms = decode_forward_ns / 1e6
    total_ms = (decode_end - prefill_start) * 1000
    prompt_len = len(tokens)
    decode_tps = decode_tokens / (decode_ms / 1000) if decode_ms > 0 else 0
    decode_fwd_tps = decode_tokens / (decode_fwd_ms / 1000) \
        if decode_fwd_ms > 0 else 0
    overall_tps = (prompt_len + max_tokens) / (total_ms / 1000) \
        if total_ms > 0 else 0
    print(f"\n\n--- Performance ---")
    print(f"  Prefill (TTFT): {prefill_ms:.1f}ms "
          f"({prompt_len} prompt + 1st token)")
    print(f"  Decode:  {decode_tokens} tokens in {decode_ms:.1f}ms "
          f"({decode_tps:.1f} tok/s)")
    print(f"    forward() only: {decode_fwd_ms:.1f}ms "
          f"({decode_fwd_tps:.1f} tok/s)")
    print(f"  Total:   {prompt_len + max_tokens} tokens in {total_ms:.1f}ms "
          f"({overall_tps:.1f} tok/s)")

    return dec_fn(generated)
