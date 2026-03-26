#!/usr/bin/env python3
"""
ONNX model inference runtime for WebGPU.

Loads ONNX GenAI models (like phi-4-mini) and runs inference using the
same WebGPU pipeline as the GGUF runtime. Weights are dequantized to
fp16 at load time; the inference engine is shared with GGUFModel.

Supports:
  - Q4 (MatMulNBits 4-bit) and Q8 (8-bit) quantized weights
  - Pre-computed Su-scaled RoPE tables (phi-4-mini)
  - Partial rotary embedding (rotary_dim < head_dim)
  - BPE tokenizer from tokenizer.json
  - Streaming text output

Usage:
    python -m runtimes.python.onnx_runtime --model path/to/onnx/dir \\
        --prompt "Hello" --max-tokens 50 [--profile]
"""

import argparse
import gc
import os
import sys
import time
from typing import Dict, List

import numpy as np

# Add runtimes/python/ and project root to path
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(os.path.dirname(_here))
for p in [_here, _root]:
    if p not in sys.path:
        sys.path.insert(0, p)

from onnx_loader import extract_onnx_config, load_onnx_weights, OnnxTokenizer
from gguf_runtime import GGUFModel


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ONNX model inference runtime (WebGPU)")
    parser.add_argument("--model", required=True,
                        help="Path to ONNX model directory")
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

    # 1. Extract config from model config
    model_dir = args.model
    if not os.path.isdir(model_dir):
        print(f"Error: {model_dir} is not a directory")
        sys.exit(1)
    if not os.path.exists(os.path.join(model_dir, "model.onnx")):
        print(f"Error: No model.onnx found in {model_dir}")
        sys.exit(1)

    print(f"Loading ONNX model: {model_dir}")
    cfg = extract_onnx_config(model_dir)

    # 2. Load weights (this updates cfg with intermediate_size)
    weights = load_onnx_weights(model_dir, cfg)

    # 3. Fix tie_word_embeddings based on actual weights
    if "lm_head.weight" in weights:
        cfg["tie_word_embeddings"] = False

    # 4. Infer rotary_dim from ONNX cos/sin cache
    if "_rope_cos" in weights:
        # cos_cache shape: [max_positions, half_rotary_dim]
        cfg["rotary_dim"] = 2 * weights["_rope_cos"].shape[1]

    print(f"Model: {cfg['arch']} ({cfg['n_layer']} layers, "
          f"E={cfg['n_embd']}, HD={cfg['head_dim']}, "
          f"V={cfg['n_vocab']}, KV={cfg['n_kv_heads']}, "
          f"IM={cfg['intermediate_size']})")
    if cfg.get("rotary_dim", cfg["head_dim"]) != cfg["head_dim"]:
        print(f"  Partial RoPE: rotary_dim={cfg['rotary_dim']} "
              f"(head_dim={cfg['head_dim']})")

    # 5. Load tokenizer
    tokenizer = OnnxTokenizer(model_dir)

    # 6. Create model (reuses the GGUFModel inference engine)
    model = GGUFModel(weights, cfg)
    t1 = time.time()
    print(f"Model loaded in {(t1-t0)*1000:.0f}ms\n")

    # 7. Enable profiling
    if args.profile:
        model.enable_profiling()
        print("GPU profiling enabled")

    # 8. Tokenize prompt
    prompt_tokens = tokenizer.encode(args.prompt)
    print(f'Prompt: "{args.prompt}"')
    print(f"Tokens ({len(prompt_tokens)}): {prompt_tokens}")

    # 9. Prefill
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

    # 10. Decode with streaming output
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

    # 11. Print profile report
    if args.profile and model.profiler:
        model.profiler.finish()
        model.profiler.report()


if __name__ == "__main__":
    main()
