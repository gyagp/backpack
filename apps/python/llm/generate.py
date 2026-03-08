#!/usr/bin/env python3
"""
LLM text generation application.

Shared application code for all LLM architectures (LLaMA, Qwen, Mistral,
Gemma, Phi, GPT-2, etc.). Uses the model-agnostic GGUF engine for inference.

Usage:
    # Direct GGUF file
    python -m apps.llm --model path/to/model.gguf --prompt "Hello"

    # Model directory (auto-discovers GGUF inside)
    python -m apps.llm --model path/to/model-dir/ --prompt "Hello"

    # From central model repo
    python -m apps.llm --model E:/workspace/project/ai-models/qwen-3-1.7B \\
        --chat "What is 2+2?"

    # With profiling
    python -m apps.llm --model model.gguf --prompt "Hello" --profile
"""

import argparse
import os
import sys
import time

import numpy as np

# Add project root to path
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'models'))
sys.path.insert(0, os.path.join(ROOT, 'runtimes', 'python'))

from model_parser.resolver import discover_model
from model_parser.gguf_parser import GGUFFile, extract_gguf_config, GGUFTokenizer
from model_parser.onnx_parser import extract_onnx_config, OnnxTokenizer
from runtimes.python.gguf_engine import GGUFModel, load_gguf_weights
from runtimes.python.onnx_loader import load_onnx_weights


# ─── Chat template formatting ────────────────────────────────────────────────

def format_chat_prompt(message: str, arch: str = "qwen3") -> str:
    """Wrap a user message in the model's chat template."""
    # Most llama-family models use ChatML-style templates
    return f"<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\n"


# ─── Text generation ─────────────────────────────────────────────────────────

def generate(model: GGUFModel, tokenizer: GGUFTokenizer,
             prompt: str, max_tokens: int = 100,
             stream: bool = True, stream_prompt: bool = True) -> str:
    """Generate text from a prompt. Returns the generated text."""
    prompt_tokens = tokenizer.encode(prompt)

    # Prefill
    t_prefill = time.time()
    logits = model.forward(np.array(prompt_tokens),
                           use_cache=True, pos_offset=0)
    prefill_ms = (time.time() - t_prefill) * 1000

    last_logits = logits[-1] if logits.ndim > 1 else logits.ravel()
    next_token = int(np.argmax(last_logits))

    if stream and stream_prompt:
        print(prompt, end="", flush=True)

    # Decode
    t_decode = time.time()
    generated_tokens = []
    generated_text = ""
    suppress_prompt = prompt if not stream_prompt else None

    for step in range(max_tokens):
        if next_token == tokenizer.eos_token_id:
            break

        generated_tokens.append(next_token)
        text = tokenizer.decode_token(next_token)
        # Skip special tokens in output
        if text.startswith('<|') and text.endswith('|>'):
            pass
        else:
            generated_text += text
            if stream:
                print(text, end="", flush=True)

        pos = len(prompt_tokens) + step
        logits = model.forward(np.array([next_token]),
                               use_cache=True, pos_offset=pos)
        last_logits = logits[-1] if logits.ndim > 1 else logits.ravel()
        next_token = int(np.argmax(last_logits))

    decode_ms = (time.time() - t_decode) * 1000

    if stream:
        print()

    # Stats
    n_decode = len(generated_tokens)
    tps = n_decode * 1000.0 / decode_ms if decode_ms > 0 else 0
    print(f"\n--- Performance ---")
    print(f"  Prefill: {prefill_ms:.0f}ms ({len(prompt_tokens)} tokens)")
    print(f"  Decode:  {n_decode} tokens in {decode_ms:.0f}ms ({tps:.1f} tok/s)")

    return generated_text


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LLM text generation (all architectures)")
    parser.add_argument("--model", required=True,
                        help="GGUF file, model directory, or model name")
    parser.add_argument("--format", default=None, choices=['gguf', 'onnx'],
                        help="Force specific model format")
    parser.add_argument("--prompt", default=None, help="Text prompt")
    parser.add_argument("--chat", default=None,
                        help="Chat message (auto-wrapped in template)")
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--backend", default=None, help="vulkan/d3d12")
    args = parser.parse_args()

    if args.backend:
        os.environ["DAWN_BACKEND"] = args.backend

    if not args.prompt and not args.chat:
        args.prompt = "Hello"

    # Discover model formats
    info = discover_model(args.model)
    print(f"Model: {info.summary()}")

    # Select format
    if args.format:
        model_format = args.format
        if model_format not in info.formats:
            print(f"  Warning: {model_format} not available, "
                  f"using {info.preferred_format}")
            model_format = info.preferred_format
    else:
        model_format = info.preferred_format

    model_path = info.formats[model_format]

    # Load model (GGUF or ONNX)
    t0 = time.time()
    print(f"Loading ({model_format}): {model_path}")

    if model_format == 'onnx':
        cfg = extract_onnx_config(model_path)
        tokenizer = OnnxTokenizer(model_path)
        weights = load_onnx_weights(model_path, cfg)
        # ONNX weights are dequantized to fp16 — use fp16w path
        cfg["model_format"] = "onnx"
    else:
        gf = GGUFFile(model_path)
        cfg = extract_gguf_config(gf)
        tokenizer = GGUFTokenizer(gf)
        weights = load_gguf_weights(gf, cfg)

    model = GGUFModel(weights, cfg)
    print(f"Model: {cfg['arch']} ({cfg['n_layer']}L, E={cfg['n_embd']}, "
          f"V={cfg['n_vocab']}) loaded in {(time.time()-t0)*1000:.0f}ms\n")

    if args.profile:
        model.enable_profiling()

    # Format prompt
    if args.chat:
        prompt = format_chat_prompt(args.chat, cfg['arch'])
        print(f"Chat: {args.chat}")
        print(f"Template: {prompt[:80]}...")
    else:
        prompt = args.prompt

    # Generate
    print("--- Output ---")
    stream_prompt = (args.chat is None)  # Don't echo template for chat mode
    generate(model, tokenizer, prompt, args.max_tokens,
             stream_prompt=stream_prompt)

    if args.profile and model.profiler:
        model.profiler.finish()
        model.profiler.report()


if __name__ == "__main__":
    main()
