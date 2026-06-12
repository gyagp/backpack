#!/usr/bin/env python3
"""Run ONNX Runtime WebGPU benchmark and output results as JSON."""

import argparse
import json
import os
import re
import subprocess
import sys


def find_ort_genai_bench():
    for name in ["onnxruntime-genai-benchmark", "onnxruntime-genai-benchmark.exe",
                 "ort-genai-bench", "ort-genai-bench.exe"]:
        for d in os.environ.get("PATH", "").split(os.pathsep):
            p = os.path.join(d, name)
            if os.path.isfile(p):
                return p
    return "onnxruntime-genai-benchmark"


def parse_model_info(model_path):
    basename = os.path.basename(model_path.rstrip("/\\"))
    quant_match = re.search(r'[.-](Q\d[_A-Za-z0-9]*)', basename, re.IGNORECASE)
    quant = quant_match.group(1) if quant_match else "unknown"
    name = re.sub(r'[.-]Q\d[_A-Za-z0-9]*', '', basename)
    name = re.sub(r'\.(onnx|gguf)$', '', name, flags=re.IGNORECASE)
    return name or basename, quant


def parse_output(text):
    prefill_toks = None
    decode_toks = None

    for line in text.splitlines():
        lower = line.lower()
        toks_match = re.search(r'([\d.]+)\s*(?:tokens?/s|tok/s)', lower)
        if not toks_match:
            continue
        val = float(toks_match.group(1))
        if "prefill" in lower or "prompt" in lower or "pp" in lower:
            prefill_toks = val
        elif "decode" in lower or "generate" in lower or "tg" in lower:
            decode_toks = val

    if prefill_toks is None and decode_toks is None:
        try:
            data = json.loads(text.strip().splitlines()[-1])
            prefill_toks = data.get("prefill_tok_s")
            decode_toks = data.get("decode_tok_s")
        except (json.JSONDecodeError, IndexError):
            pass

    return prefill_toks, decode_toks


def run_bench(model_path, bench_path=None):
    bench = bench_path or find_ort_genai_bench()
    cmd = [
        bench,
        "-m", model_path,
        "-e", "webgpu",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        print(f"ORT WebGPU benchmark failed (exit {result.returncode}):", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        sys.exit(1)

    return result.stdout


def main():
    parser = argparse.ArgumentParser(description="Run ONNX Runtime WebGPU benchmark")
    parser.add_argument("model", help="Path to ONNX model directory or file")
    parser.add_argument("--ort-bench", help="Path to ORT benchmark executable")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Model not found: {args.model}", file=sys.stderr)
        sys.exit(1)

    model_name, quant = parse_model_info(args.model)
    output = run_bench(args.model, args.ort_bench)
    prefill_toks, decode_toks = parse_output(output)

    result = {
        "model": model_name,
        "quant": quant,
        "engine": "ort-webgpu",
        "prefill_tok_s": prefill_toks,
        "decode_tok_s": decode_toks,
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
