#!/usr/bin/env python3
"""Run llama.cpp benchmark with Vulkan backend and output results as JSON."""

import argparse
import json
import os
import re
import subprocess
import sys


def find_llama_bench():
    for name in ["llama-bench", "llama-bench.exe"]:
        for d in os.environ.get("PATH", "").split(os.pathsep):
            p = os.path.join(d, name)
            if os.path.isfile(p):
                return p
    return "llama-bench"


def parse_model_info(model_path):
    basename = os.path.basename(model_path)
    name = os.path.splitext(basename)[0]
    quant_match = re.search(r'[.-](Q\d[_A-Za-z0-9]*)', basename, re.IGNORECASE)
    quant = quant_match.group(1) if quant_match else "unknown"
    return name, quant


def run_bench(model_path, llama_bench_path=None):
    bench = llama_bench_path or find_llama_bench()
    cmd = [
        bench,
        "-m", model_path,
        "-ngl", "99",
        "-o", "csv",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        print(f"llama-bench failed (exit {result.returncode}):", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        sys.exit(1)

    return result.stdout


def parse_csv_output(csv_text):
    lines = [l.strip() for l in csv_text.strip().splitlines() if l.strip()]
    if len(lines) < 2:
        print("No benchmark results found in output", file=sys.stderr)
        print("Raw output:", csv_text, file=sys.stderr)
        sys.exit(1)

    header = lines[0].split(",")
    col = {name: i for i, name in enumerate(header)}

    prefill_toks = None
    decode_toks = None

    for line in lines[1:]:
        fields = line.split(",")
        test_type = fields[col.get("test", -1)] if "test" in col else ""
        if "t/s" in col:
            toks = float(fields[col["t/s"]])
        elif "speed" in col:
            toks = float(fields[col["speed"]])
        else:
            for f in reversed(fields):
                try:
                    toks = float(f)
                    break
                except ValueError:
                    continue
            else:
                continue

        if test_type == "pp512" or "pp" in test_type:
            prefill_toks = toks
        elif test_type == "tg128" or "tg" in test_type:
            decode_toks = toks

    return prefill_toks, decode_toks


def main():
    parser = argparse.ArgumentParser(description="Run llama.cpp Vulkan benchmark")
    parser.add_argument("model", help="Path to GGUF model file")
    parser.add_argument("--llama-bench", help="Path to llama-bench executable")
    args = parser.parse_args()

    if not os.path.isfile(args.model):
        print(f"Model not found: {args.model}", file=sys.stderr)
        sys.exit(1)

    model_name, quant = parse_model_info(args.model)
    csv_output = run_bench(args.model, args.llama_bench)
    prefill_toks, decode_toks = parse_csv_output(csv_output)

    result = {
        "model": model_name,
        "quant": quant,
        "engine": "llamacpp-vulkan",
        "prefill_tok_s": prefill_toks,
        "decode_tok_s": decode_toks,
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
