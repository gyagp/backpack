#!/usr/bin/env python3
"""Performance regression test for backpack inference.

Runs bench_backpack on a small model and asserts decode tok/s stays
above a minimum threshold. Can be run standalone or via pytest.

Usage:
    python tests/test_perf_regression.py
    python tests/test_perf_regression.py --model models/Llama-3.2-1B-Q4_K_M.gguf
    python tests/test_perf_regression.py --min-decode-tok-s 5.0
    pytest tests/test_perf_regression.py -v
"""

import argparse
import json
import os
import subprocess
import sys

DEFAULT_MODEL = "models/Llama-3.2-1B-Q4_K_M.gguf"
DEFAULT_MIN_DECODE_TOK_S = 1.0
DEFAULT_BENCH_BINARY = os.path.join("build", "bench_backpack.exe") if sys.platform == "win32" else os.path.join("build", "bench_backpack")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def find_bench_binary():
    candidates = [
        os.path.join(PROJECT_ROOT, DEFAULT_BENCH_BINARY),
        os.path.join(PROJECT_ROOT, "build", "Release", "bench_backpack.exe"),
        os.path.join(PROJECT_ROOT, "build", "Debug", "bench_backpack.exe"),
        os.path.join(PROJECT_ROOT, "build", "Release", "bench_backpack"),
        os.path.join(PROJECT_ROOT, "build", "Debug", "bench_backpack"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return candidates[0]


def run_bench(model_path, bench_binary=None):
    binary = bench_binary or find_bench_binary()
    if not os.path.isfile(binary):
        raise FileNotFoundError(f"bench_backpack binary not found at {binary}")

    abs_model = os.path.join(PROJECT_ROOT, model_path) if not os.path.isabs(model_path) else model_path
    if not os.path.isfile(abs_model):
        raise FileNotFoundError(f"Model file not found at {abs_model}")

    result = subprocess.run(
        [binary, abs_model],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(f"bench_backpack failed (exit {result.returncode}):\n{result.stderr}")

    output = result.stdout.strip()
    return json.loads(output)


def test_decode_throughput():
    model = os.environ.get("BACKPACK_TEST_MODEL", DEFAULT_MODEL)
    min_tok_s = float(os.environ.get("BACKPACK_MIN_DECODE_TOK_S", DEFAULT_MIN_DECODE_TOK_S))
    bench_bin = os.environ.get("BACKPACK_BENCH_BINARY", None)

    result = run_bench(model, bench_bin)

    decode_tok_s = result["decode_tok_s"]
    print(f"Model: {result.get('model', 'unknown')}")
    print(f"Quant: {result.get('quant', 'unknown')}")
    print(f"Decode tok/s: {decode_tok_s:.2f} (threshold: {min_tok_s:.2f})")

    assert decode_tok_s >= min_tok_s, (
        f"Decode throughput regression: {decode_tok_s:.2f} tok/s < {min_tok_s:.2f} tok/s minimum"
    )


def main():
    parser = argparse.ArgumentParser(description="Performance regression test")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path to GGUF model file")
    parser.add_argument("--min-decode-tok-s", type=float, default=DEFAULT_MIN_DECODE_TOK_S,
                        help="Minimum decode tok/s threshold")
    parser.add_argument("--bench-binary", default=None, help="Path to bench_backpack binary")
    args = parser.parse_args()

    try:
        result = run_bench(args.model, args.bench_binary)
    except FileNotFoundError as e:
        print(f"SKIP: {e}", file=sys.stderr)
        sys.exit(2)
    except RuntimeError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        sys.exit(1)

    decode_tok_s = result["decode_tok_s"]
    print(f"Model: {result.get('model', 'unknown')}")
    print(f"Quant: {result.get('quant', 'unknown')}")
    print(f"Prefill tok/s: {result.get('prefill_tok_s', 0):.2f}")
    print(f"Decode tok/s: {decode_tok_s:.2f}")
    print(f"Threshold: {args.min_decode_tok_s:.2f} tok/s")

    if decode_tok_s < args.min_decode_tok_s:
        print(f"\nFAIL: Decode throughput regression: {decode_tok_s:.2f} < {args.min_decode_tok_s:.2f} tok/s")
        sys.exit(1)
    else:
        print(f"\nPASS: Decode throughput {decode_tok_s:.2f} >= {args.min_decode_tok_s:.2f} tok/s")
        sys.exit(0)


if __name__ == "__main__":
    main()
