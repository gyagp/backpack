#!/usr/bin/env python3
"""Benchmark orchestrator: runs backpack and llama.cpp benchmarks across models
and generates a comparison report."""

import argparse
import json
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(SCRIPT_DIR, "benchmark_config.json")
RUN_LLAMACPP = os.path.join(SCRIPT_DIR, "run_llamacpp.py")
RUN_ORT_WEBGPU = os.path.join(SCRIPT_DIR, "run_ort_webgpu.py")


def find_bench_backpack():
    build_dir = os.path.join(SCRIPT_DIR, "..", "build")
    for subdir in ["Release", "Debug", "RelWithDebInfo", "."]:
        for name in ["bench_backpack.exe", "bench_backpack"]:
            p = os.path.join(build_dir, subdir, name)
            if os.path.isfile(p):
                return os.path.abspath(p)
    for name in ["bench_backpack", "bench_backpack.exe"]:
        for d in os.environ.get("PATH", "").split(os.pathsep):
            p = os.path.join(d, name)
            if os.path.isfile(p):
                return p
    return "bench_backpack"


def run_backpack(model_path, bench_path):
    try:
        result = subprocess.run(
            [bench_path, model_path],
            capture_output=True, text=True, timeout=600,
        )
    except FileNotFoundError:
        print(f"  [backpack] bench_backpack not found at {bench_path}", file=sys.stderr)
        return None
    except subprocess.TimeoutExpired:
        print(f"  [backpack] timed out for {model_path}", file=sys.stderr)
        return None

    if result.returncode != 0:
        print(f"  [backpack] failed (exit {result.returncode}): {result.stderr.strip()}", file=sys.stderr)
        return None

    try:
        data = json.loads(result.stdout.strip().splitlines()[-1])
        data["engine"] = "backpack"
        return data
    except (json.JSONDecodeError, IndexError) as e:
        print(f"  [backpack] failed to parse output: {e}", file=sys.stderr)
        return None


def run_llamacpp(model_path):
    try:
        result = subprocess.run(
            [sys.executable, RUN_LLAMACPP, model_path],
            capture_output=True, text=True, timeout=600,
        )
    except subprocess.TimeoutExpired:
        print(f"  [llamacpp] timed out for {model_path}", file=sys.stderr)
        return None

    if result.returncode != 0:
        print(f"  [llamacpp] failed (exit {result.returncode}): {result.stderr.strip()}", file=sys.stderr)
        return None

    try:
        data = json.loads(result.stdout.strip().splitlines()[-1])
        return data
    except (json.JSONDecodeError, IndexError) as e:
        print(f"  [llamacpp] failed to parse output: {e}", file=sys.stderr)
        return None


def run_ort_webgpu(model_path):
    try:
        result = subprocess.run(
            [sys.executable, RUN_ORT_WEBGPU, model_path],
            capture_output=True, text=True, timeout=600,
        )
    except subprocess.TimeoutExpired:
        print(f"  [ort-webgpu] timed out for {model_path}", file=sys.stderr)
        return None

    if result.returncode != 0:
        print(f"  [ort-webgpu] failed (exit {result.returncode}): {result.stderr.strip()}", file=sys.stderr)
        return None

    try:
        data = json.loads(result.stdout.strip().splitlines()[-1])
        return data
    except (json.JSONDecodeError, IndexError) as e:
        print(f"  [ort-webgpu] failed to parse output: {e}", file=sys.stderr)
        return None


def generate_markdown(results):
    lines = [
        "# Benchmark Results",
        "",
        "| Model | Quant | Engine | Prefill tok/s | Decode tok/s |",
        "|-------|-------|--------|---------------|--------------|",
    ]
    for r in results:
        prefill = f"{r['prefill_tok_s']:.2f}" if r.get("prefill_tok_s") is not None else "N/A"
        decode = f"{r['decode_tok_s']:.2f}" if r.get("decode_tok_s") is not None else "N/A"
        lines.append(f"| {r.get('model', 'unknown')} | {r.get('quant', 'unknown')} | {r.get('engine', 'unknown')} | {prefill} | {decode} |")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Run benchmarks across models")
    parser.add_argument(
        "--config", default=DEFAULT_CONFIG,
        help="JSON config file with model paths (default: benchmark_config.json)",
    )
    parser.add_argument(
        "--models", nargs="+",
        help="Model GGUF paths (overrides config file)",
    )
    parser.add_argument("--bench-backpack", help="Path to bench_backpack executable")
    parser.add_argument(
        "--output-dir", default=SCRIPT_DIR,
        help="Directory for results.json and results.md",
    )
    args = parser.parse_args()

    if args.models:
        model_paths = args.models
    elif os.path.isfile(args.config):
        with open(args.config) as f:
            config = json.load(f)
        model_paths = config.get("models", [])
    else:
        print(f"No models specified and config not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    if not model_paths:
        print("No model paths provided.", file=sys.stderr)
        sys.exit(1)

    bench_path = args.bench_backpack or find_bench_backpack()
    all_results = []

    for model_path in model_paths:
        print(f"Benchmarking: {model_path}")

        if not os.path.isfile(model_path):
            print(f"  Model file not found, skipping: {model_path}", file=sys.stderr)
            continue

        bp_result = run_backpack(model_path, bench_path)
        if bp_result:
            all_results.append(bp_result)
            print(f"  [backpack] prefill={bp_result.get('prefill_tok_s'):.2f} tok/s, decode={bp_result.get('decode_tok_s'):.2f} tok/s")

        lc_result = run_llamacpp(model_path)
        if lc_result:
            all_results.append(lc_result)
            print(f"  [llamacpp] prefill={lc_result.get('prefill_tok_s'):.2f} tok/s, decode={lc_result.get('decode_tok_s'):.2f} tok/s")

        ort_result = run_ort_webgpu(model_path)
        if ort_result:
            all_results.append(ort_result)
            prefill = ort_result.get('prefill_tok_s')
            decode = ort_result.get('decode_tok_s')
            print(f"  [ort-webgpu] prefill={prefill:.2f} tok/s, decode={decode:.2f} tok/s"
                  if prefill is not None and decode is not None
                  else f"  [ort-webgpu] prefill={prefill} tok/s, decode={decode} tok/s")

    os.makedirs(args.output_dir, exist_ok=True)

    json_path = os.path.join(args.output_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults written to {json_path}")

    md_path = os.path.join(args.output_dir, "results.md")
    with open(md_path, "w") as f:
        f.write(generate_markdown(all_results))
    print(f"Report written to {md_path}")


if __name__ == "__main__":
    main()
