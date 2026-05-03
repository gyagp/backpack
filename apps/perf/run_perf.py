#!/usr/bin/env python3
"""
Generate per-model perf JSON files in apps/perf/ with Backpack + baseline comparison.
Each file contains: backpack_gguf, backpack_onnx, llamacpp_vulkan, ort_webgpu benchmarks.
"""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

MODELS_DIR = Path(r"E:\workspace\project\agents\ai-models")
BACKPACK_EXE = Path(r"E:\workspace\project\agents\backpack\runtime\build\backpack_llm.exe")
LLAMA_BENCH = Path(r"D:\backup\x64\llamacpp\b8981\llama-bench.exe")
ORT_BENCH = Path(r"D:\backup\x64\ort\20260430\model_benchmark.exe")
PERF_DIR = Path(r"E:\workspace\project\agents\backpack\apps\perf")

PROMPT_LENGTHS = [128, 256, 512]
DECODE_TOKENS = 128
SKIP_MODELS = {"Phi-4-multimodal-instruct", "whisper-tiny"}
TIMEOUT = 600


def run(cmd, timeout=TIMEOUT):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           encoding="utf-8", errors="replace")
        return r.stdout, r.stderr, r.returncode
    except subprocess.TimeoutExpired:
        return "", "TIMEOUT", -1
    except Exception as e:
        return "", str(e), -1


def get_system_info():
    """Get basic system info."""
    import platform
    return {
        "cpu": platform.processor() or "unknown",
        "memory_gb": 0,
        "os": platform.system()
    }


def discover_models():
    models = []
    for d in sorted(MODELS_DIR.iterdir()):
        if not d.is_dir() or d.name in SKIP_MODELS or d.name == "README.md":
            continue
        gguf_dir = d / "gguf"
        onnx_dir = d / "onnx-webgpu"
        has_gguf = gguf_dir.is_dir() and any(gguf_dir.glob("*.gguf"))
        has_onnx = onnx_dir.is_dir() and (onnx_dir / "model.onnx").exists()
        if has_gguf or has_onnx:
            gguf_file = None
            if has_gguf:
                gf = list(gguf_dir.glob("*.gguf"))
                if gf: gguf_file = str(gf[0])
            models.append({
                "name": d.name, "path": str(d),
                "has_gguf": has_gguf, "has_onnx": has_onnx,
                "gguf_file": gguf_file,
                "onnx_dir": str(onnx_dir) if has_onnx else None,
            })
    return models


def parse_backpack_bench(stdout, stderr):
    results = []
    for line in (stdout + stderr).split("\n"):
        m = re.match(r"\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", line)
        if m:
            results.append({
                "input_tokens": int(m.group(1)),
                "prefill_ms": float(m.group(2)), "prefill_tok_s": float(m.group(3)),
                "decode_ms": float(m.group(4)), "decode_tok_s": float(m.group(5)),
            })
    return results


def run_backpack(model, fmt):
    cmd = [str(BACKPACK_EXE), "--model", model["path"], "--benchmark",
           "--bench-gen-tokens", str(DECODE_TOKENS)]
    if fmt: cmd += ["--format", fmt]
    stdout, stderr, rc = run(cmd)
    results = parse_backpack_bench(stdout, stderr)
    return results if results else None


def run_llama_bench(model):
    if not model["gguf_file"]: return None
    # Prefill
    pp = ",".join(str(x) for x in PROMPT_LENGTHS)
    stdout, stderr, rc = run([str(LLAMA_BENCH), "-m", model["gguf_file"],
                               "-p", pp, "-n", "0", "-r", "3", "-o", "json"])
    prefill = {}
    if rc == 0:
        try:
            for e in json.loads(stdout):
                if e.get("n_prompt", 0) > 0: prefill[e["n_prompt"]] = e.get("avg_ts", 0)
        except: pass
    # Decode
    stdout, stderr, rc = run([str(LLAMA_BENCH), "-m", model["gguf_file"],
                               "-p", "0", "-n", str(DECODE_TOKENS), "-r", "3", "-o", "json"])
    decode_ts = 0
    if rc == 0:
        try:
            for e in json.loads(stdout):
                if e.get("n_gen", 0) > 0: decode_ts = e.get("avg_ts", 0)
        except: pass
    results = []
    for pl in PROMPT_LENGTHS:
        results.append({
            "input_tokens": pl,
            "prefill_tok_s": prefill.get(pl, 0),
            "decode_tok_s": decode_ts,
        })
    return results if any(r["decode_tok_s"] > 0 for r in results) else None


def run_ort_bench(model):
    if not model["onnx_dir"]: return None
    results = []
    for pl in [128]:  # single prompt length
        cmd = [str(ORT_BENCH), "-i", model["onnx_dir"], "-l", str(pl),
               "-g", str(DECODE_TOKENS), "-r", "1"]
        stdout, stderr, rc = run(cmd, timeout=300)
        if rc != 0:
            results.append({"input_tokens": pl, "prefill_tok_s": 0, "decode_tok_s": 0})
            continue
        text = stdout + stderr
        prefill_ts = decode_ts = 0
        for line in text.split("\n"):
            if "tokens/s" in line and prefill_ts == 0:
                m = re.search(r"avg\s+\(tokens/s\):\s+([\d.]+)", line)
                if m: prefill_ts = float(m.group(1)); continue
            if "tokens/s" in line and prefill_ts > 0 and decode_ts == 0:
                m = re.search(r"avg\s+\(tokens/s\):\s+([\d.]+)", line)
                if m: decode_ts = float(m.group(1))
        results.append({"input_tokens": pl, "prefill_tok_s": prefill_ts, "decode_tok_s": decode_ts})
    return results if results and results[0]["decode_tok_s"] > 0 else None


def save_perf(model_name, benchmarks):
    """Save or update perf JSON for a model."""
    path = PERF_DIR / f"{model_name}.json"

    # Load existing if present
    existing = {}
    if path.exists():
        try:
            with open(path) as f: existing = json.load(f)
        except: pass

    # Merge benchmarks into existing
    if "benchmarks" not in existing:
        existing["benchmarks"] = {}
    existing["benchmarks"].update(benchmarks)
    existing["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    with open(path, "w") as f:
        json.dump(existing, f, indent=2)
    return path


def main():
    PERF_DIR.mkdir(parents=True, exist_ok=True)
    models = discover_models()
    print(f"Discovered {len(models)} models\n")

    for i, model in enumerate(models):
        name = model["name"]
        perf_file = PERF_DIR / f"{name}.json"

        # Skip if already has all 4 benchmarks
        if perf_file.exists():
            try:
                with open(perf_file) as f:
                    existing = json.load(f)
                bkeys = set(existing.get("benchmarks", {}).keys())
                needed = set()
                if model["has_gguf"]: needed |= {"backpack_gguf", "llamacpp_vulkan"}
                if model["has_onnx"]: needed |= {"backpack_onnx", "ort_webgpu"}
                if needed.issubset(bkeys):
                    print(f"[{i+1}/{len(models)}] {name}: already complete, skipping")
                    continue
            except:
                pass

        print(f"[{i+1}/{len(models)}] {name}:")
        benchmarks = {}

        # Backpack GGUF
        if model["has_gguf"]:
            print(f"  backpack gguf...", end="", flush=True)
            results = run_backpack(model, "gguf")
            if results:
                r128 = next((r for r in results if r["input_tokens"] == 128), None)
                print(f" {r128['decode_tok_s']:.1f} tok/s" if r128 else " (no 128)")
                benchmarks["backpack_gguf"] = {
                    "format": "gguf", "quantization": "Q4_K",
                    "decode_tokens": DECODE_TOKENS, "results": results
                }
            else:
                print(" FAIL")

        # Backpack ONNX
        if model["has_onnx"]:
            print(f"  backpack onnx...", end="", flush=True)
            results = run_backpack(model, "onnx")
            if results:
                r128 = next((r for r in results if r["input_tokens"] == 128), None)
                print(f" {r128['decode_tok_s']:.1f} tok/s" if r128 else " (no 128)")
                benchmarks["backpack_onnx"] = {
                    "format": "onnx", "quantization": "Q4",
                    "decode_tokens": DECODE_TOKENS, "results": results
                }
            else:
                print(" FAIL")

        # llama.cpp Vulkan
        if model["has_gguf"] and LLAMA_BENCH.exists():
            print(f"  llama.cpp...", end="", flush=True)
            results = run_llama_bench(model)
            if results:
                r128 = next((r for r in results if r["input_tokens"] == 128), None)
                print(f" {r128['decode_tok_s']:.1f} tok/s" if r128 else " (no 128)")
                benchmarks["llamacpp_vulkan"] = {
                    "format": "gguf", "quantization": "Q4_K",
                    "decode_tokens": DECODE_TOKENS, "results": results
                }
            else:
                print(" FAIL")

        # ORT WebGPU
        if model["has_onnx"] and ORT_BENCH.exists():
            print(f"  ort webgpu...", end="", flush=True)
            results = run_ort_bench(model)
            if results:
                r128 = next((r for r in results if r["input_tokens"] == 128), None)
                print(f" {r128['decode_tok_s']:.1f} tok/s" if r128 else " (no 128)")
                benchmarks["ort_webgpu"] = {
                    "format": "onnx", "quantization": "Q4",
                    "decode_tokens": DECODE_TOKENS, "results": results
                }
            else:
                print(" FAIL")

        if benchmarks:
            path = save_perf(name, benchmarks)
            print(f"  -> {path}")
        print()


if __name__ == "__main__":
    main()
