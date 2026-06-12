#!/usr/bin/env python3
"""Test matrix: iterate all models x formats (GGUF/ONNX) x backends (D3D12/Vulkan),
run inference, and report pass/fail as JSON + markdown."""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_MODELS_DIR = Path("E:/workspace/project/agents/ai-models")
DEFAULT_EXE = PROJECT_ROOT / "build" / "backpack.exe"
FORMATS = ["gguf", "onnx"]
FORMAT_DIR_NAMES = {"gguf": "gguf", "onnx": "onnx-webgpu"}
BACKENDS = ["d3d12", "vulkan"]
DEFAULT_TIMEOUT = 300


def discover_models(models_dir: Path) -> list[dict]:
    """Walk models_dir and find model directories containing gguf/onnx sub-folders."""
    models = []
    if not models_dir.is_dir():
        print(f"Warning: models directory not found: {models_dir}", file=sys.stderr)
        return models

    for entry in sorted(models_dir.iterdir()):
        if not entry.is_dir():
            continue
        formats_found = {}
        for fmt in FORMATS:
            fmt_dir = entry / FORMAT_DIR_NAMES.get(fmt, fmt)
            if fmt_dir.is_dir():
                files = list(fmt_dir.iterdir())
                if files:
                    formats_found[fmt] = [f.name for f in files if f.is_file()]
        if formats_found:
            models.append({
                "name": entry.name,
                "path": str(entry),
                "formats": formats_found,
            })
    return models


def run_inference(exe: Path, model_path: str, fmt: str, backend: str,
                  timeout: int) -> dict:
    """Run a single inference test. Returns a result dict."""
    result = {
        "model_path": model_path,
        "format": fmt,
        "backend": backend,
        "status": "skip",
        "duration_s": 0.0,
        "error": None,
    }

    if not exe.is_file():
        result["status"] = "skip"
        result["error"] = f"executable not found: {exe}"
        return result

    cmd = [str(exe), "--model", model_path, "--backend", backend]

    start = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        result["duration_s"] = round(time.monotonic() - start, 2)

        if proc.returncode == 0:
            result["status"] = "pass"
        elif proc.returncode == 0xC0000005 or proc.returncode == -11:
            result["status"] = "fail"
            result["error"] = "access violation / segfault"
        else:
            result["status"] = "fail"
            result["error"] = (proc.stderr.strip() or f"exit code {proc.returncode}")[:200]
    except subprocess.TimeoutExpired:
        result["duration_s"] = round(time.monotonic() - start, 2)
        result["status"] = "timeout"
        result["error"] = f"exceeded {timeout}s timeout"
    except MemoryError:
        result["duration_s"] = round(time.monotonic() - start, 2)
        result["status"] = "oom"
        result["error"] = "out of memory"
    except OSError as e:
        result["duration_s"] = round(time.monotonic() - start, 2)
        result["status"] = "error"
        result["error"] = str(e)[:200]

    return result


def generate_results_json(results: list[dict], output_path: Path):
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total": len(results),
        "passed": sum(1 for r in results if r["status"] == "pass"),
        "failed": sum(1 for r in results if r["status"] == "fail"),
        "timeout": sum(1 for r in results if r["status"] == "timeout"),
        "oom": sum(1 for r in results if r["status"] == "oom"),
        "skipped": sum(1 for r in results if r["status"] == "skip"),
        "error": sum(1 for r in results if r["status"] == "error"),
    }
    payload = {"summary": summary, "results": results}
    output_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {output_path}")


def generate_results_md(results: list[dict], models: list[dict], output_path: Path):
    lines = ["# Test Matrix Results", ""]
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines.append(f"Generated: {ts}")
    lines.append("")

    total = len(results)
    passed = sum(1 for r in results if r["status"] == "pass")
    failed = sum(1 for r in results if r["status"] == "fail")
    timed = sum(1 for r in results if r["status"] == "timeout")
    skipped = sum(1 for r in results if r["status"] == "skip")
    lines.append(f"**Total: {total}** | Pass: {passed} | Fail: {failed} "
                 f"| Timeout: {timed} | Skip: {skipped}")
    lines.append("")

    # Build table: Model | GGUF+D3D12 | GGUF+Vulkan | ONNX+D3D12 | ONNX+Vulkan
    combos = [(f, b) for f in FORMATS for b in BACKENDS]
    header_cols = ["Model"] + [f"{f.upper()}+{b.upper()}" for f, b in combos]
    lines.append("| " + " | ".join(header_cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(header_cols)) + " |")

    STATUS_ICON = {
        "pass": "PASS",
        "fail": "FAIL",
        "timeout": "TIMEOUT",
        "oom": "OOM",
        "skip": "-",
        "error": "ERROR",
    }

    lookup = {}
    for r in results:
        key = (r["model"], r["format"], r["backend"])
        lookup[key] = r["status"]

    for model in models:
        row = [model["name"]]
        for fmt, backend in combos:
            status = lookup.get((model["name"], fmt, backend), "skip")
            row.append(STATUS_ICON.get(status, status))
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    output_path.write_text("\n".join(lines))
    print(f"Wrote {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run model test matrix")
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR,
                        help="Directory containing model sub-folders")
    parser.add_argument("--exe", type=Path, default=DEFAULT_EXE,
                        help="Path to backpack executable")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                        help="Per-test timeout in seconds")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "scripts",
                        help="Directory for results.json and results.md")
    parser.add_argument("--format", choices=FORMATS, action="append", dest="formats",
                        help="Only test specific format(s)")
    parser.add_argument("--backend", choices=BACKENDS, action="append", dest="backends",
                        help="Only test specific backend(s)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Discover models and print plan without running")
    args = parser.parse_args()

    formats = args.formats or FORMATS
    backends = args.backends or BACKENDS

    print(f"Discovering models in {args.models_dir} ...")
    models = discover_models(args.models_dir)
    if not models:
        print("No models found.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(models)} models")
    total_combos = sum(
        1 for m in models for f in formats if f in m["formats"] for _ in backends
    )
    print(f"Test combinations: {total_combos} "
          f"(models={len(models)}, formats={formats}, backends={backends})")

    if args.dry_run:
        for m in models:
            for f in formats:
                if f not in m["formats"]:
                    continue
                for b in backends:
                    print(f"  {m['name']} x {f} x {b}")
        sys.exit(0)

    results = []
    done = 0
    for m in models:
        for fmt in formats:
            if fmt not in m["formats"]:
                continue
            model_files = m["formats"][fmt]
            model_file = model_files[0]
            fmt_dir_name = FORMAT_DIR_NAMES.get(fmt, fmt)
            model_path = str(Path(m["path"]) / fmt_dir_name / model_file)
            for backend in backends:
                done += 1
                print(f"[{done}/{total_combos}] {m['name']} x {fmt} x {backend} ... ",
                      end="", flush=True)
                r = run_inference(args.exe, model_path, fmt, backend, args.timeout)
                r["model"] = m["name"]
                results.append(r)
                print(r["status"].upper())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    generate_results_json(results, args.output_dir / "results.json")
    generate_results_md(results, models, args.output_dir / "results.md")

    passed = sum(1 for r in results if r["status"] == "pass")
    failed = sum(1 for r in results if r["status"] != "pass" and r["status"] != "skip")
    print(f"\nDone: {passed} passed, {failed} failed/timeout/error out of {len(results)}")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
