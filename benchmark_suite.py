#!/usr/bin/env python3
"""
Backpack Full Model Suite Benchmark

Discovers models in ai-models/, runs Backpack + baseline tools,
collects results into JSON, and generates an HTML comparison report.
"""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# ─── Configuration ────────────────────────────────────────────────────────────

MODELS_DIR = Path(r"E:\workspace\project\agents\ai-models")
BACKPACK_EXE = Path(r"E:\workspace\project\agents\backpack\runtime\build\backpack_llm.exe")
LLAMA_BENCH = Path(r"D:\backup\x64\llamacpp\b8981\llama-bench.exe")
ORT_BENCH = Path(r"D:\backup\x64\ort\20260430\model_benchmark.exe")

RESULTS_FILE = Path("benchmark_results.json")
REPORT_FILE = Path("benchmark_report.html")

PROMPT_LENGTHS = [128, 256, 512, 1024, 2048, 4096]
DECODE_TOKENS = 128
LLAMA_REPEATS = 3
ORT_REPEATS = 1
TIMEOUT_SECONDS = 600  # 10 min per run

SKIP_MODELS = {"Phi-4-multimodal-instruct", "whisper-tiny"}


def discover_models():
    """Scan ai-models/ for models with gguf/ and/or onnx-webgpu/ subdirs."""
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
                gguf_files = list(gguf_dir.glob("*.gguf"))
                if gguf_files:
                    gguf_file = str(gguf_files[0])
            models.append({
                "name": d.name,
                "path": str(d),
                "has_gguf": has_gguf,
                "has_onnx": has_onnx,
                "gguf_file": gguf_file,
                "onnx_dir": str(onnx_dir) if has_onnx else None,
            })
    return models


def run_command(cmd, timeout=TIMEOUT_SECONDS):
    """Run a command, return (stdout, stderr, returncode)."""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            encoding="utf-8", errors="replace"
        )
        return result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return "", "TIMEOUT", -1
    except Exception as e:
        return "", str(e), -1


def parse_backpack_benchmark(stdout, stderr):
    """Parse backpack_llm --benchmark output into results list."""
    text = stdout + stderr
    results = []
    for line in text.split("\n"):
        m = re.match(
            r"\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)",
            line,
        )
        if m:
            results.append({
                "prompt_len": int(m.group(1)),
                "prefill_ms": float(m.group(2)),
                "prefill_tok_s": float(m.group(3)),
                "decode_ms": float(m.group(4)),
                "decode_tok_s": float(m.group(5)),
            })
    return results


def run_backpack(model, fmt):
    """Run backpack_llm --benchmark for a model in given format."""
    cmd = [
        str(BACKPACK_EXE),
        "--model", model["path"],
        "--benchmark",
        "--bench-gen-tokens", str(DECODE_TOKENS),
    ]
    if fmt:
        cmd += ["--format", fmt]
    print(f"    Running: backpack ({fmt or 'auto'})...", flush=True)
    stdout, stderr, rc = run_command(cmd)
    # Try to parse results even if rc != 0 (DAWN warnings cause non-zero on some systems)
    results = parse_backpack_benchmark(stdout, stderr)
    if rc != 0 and not results:
        print(f"    FAILED (rc={rc}): {stderr[:200]}")
        return None, stderr[:500]
    if rc != 0 and results:
        print(f"    WARNING: rc={rc} but got {len(results)} results (likely DAWN warnings)")
    if results:
        print(f"    OK: {len(results)} prompt lengths, "
              f"decode={results[0]['decode_tok_s']:.1f} tok/s @ {results[0]['prompt_len']}")
    else:
        print(f"    WARNING: no results parsed from stdout")
        return None, f"No benchmark output parsed. stderr: {stderr[:300]}"
    return results, None


def parse_llama_bench_json(stdout):
    """Parse llama-bench -o json output."""
    results = []
    try:
        data = json.loads(stdout)
        if isinstance(data, list):
            for entry in data:
                pp = entry.get("n_prompt", 0)
                tg = entry.get("n_gen", 0)
                tok_s = entry.get("avg_ts", 0)
                if pp > 0 and tg == 0:
                    results.append({"prompt_len": pp, "prefill_tok_s": tok_s})
                elif tg > 0 and pp == 0:
                    results.append({"prompt_len": 0, "decode_tok_s": tok_s})
    except json.JSONDecodeError:
        pass
    return results


def run_llama_bench(model):
    """Run llama-bench for a GGUF model."""
    if not model["gguf_file"] or not LLAMA_BENCH.exists():
        return None, "llama-bench not available"

    print(f"    Running: llama-bench...", flush=True)
    # Prefill benchmark (prompt processing)
    pp_str = ",".join(str(x) for x in PROMPT_LENGTHS)
    cmd = [
        str(LLAMA_BENCH),
        "-m", model["gguf_file"],
        "-p", pp_str,
        "-n", "0",
        "-r", str(LLAMA_REPEATS),
        "-o", "json",
    ]
    stdout, stderr, rc = run_command(cmd)
    if rc != 0:
        print(f"    llama-bench prefill FAILED (rc={rc}): {stderr[:200]}")
        return None, stderr[:500]

    prefill_results = {}
    try:
        data = json.loads(stdout)
        for entry in data:
            pp = entry.get("n_prompt", 0)
            if pp > 0:
                prefill_results[pp] = entry.get("avg_ts", 0)
    except (json.JSONDecodeError, TypeError):
        print(f"    WARNING: could not parse llama-bench prefill JSON")

    # Decode benchmark
    cmd = [
        str(LLAMA_BENCH),
        "-m", model["gguf_file"],
        "-p", "0",
        "-n", str(DECODE_TOKENS),
        "-r", str(LLAMA_REPEATS),
        "-o", "json",
    ]
    stdout, stderr, rc = run_command(cmd)
    decode_tok_s = 0
    if rc == 0:
        try:
            data = json.loads(stdout)
            for entry in data:
                if entry.get("n_gen", 0) > 0:
                    decode_tok_s = entry.get("avg_ts", 0)
        except (json.JSONDecodeError, TypeError):
            pass

    results = []
    for pl in PROMPT_LENGTHS:
        results.append({
            "prompt_len": pl,
            "prefill_tok_s": prefill_results.get(pl, 0),
            "decode_tok_s": decode_tok_s,
        })

    if results:
        print(f"    OK: prefill={results[0]['prefill_tok_s']:.1f}, decode={decode_tok_s:.1f}")
    return results, None


def run_ort_bench(model):
    """Run ORT model_benchmark (WebGPU default EP) for an ONNX model."""
    if not model["onnx_dir"] or not ORT_BENCH.exists():
        return None, "ORT bench not available"

    print(f"    Running: ORT WebGPU benchmark...", flush=True)
    results = []
    # Only run at 128 prompt length for ORT (sufficient for comparison)
    ort_lengths = [128]
    for pl in ort_lengths:
        cmd = [
            str(ORT_BENCH),
            "-i", model["onnx_dir"],
            "-l", str(pl),
            "-g", str(DECODE_TOKENS),
            "-r", str(ORT_REPEATS),
        ]
        stdout, stderr, rc = run_command(cmd, timeout=300)
        if rc != 0:
            results.append({"prompt_len": pl, "prefill_tok_s": 0, "decode_tok_s": 0,
                            "error": stderr[:200]})
            continue

        # Parse ORT output
        text = stdout + stderr
        prefill_ts = 0
        decode_ts = 0
        for line in text.split("\n"):
            # "Prompt processing" section: "avg (tokens/s): 3763.57"
            if "tokens/s" in line and prefill_ts == 0:
                m = re.search(r"avg\s+\(tokens/s\):\s+([\d.]+)", line)
                if m:
                    prefill_ts = float(m.group(1))
                    continue
            # "Token generation" section: second "avg (tokens/s)"
            if "tokens/s" in line and prefill_ts > 0 and decode_ts == 0:
                m = re.search(r"avg\s+\(tokens/s\):\s+([\d.]+)", line)
                if m:
                    decode_ts = float(m.group(1))
        results.append({
            "prompt_len": pl,
            "prefill_tok_s": prefill_ts,
            "decode_tok_s": decode_ts,
        })

    if results and results[0]["prefill_tok_s"] > 0:
        print(f"    OK: prefill={results[0]['prefill_tok_s']:.1f}, "
              f"decode={results[0]['decode_tok_s']:.1f}")
    else:
        print(f"    WARNING: no valid ORT results")
    return results, None


def load_partial_results():
    """Load existing results for crash recovery."""
    if RESULTS_FILE.exists():
        try:
            with open(RESULTS_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {"models": {}, "errors": []}


def save_results(results):
    """Save results to JSON (atomic write)."""
    tmp = str(RESULTS_FILE) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(results, f, indent=2)
    os.replace(tmp, str(RESULTS_FILE))


def benchmark_all():
    """Run benchmarks for all discovered models."""
    models = discover_models()
    print(f"Discovered {len(models)} models\n")

    results = load_partial_results()
    completed = set(results.get("_completed", []))

    for i, model in enumerate(models):
        name = model["name"]
        tag = f"[{i+1}/{len(models)}]"

        if name in completed:
            print(f"{tag} {name}: already completed, skipping")
            continue

        print(f"{tag} {name}:", flush=True)
        model_result = {"name": name, "has_gguf": model["has_gguf"], "has_onnx": model["has_onnx"]}

        # Backpack GGUF
        if model["has_gguf"]:
            data, err = run_backpack(model, "gguf")
            model_result["backpack_gguf"] = data
            if err:
                results.setdefault("errors", []).append(
                    {"model": name, "backend": "backpack_gguf", "error": err})

        # Backpack ONNX
        if model["has_onnx"]:
            data, err = run_backpack(model, "onnx")
            model_result["backpack_onnx"] = data
            if err:
                results.setdefault("errors", []).append(
                    {"model": name, "backend": "backpack_onnx", "error": err})

        # llama.cpp Vulkan
        if model["has_gguf"]:
            data, err = run_llama_bench(model)
            model_result["llama_cpp"] = data
            if err:
                results.setdefault("errors", []).append(
                    {"model": name, "backend": "llama_cpp", "error": err})

        # ORT WebGPU
        if model["has_onnx"]:
            data, err = run_ort_bench(model)
            model_result["ort"] = data
            if err:
                results.setdefault("errors", []).append(
                    {"model": name, "backend": "ort", "error": err})

        results["models"][name] = model_result
        completed.add(name)
        results["_completed"] = list(completed)
        save_results(results)
        print()

    return results


def generate_report(results):
    """Generate self-contained HTML report with Chart.js."""
    models_data = results.get("models", {})
    errors = results.get("errors", [])

    # Build comparison data
    gguf_comparisons = []  # (name, backpack_prefill, llama_prefill, backpack_decode, llama_decode)
    onnx_comparisons = []  # (name, backpack_prefill, ort_prefill, backpack_decode, ort_decode)

    for name, m in sorted(models_data.items()):
        # GGUF comparison at prompt_len=512
        bp_gguf = m.get("backpack_gguf") or []
        llama = m.get("llama_cpp") or []
        bp_512 = next((r for r in bp_gguf if r.get("prompt_len") == 512), None)
        ll_512 = next((r for r in llama if r.get("prompt_len") == 512), None)
        if bp_512 and ll_512:
            gguf_comparisons.append({
                "name": name,
                "bp_prefill": bp_512.get("prefill_tok_s", 0),
                "ll_prefill": ll_512.get("prefill_tok_s", 0),
                "bp_decode": bp_512.get("decode_tok_s", 0),
                "ll_decode": ll_512.get("decode_tok_s", 0),
            })

        # ONNX comparison at prompt_len=512
        bp_onnx = m.get("backpack_onnx") or []
        ort = m.get("ort") or []
        bp_o512 = next((r for r in bp_onnx if r.get("prompt_len") == 512), None)
        ort_512 = next((r for r in ort if r.get("prompt_len") == 512), None)
        if bp_o512 and ort_512:
            onnx_comparisons.append({
                "name": name,
                "bp_prefill": bp_o512.get("prefill_tok_s", 0),
                "ort_prefill": ort_512.get("prefill_tok_s", 0),
                "bp_decode": bp_o512.get("decode_tok_s", 0),
                "ort_decode": ort_512.get("decode_tok_s", 0),
            })

    # Build summary table rows
    summary_rows = []
    for name, m in sorted(models_data.items()):
        bp_gguf = m.get("backpack_gguf") or []
        llama = m.get("llama_cpp") or []
        bp_onnx = m.get("backpack_onnx") or []
        ort = m.get("ort") or []

        bp_g128 = next((r for r in bp_gguf if r.get("prompt_len") == 128), None)
        ll_128 = next((r for r in llama if r.get("prompt_len") == 128), None)
        bp_o128 = next((r for r in bp_onnx if r.get("prompt_len") == 128), None)
        ort_128 = next((r for r in ort if r.get("prompt_len") == 128), None)

        def ratio(a, b):
            if a and b and b > 0:
                return f"{a/b:.2f}x"
            return "—"

        row = {
            "name": name,
            "bp_gguf_decode": f"{bp_g128['decode_tok_s']:.1f}" if bp_g128 else "—",
            "llama_decode": f"{ll_128['decode_tok_s']:.1f}" if ll_128 else "—",
            "gguf_ratio": ratio(
                bp_g128["decode_tok_s"] if bp_g128 else 0,
                ll_128["decode_tok_s"] if ll_128 else 0,
            ),
            "bp_onnx_decode": f"{bp_o128['decode_tok_s']:.1f}" if bp_o128 else "—",
            "ort_decode": f"{ort_128['decode_tok_s']:.1f}" if ort_128 else "—",
            "onnx_ratio": ratio(
                bp_o128["decode_tok_s"] if bp_o128 else 0,
                ort_128["decode_tok_s"] if ort_128 else 0,
            ),
        }
        summary_rows.append(row)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Backpack Benchmark Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         max-width: 1400px; margin: 40px auto; padding: 0 20px; background: #f8f9fa; }}
  h1 {{ color: #1a1a2e; }}
  h2 {{ color: #16213e; margin-top: 40px; }}
  .chart-container {{ background: white; border-radius: 8px; padding: 20px;
                      margin: 20px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
  canvas {{ max-height: 500px; }}
  table {{ border-collapse: collapse; width: 100%; background: white;
           border-radius: 8px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
  th {{ background: #1a1a2e; color: white; padding: 12px 8px; text-align: left; font-size: 13px; }}
  td {{ padding: 10px 8px; border-bottom: 1px solid #eee; font-size: 13px; }}
  tr:hover td {{ background: #f0f4ff; }}
  .error-log {{ background: #fff3f3; border: 1px solid #ffcdd2; border-radius: 8px;
                padding: 15px; margin: 20px 0; }}
  .error-log pre {{ font-size: 12px; white-space: pre-wrap; }}
  .good {{ color: #2e7d32; font-weight: bold; }}
  .bad {{ color: #c62828; font-weight: bold; }}
</style>
</head>
<body>
<h1>Backpack Full Model Suite Benchmark</h1>
<p>Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
<p>Models tested: {len(models_data)} | Errors: {len(errors)}</p>

<h2>GGUF: Backpack vs llama.cpp (Decode tok/s @ 512 prompt)</h2>
<div class="chart-container"><canvas id="ggufChart"></canvas></div>

<h2>ONNX: Backpack vs ORT WebGPU (Decode tok/s @ 512 prompt)</h2>
<div class="chart-container"><canvas id="onnxChart"></canvas></div>

<h2>Summary Table (@ 128 prompt tokens)</h2>
<table>
<tr>
  <th>Model</th>
  <th>BP GGUF<br>decode</th><th>llama.cpp<br>decode</th><th>Ratio</th>
  <th>BP ONNX<br>decode</th><th>ORT<br>decode</th><th>Ratio</th>
</tr>
"""
    for row in summary_rows:
        # Color the ratio
        def color_ratio(r):
            if r == "—":
                return r
            try:
                v = float(r.replace("x", ""))
                cls = "good" if v >= 1.0 else "bad"
                return f'<span class="{cls}">{r}</span>'
            except ValueError:
                return r

        html += f"""<tr>
  <td>{row['name']}</td>
  <td>{row['bp_gguf_decode']}</td><td>{row['llama_decode']}</td>
  <td>{color_ratio(row['gguf_ratio'])}</td>
  <td>{row['bp_onnx_decode']}</td><td>{row['ort_decode']}</td>
  <td>{color_ratio(row['onnx_ratio'])}</td>
</tr>
"""
    html += "</table>\n"

    # Error log
    if errors:
        html += '<h2>Failure Log</h2>\n<div class="error-log">\n'
        for e in errors:
            html += f"<p><b>{e['model']}</b> ({e['backend']}): <pre>{e['error'][:300]}</pre></p>\n"
        html += "</div>\n"

    # Chart.js scripts
    gguf_labels = json.dumps([c["name"] for c in gguf_comparisons])
    gguf_bp = json.dumps([c["bp_decode"] for c in gguf_comparisons])
    gguf_ll = json.dumps([c["ll_decode"] for c in gguf_comparisons])

    onnx_labels = json.dumps([c["name"] for c in onnx_comparisons])
    onnx_bp = json.dumps([c["bp_decode"] for c in onnx_comparisons])
    onnx_ort = json.dumps([c["ort_decode"] for c in onnx_comparisons])

    html += f"""
<script>
new Chart(document.getElementById('ggufChart'), {{
  type: 'bar',
  data: {{
    labels: {gguf_labels},
    datasets: [
      {{ label: 'Backpack GGUF', data: {gguf_bp}, backgroundColor: '#3498db' }},
      {{ label: 'llama.cpp Vulkan', data: {gguf_ll}, backgroundColor: '#e74c3c' }}
    ]
  }},
  options: {{
    responsive: true,
    plugins: {{ title: {{ display: true, text: 'Decode tok/s (higher is better)' }} }},
    scales: {{
      x: {{ ticks: {{ maxRotation: 45 }} }},
      y: {{ beginAtZero: true, title: {{ display: true, text: 'tok/s' }} }}
    }}
  }}
}});

new Chart(document.getElementById('onnxChart'), {{
  type: 'bar',
  data: {{
    labels: {onnx_labels},
    datasets: [
      {{ label: 'Backpack ONNX', data: {onnx_bp}, backgroundColor: '#2ecc71' }},
      {{ label: 'ORT WebGPU', data: {onnx_ort}, backgroundColor: '#f39c12' }}
    ]
  }},
  options: {{
    responsive: true,
    plugins: {{ title: {{ display: true, text: 'Decode tok/s (higher is better)' }} }},
    scales: {{
      x: {{ ticks: {{ maxRotation: 45 }} }},
      y: {{ beginAtZero: true, title: {{ display: true, text: 'tok/s' }} }}
    }}
  }}
}});
</script>
</body>
</html>
"""
    with open(REPORT_FILE, "w") as f:
        f.write(html)
    print(f"\nReport written to {REPORT_FILE}")


def main():
    if "--report-only" in sys.argv:
        results = load_partial_results()
        generate_report(results)
        return

    print("=" * 60)
    print("Backpack Full Model Suite Benchmark")
    print("=" * 60)
    print()

    if not BACKPACK_EXE.exists():
        print(f"ERROR: {BACKPACK_EXE} not found. Build first.")
        sys.exit(1)

    results = benchmark_all()
    generate_report(results)
    print("\nDone!")


if __name__ == "__main__":
    main()
