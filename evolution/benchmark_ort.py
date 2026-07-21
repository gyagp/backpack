from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


def find_bin_dir(root: Path) -> Path:
    candidates = [p.parent for p in root.glob("*/model_benchmark.exe")
                  if (p.parent / "model_chat.exe").is_file()]
    if not candidates:
        raise RuntimeError(f"No complete ORT GenAI backup found below {root}")
    return max(candidates, key=lambda p: ((p / "build-manifest.json").stat().st_mtime
                                          if (p / "build-manifest.json").exists() else p.stat().st_mtime))


def revision(bin_dir: Path) -> str:
    manifest = json.loads((bin_dir / "build-manifest.json").read_text(encoding="utf-8"))
    ort = str(manifest["onnxruntime_revision"])[:10]
    genai = str(manifest["onnxruntime_genai_revision"])[:10]
    date = str(manifest["date"]).replace("-", "")
    return f"ort-{ort}-genai-{genai}-{date}"


def graph_capture_enabled(model: Path) -> bool | None:
    config = json.loads((model / "genai_config.json").read_text(encoding="utf-8"))

    def visit(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key.lower() == "enablegraphcapture":
                    return str(child).lower() in {"1", "true", "yes"}
                result = visit(child)
                if result is not None:
                    return result
        elif isinstance(value, list):
            for child in value:
                result = visit(child)
                if result is not None:
                    return result
        return None

    return visit(config)


def parse_benchmark(output: str) -> tuple[float, float]:
    prefill = re.search(r"Prompt processing.*?avg \(tokens/s\):\s*([0-9.]+)", output, re.S)
    decode = re.search(r"Token generation.*?avg \(tokens/s\):\s*([0-9.]+)", output, re.S)
    if not prefill or not decode:
        raise RuntimeError("model_benchmark output did not contain separate prefill/decode TPS")
    return float(prefill.group(1)), float(decode.group(1))


def run(command: list[str], cwd: Path, timeout: int) -> str:
    completed = subprocess.run(command, cwd=cwd, text=True, encoding="utf-8", errors="replace",
                               capture_output=True, timeout=timeout, shell=False)
    output = completed.stdout + "\n" + completed.stderr
    if completed.returncode:
        raise RuntimeError(f"command exited {completed.returncode}:\n{output[-4000:]}")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate and benchmark source-built ORT WebGPU")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--required-fact", required=True)
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--generation-tokens", type=int, default=128)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--root", type=Path, default=Path(r"D:\backup\x64\ort"))
    parser.add_argument("--bin-dir", type=Path)
    args = parser.parse_args()

    bin_dir = args.bin_dir or find_bin_dir(args.root)
    runtime_revision = revision(bin_dir)
    chat = run([str(bin_dir / "model_chat.exe"), "-m", str(args.model),
                "--user_prompt", args.prompt, "--non_interactive", "-l", "128"], bin_dir, 300)
    passed = args.required_fact.lower() in chat.lower()
    print("EVOLUTION_CONFORMANCE " + json.dumps({
        "passed": passed, "prompt": args.prompt, "required_fact": args.required_fact,
        "output": chat[-4000:], "revision": runtime_revision,
    }, separators=(",", ":")))
    if not passed:
        return 2

    benchmark = run([str(bin_dir / "model_benchmark.exe"), "-i", str(args.model),
                     "-l", str(args.prompt_tokens), "-g", str(args.generation_tokens),
                     "-r", str(args.repetitions), "--reuse_generator"], bin_dir, 1800)
    prefill, decode = parse_benchmark(benchmark)
    print("EVOLUTION_METRICS " + json.dumps({
        "prompt_tokens": args.prompt_tokens, "generation_tokens": args.generation_tokens,
        "prefill_tok_s": prefill, "decode_tok_s": decode,
        "graph_capture": graph_capture_enabled(args.model), "reuse_generator": True,
    }, separators=(",", ":")))
    print(f"RUNTIME_REVISION {runtime_revision}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
