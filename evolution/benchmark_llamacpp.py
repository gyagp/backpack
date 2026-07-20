from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run llama.cpp Vulkan prefill and decode benchmark")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--generation-tokens", type=int, default=128)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--root", type=Path,
                        default=Path(r"D:\backup\x64\llamacpp"))
    args = parser.parse_args()

    versions = sorted(
        (path for path in args.root.glob("b*/vulkan/llama-bench.exe")
         if re.fullmatch(r"b\d+", path.parent.parent.name)),
        key=lambda path: int(path.parent.parent.name[1:]), reverse=True,
    )
    if not versions:
        raise SystemExit(f"No llama-bench.exe found below {args.root}")
    executable = versions[0]
    command = [str(executable), "-m", str(args.model), "-p", str(args.prompt_tokens),
               "-n", str(args.generation_tokens), "-r", str(args.repetitions),
               "-ngl", "99", "-o", "json"]
    completed = subprocess.run(command, cwd=executable.parent, text=True, encoding="utf-8",
                               errors="replace", capture_output=True, timeout=1800, shell=False)
    if completed.returncode:
        print(completed.stdout)
        print(completed.stderr)
        return completed.returncode
    rows = json.loads(completed.stdout)
    prefill = next((row for row in rows if int(row.get("n_prompt", 0)) > 0), None)
    decode = next((row for row in rows if int(row.get("n_gen", 0)) > 0), None)
    if not prefill or not decode:
        raise SystemExit("llama-bench did not return separate prompt and generation records")
    metrics = {
        "prompt_tokens": args.prompt_tokens,
        "generation_tokens": args.generation_tokens,
        "prefill_tok_s": float(prefill["avg_ts"]),
        "decode_tok_s": float(decode["avg_ts"]),
        "prefill_stddev_tok_s": float(prefill.get("stddev_ts", 0)),
        "decode_stddev_tok_s": float(decode.get("stddev_ts", 0)),
    }
    print("EVOLUTION_METRICS " + json.dumps(metrics, separators=(",", ":")))
    print(f"LLAMACPP_REVISION {executable.parent.parent.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
