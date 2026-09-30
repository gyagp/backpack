"""Compare the shared LLM application's Qwen GGUF output with llama.cpp.

Sequential, bounded runs preserve both streams, commands, binary hashes, and
each exact-output verdict under gitignore/. This is conformance, not timing.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--app", type=Path, default=root / "gitignore/runtime/build/backpack_llm.exe")
    parser.add_argument("--app-revision", help="Source revision of an externally built application")
    parser.add_argument("--output", type=Path, default=root / "gitignore/logs/qwen38-native/conformance")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--context", type=int, default=1024)
    parser.add_argument("--include-batched", action="store_true")
    args = parser.parse_args()
    if socket.gethostname().lower() != "webgfx-104":
        parser.error("this goal targets webgfx-104 only")
    out = args.output.resolve()
    if not out.is_relative_to(root / "gitignore"):
        parser.error("output must stay below gitignore/")
    out.mkdir(parents=True, exist_ok=True)
    prompts = ["Answer with only the number. What is 2 + 2?",
               "What is the capital of France? Answer with only the city name.",
               "Continue the sequence: 2, 4, 6, 8,",
               "Explain in one sentence why the sky appears blue.",
               "中国的首都是哪里？只回答城市名。",
               "Reply with only this number: 123456"]
    report = {"device": socket.gethostname(), "model": str(args.model.resolve()), "binaries": {}, "results": []}
    for path in [args.app, args.app.with_name("backpack.dll"), args.reference]:
        with path.open("rb") as file:
            report["binaries"][str(path.resolve())] = hashlib.file_digest(file, "sha256").hexdigest()
    report["validator_revision"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    report["app_revision"] = args.app_revision

    def run(name: str, command: list[str]) -> tuple[str, int]:
        start = time.monotonic()
        env = dict(os.environ, BP_DUMP_TOKENS="1")
        if name.startswith("serial-"):
            env["BP_QWEN35_SERIAL_PREFILL"] = "1"
        elif name.startswith("batched-"):
            env.pop("BP_QWEN35_SERIAL_PREFILL", None)
        result = subprocess.run(command, capture_output=True, encoding="utf-8", errors="replace",
                                stdin=subprocess.DEVNULL, env=env, timeout=300)
        (out / (name + ".stdout.log")).write_text(result.stdout, encoding="utf-8")
        (out / (name + ".stderr.log")).write_text(result.stderr, encoding="utf-8")
        report["results"].append({"name": name, "command": command, "exit_code": result.returncode,
                                  "environment": {k: v for k, v in env.items() if k.startswith("BP_")},
                                  "elapsed_seconds": time.monotonic() - start})
        return result.stdout + "\n" + result.stderr, result.returncode

    passed = True
    for index, prompt in enumerate(prompts):
        formatted = ("<|im_start|>system\nYou are a helpful AI assistant.<|im_end|>\n"
                     "<|im_start|>user\n" + prompt + "<|im_end|>\n"
                     "<|im_start|>assistant\n<think>\n\n</think>\n\n")
        name = f"reference-{index}"
        _, code = run(name, [str(args.reference.resolve()), "-m", str(args.model.resolve()),
                            "-p", formatted, "--no-conversation", "--no-display-prompt",
                            "--temp", "0", "-ngl", "99", "-c", str(args.context), "-n", str(args.max_tokens)])
        reference = (out / (name + ".stdout.log")).read_text(encoding="utf-8")
        reference = re.sub(r"\s*\[end of text\]\s*$", "", reference).strip()
        for mode in (["serial", "batched"] if args.include_batched else ["serial"]):
            text, status = run(f"{mode}-{index}", [str(args.app.resolve()), "--model", str(args.model.resolve()),
                               "--backend", "d3d12", "--chat", prompt, "--temperature", "0",
                               "--max-tokens", str(args.max_tokens), "--max-seq-len", str(args.context)]
                               + (["--fast-prefill"] if mode == "batched" else []))
            match = re.search(r"--- Output ---\s*(.*?)\s*--- Performance ---", text, re.S)
            actual = match.group(1).strip() if match else ""
            # Prompt-token debug traces can occur inside the output region.
            actual = re.sub(r"\[debug\] prompt tokens:[^\n]*\n?", "", actual).strip()
            good = code == 0 and status == 0 and actual == reference and bool(actual)
            if index == 0: good = good and actual == "4"
            if index == 1: good = good and actual == "Paris"
            report["results"][-1].update(reference=reference, actual=actual, matches=good)
            passed = passed and good
            print(f"{mode} prompt {index}: {'PASS' if good else 'FAIL'}", flush=True)
            (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
