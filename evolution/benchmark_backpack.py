from __future__ import annotations

import argparse
import hashlib
import json
import math
import socket
import subprocess
import tempfile
from pathlib import Path

from evolution.benchmark_ort import artifact_fingerprint


def validate_results(data: dict, prompt_tokens: int, outputs: int, repetitions: int,
                     warmup: int, capacity: int, prompt: str) -> dict:
    protocol = data.get("benchmark_protocol", "")
    chunk = data.get("prefill_chunk", 0)
    expected_protocol = "llm-fixed-text-reuse-first-token-v2" + (f"-chunk{chunk}" if chunk else "")
    if protocol != expected_protocol:
        raise RuntimeError("Unsupported Backpack benchmark protocol")
    if (data.get("warmup_runs"), data.get("reuse_generator"),
        data.get("prefill_includes_first_token"), data.get("max_seq_len")) != (
            warmup, True, True, capacity):
        raise RuntimeError("Benchmark options differ from the requested protocol")
    if str(data.get("backend", "")).lower() != "d3d12":
        raise RuntimeError("Benchmark did not use D3D12")
    runs = data.get("runs", [])
    if len(runs) != repetitions:
        raise RuntimeError("Wrong benchmark repetition count")
    for index, run in enumerate(runs):
        if (run.get("input_tokens"), run.get("generated_tokens"), run.get("decode_sample_tokens"),
            run.get("final_position"), run.get("repetition"), run.get("prompt")) != (
                prompt_tokens, outputs, outputs - 1, prompt_tokens + outputs - 1, index, prompt):
            raise RuntimeError("Benchmark token/count/position mismatch")
        if len(run.get("prompt_token_ids", [])) != prompt_tokens or len(run.get("generated_token_ids", [])) != outputs:
            raise RuntimeError("Benchmark token arrays do not match declared counts")
        if min(run["prompt_token_ids"] + run["generated_token_ids"]) < 0:
            raise RuntimeError("Benchmark returned an invalid token")
        if index and (run["prompt_token_ids"] != runs[0]["prompt_token_ids"] or
                      run["generated_token_ids"] != runs[0]["generated_token_ids"]):
            raise RuntimeError("Benchmark continuation is not deterministic")
        for key in ["prefill_ms", "decode_ms"]:
            value = run.get(key, -1)
            if not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
                raise RuntimeError("Invalid benchmark duration")
        if run["prefill_ms"] <= 0 or (outputs > 1 and run["decode_ms"] <= 0):
            raise RuntimeError("Benchmark reported zero elapsed time")
    return {"measured_processes": 1, "measured_repetitions_per_process": repetitions,
            "measured_repetitions": repetitions,
            "prefill_tok_s": prompt_tokens * repetitions * 1000 / sum(r["prefill_ms"] for r in runs),
            "decode_tok_s": ((outputs - 1) * repetitions * 1000 / sum(r["decode_ms"] for r in runs)
                             if outputs > 1 else 0.0),
            "prefill_samples": [prompt_tokens * 1000 / r["prefill_ms"] for r in runs],
            "decode_samples": [(outputs - 1) * 1000 / r["decode_ms"] if outputs > 1 else 0.0 for r in runs]}


def main() -> int:
    parser = argparse.ArgumentParser(description="Conform and benchmark exact Backpack workloads on webgfx-104")
    parser.add_argument("--exe", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--prompt", required=True, help="Conformance chat prompt")
    parser.add_argument("--expected-output", required=True)
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--generation-tokens", type=int, default=128)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--prefill-chunk", type=int, default=0)
    parser.add_argument("--fast-prefill", action="store_true")
    parser.add_argument("--benchmark-prompt-file", type=Path)
    parser.add_argument("--evidence-root", type=Path,
                        default=Path(__file__).resolve().parents[1] / "gitignore/logs/backpack-reference")
    args = parser.parse_args()
    if socket.gethostname().lower() != "webgfx-104":
        parser.error("this goal only permits execution on webgfx-104")
    if min(args.prompt_tokens, args.generation_tokens, args.repetitions) <= 0 or min(args.warmup, args.prefill_chunk) < 0:
        parser.error("Invalid workload counts")
    args.exe = args.exe.resolve()
    args.model = args.model.resolve()
    capacity = args.prompt_tokens + args.generation_tokens
    args.evidence_root.mkdir(parents=True, exist_ok=True)
    evidence = Path(tempfile.mkdtemp(prefix="backpack-", dir=args.evidence_root.resolve()))
    if args.model.is_file():
        with args.model.open("rb") as file:
            fingerprint = hashlib.file_digest(file, "sha256").hexdigest()
    else:
        fingerprint = artifact_fingerprint(args.model)
    prompt = (args.benchmark_prompt_file.read_text(encoding="utf-8") if args.benchmark_prompt_file
              else " ".join(["A"] * args.prompt_tokens))
    prompt_file = evidence / "prompt.txt"
    prompt_file.write_text(prompt, encoding="utf-8", newline="")
    base = [str(args.exe), "--model", str(args.model), "--backend", "d3d12",
            "--max-seq-len", str(capacity), "--prefill-chunk", str(args.prefill_chunk)]
    if args.fast_prefill:
        base.append("--fast-prefill")

    def run(name: str, command: list[str], timeout: int) -> str:
        (evidence / (name + "-command.json")).write_text(json.dumps(command, indent=2), encoding="utf-8")
        result = subprocess.run(command, text=True, encoding="utf-8", errors="replace",
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, timeout=timeout)
        (evidence / (name + ".log")).write_text(result.stdout, encoding="utf-8")
        if result.returncode or "[DAWN " in result.stdout:
            raise RuntimeError(f"Backpack {name} failed; see {evidence}")
        return result.stdout

    log = run("conformance", base + ["--chat", args.prompt, "--max-tokens", "64"], 600)
    try:
        answer = log.split("--- Output ---", 1)[1].split("--- Performance ---", 1)[0].strip()
    except IndexError as error:
        raise RuntimeError("Missing conformance output") from error
    passed = answer == args.expected_output.strip()
    print("EVOLUTION_CONFORMANCE " + json.dumps({"passed": passed, "output": answer,
        "expected_output": args.expected_output, "artifact_fingerprint": fingerprint,
        "revision": args.revision}, separators=(",", ":")))
    if not passed:
        return 2
    result_file = evidence / "result.json"
    command = base + ["--benchmark", "--bench-prompt-len", str(args.prompt_tokens),
        "--bench-prompt-file", str(prompt_file), "--bench-gen-tokens", str(args.generation_tokens),
        "--bench-repetitions", str(args.repetitions), "--bench-warmup", str(args.warmup),
        "--bench-json", str(result_file)]
    run("benchmark", command, 3600)
    data = json.loads(result_file.read_text(encoding="utf-8"))
    metrics = validate_results(data, args.prompt_tokens, args.generation_tokens, args.repetitions,
                               args.warmup, capacity, prompt)
    metrics.update(prompt_tokens=args.prompt_tokens, generation_tokens=args.generation_tokens,
        decode_sample_tokens=args.generation_tokens-1, max_seq_len=capacity, warmup_runs=args.warmup,
        reuse_generator=True, prefill_includes_first_token=True, prefill_chunk=data["prefill_chunk"],
        benchmark_protocol=data["benchmark_protocol"], artifact_fingerprint=fingerprint,
        prompt_sha256=hashlib.sha256(prompt_file.read_bytes()).hexdigest(), backend="d3d12")
    (evidence / "provenance.json").write_text(json.dumps({"revision": args.revision,
        "binary_sha256": hashlib.sha256(args.exe.read_bytes()).hexdigest(),
        "library_sha256": hashlib.sha256(args.exe.with_name("backpack.dll").read_bytes()).hexdigest(),
        "model": str(args.model), "metrics": metrics}, indent=2), encoding="utf-8")
    print("EVOLUTION_METRICS " + json.dumps(metrics, separators=(",", ":")))
    print(f"RUNTIME_REVISION {args.revision}")
    print(f"RUNTIME_EVIDENCE {evidence}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
