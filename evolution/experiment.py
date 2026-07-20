from __future__ import annotations
import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def parse_argv(value: str) -> list[str]:
    result = json.loads(value)
    if not isinstance(result, list) or not result or not all(isinstance(x, str) for x in result):
        raise argparse.ArgumentTypeError("command must be a JSON array of strings")
    return result


def extract_metric(stdout: str, metric: str) -> float:
    for line in reversed(stdout.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and isinstance(value.get(metric), (int, float)):
            return float(value[metric])
    raise RuntimeError(f"command output has no JSON object containing numeric {metric!r}")


def execute(argv: list[str], cwd: Path, timeout: int) -> tuple[float, str, str, int]:
    started = time.monotonic()
    result = subprocess.run(argv, cwd=cwd, text=True, capture_output=True, timeout=timeout, shell=False)
    duration = time.monotonic() - started
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): {result.stderr[-2000:]}")
    return duration, result.stdout, result.stderr, result.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a paired Backpack base/candidate experiment")
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--machine-id", required=True)
    parser.add_argument("--base-sha", required=True)
    parser.add_argument("--candidate-sha", required=True)
    parser.add_argument("--base-command", required=True, type=parse_argv)
    parser.add_argument("--candidate-command", required=True, type=parse_argv)
    parser.add_argument("--metric", default="decode_tok_s")
    parser.add_argument("--unit", default="tok/s")
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--cwd", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.samples < 2 or args.warmups < 0:
        parser.error("samples must be >= 2 and warmups must be >= 0")

    output_dir = args.output or ROOT / "gitignore" / "evolution" / "experiments" / args.task_id / args.machine_id
    output_dir.mkdir(parents=True, exist_ok=True)
    commands = {"base": args.base_command, "candidate": args.candidate_command}
    collected: dict[str, list[float]] = {"base": [], "candidate": []}
    raw: list[dict[str, Any]] = []
    order = ["base", "candidate"] * (args.warmups + args.samples)
    # Randomize each pair while keeping both variants equally represented over time.
    for pair in range(0, len(order), 2):
        if random.Random(f"{args.task_id}-{args.machine_id}-{pair}").random() < 0.5:
            order[pair:pair + 2] = reversed(order[pair:pair + 2])
    seen = {"base": 0, "candidate": 0}
    for index, variant in enumerate(order):
        duration, stdout, stderr, code = execute(commands[variant], args.cwd, args.timeout)
        value = extract_metric(stdout, args.metric)
        is_warmup = seen[variant] < args.warmups
        seen[variant] += 1
        if not is_warmup:
            collected[variant].append(value)
        raw.append({"index": index, "variant": variant, "warmup": is_warmup,
                    "value": value, "duration_seconds": duration, "exit_code": code,
                    "stdout": stdout, "stderr": stderr})
        print(f"{index + 1}/{len(order)} {variant}: {value:g} {args.unit}{' (warmup)' if is_warmup else ''}")

    raw_path = output_dir / "raw.json"
    raw_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    for variant, sha in (("base", args.base_sha), ("candidate", args.candidate_sha)):
        evidence = {
            "task_id": args.task_id, "machine_id": args.machine_id, "variant": variant,
            "metric": args.metric, "unit": args.unit, "samples": collected[variant],
            "correctness": {"passed": True, "source": "benchmark-process-exit"},
            "environment": {"cwd": str(args.cwd), "command": commands[variant]},
            "artifacts": [str(raw_path)], "commit_sha": sha,
        }
        path = output_dir / f"{variant}-evidence.json"
        path.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
        print(f"Evidence: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
