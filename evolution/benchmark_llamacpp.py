from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


def conformance_passed(output: str, required_fact: str, expected_output: str = "") -> bool:
    if expected_output.strip():
        return output.strip() == expected_output.strip()
    return bool(required_fact.strip() and required_fact.lower() in output.lower())


def final_answer(output: str) -> str:
    """Remove an optional reasoning trace and llama.cpp terminal marker."""
    if "</think>" in output:
        output = output.rsplit("</think>", 1)[1]
    output = re.sub(r"\s*\[end of text\]\s*$", "", output, flags=re.I)
    return output.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run llama.cpp Vulkan prefill and decode benchmark")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--generation-tokens", type=int, default=128)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--prompt")
    parser.add_argument("--required-fact")
    parser.add_argument("--expected-output")
    parser.add_argument("--prompt-format", choices=("conversation", "qwen-nothink"), default="conversation")
    parser.add_argument("--conformance-tokens", type=int, default=256)
    parser.add_argument("--root", type=Path,
                        default=Path(__file__).resolve().parents[1] / "gitignore/evolution/backups/llamacpp")
    args = parser.parse_args()
    args.model = args.model.resolve()
    args.root = args.root.resolve()
    if min(args.prompt_tokens, args.generation_tokens, args.repetitions) <= 0:
        parser.error("prompt tokens, generation tokens, and repetitions must be positive")

    versions = sorted(
        (path for path in args.root.glob("b*/vulkan/llama-bench.exe")
         if re.fullmatch(r"b\d+", path.parent.parent.name)),
        key=lambda path: int(path.parent.parent.name[1:]), reverse=True,
    )
    if not versions:
        raise SystemExit(f"No llama-bench.exe found below {args.root}")
    executable = versions[0]
    revision = executable.parent.parent.name
    if args.required_fact:
        completion = executable.with_name("llama-completion.exe")
        if not completion.exists():
            raise SystemExit(f"llama-completion.exe is missing beside {executable}")
        prompt = args.prompt or "What is 2 + 2?"
        formatted_prompt = prompt
        chat_flags = ["--conversation", "-st", "--jinja", "--reasoning", "off", "--reasoning-budget", "0"]
        if args.prompt_format == "qwen-nothink":
            # Official Qwen enable_thinking=false prefix, also used by backpack_llm.
            formatted_prompt = ("<|im_start|>user\n" + prompt + "<|im_end|>\n"
                                "<|im_start|>assistant\n<think>\n\n</think>\n\n")
            chat_flags = ["--no-conversation"]
        check_command = [str(completion), "-m", str(args.model), "-p", formatted_prompt,
                         "--temp", "0", "-ngl", "99", "--no-display-prompt",
                         *chat_flags, "-c", str(max(1024, args.prompt_tokens + args.generation_tokens)),
                         "-n", str(args.conformance_tokens)]
        # Conversation mode reads stdin when a turn ends without stopping. An
        # inherited console blocks that read forever while the process still
        # holds the GPU, which presents as a wedged driver that resists
        # termination rather than as a timeout. DEVNULL makes the read EOF.
        checked = subprocess.run(check_command, cwd=completion.parent, text=True, encoding="utf-8",
                                 errors="replace", capture_output=True, timeout=300, shell=False,
                                 stdin=subprocess.DEVNULL)
        # --no-display-prompt keeps stdout limited to generated text; llama.cpp
        # diagnostics remain on stderr and must not satisfy an exact-output gate.
        output = final_answer(checked.stdout)
        passed = checked.returncode == 0 and conformance_passed(
            output, args.required_fact, args.expected_output or "")
        print("EVOLUTION_CONFORMANCE " + json.dumps({
            "passed": passed, "prompt": prompt, "required_fact": args.required_fact,
            "expected_output": args.expected_output, "output": output[-4000:],
            "revision": revision, "command": check_command,
        }, separators=(",", ":")))
        if not passed:
            return checked.returncode or 2
    # llama-bench -p P -n G runs independent pp and tg tests; tg starts at
    # context depth zero. Populate P tokens outside the timed decode region
    # so the result can be compared with Backpack's decode after prefill.
    common = [str(executable), "-m", str(args.model), "-r", str(args.repetitions),
              "-ngl", "99", "-o", "json"]
    commands = [common + ["-p", str(args.prompt_tokens), "-n", "0", "-d", "0"],
                common + ["-p", "0", "-n", str(args.generation_tokens), "-d", str(args.prompt_tokens)]]
    rows = []
    for command in commands:
        completed = subprocess.run(command, cwd=executable.parent, text=True, encoding="utf-8",
                                   errors="replace", capture_output=True, timeout=1800, shell=False,
                                   stdin=subprocess.DEVNULL)
        if completed.returncode:
            print(completed.stdout)
            print(completed.stderr)
            return completed.returncode
        rows.extend(json.loads(completed.stdout))
    prefill = next((row for row in rows if (row.get("n_prompt"), row.get("n_gen"), row.get("n_depth"))
                    == (args.prompt_tokens, 0, 0)), None)
    decode = next((row for row in rows if (row.get("n_prompt"), row.get("n_gen"), row.get("n_depth"))
                   == (0, args.generation_tokens, args.prompt_tokens)), None)
    if not prefill or not decode:
        raise SystemExit("llama-bench did not return the requested prefill and populated-context decode records")
    metrics = {
        "prompt_tokens": args.prompt_tokens,
        "generation_tokens": args.generation_tokens,
        "decode_context_tokens": args.prompt_tokens,
        "prefill_tok_s": float(prefill["avg_ts"]),
        "decode_tok_s": float(decode["avg_ts"]),
        "prefill_stddev_tok_s": float(prefill.get("stddev_ts", 0)),
        "decode_stddev_tok_s": float(decode.get("stddev_ts", 0)),
        "prefill_samples_tok_s": prefill.get("samples_ts", []),
        "decode_samples_tok_s": decode.get("samples_ts", []),
    }
    print("EVOLUTION_METRICS " + json.dumps(metrics, separators=(",", ":")))
    print(f"LLAMACPP_REVISION {revision}")
    print(f"RUNTIME_ARTIFACT {executable.parent}")
    print("RUNTIME_COMMAND " + json.dumps(commands, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
