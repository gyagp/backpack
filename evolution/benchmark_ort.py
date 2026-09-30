from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
import os
import shutil
import hashlib
from pathlib import Path
from datetime import datetime, timezone


def build_metadata(bin_dir: Path) -> dict:
    manifest_path = bin_dir / "build-manifest.json"
    if manifest_path.is_file():
        return json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    metadata = json.loads((bin_dir / "build-metadata.json").read_text(encoding="utf-8-sig"))
    return {
        "onnxruntime_revision": metadata["repositories"]["onnxruntime"]["commit"],
        "onnxruntime_genai_revision": metadata["repositories"]["onnxruntime-genai"]["commit"],
        "date": metadata["builtAt"],
    }


def find_bin_dir(root: Path) -> Path:
    candidates = [p.parent for p in root.glob("*/model_benchmark.exe")
                  if ((p.parent / "model_chat.exe").is_file() or
                      (p.parent / "genai_state_reference.exe").is_file())]
    if not candidates:
        raise RuntimeError(f"No complete ORT GenAI backup found below {root}")
    def date_key(path: Path) -> float:
        date = datetime.fromisoformat(str(build_metadata(path)["date"]).replace("Z", "+00:00"))
        return (date if date.tzinfo else date.replace(tzinfo=timezone.utc)).timestamp()
    return max(candidates, key=date_key)


def revision(bin_dir: Path) -> str:
    manifest = build_metadata(bin_dir)
    ort = str(manifest["onnxruntime_revision"])[:10]
    genai = str(manifest["onnxruntime_genai_revision"])[:10]
    date = str(manifest["date"])[:10].replace("-", "")
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

    # Media encoders may disable capture independently of the text decoder.
    return visit(config.get("model", {}).get("decoder", {}).get("session_options", {}))


def require_warm_rewind_support(model: Path, metadata: dict) -> None:
    """Exclude reproduced native rewind failures from automatic warm results."""
    model_type = json.loads((model / "genai_config.json").read_text(encoding="utf-8-sig"))["model"]["type"]
    # This pinned MultiModalPipelineState inherits the no-op State::RewindTo.
    # Arithmetic/sky repeats fail although the all-A benchmark can match.
    # Evidence: docs/webgfx-104-enablement.md, Qwen3.8 reference/rewind audit.
    if (metadata["onnxruntime_genai_revision"] == "66a5cacc86171d866e2125846521a32e17c7efce"
            and model_type in {"gemma4", "qwen3_5"}):
        raise RuntimeError(
            "Native GenAI multimodal rewind is not conformant for this revision/model; "
            "reused-generator warm benchmarks are disabled until reset is repaired")


def parse_benchmark(output: str) -> tuple[float, float]:
    prefill = re.search(r"Prompt processing.*?avg \(tokens/s\):\s*([0-9.]+)", output, re.S)
    decode = re.search(r"Token generation.*?avg \(tokens/s\):\s*([0-9.]+)", output, re.S)
    if not prefill or not decode:
        raise RuntimeError("model_benchmark output did not contain separate prefill/decode TPS")
    return float(prefill.group(1)), float(decode.group(1))


def chat_answer(output: str) -> str:
    """Extract only the generated response from model_chat's report."""
    match = re.search(r"(?m)^Output:\s*(.*?)(?:\r?\n-{3,}|\Z)", output, re.S)
    answer = match.group(1) if match else output
    if "</think>" in answer:
        answer = answer.rsplit("</think>", 1)[1]
    return answer.strip()


def run(command: list[str], cwd: Path, timeout: int, environment: dict[str, str] | None = None) -> str:
    # model_chat.exe is an interactive binary held non-interactive only by a
    # flag. Closing stdin means a future flag change degrades into an EOF
    # rather than a child blocked on a console read while holding the GPU.
    completed = subprocess.run(command, cwd=cwd, text=True, encoding="utf-8", errors="replace",
                               capture_output=True, timeout=timeout, shell=False,
                               stdin=subprocess.DEVNULL,
                               env={**os.environ, **environment} if environment else None)
    output = completed.stdout + "\n" + completed.stderr
    if completed.returncode:
        raise RuntimeError(f"command exited {completed.returncode}:\n{output[-4000:]}")
    return output


def validate_reference_view(source: Path, view: Path) -> None:
    """A native configuration view may not replace graphs, weights or tokenization."""
    checked = 0
    for original in source.rglob("*"):
        if not original.is_file() or not (".onnx" in original.name or original.name in {
                "tokenizer.json", "tokenizer_config.json", "config.json"}):
            continue
        prepared = view / original.relative_to(source)
        if not prepared.is_file():
            raise RuntimeError(f"Reference view is missing original artifact: {original.name}")
        if not os.path.samefile(original, prepared):
            def digest(path: Path) -> str:
                with path.open("rb") as stream:
                    return hashlib.file_digest(stream, "sha256").hexdigest()
            if digest(original) != digest(prepared):
                raise RuntimeError(f"Reference view changed original artifact: {original.name}")
        checked += 1
    if not checked:
        raise RuntimeError("Source model has no artifacts to validate against the reference view")


def qwen_static_capture_environment(model: Path, bin_dir: Path, metadata: dict,
                                    prompt_tokens: int, generation_tokens: int) -> dict[str, str]:
    if "qwen38_static_capture_b1_c640" not in metadata.get("capabilities", []):
        raise RuntimeError("Selected native package does not declare validated Qwen3.8 static capture")
    config = json.loads((model / "genai_config.json").read_text(encoding="utf-8-sig"))
    values, search = config["model"], config["search"]
    decoder = values["decoder"]
    if ((values.get("type"), decoder.get("num_hidden_layers"), decoder.get("hidden_size"), values.get("vocab_size"))
            != ("qwen3_5", 64, 5120, 248320) or values.get("context_length") != 640
            or search.get("max_length") != 640 or search.get("batch_size", 1) != 1
            or search.get("num_beams", 1) != 1 or search.get("chunk_size") != 32
            or graph_capture_enabled(model) is not True):
        raise RuntimeError("Qwen static capture requires the validated batch1/context640/chunk32 model view")
    if prompt_tokens + generation_tokens != 640:
        raise RuntimeError("Qwen static capture benchmark requires total context capacity640")
    hashes = metadata.get("artifact_hashes", metadata.get("artifacts", {}))
    if not isinstance(hashes, dict):
        raise RuntimeError("Native capture package must provide an artifact hash mapping")
    for name in ["onnxruntime.dll", "onnxruntime-genai.dll", "genai_state_reference.exe", "model_benchmark.exe", "dxcompiler.dll"]:
        path = bin_dir / name
        if not path.is_file() or hashes.get(name) != hashlib.sha256(path.read_bytes()).hexdigest():
            raise RuntimeError(f"Native capture package artifact hash differs: {name}")
    return {"ORTGENAI_QWEN_STATIC_CAPTURE": "1"}


def prepare_native_model(model: Path, evidence_root: Path) -> Path:
    """Pin native D3D12 selection without changing the source model package."""
    evidence_root.mkdir(parents=True, exist_ok=True)
    original = model / "genai_config.json"
    config = json.loads(original.read_text(encoding="utf-8-sig"))
    providers = config.get("model", {}).get("decoder", {}).get("session_options", {}).get("provider_options", [])
    if not any(any(key.lower() == "webgpu" for key in provider) for provider in providers):
        raise RuntimeError("Decoder is not configured for WebGPU")
    target = Path(tempfile.mkdtemp(prefix="ort-model-", dir=evidence_root))
    for source in model.rglob("*"):
        if not source.is_file() or source == original:
            continue
        dest = target / source.relative_to(model)
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source, dest)
        except OSError:
            shutil.copy2(source, dest)
    def set_backend(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key.lower() == "webgpu" and isinstance(child, dict):
                    child["dawnBackendType"] = "D3D12"
                    child["powerPreference"] = "high-performance"
                    # GenAI's initialization session does not forward this
                    # selector to ORT's process-wide WebGPU context.
                    child.pop("adapterIndex", None)
                else:
                    set_backend(child)
        elif isinstance(value, list):
            for child in value:
                set_backend(child)
    set_backend(config)
    # This file is independently written, never hard-linked to the source.
    (target / "genai_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (target / "reference-view.json").write_text(json.dumps({"source": str(model),
        "source_config_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
        "overrides": {"dawnBackendType": "D3D12", "powerPreference": "high-performance"}}, indent=2), encoding="utf-8")
    return target


def artifact_fingerprint(model: Path) -> str:
    entries = []
    for path in sorted(model.rglob("*")):
        if path.is_file() and (".onnx" in path.name or path.name in {
                "genai_config.json", "tokenizer.json", "tokenizer_config.json"}):
            with path.open("rb") as file:
                digest = hashlib.file_digest(file, "sha256").hexdigest()
            entries.append((path.relative_to(model).as_posix(), digest))
    if not entries:
        raise RuntimeError("No model artifacts found for fingerprinting")
    return hashlib.sha256(json.dumps(entries, separators=(",", ":")).encode()).hexdigest()


def benchmark_token_counts(output: str) -> tuple[int, int]:
    match = re.search(r"prompt tokens:\s*(\d+), tokens to generate:\s*(\d+)", output)
    if not match:
        raise RuntimeError("Benchmark did not report actual token counts")
    return int(match[1]), int(match[2])


def run_conformance(bin_dir: Path, model: Path, prompt: str, capture: bool | None,
                    evidence_root: Path, *, capacity: int = 1024,
                    environment: dict[str, str] | None = None) -> tuple[str, list[str]]:
    reference = bin_dir / "genai_state_reference.exe"
    if reference.is_file():
        evidence_root.mkdir(parents=True, exist_ok=True)
        directory = Path(tempfile.mkdtemp(prefix="ort-conformance-", dir=evidence_root))
        request = directory / "request.json"
        result = directory / "result.json"
        payload = {"prompt": prompt, "max_new_tokens": 128,
                   "max_seq_len": capacity, "graph_capture": capture is True,
                   "repetitions": 2, "warmup_runs": 1}
        request.write_text(json.dumps(payload), encoding="utf-8")
        command = [str(reference), str(model), str(request), str(result)]
        log = run(command, bin_dir, 300, environment) if environment else run(command, bin_dir, 300)
        (directory / "run.log").write_text(log, encoding="utf-8")
        (directory / "command.json").write_text(json.dumps({"argv": command, "environment": environment or {}}, indent=2), encoding="utf-8")
        data = json.loads(result.read_text(encoding="utf-8"))
        if data.get("graph_capture_requested") is not (capture is True):
            raise RuntimeError("Reference used a different graph-capture setting")
        runs = data.get("runs", [])
        if (data.get("warmup_runs") != 1 or len(runs) != 2 or not data.get("tokens")
                or any(item.get("tokens") != data["tokens"] for item in runs)):
            raise RuntimeError("Native conformance did not preserve its continuation across resets")
        return str(data["text"]).strip(), command
    command = [str(bin_dir / "model_chat.exe"), "-m", str(model),
               "--user_prompt", prompt, "--non_interactive", "-l", "128"]
    return chat_answer(run(command, bin_dir, 300)), command


def parse_comparable_benchmark(output: str, prompt_tokens: int, generation_tokens: int,
                               repetitions: int) -> dict:
    """Count actual timed calls and include the first prediction in prefill."""
    def section(label: str) -> tuple[float, int, int]:
        match = re.search(re.escape(label) + r":\s*\n((?:[ \t]+[^\n]*(?:\n|$))+)", output)
        if not match:
            raise RuntimeError(f"Missing native benchmark section: {label}")
        body = match.group(1)
        average = re.search(r"avg \(us\):\s*([0-9.eE+-]+)", body)
        count = re.search(r"\bn:\s*(\d+)\s*\*\s*(\d+)", body)
        if not average or not count:
            raise RuntimeError(f"Missing native timing/counts: {label}")
        value = float(average.group(1))
        if not 0 <= value < float("inf"):
            raise RuntimeError(f"Invalid native timing: {label}")
        return value, int(count.group(1)), int(count.group(2))

    prompt_us, prompt_calls, input_count = section("Prompt processing (time to first token)")
    sample_us, sample_calls, sample_count = section("Token sampling")
    decode_us, decode_calls, decode_count = section("Token generation")
    if (prompt_calls, input_count, sample_calls, sample_count, decode_calls, decode_count) != (
            repetitions, prompt_tokens, repetitions, 1, repetitions * (generation_tokens - 1), 1):
        raise RuntimeError("Native timed call counts differ from the requested workload")
    if prompt_us + sample_us <= 0 or (generation_tokens > 1 and decode_us <= 0):
        raise RuntimeError("Native benchmark reported zero elapsed time")
    return {"prefill_tok_s": prompt_tokens * 1e6 / (prompt_us + sample_us),
            "decode_tok_s": 1e6 / decode_us if generation_tokens > 1 else 0.0,
            "prompt_processing_ms": prompt_us / 1000,
            "first_token_sample_ms": sample_us / 1000}


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate and benchmark source-built ORT WebGPU")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--required-fact", required=True)
    parser.add_argument("--expected-output")
    parser.add_argument("--allow-disabled-graph-capture", action="store_true")
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--generation-tokens", type=int, default=128)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--benchmark-prompt-file", type=Path)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1] / "gitignore/evolution/backups/ort")
    parser.add_argument("--bin-dir", type=Path)
    parser.add_argument("--reference-model", type=Path,
                        help="Native configuration view preserving the source graphs, weights and tokenizer")
    parser.add_argument("--qwen-static-capture", action="store_true")
    parser.add_argument("--evidence-root", type=Path, default=Path(__file__).resolve().parents[1] / "gitignore/logs/ort-reference")
    args = parser.parse_args()
    if min(args.prompt_tokens, args.generation_tokens, args.repetitions) <= 0:
        parser.error("Token counts and repetitions must be positive")

    bin_dir = (args.bin_dir or find_bin_dir(args.root)).resolve()
    metadata = build_metadata(bin_dir)
    source_model = args.model.resolve()
    reference_model = (args.reference_model or source_model).resolve()
    if args.reference_model:
        validate_reference_view(source_model, reference_model)
    require_warm_rewind_support(reference_model, metadata)
    runtime_revision = revision(bin_dir)
    capture = graph_capture_enabled(reference_model)
    if capture is not True and not args.allow_disabled_graph_capture:
        raise RuntimeError("ORT graph capture is not enabled in the model configuration")
    environment = (qwen_static_capture_environment(reference_model, bin_dir, metadata,
                                                    args.prompt_tokens, args.generation_tokens)
                   if args.qwen_static_capture else None)
    fingerprint = artifact_fingerprint(source_model)
    args.model = prepare_native_model(reference_model, args.evidence_root.resolve())
    config = json.loads((args.model / "genai_config.json").read_text(encoding="utf-8-sig"))
    capacity = min(1024, int(config["model"].get("context_length", 1024)))
    answer, chat_command = run_conformance(bin_dir, args.model, args.prompt, capture,
                                          args.evidence_root.resolve(), capacity=capacity, environment=environment)
    passed = (answer == args.expected_output.strip() if args.expected_output is not None
              else args.required_fact.lower() in answer.lower())
    print("EVOLUTION_CONFORMANCE " + json.dumps({
        "passed": passed, "prompt": args.prompt, "required_fact": args.required_fact,
        "expected_output": args.expected_output, "output": answer[-4000:],
        "revision": runtime_revision, "command": chat_command,
        "graph_capture": "enabled" if capture else "disabled-temporary",
        "artifact_fingerprint": fingerprint,
        "reference_model": str(reference_model), "reference_environment": environment or {},
    }, separators=(",", ":")))
    if not passed:
        return 2

    prompt_file = args.benchmark_prompt_file
    if prompt_file is None:
        # Avoid generating and then retokenizing a model-dependent prompt.
        # The count reported by the native tokenizer is checked below.
        prompt_file = args.model / "benchmark-prompt.txt"
        prompt_file.write_text(("A " * args.prompt_tokens).rstrip(), encoding="utf-8")
    prompt_file = prompt_file.resolve()
    benchmark_command = [str(bin_dir / "model_benchmark.exe"), "-i", str(args.model),
                         "--prompt_file", str(prompt_file), "-g", str(args.generation_tokens),
                         "-r", str(args.repetitions), "--reuse_generator"]
    benchmark = (run(benchmark_command, bin_dir, 1800, environment) if environment
                 else run(benchmark_command, bin_dir, 1800))
    args.evidence_root.mkdir(parents=True, exist_ok=True)
    benchmark_dir = Path(tempfile.mkdtemp(prefix="ort-benchmark-", dir=args.evidence_root.resolve()))
    (benchmark_dir / "run.log").write_text(benchmark, encoding="utf-8")
    (benchmark_dir / "command.json").write_text(json.dumps({"argv": benchmark_command,
        "revision": runtime_revision, "graph_capture": capture, "repetitions": args.repetitions,
        "environment": environment or {}, "source_model": str(source_model),
        "reference_model": str(reference_model)}, indent=2), encoding="utf-8")
    if benchmark_token_counts(benchmark) != (args.prompt_tokens, args.generation_tokens):
        raise RuntimeError("Actual benchmark token counts differ from the requested workload")
    comparable = parse_comparable_benchmark(benchmark, args.prompt_tokens,
                                            args.generation_tokens, args.repetitions)
    print("EVOLUTION_METRICS " + json.dumps({
        "prompt_tokens": args.prompt_tokens, "generation_tokens": args.generation_tokens,
        **comparable,
        "graph_capture": capture, "reuse_generator": True, "backend": "d3d12",
        "prompt_sha256": hashlib.sha256(prompt_file.read_bytes()).hexdigest(),
        "artifact_fingerprint": fingerprint,
        "benchmark_protocol": ("llm-fixed-text-reuse-first-token-v2-chunk32" if environment
                               else "llm-fixed-text-reuse-first-token-v2"),
        "prefill_chunk": int(config.get("search", {}).get("chunk_size", 0)),
        "reference_environment": environment or {},
        "prefill_includes_first_token": True,
        "warmup_runs": 1, "max_seq_len": args.prompt_tokens + args.generation_tokens,
        "decode_sample_tokens": args.generation_tokens - 1,
    }, separators=(",", ":")))
    print(f"RUNTIME_REVISION {runtime_revision}")
    print(f"RUNTIME_ARTIFACT {bin_dir}")
    print(f"RUNTIME_EVIDENCE {benchmark_dir}")
    print("RUNTIME_COMMAND " + json.dumps(benchmark_command, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
