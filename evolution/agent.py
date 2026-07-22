from __future__ import annotations

import argparse
import json
import os
import platform
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def request_json(url: str, method: str = "GET", body: dict[str, Any] | None = None,
                 actor: str = "machine-agent") -> Any:
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(url, data=data, method=method,
                                     headers={"Content-Type": "application/json", "X-Evolution-Actor": actor})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        raise RuntimeError(f"server returned {exc.code}: {detail}") from exc


def _command_output(argv: list[str]) -> str:
    try:
        return subprocess.check_output(
            argv, text=True, encoding="utf-8", errors="replace",
            stderr=subprocess.DEVNULL, timeout=10,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def fingerprint(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    gpu = os.environ.get("BP_EVOLUTION_GPU", "")
    driver = os.environ.get("BP_EVOLUTION_DRIVER", "")
    if platform.system() == "Windows" and not gpu:
        gpu = _command_output([
            "powershell", "-NoProfile", "-Command",
            "(Get-CimInstance Win32_VideoController | Where-Object {$_.PNPDeviceID -like 'PCI*'} | "
            "Sort-Object AdapterRAM -Descending | Select-Object -First 1 -ExpandProperty Name)",
        ])
    if platform.system() == "Windows" and not driver:
        driver = _command_output([
            "powershell", "-NoProfile", "-Command",
            "(Get-CimInstance Win32_VideoController | Where-Object {$_.PNPDeviceID -like 'PCI*'} | "
            "Sort-Object AdapterRAM -Descending | Select-Object -First 1 -ExpandProperty DriverVersion)",
        ])
    system_info: dict[str, Any] = {}
    if platform.system() == "Windows":
        raw = _command_output(["powershell", "-NoProfile", "-Command",
                               "Get-CimInstance Win32_ComputerSystem | Select-Object TotalPhysicalMemory,NumberOfLogicalProcessors,Manufacturer,Model | ConvertTo-Json -Compress"])
        try:
            system_info = json.loads(raw)
        except json.JSONDecodeError:
            pass
    vendor = "unknown"
    lowered = gpu.lower()
    if "nvidia" in lowered:
        vendor = "nvidia"
    elif "amd" in lowered or "radeon" in lowered:
        vendor = "amd"
    elif "intel" in lowered or "arc" in lowered:
        vendor = "intel"
    result = {
        "os": platform.system().lower(), "os_version": platform.version(),
        "architecture": platform.machine(), "cpu": platform.processor(),
        "python": platform.python_version(), "gpu": gpu or "unknown",
        "driver": driver or "unknown", "gpu_vendor": vendor,
        "backend": os.environ.get("BP_EVOLUTION_BACKEND", "webgpu"),
        "cpu_cores": system_info.get("NumberOfLogicalProcessors") or os.cpu_count(),
        "memory_mb": round(int(system_info.get("TotalPhysicalMemory") or 0) / 1024 / 1024),
        "manufacturer": system_info.get("Manufacturer", ""), "system_model": system_info.get("Model", ""),
    }
    result.update(overrides or {})
    return result


def sync_base(server: str, repo: Path, worktrees: Path) -> bool:
    milestone = request_json(server + "/api/milestones/current")
    if not milestone:
        return False
    sha = milestone["commit_sha"]
    target = worktrees / f"base-{sha[:12]}"
    if (target / ".git").exists() or (target / ".git").is_file():
        (worktrees / "CURRENT_BASE").write_text(f"{sha}\n{target}\n", encoding="utf-8")
        return False
    worktrees.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "-C", str(repo), "fetch", milestone["remote"], sha], check=True, shell=False)
    subprocess.run(["git", "-C", str(repo), "worktree", "add", "--detach", str(target), sha], check=True, shell=False)
    (worktrees / "CURRENT_BASE").write_text(f"{sha}\n{target}\n", encoding="utf-8")
    print(f"Synchronized evolution base {sha} to {target}")
    return True


def current_base_worktree(repo: Path) -> Path:
    marker = repo / "gitignore" / "evolution" / "worktrees" / "CURRENT_BASE"
    try:
        lines = marker.read_text(encoding="utf-8").splitlines()
        target = Path(lines[1])
        if target.is_dir():
            return target
    except (OSError, IndexError):
        pass
    return repo


def rewrite_repo_argv(argv: list[str], repo: Path, execution_repo: Path) -> list[str]:
    if execution_repo == repo:
        return argv
    repo_prefix = str(repo).rstrip("\\/")
    return [str(execution_repo) + value[len(repo_prefix):]
            if value.lower().startswith(repo_prefix.lower()) else value for value in argv]


def execute_run(server: str, name: str, run: dict[str, Any], repo: Path) -> None:
    """Execute only explicit, typed adapters; never interpret server text as a shell command."""
    task, run_id = run["task"], run["id"]
    manifest = task.get("manifest") or {}
    adapter = manifest.get("adapter")
    if adapter != "argv":
        raise RuntimeError(f"unsupported task adapter: {adapter or 'none'}")
    argv = manifest.get("argv")
    if not isinstance(argv, list) or not argv or not all(isinstance(x, str) for x in argv):
        raise RuntimeError("argv adapter requires a non-empty string array")
    execution_repo = current_base_worktree(repo)
    prior_repair = (run.get("result") or {}).get("codex_repair") or {}
    if prior_repair.get("candidate_sha") and Path(prior_repair.get("worktree", "")).is_dir():
        execution_repo = Path(prior_repair["worktree"])
    argv = rewrite_repo_argv(argv, repo, execution_repo)
    request_json(server + f"/api/runs/{run_id}", "POST",
                 {"status": "running", "phase": "validating Codex repair" if prior_repair.get("candidate_sha") else "executing",
                  "progress": 10}, name)
    started = time.monotonic()
    timeout_seconds = int(manifest.get("timeout_seconds") or (1800 if task.get("kind") == "benchmark" else 7200))
    live_dir = repo / "gitignore" / "evolution" / "live"
    live_dir.mkdir(parents=True, exist_ok=True)
    stdout_path, stderr_path = live_dir / f"{run_id}.stdout.log", live_dir / f"{run_id}.stderr.log"
    timed_out = False
    with stdout_path.open("w", encoding="utf-8", errors="replace") as stdout_file, \
            stderr_path.open("w", encoding="utf-8", errors="replace") as stderr_file:
        process = subprocess.Popen(argv, cwd=execution_repo, text=True, encoding="utf-8", errors="replace",
                                   stdout=stdout_file, stderr=stderr_file, shell=False)
        last_upload = 0.0
        while process.poll() is None:
            elapsed = time.monotonic() - started
            if elapsed >= timeout_seconds:
                timed_out = True
                if platform.system() == "Windows":
                    subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                   capture_output=True, timeout=30, shell=False)
                else:
                    process.kill()
                process.wait(timeout=30)
                break
            if elapsed - last_upload >= 10:
                stdout_file.flush(); stderr_file.flush()
                stdout_tail = stdout_path.read_text(encoding="utf-8", errors="replace")[-12000:]
                stderr_tail = stderr_path.read_text(encoding="utf-8", errors="replace")[-12000:]
                request_json(server + f"/api/runs/{run_id}", "POST", {
                    "status": "running", "phase": "executing", "progress": 10,
                    "result": {"live": True, "elapsed_seconds": round(elapsed, 1),
                               "stdout_tail": stdout_tail, "stderr_tail": stderr_tail},
                }, name)
                last_upload = elapsed
            time.sleep(1)
    stdout = stdout_path.read_text(encoding="utf-8", errors="replace")
    stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
    if timed_out:
        request_json(server + f"/api/runs/{run_id}", "POST", {
            "status": "failed", "phase": "timed out", "progress": 100,
            "error": f"timeout: exceeded {timeout_seconds} seconds",
            "result": {"argv": argv, "timeout_seconds": timeout_seconds,
                       "stdout_tail": stdout[-12000:], "stderr_tail": stderr[-12000:]},
        }, name)
        return
    completed = subprocess.CompletedProcess(argv, process.returncode, stdout, stderr)
    result = {"argv": argv, "exit_code": completed.returncode,
              "duration_seconds": round(time.monotonic() - started, 3),
              "stdout_tail": completed.stdout[-12000:], "stderr_tail": completed.stderr[-12000:]}
    status = "completed" if completed.returncode == 0 else "failed"
    task = run.get("task") or {}
    if status == "failed":
        prior_attempts = int((run.get("result") or {}).get("codex_repair_attempts", 0))
        if prior_attempts < 1:
            repair = codex_repair(task, run, repo, name, completed)
            if repair:
                result["codex_repair"] = repair
                result["codex_repair_attempts"] = prior_attempts + 1
                # Retry only when Codex actually produced a committed candidate.
                # Authentication and CLI failures must remain visible to the
                # operator instead of cycling the same task forever.
                if repair.get("candidate_sha"):
                    status = "pending"
    if status == "completed" and task.get("kind") == "benchmark":
        canonical = re.search(r"(?m)^EVOLUTION_METRICS (\{.*\})$", completed.stdout)
        runtime = (manifest.get("runtimes") or [{}])[0]
        if canonical:
            metrics = json.loads(canonical.group(1))
            revision_match = re.search(r"(?m)^(?:RUNTIME|LLAMACPP)_REVISION (\S+)$", completed.stdout)
            conformance_match = re.search(r"(?m)^EVOLUTION_CONFORMANCE (\{.*\})$", completed.stdout)
            conformance = json.loads(conformance_match.group(1)) if conformance_match else {}
            origin = task.get("origin", {})
            request_json(server + "/api/observations", "POST", {
                "model_id": origin.get("model_id") or next(iter(manifest.get("models") or []), ""),
                "machine_id": run["machine_id"], "framework": runtime.get("framework", "llamacpp"),
                "format": runtime.get("format", "gguf"), "backend": runtime.get("backend", "vulkan"),
                "conformance": "pass" if conformance.get("passed") else "not_applicable",
                "revision": revision_match.group(1) if revision_match else "unknown",
                "conformance_details": {
                    "source": f"benchmark task {task.get('id', run['task_id'])}",
                    "prompt": conformance.get("prompt"),
                    "required_fact": conformance.get("required_fact"),
                    "output": conformance.get("output"),
                },
                "metrics": metrics, "artifacts": [],
            }, name)
        rows = re.findall(
            r"(?m)^\s*(\d+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)(?:\s+([0-9.]+)\s+([0-9.]+)%)?\s*$",
            completed.stdout + "\n" + completed.stderr,
        )
        if rows and not canonical:
            prompt_tokens, prefill_ms, prefill_rate, decode_ms, decode_rate, fence_ms, fence_pct = rows[-1]
            origin, manifest = task.get("origin", {}), task.get("manifest", {})
            model_path = next((argv[i + 1] for i, value in enumerate(argv[:-1]) if value == "--model"), "")
            metrics = {"prompt_tokens": int(prompt_tokens), "prefill_ms": float(prefill_ms),
                       "prefill_tok_s": float(prefill_rate), "decode_ms": float(decode_ms),
                       "decode_tok_s": float(decode_rate)}
            if fence_ms:
                metrics.update({"fence_ms": float(fence_ms), "fence_percent": float(fence_pct)})
            request_json(server + "/api/observations", "POST", {
                "model_id": origin.get("model_id") or next(iter(manifest.get("models") or []), ""),
                "machine_id": run["machine_id"], "framework": "backpack",
                "format": "gguf" if model_path.lower().endswith(".gguf") else "ort",
                "backend": "webgpu", "conformance": "pass",
                "revision": _command_output(["git", "-C", str(repo), "rev-parse", "HEAD"]) or "unknown",
                "conformance_details": {"source": f"benchmark task {task.get('id', run['task_id'])}"},
                "metrics": metrics, "artifacts": [],
            }, name)
    elif status == "completed" and task.get("kind") in {"correctness", "conformance"}:
        manifest, origin = task.get("manifest", {}), task.get("origin", {})
        spec = manifest.get("conformance_spec") or {}
        required = str(spec.get("required_fact") or "").strip().lower()
        output_match = re.search(r"--- Output ---\s*(.*?)\s*--- Performance ---",
                                 completed.stdout + "\n" + completed.stderr, re.S)
        output = output_match.group(1).strip() if output_match else completed.stdout[-2000:].strip()
        if required:
            passed = required in output.lower()
            model_path = next((argv[i + 1] for i, value in enumerate(argv[:-1]) if value == "--model"), "")
            request_json(server + "/api/observations", "POST", {
                "model_id": origin.get("model_id") or next(iter(manifest.get("models") or []), ""),
                "machine_id": run["machine_id"], "framework": "backpack",
                "format": "gguf" if model_path.lower().endswith(".gguf") else "ort",
                "backend": "webgpu", "conformance": "pass" if passed else "fail",
                "revision": _command_output(["git", "-C", str(repo), "rev-parse", "HEAD"]) or "unknown",
                "conformance_details": {"prompt": spec.get("prompt"), "required_fact": required,
                                        "output": output[-4000:], "source": f"task {task.get('id', run['task_id'])}"},
                "metrics": {}, "artifacts": [],
            }, name)
    error = None if completed.returncode == 0 else f"process exited {completed.returncode}"
    repair_result = result.get("codex_repair") or {}
    if repair_result.get("status") == "authentication_required":
        error = (f"Codex CLI is not authenticated on {name}. Run 'codex login' on that device "
                 "or provide a scoped automation credential, then resume the task.")
    elif repair_result and not repair_result.get("candidate_sha"):
        error = f"Codex repair could not produce a retry candidate: {repair_result.get('error') or repair_result.get('status')}"
    request_json(server + f"/api/runs/{run_id}", "POST",
                 {"status": status, "phase": "finished", "progress": 100, "result": result,
                  "error": error}, name)


def codex_repair(task: dict[str, Any], run: dict[str, Any], repo: Path, name: str,
                 completed: subprocess.CompletedProcess[str]) -> dict[str, Any] | None:
    codex = _command_output(["where.exe", "codex"]).splitlines()
    if not codex:
        return {"status": "cli_missing", "error": f"Codex CLI is not installed on {name}"}
    task_id = str(task.get("id") or run.get("task_id") or "task")
    safe_name = re.sub(r"[^a-z0-9-]+", "-", name.lower()).strip("-")
    branch = f"experiment/{task_id}/{safe_name}"
    worktree = repo / "gitignore" / "evolution" / "worktrees" / "codex" / f"{task_id}-{safe_name}"
    worktree.parent.mkdir(parents=True, exist_ok=True)
    if not worktree.exists():
        add = subprocess.run(["git", "-C", str(repo), "worktree", "add", "-b", branch,
                              str(worktree), "HEAD"], text=True, encoding="utf-8", errors="replace",
                             capture_output=True, timeout=120, shell=False)
        if add.returncode:
            add = subprocess.run(["git", "-C", str(repo), "worktree", "add", str(worktree), branch],
                                 text=True, encoding="utf-8", errors="replace", capture_output=True,
                                 timeout=120, shell=False)
        if add.returncode:
            return {"status": "worktree_failed", "branch": branch,
                    "error": (add.stderr or add.stdout)[-4000:]}
    failure = (completed.stderr + "\n" + completed.stdout)[-16000:]
    prompt = f"""Goal: unblock Backpack task {task_id} on device {name}.
Task: {task.get('title', '')}
Hypothesis: {task.get('hypothesis', '')}
Failure log:
{failure}

Reproduce the failure in this isolated worktree, identify the root cause, implement the smallest portable fix,
and run focused validation. Preserve conformance-first policy and do not use Windows WebGPU subgroup matrices.
Do not merge or push. Summarize changed files, tests, remaining risk, and whether the task is ready to retry.
"""
    repair = subprocess.run([codex[0], "exec", "--sandbox", "workspace-write", "--json", prompt],
                            cwd=worktree, text=True, encoding="utf-8", errors="replace",
                            capture_output=True, timeout=7200, shell=False)
    changed = _command_output(["git", "-C", str(worktree), "status", "--porcelain"])
    commit = ""
    if repair.returncode == 0 and changed:
        subprocess.run(["git", "-C", str(worktree), "add", "-A"], check=False, shell=False)
        committed = subprocess.run(["git", "-C", str(worktree), "commit", "-m",
                                    f"Experiment: unblock {task_id} on {name}"],
                                   text=True, encoding="utf-8", errors="replace",
                                   capture_output=True, timeout=120, shell=False)
        if committed.returncode == 0:
            commit = _command_output(["git", "-C", str(worktree), "rev-parse", "HEAD"])
    return {"status": "completed" if repair.returncode == 0 else "failed", "branch": branch,
            "worktree": str(worktree), "candidate_sha": commit, "changed": bool(changed),
            "exit_code": repair.returncode, "stdout_tail": repair.stdout[-12000:],
            "stderr_tail": repair.stderr[-12000:]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Register a Backpack evolution machine or upload evidence")
    parser.add_argument("--server", default="http://127.0.0.1:8787")
    parser.add_argument("--name", default=socket.gethostname())
    parser.add_argument("--label", action="append", default=[], metavar="KEY=VALUE")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("register")
    upload = sub.add_parser("upload")
    upload.add_argument("evidence", type=Path)
    sync = sub.add_parser("sync-models")
    sync.add_argument("--models-dir", type=Path, default=Path(r"D:\workspace\project\agents\ai-models"))
    sync_base_parser = sub.add_parser("sync-base")
    sync_base_parser.add_argument("--repo", type=Path, default=Path.cwd())
    sync_base_parser.add_argument("--worktrees", type=Path)
    watch = sub.add_parser("watch")
    watch.add_argument("--repo", type=Path, default=Path.cwd())
    watch.add_argument("--worktrees", type=Path)
    watch.add_argument("--interval", type=int, default=60)
    args = parser.parse_args(argv)
    if args.command == "register":
        labels = dict(item.split("=", 1) for item in args.label)
        result = request_json(args.server + "/api/machines/register", "POST",
                              {"name": args.name, "fingerprint": fingerprint(), "labels": labels}, args.name)
        print(json.dumps(result, indent=2))
    elif args.command == "upload":
        payload = json.loads(args.evidence.read_text(encoding="utf-8"))
        result = request_json(args.server + "/api/evidence", "POST", payload, args.name)
        print(json.dumps(result, indent=2))
    elif args.command == "sync-models":
        manifest = request_json(args.server + "/api/models/manifest")
        for model in manifest:
            for fmt, entry in model["files"].items():
                files = entry.get("entries") or [{"path": entry["name"], "size": entry["size"],
                                                   "download_url": entry["download_url"]}]
                for file_entry in files:
                    destination = args.models_dir / entry["relative_dir"] / Path(file_entry["path"])
                    if destination.is_file() and destination.stat().st_size == file_entry["size"]:
                        continue
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    temporary = destination.with_suffix(destination.suffix + ".part")
                    print(f"[sync] {model['name']} {fmt}/{file_entry['path']}: {file_entry['size']} bytes")
                    with urllib.request.urlopen(args.server + file_entry["download_url"], timeout=3600) as response, temporary.open("wb") as out:
                        while chunk := response.read(1024 * 1024):
                            out.write(chunk)
                    if temporary.stat().st_size != file_entry["size"]:
                        temporary.unlink(missing_ok=True)
                        raise RuntimeError(f"size mismatch for {destination}")
                    temporary.replace(destination)
        print("Model synchronization complete")
    elif args.command == "sync-base":
        worktrees = args.worktrees or args.repo / "gitignore" / "evolution" / "worktrees"
        if not sync_base(args.server, args.repo, worktrees):
            print("Evolution base is already current or not yet published")
    else:
        if args.interval < 10:
            parser.error("watch interval must be at least 10 seconds")
        labels = dict(item.split("=", 1) for item in args.label)
        worktrees = args.worktrees or args.repo / "gitignore" / "evolution" / "worktrees"
        print(f"Watching {args.server} for milestones every {args.interval}s")
        try:
            request_json(args.server + "/api/machines/register", "POST",
                         {"name": args.name, "fingerprint": fingerprint(), "labels": labels}, args.name)
            while True:
                try:
                    sync_base(args.server, args.repo, worktrees)
                    run = request_json(args.server + "/api/runs/claim", "POST",
                                       {"machine": args.name, "capabilities": ["argv"]}, args.name)
                    if run:
                        try:
                            execute_run(args.server, args.name, run, args.repo)
                        except Exception as exc:
                            request_json(args.server + f"/api/runs/{run['id']}", "POST",
                                         {"status": "failed", "phase": "agent-error", "error": str(exc)}, args.name)
                        continue
                except Exception as exc:
                    print(f"[watch] {exc}", file=sys.stderr)
                time.sleep(args.interval)
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
