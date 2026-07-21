from __future__ import annotations

import argparse
import json
import mimetypes
import os
import queue
import re
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlparse

from .domain import DomainError
from .policy import PolicyEngine
from .milestone import MilestonePublisher
from .provisioner import provision_winrm
from .store import Store


ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = Path(__file__).with_name("static")
DEFAULT_DB = ROOT / "gitignore" / "evolution" / "state.db"


class EventBus:
    def __init__(self) -> None:
        self._subscribers: set[queue.Queue[dict[str, Any]]] = set()
        self._lock = threading.Lock()

    def publish(self, event: str, data: dict[str, Any]) -> None:
        with self._lock:
            for subscriber in list(self._subscribers):
                try:
                    subscriber.put_nowait({"event": event, "data": data})
                except queue.Full:
                    self._subscribers.discard(subscriber)

    def subscribe(self) -> queue.Queue[dict[str, Any]]:
        result: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=100)
        with self._lock:
            self._subscribers.add(result)
        return result

    def unsubscribe(self, subscriber: queue.Queue[dict[str, Any]]) -> None:
        with self._lock:
            self._subscribers.discard(subscriber)


class EvolutionServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], store: Store):
        super().__init__(address, Handler)
        self.store = store
        self.policy = PolicyEngine(store)
        self.events = EventBus()
        self.milestones = MilestonePublisher(store, ROOT)
        catalog_path = Path(__file__).with_name("models.json")
        for model in json.loads(catalog_path.read_text(encoding="utf-8")):
            self.store.upsert_model(model)
        seed_path = Path(__file__).with_name("seed-observations.json")
        if seed_path.is_file():
            machines = {machine["name"]: machine["id"] for machine in self.store.list_machines()}
            for observation in json.loads(seed_path.read_text(encoding="utf-8")):
                item = dict(observation)
                machine_id = machines.get(item.pop("machine_name"))
                if machine_id:
                    item["machine_id"] = machine_id
                    self.store.add_observation(item, "documented-status-import")
        learning_path = Path(__file__).with_name("learned_tasks.json")
        if learning_path.is_file():
            existing_keys = {task.get("origin", {}).get("automation_key") for task in self.store.list_tasks()}
            for learned in json.loads(learning_path.read_text(encoding="utf-8")):
                key = f"learning:{learned['key']}"
                if key in existing_keys:
                    continue
                self.store.create_task({
                    "title": learned["title"], "kind": learned["kind"],
                    "hypothesis": learned["hypothesis"],
                    "origin": {"type": "continuous-learning", "automation_key": key,
                               "source": learned.get("source", {})},
                    "manifest": learned.get("manifest", {}),
                    "device_policy": learned.get("device_policy", {}),
                }, "continuous-learning")
        self.store.ensure_automatic_tasks()
        self.store.ensure_runnable_automatic_tasks()
        self.store.ensure_task_runs()
        self.store.ensure_daily_upstream_tasks()
        studies_path = Path(__file__).with_name("studies.json")
        if studies_path.is_file():
            for study in json.loads(studies_path.read_text(encoding="utf-8")):
                self.store.add_learning_study(study, "learning-agent")


class Handler(BaseHTTPRequestHandler):
    server: EvolutionServer

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[evolution] {self.address_string()} {fmt % args}")

    def _actor(self) -> str:
        return self.headers.get("X-Evolution-Actor", "local-user")[:100]

    def _json_body(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 2 * 1024 * 1024:
                raise DomainError("request body is too large")
            value = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(value, dict):
                raise DomainError("JSON body must be an object")
            return value
        except json.JSONDecodeError as exc:
            raise DomainError(f"invalid JSON: {exc.msg}") from exc

    def _send_json(self, value: Any, status: int = 200) -> None:
        data = json.dumps(value, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _error(self, status: int, message: str) -> None:
        self._send_json({"error": message}, status)

    def do_GET(self) -> None:
        try:
            path = urlparse(self.path).path
            if path == "/api/status":
                return self._send_json(self.server.store.status())
            if path == "/api/activity":
                self.server.store.ensure_daily_upstream_tasks()
                query = parse_qs(urlparse(self.path).query)
                return self._send_json(self.server.store.activity(int((query.get("limit") or [40])[0])))
            if path == "/api/tasks":
                return self._send_json(self.server.store.list_tasks())
            match = re.fullmatch(r"/api/tasks/([^/]+)", path)
            if match:
                task = self.server.store.task_detail(match.group(1))
                return self._send_json(task) if task else self._error(404, "task not found")
            if path == "/api/machines":
                return self._send_json(self.server.store.list_machines())
            if path == "/api/models":
                return self._send_json(self.server.store.list_models())
            if path == "/api/models/matrix":
                return self._send_json(self.server.store.model_matrix())
            if path == "/api/observations":
                query = parse_qs(urlparse(self.path).query)
                return self._send_json(self.server.store.list_observations(
                    {key: values[0] for key, values in query.items() if values}))
            if path == "/api/models/manifest":
                return self._send_json(self._model_manifest())
            match = re.fullmatch(r"/api/models/([^/]+)/files/(gguf|ort)(?:/(.+))?", path)
            if match:
                return self._model_file(match.group(1), match.group(2), match.group(3))
            if path == "/api/history":
                return self._send_json(self.server.store.list_history())
            if path == "/api/todos/dismissed":
                return self._send_json(self.server.store.list_todo_dismissals())
            if path == "/api/regressions/confirmed":
                return self._send_json(self.server.store.confirmed_regressions())
            if path == "/api/studies":
                return self._send_json(self.server.store.list_learning_studies())
            if path == "/api/runs":
                return self._send_json(self.server.store.list_runs())
            if path == "/api/milestones":
                return self._send_json(self.server.store.list_milestones())
            if path == "/api/milestones/current":
                return self._send_json(self.server.store.current_milestone())
            if path == "/api/decisions":
                query = parse_qs(urlparse(self.path).query)
                return self._send_json(self.server.store.list_decisions((query.get("status") or [None])[0]))
            if path == "/api/events":
                return self._events()
            if path == "/api/agent.py":
                return self._source_file(Path(__file__).with_name("agent.py"), "text/x-python")
            if path == "/api/bootstrap.ps1":
                return self._source_file(Path(__file__).with_name("bootstrap-agent.ps1"), "text/plain")
            return self._static(path)
        except (DomainError, ValueError) as exc:
            self._error(400, str(exc))
        except Exception as exc:
            self._error(500, f"internal error: {exc}")

    def do_POST(self) -> None:
        try:
            path, body, actor = urlparse(self.path).path, self._json_body(), self._actor()
            if path == "/api/tasks":
                result = self.server.store.create_task(body, actor)
                self.server.store.ensure_task_runs()
                self.server.events.publish("task-created", result)
                return self._send_json(result, HTTPStatus.CREATED)
            match = re.fullmatch(r"/api/tasks/([^/]+)/transition", path)
            if match:
                task_id, target = match.group(1), body.get("state", "")
                milestone = None
                if target == "integrated":
                    milestone = self.server.milestones.publish(task_id)
                result = self.server.store.transition_task(task_id, target, actor, body.get("reason", ""))
                self.server.events.publish("task-updated", result)
                if milestone:
                    self.server.events.publish("milestone-published", milestone)
                return self._send_json({"task": result, "milestone": milestone} if milestone else result)
            match = re.fullmatch(r"/api/tasks/([^/]+)/candidate", path)
            if match:
                result = self.server.store.set_candidate(match.group(1), body.get("base_sha", ""), body.get("candidate_sha", ""), actor)
                self.server.events.publish("task-updated", result)
                return self._send_json(result)
            if path == "/api/evidence":
                result = self.server.store.add_evidence(body, actor)
                self.server.events.publish("evidence-added", {"task_id": result["task_id"], "id": result["id"]})
                return self._send_json(result, HTTPStatus.CREATED)
            if path == "/api/observations":
                result = self.server.store.add_observation(body, actor)
                created = self.server.store.ensure_automatic_tasks()
                self.server.store.ensure_runnable_automatic_tasks()
                self.server.store.ensure_task_runs()
                self.server.events.publish("observation-added", result)
                for task in created:
                    self.server.events.publish("task-created", task)
                return self._send_json(result, HTTPStatus.CREATED)
            if path == "/api/runs/claim":
                result = self.server.store.claim_run(str(body.get("machine", "")),
                                                     body.get("capabilities") or [], actor)
                if result:
                    self.server.events.publish("run-updated", result)
                return self._send_json(result)
            match = re.fullmatch(r"/api/runs/([^/]+)", path)
            if match:
                result = self.server.store.update_run(match.group(1), body, actor)
                self.server.events.publish("run-updated", result)
                return self._send_json(result)
            if path == "/api/history":
                result = self.server.store.add_history(body, actor)
                self.server.events.publish("history-added", result)
                return self._send_json(result, HTTPStatus.CREATED)
            if path == "/api/todos/dismiss":
                ids = body.get("ids") or []
                if not isinstance(ids, list):
                    raise DomainError("todo ids must be an array")
                return self._send_json(self.server.store.dismiss_todos(ids, actor))
            if path == "/api/studies":
                result = self.server.store.add_learning_study(body, actor)
                self.server.events.publish("study-completed", result)
                return self._send_json(result, HTTPStatus.CREATED)
            match = re.fullmatch(r"/api/tasks/([^/]+)/evaluate", path)
            if match:
                result = self.server.policy.evaluate(match.group(1))
                self.server.events.publish("task-evaluated", {"task_id": match.group(1), **result})
                return self._send_json(result)
            if path == "/api/machines/register":
                result = self.server.store.register_machine(body)
                self.server.events.publish("machine-updated", result)
                return self._send_json(result)
            if path == "/api/machines/configure":
                result = self.server.store.configure_machine(body, actor)
                self.server.events.publish("machine-updated", result)
                return self._send_json(result, HTTPStatus.CREATED)
            match = re.fullmatch(r"/api/machines/([^/]+)/activity", path)
            if match:
                action = body.get("action")
                if action not in {"pause", "resume"}:
                    raise DomainError("activity action must be pause or resume")
                result = self.server.store.set_machine_activity(match.group(1), action == "pause", actor)
                self.server.events.publish("machine-updated", result)
                return self._send_json(result)
            if path == "/api/machines/provision":
                name = str(body.get("name", "")).strip().lower()
                if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,62}", name):
                    raise DomainError("device name must contain only letters, numbers, and hyphens")
                existing = next((item for item in self.server.store.list_machines()
                                 if item["name"].lower() == name), None)
                address = str(body.get("address") or (existing or {}).get("labels", {}).get("address") or "").strip()
                fingerprint = body.get("fingerprint") or (existing or {}).get("fingerprint", {})
                labels = {**(existing or {}).get("labels", {}), **(body.get("labels") or {}), "role": "required"}
                if address:
                    labels["address"] = address
                labels["inventory_status"] = "awaiting-enrollment"
                result = self.server.store.configure_machine({
                    "name": name, "status": "offline", "fingerprint": fingerprint, "labels": labels,
                }, actor)
                server_url = os.environ.get("BP_EVOLUTION_PUBLIC_URL", "http://10.172.21.28:8787").rstrip("/")
                target = address or name
                provision = provision_winrm(target, server_url)
                self.server.events.publish("machine-updated", result)
                return self._send_json({"machine": result, "provision": provision}, HTTPStatus.CREATED)
            match = re.fullmatch(r"/api/decisions/([^/]+)/resolve", path)
            if match:
                result = self.server.store.resolve_decision(match.group(1), body.get("status", ""), actor, body.get("rationale", ""))
                self.server.events.publish("decision-resolved", result)
                return self._send_json(result)
            self._error(404, "endpoint not found")
        except (DomainError, ValueError) as exc:
            self._error(400, str(exc))
        except Exception as exc:
            self._error(500, f"internal error: {exc}")

    def do_DELETE(self) -> None:
        try:
            path, actor = urlparse(self.path).path, self._actor()
            match = re.fullmatch(r"/api/tasks/([^/]+)", path)
            if not match:
                return self._error(404, "not found")
            if not self.server.store.delete_task(match.group(1), actor):
                return self._error(404, "task not found")
            return self._send_json({"deleted": True, "id": match.group(1)})
        except Exception as exc:
            self._error(500, f"internal error: {exc}")

    def _events(self) -> None:
        subscriber = self.server.events.subscribe()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        try:
            self.wfile.write(b"event: connected\ndata: {}\n\n")
            self.wfile.flush()
            while True:
                try:
                    message = subscriber.get(timeout=20)
                    line = f"event: {message['event']}\ndata: {json.dumps(message['data'])}\n\n"
                except queue.Empty:
                    line = ": keepalive\n\n"
                self.wfile.write(line.encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            self.server.events.unsubscribe(subscriber)

    def _static(self, path: str) -> None:
        app_routes = {
            "", "/", "/tasks", "/evolution", "/human-intervention",
            "/status", "/performance-analysis", "/history", "/devices",
        }
        relative = "index.html" if path in app_routes else path.lstrip("/")
        candidate = (STATIC_DIR / relative).resolve()
        if STATIC_DIR.resolve() not in candidate.parents or not candidate.is_file():
            return self._error(404, "not found")
        data = candidate.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(candidate.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _model_manifest(self) -> list[dict[str, Any]]:
        result = []
        for model in self.server.store.list_models(cared_only=True):
            files = {}
            for fmt, entry in model["files"].items():
                source = Path(entry["path"])
                if source.is_file():
                    files[fmt] = {"name": source.name, "size": source.stat().st_size,
                                  "relative_dir": entry.get("relative_dir", model["id"]),
                                  "download_url": f"/api/models/{model['id']}/files/{fmt}"}
                elif source.is_dir():
                    entries = []
                    for child in sorted(path for path in source.rglob("*") if path.is_file()):
                        relative = child.relative_to(source).as_posix()
                        entries.append({"path": relative, "size": child.stat().st_size,
                                        "download_url": f"/api/models/{model['id']}/files/{fmt}/{quote(relative)}"})
                    files[fmt] = {"relative_dir": entry.get("relative_dir", model["id"]),
                                  "size": sum(item["size"] for item in entries), "entries": entries}
            result.append({"id": model["id"], "name": model["name"], "files": files})
        return result

    def _model_file(self, model_id: str, fmt: str, relative: str | None = None) -> None:
        model = self.server.store.get_model(model_id)
        entry = (model or {}).get("files", {}).get(fmt)
        source = Path(entry["path"]) if entry else None
        if source and source.is_dir() and relative:
            root = source.resolve()
            candidate = (root / unquote(relative)).resolve()
            if root not in candidate.parents:
                return self._error(400, "invalid model path")
            source = candidate
        if not source or not source.is_file():
            return self._error(404, "model format not available")
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(source.stat().st_size))
        self.send_header("Content-Disposition", f'attachment; filename="{source.name}"')
        self.end_headers()
        with source.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                self.wfile.write(chunk)

    def _source_file(self, source: Path, content_type: str) -> None:
        data = source.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type + "; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Backpack evolution control plane")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    args = parser.parse_args(argv)
    store = Store(args.db)
    server = EvolutionServer((args.host, args.port), store)
    print(f"Backpack evolution dashboard: http://{args.host}:{args.port}")
    print(f"State database: {args.db}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
