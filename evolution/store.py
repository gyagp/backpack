from __future__ import annotations

import sqlite3
import sys
import threading
import uuid
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .domain import DomainError, json_text, parse_json, require, utc_now, validate_transition


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS tasks (
  id TEXT PRIMARY KEY, title TEXT NOT NULL, kind TEXT NOT NULL,
  state TEXT NOT NULL, hypothesis TEXT NOT NULL, origin_json TEXT NOT NULL,
  base_sha TEXT, candidate_sha TEXT, manifest_json TEXT NOT NULL,
  device_policy_json TEXT NOT NULL, decision_policy_json TEXT NOT NULL,
  aggregate_verdict TEXT, verdict_reason TEXT,
  created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS machines (
  id TEXT PRIMARY KEY, name TEXT NOT NULL UNIQUE, status TEXT NOT NULL,
  fingerprint_json TEXT NOT NULL, labels_json TEXT NOT NULL,
  current_run_id TEXT, last_seen_at TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS evidence (
  id TEXT PRIMARY KEY, task_id TEXT NOT NULL REFERENCES tasks(id),
  machine_id TEXT NOT NULL REFERENCES machines(id), variant TEXT NOT NULL,
  metric TEXT NOT NULL, unit TEXT NOT NULL, samples_json TEXT NOT NULL,
  correctness_json TEXT NOT NULL, environment_json TEXT NOT NULL,
  artifacts_json TEXT NOT NULL, commit_sha TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE(task_id, machine_id, variant, metric)
);
CREATE TABLE IF NOT EXISTS evaluations (
  id TEXT PRIMARY KEY, task_id TEXT NOT NULL REFERENCES tasks(id),
  machine_id TEXT NOT NULL REFERENCES machines(id), metric TEXT NOT NULL,
  verdict TEXT NOT NULL, base_median REAL, candidate_median REAL,
  delta_percent REAL, details_json TEXT NOT NULL, created_at TEXT NOT NULL,
  UNIQUE(task_id, machine_id, metric)
);
CREATE TABLE IF NOT EXISTS decisions (
  id TEXT PRIMARY KEY, task_id TEXT NOT NULL REFERENCES tasks(id),
  question TEXT NOT NULL, options_json TEXT NOT NULL,
  recommendation TEXT, status TEXT NOT NULL, rationale TEXT,
  resolved_by TEXT, created_at TEXT NOT NULL, resolved_at TEXT
);
CREATE TABLE IF NOT EXISTS audit_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT, entity_type TEXT NOT NULL,
  entity_id TEXT NOT NULL, event_type TEXT NOT NULL, actor TEXT NOT NULL,
  payload_json TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS models (
  id TEXT PRIMARY KEY, name TEXT NOT NULL, cared INTEGER NOT NULL DEFAULT 1,
  files_json TEXT NOT NULL, conformance_json TEXT NOT NULL,
  created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS observations (
  id TEXT PRIMARY KEY, model_id TEXT NOT NULL REFERENCES models(id),
  machine_id TEXT NOT NULL REFERENCES machines(id), framework TEXT NOT NULL,
  format TEXT NOT NULL, backend TEXT NOT NULL, conformance TEXT NOT NULL,
  conformance_details_json TEXT NOT NULL, metrics_json TEXT NOT NULL,
  revision TEXT, artifacts_json TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS optimization_history (
  id TEXT PRIMARY KEY, task_id TEXT REFERENCES tasks(id), title TEXT NOT NULL,
  summary TEXT NOT NULL, model_id TEXT, before_json TEXT NOT NULL,
  after_json TEXT NOT NULL, gains_json TEXT NOT NULL, evidence_json TEXT NOT NULL,
  commit_sha TEXT, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS milestones (
  id TEXT PRIMARY KEY, task_id TEXT NOT NULL REFERENCES tasks(id),
  commit_sha TEXT NOT NULL, remote TEXT NOT NULL, remote_ref TEXT NOT NULL,
  status TEXT NOT NULL, error TEXT, created_at TEXT NOT NULL, published_at TEXT
);
CREATE TABLE IF NOT EXISTS task_runs (
  id TEXT PRIMARY KEY, task_id TEXT NOT NULL REFERENCES tasks(id),
  machine_id TEXT NOT NULL REFERENCES machines(id), status TEXT NOT NULL,
  phase TEXT NOT NULL, progress INTEGER NOT NULL DEFAULT 0,
  result_json TEXT NOT NULL, error TEXT, started_at TEXT, completed_at TEXT,
  created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
  UNIQUE(task_id,machine_id)
);
CREATE TABLE IF NOT EXISTS todo_dismissals (
  id TEXT PRIMARY KEY, dismissed_by TEXT NOT NULL, dismissed_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS learning_studies (
  id TEXT PRIMARY KEY, source TEXT NOT NULL, title TEXT NOT NULL,
  revision TEXT, status TEXT NOT NULL, summary TEXT NOT NULL,
  scope_json TEXT NOT NULL, findings_json TEXT NOT NULL,
  references_json TEXT NOT NULL, generated_tasks_json TEXT NOT NULL,
  started_at TEXT NOT NULL, completed_at TEXT, created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS evidence_task_idx ON evidence(task_id);
CREATE INDEX IF NOT EXISTS evaluation_task_idx ON evaluations(task_id);
CREATE INDEX IF NOT EXISTS audit_entity_idx ON audit_events(entity_type, entity_id, id);
"""


JSON_FIELDS = {
    "origin_json": "origin", "manifest_json": "manifest",
    "device_policy_json": "device_policy", "decision_policy_json": "decision_policy",
    "fingerprint_json": "fingerprint", "labels_json": "labels",
    "samples_json": "samples", "correctness_json": "correctness",
    "environment_json": "environment", "artifacts_json": "artifacts",
    "details_json": "details", "options_json": "options", "payload_json": "payload",
    "files_json": "files", "conformance_json": "conformance_spec",
    "conformance_details_json": "conformance_details", "metrics_json": "metrics",
    "before_json": "before", "after_json": "after", "gains_json": "gains",
    "evidence_json": "evidence",
    "result_json": "result",
    "scope_json": "scope", "findings_json": "findings",
    "references_json": "references", "generated_tasks_json": "generated_tasks",
}


class Store:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(SCHEMA)
        with self._db:
            self._db.execute("UPDATE observations SET backend='webgpu' WHERE framework='backpack' AND backend='d3d12'")
            self._db.execute("UPDATE observations SET backend='webgpu' WHERE framework='ort' AND backend='webgpu-native'")

    def close(self) -> None:
        self._db.close()

    def _row(self, row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        result = dict(row)
        for source, target in JSON_FIELDS.items():
            if source in result:
                result[target] = parse_json(result.pop(source), {} if source != "samples_json" else [])
        return result

    def _all(self, sql: str, args: Iterable[Any] = ()) -> list[dict[str, Any]]:
        return [self._row(row) for row in self._db.execute(sql, tuple(args)).fetchall()]  # type: ignore[misc]

    def audit(self, entity_type: str, entity_id: str, event_type: str,
              actor: str, payload: dict[str, Any] | None = None) -> None:
        self._db.execute(
            "INSERT INTO audit_events(entity_type,entity_id,event_type,actor,payload_json,created_at) VALUES(?,?,?,?,?,?)",
            (entity_type, entity_id, event_type, actor, json_text(payload or {}), utc_now()),
        )

    def create_task(self, data: dict[str, Any], actor: str = "human") -> dict[str, Any]:
        now = utc_now()
        task_id = data.get("id") or f"evo-{uuid.uuid4().hex[:10]}"
        manifest = data.get("manifest") or {}
        device_policy = data.get("device_policy") or {}
        if not isinstance(manifest, dict) or not isinstance(device_policy, dict):
            raise DomainError("manifest and device_policy must be objects")
        kind = data.get("kind", "optimization")
        if kind == "optimization":
            slug = re.sub(r"[^a-z0-9]+", "-", str(data.get("title", "experiment")).lower()).strip("-")[:42]
            manifest = {**manifest, "atomic_experiment": True,
                        "experiment_branch": manifest.get("experiment_branch") or f"experiment/{task_id}-{slug}"}
        initial_state = "blocked" if kind == "optimization" and self.has_conformance_gaps() else "proposed"
        values = (
            task_id, require(data.get("title"), "title"), data.get("kind", "optimization"),
            initial_state, require(data.get("hypothesis"), "hypothesis"),
            json_text(data.get("origin") or {"type": "human"}), data.get("base_sha"),
            data.get("candidate_sha"), json_text(manifest), json_text(device_policy),
            json_text(data.get("decision_policy") or {}), now, now,
        )
        with self._lock, self._db:
            self._db.execute("""INSERT INTO tasks(
              id,title,kind,state,hypothesis,origin_json,base_sha,candidate_sha,
              manifest_json,device_policy_json,decision_policy_json,created_at,updated_at
              ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""", values)
            self.audit("task", task_id, "created", actor, {"title": data["title"], "state": initial_state,
                                                             "reason": "conformance-first gate" if initial_state == "blocked" else ""})
        return self.get_task(task_id)  # type: ignore[return-value]

    def list_tasks(self) -> list[dict[str, Any]]:
        return self._all("""SELECT * FROM tasks ORDER BY
          CASE kind WHEN 'correctness' THEN 0 WHEN 'conformance' THEN 0 WHEN 'optimization' THEN 2 ELSE 1 END,
          created_at DESC""")

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        return self._row(self._db.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone())

    def transition_task(self, task_id: str, target: str, actor: str, reason: str = "") -> dict[str, Any]:
        task = self.get_task(task_id)
        if not task:
            raise DomainError("task not found")
        if task["kind"] == "optimization" and target in {"implementing", "candidate_ready", "validating", "evaluating"} and self.has_conformance_gaps():
            raise DomainError("optimization is blocked until every cared model/device passes Backpack conformance")
        validate_transition(task["state"], target)
        with self._lock, self._db:
            self._db.execute("UPDATE tasks SET state=?,updated_at=? WHERE id=?", (target, utc_now(), task_id))
            self.audit("task", task_id, "transition", actor, {"from": task["state"], "to": target, "reason": reason})
        return self.get_task(task_id)  # type: ignore[return-value]

    def has_conformance_gaps(self) -> bool:
        models = self.list_models(cared_only=True)
        machines = self.list_machines()
        if not models or not machines:
            return False
        latest = self.latest_observations()
        passed = {(item["model_id"], item["machine_id"])
                  for item in latest if item["framework"] == "backpack" and item["conformance"] == "pass"}
        return any((model["id"], machine["id"]) not in passed for model in models for machine in machines)

    def set_candidate(self, task_id: str, base_sha: str, candidate_sha: str, actor: str) -> dict[str, Any]:
        if not self.get_task(task_id):
            raise DomainError("task not found")
        with self._lock, self._db:
            self._db.execute("UPDATE tasks SET base_sha=?,candidate_sha=?,updated_at=? WHERE id=?",
                             (require(base_sha, "base_sha"), require(candidate_sha, "candidate_sha"), utc_now(), task_id))
            self.audit("task", task_id, "candidate_set", actor, {"base_sha": base_sha, "candidate_sha": candidate_sha})
        return self.get_task(task_id)  # type: ignore[return-value]

    def register_machine(self, data: dict[str, Any]) -> dict[str, Any]:
        now = utc_now()
        machine_id = data.get("id") or f"machine-{uuid.uuid4().hex[:10]}"
        name = require(data.get("name"), "name")
        with self._lock, self._db:
            existing = self._db.execute("SELECT id,labels_json FROM machines WHERE name=?", (name,)).fetchone()
            if existing:
                machine_id = existing["id"]
                labels = {**parse_json(existing["labels_json"], {}), **(data.get("labels") or {})}
                self._db.execute("UPDATE machines SET status='online',fingerprint_json=?,labels_json=?,last_seen_at=? WHERE id=?",
                                 (json_text(data.get("fingerprint") or {}), json_text(labels), now, machine_id))
                event = "registration_refreshed"
            else:
                self._db.execute("INSERT INTO machines VALUES(?,?,?,?,?,?,?,?)",
                                 (machine_id, name, "online", json_text(data.get("fingerprint") or {}),
                                  json_text(data.get("labels") or {}), None, now, now))
                event = "registered"
            self.audit("machine", machine_id, event, name, {})
        return self.get_machine(machine_id)  # type: ignore[return-value]

    def configure_machine(self, data: dict[str, Any], actor: str = "operator") -> dict[str, Any]:
        """Create/update an expected fleet member without claiming it is online."""
        now = utc_now()
        machine_id = data.get("id") or f"machine-{uuid.uuid4().hex[:10]}"
        name = require(data.get("name"), "name")
        status = data.get("status", "offline")
        if status not in {"offline", "unreachable", "maintenance"}:
            raise DomainError("configured machine status must be offline, unreachable, or maintenance")
        with self._lock, self._db:
            existing = self._db.execute("SELECT id FROM machines WHERE name=?", (name,)).fetchone()
            if existing:
                machine_id = existing["id"]
                self._db.execute(
                    "UPDATE machines SET status=?,fingerprint_json=?,labels_json=? WHERE id=?",
                    (status, json_text(data.get("fingerprint") or {}),
                     json_text(data.get("labels") or {}), machine_id),
                )
                event = "configuration_updated"
            else:
                self._db.execute("INSERT INTO machines VALUES(?,?,?,?,?,?,?,?)", (
                    machine_id, name, status, json_text(data.get("fingerprint") or {}),
                    json_text(data.get("labels") or {}), None, now, now,
                ))
                event = "configured"
            self.audit("machine", machine_id, event, actor, {"status": status})
        return self.get_machine(machine_id)  # type: ignore[return-value]

    def get_machine(self, machine_id: str) -> dict[str, Any] | None:
        return self._row(self._db.execute("SELECT * FROM machines WHERE id=?", (machine_id,)).fetchone())

    def list_machines(self) -> list[dict[str, Any]]:
        return self._all("SELECT * FROM machines ORDER BY name")

    def set_machine_activity(self, machine_id: str, paused: bool, actor: str) -> dict[str, Any]:
        machine = self.get_machine(machine_id)
        if not machine:
            raise DomainError("machine not found")
        labels = {**machine.get("labels", {}), "activity_paused": paused}
        with self._lock, self._db:
            self._db.execute("UPDATE machines SET labels_json=? WHERE id=?", (json_text(labels), machine_id))
            self.audit("machine", machine_id, "activity_paused" if paused else "activity_resumed",
                       actor, {"device": machine["name"]})
        return self.get_machine(machine_id)  # type: ignore[return-value]

    def list_todo_dismissals(self) -> list[str]:
        return [row["id"] for row in self._db.execute("SELECT id FROM todo_dismissals ORDER BY dismissed_at")]

    def dismiss_todos(self, ids: list[str], actor: str) -> dict[str, Any]:
        clean = sorted({str(item)[:300] for item in ids if str(item).strip()})
        with self._lock, self._db:
            for item in clean:
                self._db.execute("INSERT OR REPLACE INTO todo_dismissals VALUES(?,?,?)", (item, actor, utc_now()))
                self.audit("todo", item, "dismissed", actor, {})
        return {"dismissed": clean}

    def list_learning_studies(self) -> list[dict[str, Any]]:
        return self._all("SELECT * FROM learning_studies ORDER BY COALESCE(completed_at,started_at) DESC")

    def add_learning_study(self, data: dict[str, Any], actor: str) -> dict[str, Any]:
        study_id = data.get("id") or f"study-{uuid.uuid4().hex[:12]}"
        superseded = [str(value) for value in data.get("supersedes") or [] if value]
        if superseded:
            with self._lock, self._db:
                for obsolete_id in superseded:
                    self._db.execute("DELETE FROM learning_studies WHERE id=?", (obsolete_id,))
                    self.audit("study", obsolete_id, "study_superseded", actor, {"replacement": study_id})
        existing = self._row(self._db.execute("SELECT * FROM learning_studies WHERE id=?", (study_id,)).fetchone())
        if existing:
            return existing
        status = data.get("status", "completed")
        if status not in {"running", "completed", "failed"}:
            raise DomainError("invalid learning study status")
        now = utc_now()
        generated = []
        for index, proposal in enumerate(data.get("task_proposals") or []):
            key = f"study:{study_id}:{index}"
            duplicate = next((task for task in self.list_tasks()
                              if task.get("origin", {}).get("automation_key") == key), None)
            task = duplicate or self.create_task({
                "title": require(proposal.get("title"), "task title"),
                "kind": proposal.get("kind", "correctness"),
                "hypothesis": require(proposal.get("hypothesis"), "task hypothesis"),
                "origin": {"type": "learning-study", "study_id": study_id,
                           "source": data.get("source"), "automation_key": key},
                "manifest": proposal.get("manifest") or {},
                "device_policy": proposal.get("device_policy") or {},
            }, actor)
            generated.append(task["id"])
        with self._lock, self._db:
            self._db.execute("INSERT INTO learning_studies VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                study_id, require(data.get("source"), "source"), require(data.get("title"), "title"),
                data.get("revision"), status, require(data.get("summary"), "summary"),
                json_text(data.get("scope") or []), json_text(data.get("findings") or []),
                json_text(data.get("references") or []), json_text(generated),
                data.get("started_at") or now, data.get("completed_at") or (now if status == "completed" else None), now,
            ))
            self.audit("study", study_id, "study_completed" if status == "completed" else "study_updated",
                       actor, {"source": data.get("source"), "generated_tasks": generated})
        self.ensure_task_runs()
        self.ensure_runnable_automatic_tasks()
        return self._row(self._db.execute("SELECT * FROM learning_studies WHERE id=?", (study_id,)).fetchone())  # type: ignore[return-value]

    def add_evidence(self, data: dict[str, Any], actor: str) -> dict[str, Any]:
        task_id = require(data.get("task_id"), "task_id")
        machine_id = require(data.get("machine_id"), "machine_id")
        variant = require(data.get("variant"), "variant")
        if variant not in {"base", "candidate"}:
            raise DomainError("variant must be base or candidate")
        task = self.get_task(task_id)
        if not task or not self.get_machine(machine_id):
            raise DomainError("task or machine not found")
        samples = data.get("samples")
        if not isinstance(samples, list) or not samples or not all(isinstance(v, (int, float)) for v in samples):
            raise DomainError("samples must be a non-empty numeric array")
        expected_sha = task.get("base_sha" if variant == "base" else "candidate_sha")
        commit_sha = require(data.get("commit_sha"), "commit_sha")
        if expected_sha and commit_sha != expected_sha:
            raise DomainError(f"evidence commit {commit_sha} does not match frozen {variant} SHA {expected_sha}")
        evidence_id = data.get("id") or f"ev-{uuid.uuid4().hex[:12]}"
        values = (
            evidence_id, task_id, machine_id, variant, data.get("metric", "decode_tok_s"),
            data.get("unit", "tok/s"), json_text(samples), json_text(data.get("correctness") or {"passed": True}),
            json_text(data.get("environment") or {}), json_text(data.get("artifacts") or []), commit_sha, utc_now(),
        )
        with self._lock, self._db:
            self._db.execute("""INSERT INTO evidence VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
              ON CONFLICT(task_id,machine_id,variant,metric) DO UPDATE SET
              id=excluded.id,unit=excluded.unit,samples_json=excluded.samples_json,
              correctness_json=excluded.correctness_json,environment_json=excluded.environment_json,
              artifacts_json=excluded.artifacts_json,commit_sha=excluded.commit_sha,created_at=excluded.created_at""", values)
            self.audit("task", task_id, "evidence_added", actor,
                       {"evidence_id": evidence_id, "machine_id": machine_id, "variant": variant})
        return self._row(self._db.execute("SELECT * FROM evidence WHERE id=?", (evidence_id,)).fetchone())  # type: ignore[return-value]

    def list_evidence(self, task_id: str) -> list[dict[str, Any]]:
        return self._all("SELECT * FROM evidence WHERE task_id=? ORDER BY machine_id,metric,variant", (task_id,))

    def replace_evaluations(self, task_id: str, rows: list[dict[str, Any]], aggregate: str, reason: str) -> None:
        with self._lock, self._db:
            self._db.execute("DELETE FROM evaluations WHERE task_id=?", (task_id,))
            for row in rows:
                self._db.execute("INSERT INTO evaluations VALUES(?,?,?,?,?,?,?,?,?,?)", (
                    f"eval-{uuid.uuid4().hex[:12]}", task_id, row["machine_id"], row["metric"], row["verdict"],
                    row.get("base_median"), row.get("candidate_median"), row.get("delta_percent"),
                    json_text(row.get("details") or {}), utc_now(),
                ))
            self._db.execute("UPDATE tasks SET aggregate_verdict=?,verdict_reason=?,updated_at=? WHERE id=?",
                             (aggregate, reason, utc_now(), task_id))
            self.audit("task", task_id, "evaluated", "policy-engine", {"verdict": aggregate, "reason": reason})

    def list_evaluations(self, task_id: str) -> list[dict[str, Any]]:
        return self._all("SELECT * FROM evaluations WHERE task_id=? ORDER BY machine_id,metric", (task_id,))

    def create_decision(self, task_id: str, question: str, options: list[str], recommendation: str | None) -> dict[str, Any]:
        if not self.get_task(task_id):
            raise DomainError("task not found")
        decision_id = f"decision-{uuid.uuid4().hex[:10]}"
        with self._lock, self._db:
            self._db.execute("INSERT INTO decisions VALUES(?,?,?,?,?,?,?,?,?,?)", (
                decision_id, task_id, require(question, "question"), json_text(require(options, "options")),
                recommendation, "pending", None, None, utc_now(), None,
            ))
            self.audit("decision", decision_id, "created", "policy-engine", {"task_id": task_id})
        return self.get_decision(decision_id)  # type: ignore[return-value]

    def get_decision(self, decision_id: str) -> dict[str, Any] | None:
        return self._row(self._db.execute("SELECT * FROM decisions WHERE id=?", (decision_id,)).fetchone())

    def list_decisions(self, status: str | None = None) -> list[dict[str, Any]]:
        if status:
            return self._all("SELECT * FROM decisions WHERE status=? ORDER BY created_at DESC", (status,))
        return self._all("SELECT * FROM decisions ORDER BY created_at DESC")

    def resolve_decision(self, decision_id: str, status: str, actor: str, rationale: str) -> dict[str, Any]:
        decision = self.get_decision(decision_id)
        if not decision or decision["status"] != "pending":
            raise DomainError("pending decision not found")
        if status not in {"approved", "rejected", "more_evidence_requested"}:
            raise DomainError("invalid decision resolution")
        with self._lock, self._db:
            self._db.execute("UPDATE decisions SET status=?,rationale=?,resolved_by=?,resolved_at=? WHERE id=?",
                             (status, require(rationale, "rationale"), actor, utc_now(), decision_id))
            self.audit("decision", decision_id, "resolved", actor, {"status": status, "rationale": rationale})
        return self.get_decision(decision_id)  # type: ignore[return-value]

    def task_detail(self, task_id: str) -> dict[str, Any] | None:
        task = self.get_task(task_id)
        if not task:
            return None
        task["evidence"] = self.list_evidence(task_id)
        task["evaluations"] = self.list_evaluations(task_id)
        task["decisions"] = self._all("SELECT * FROM decisions WHERE task_id=? ORDER BY created_at DESC", (task_id,))
        task["audit"] = self._all("SELECT * FROM audit_events WHERE entity_type='task' AND entity_id=? ORDER BY id", (task_id,))
        task["runs"] = self.list_runs(task_id)
        return task

    def delete_task(self, task_id: str, actor: str) -> bool:
        """Remove an obsolete task and its operational records."""
        if not self.get_task(task_id):
            return False
        with self._lock, self._db:
            run_ids = [row[0] for row in self._db.execute(
                "SELECT id FROM task_runs WHERE task_id=?", (task_id,)).fetchall()]
            self._db.execute("UPDATE machines SET current_run_id=NULL WHERE current_run_id IN "
                             "(SELECT id FROM task_runs WHERE task_id=?)", (task_id,))
            for table in ("evidence", "evaluations", "decisions", "milestones", "task_runs"):
                self._db.execute(f"DELETE FROM {table} WHERE task_id=?", (task_id,))
            self._db.execute("UPDATE optimization_history SET task_id=NULL WHERE task_id=?", (task_id,))
            self._db.execute("DELETE FROM audit_events WHERE entity_type='task' AND entity_id=?", (task_id,))
            for run_id in run_ids:
                self._db.execute("DELETE FROM audit_events WHERE entity_type='run' AND entity_id=?", (run_id,))
            self._db.execute("DELETE FROM tasks WHERE id=?", (task_id,))
            self.audit("task", task_id, "deleted", actor, {"reason": "obsolete task"})
        return True

    def ensure_task_runs(self) -> list[dict[str, Any]]:
        created = []
        machines = self.list_machines()
        terminal = {"integrated", "rejected", "failed", "reverted"}
        for task in self.list_tasks():
            if task["state"] in terminal:
                continue
            machine_ids = task.get("device_policy", {}).get("machine_ids") or []
            if not machine_ids and task.get("origin", {}).get("machine_id"):
                machine_ids = [task["origin"]["machine_id"]]
            if not machine_ids:
                required = task.get("device_policy", {}).get("required", [])
                named = [item for item in required if isinstance(item, str)]
                machine_ids = [machine["id"] for machine in machines
                               if machine["id"] in named or machine.get("name") in named]
                selectors = [item.get("selector", {}) for item in required if isinstance(item, dict)]
                if selectors:
                    machine_ids.extend(machine["id"] for machine in machines
                                       if any(all({**machine.get("fingerprint", {}), **machine.get("labels", {})}.get(k) == v
                                                  for k, v in selector.items()) for selector in selectors)
                                       and machine["id"] not in machine_ids)
                if not required:
                    machine_ids = [machine["id"] for machine in machines]
            for machine_id in machine_ids:
                if not any(machine["id"] == machine_id for machine in machines):
                    continue
                exists = self._db.execute("SELECT id FROM task_runs WHERE task_id=? AND machine_id=?",
                                          (task["id"], machine_id)).fetchone()
                if exists:
                    continue
                run_id, now = f"run-{uuid.uuid4().hex[:12]}", utc_now()
                status = "blocked" if task["state"] == "blocked" else "pending"
                with self._lock, self._db:
                    self._db.execute("INSERT INTO task_runs VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", (
                        run_id, task["id"], machine_id, status, task["state"], 0,
                        json_text({}), None, None, None, now, now,
                    ))
                    self.audit("run", run_id, "queued", "scheduler",
                               {"task_id": task["id"], "machine_id": machine_id, "status": status})
                created.append(self.get_run(run_id))
        return created

    def ensure_runnable_automatic_tasks(self) -> int:
        """Attach concrete Backpack commands to automatically generated device work."""
        updated = 0
        for task in self.list_tasks():
            origin = task.get("origin", {})
            if origin.get("type") not in {"automatic", "continuous-learning", "learning-study"} or task["kind"] not in {"correctness", "benchmark"}:
                continue
            manifest = task.get("manifest", {})
            if manifest.get("adapter"):
                argv = manifest.get("argv") or []
                model_id = origin.get("model_id") or next(iter(manifest.get("models") or []), "")
                model = self.get_model(model_id)
                if task["kind"] == "correctness" and model:
                    manifest = {**manifest, "conformance_spec": model.get("conformance_spec", {})}
                if argv and argv[0] == "gitignore/runtime/build/backpack_llm.exe":
                    manifest = {**manifest, "argv": [r"D:\workspace\project\backpack\gitignore\runtime\build\backpack_llm.exe", *argv[1:]]}
                if manifest != task.get("manifest", {}):
                    with self._lock, self._db:
                        self._db.execute("UPDATE tasks SET manifest_json=?,updated_at=? WHERE id=?",
                                         (json_text(manifest), utc_now(), task["id"]))
                    updated += 1
                continue
            model_id = origin.get("model_id") or next(iter(manifest.get("models") or []), "")
            model = self.get_model(model_id)
            files = (model or {}).get("files", {})
            model_entry = files.get("gguf") or files.get("ort")
            if not model_entry or not model_entry.get("path"):
                continue
            runtime = next(iter(manifest.get("runtimes") or []), {})
            if task["kind"] == "benchmark" and runtime.get("framework") == "llamacpp":
                argv = [sys.executable, r"D:\workspace\project\backpack\evolution\benchmark_llamacpp.py",
                        "--model", model_entry["path"], "--prompt-tokens", "128",
                        "--generation-tokens", "128", "--repetitions", "5"]
            else:
                argv = [r"D:\workspace\project\backpack\gitignore\runtime\build\backpack_llm.exe", "--model", model_entry["path"]]
            if task["kind"] == "benchmark":
                if runtime.get("framework") != "llamacpp":
                    argv.append("--benchmark")
            else:
                spec = (model or {}).get("conformance_spec", {})
                # Some conformant models emit a short reasoning/preamble before
                # the required fact. Keep this bounded, but avoid false failures
                # caused solely by truncating a correct answer.
                argv += ["--prompt", spec.get("prompt", "What is 2 + 2?"),
                         "--max-tokens", str(spec.get("max_tokens", 64))]
            manifest = {**manifest, "adapter": "argv", "argv": argv}
            with self._lock, self._db:
                self._db.execute("UPDATE tasks SET manifest_json=?,updated_at=? WHERE id=?",
                                 (json_text(manifest), utc_now(), task["id"]))
                self.audit("task", task["id"], "made_runnable", "scheduler", {"adapter": "argv"})
            updated += 1
        return updated

    def ensure_daily_upstream_tasks(self) -> list[dict[str, Any]]:
        day = datetime.now(timezone.utc).date().isoformat()
        server = next((machine for machine in self.list_machines() if machine["name"].lower() == "webgfx-104"), None)
        if not server:
            return []
        existing = {task.get("origin", {}).get("automation_key"): task for task in self.list_tasks()}
        specs = [
            ("ort-webgpu", "Update ORT and ORT GenAI; build and measure native WebGPU",
             ["node", r"D:\workspace\project\agents\webgfx-agents\ai-test\scripts\build-ort.js"]),
            ("llamacpp-vulkan", "Download latest llama.cpp Vulkan and refresh reference performance",
             ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
              r"D:\workspace\project\backpack\evolution\refresh_llamacpp.ps1"]),
        ]
        created = []
        for name, title, argv in specs:
            key = f"daily:{name}:{day}"
            if key in existing:
                current = existing[key]
                manifest = {**current.get("manifest", {}), "adapter": "argv", "argv": argv}
                if manifest != current.get("manifest", {}):
                    with self._lock, self._db:
                        self._db.execute("UPDATE tasks SET manifest_json=?,updated_at=? WHERE id=?",
                                         (json_text(manifest), utc_now(), current["id"]))
                continue
            created.append(self.create_task({
                "title": title, "kind": "maintenance",
                "hypothesis": "Daily upstream references detect conformance changes and performance regressions early.",
                "origin": {"type": "scheduled", "automation_key": key, "cadence": "daily", "date": day},
                "manifest": {"adapter": "argv", "argv": argv, "conformance_first": True,
                             "models": ["gemma-4-e2b-it-qat", "qwen3.5-2b", "qwen3.5-4b"]},
                "device_policy": {"machine_ids": [server["id"]]},
            }, "daily-scheduler"))
        if created:
            self.ensure_task_runs()
        return created

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        return self._row(self._db.execute("SELECT * FROM task_runs WHERE id=?", (run_id,)).fetchone())

    def list_runs(self, task_id: str | None = None) -> list[dict[str, Any]]:
        if task_id:
            return self._all("SELECT * FROM task_runs WHERE task_id=? ORDER BY created_at", (task_id,))
        return self._all("SELECT * FROM task_runs ORDER BY created_at DESC")

    def claim_run(self, machine_name: str, capabilities: list[str], actor: str) -> dict[str, Any] | None:
        """Atomically claim the oldest pending run this worker knows how to execute."""
        self.expire_stale_runs()
        machine = self._row(self._db.execute(
            "SELECT * FROM machines WHERE lower(name)=lower(?)", (machine_name,)).fetchone())
        if not machine:
            raise DomainError("machine is not registered")
        if machine.get("labels", {}).get("activity_paused"):
            return None
        allowed = {str(item) for item in capabilities}
        with self._lock, self._db:
            rows = self._db.execute("""SELECT r.id,t.manifest_json FROM task_runs r
              JOIN tasks t ON t.id=r.task_id
              WHERE r.machine_id=? AND r.status='pending'
              ORDER BY CASE t.kind WHEN 'correctness' THEN 0 WHEN 'conformance' THEN 0 ELSE 1 END,
                       r.created_at,r.id""", (machine["id"],)).fetchall()
            chosen = None
            for row in rows:
                adapter = str(parse_json(row["manifest_json"], {}).get("adapter", ""))
                if adapter and adapter in allowed:
                    chosen = row["id"]
                    break
            if not chosen:
                return None
            now = utc_now()
            changed = self._db.execute("""UPDATE task_runs SET status='running',phase='claimed',
              progress=1,started_at=?,updated_at=? WHERE id=? AND status='pending'""",
                                       (now, now, chosen)).rowcount
            if not changed:
                return None
            self._db.execute("UPDATE machines SET current_run_id=? WHERE id=?", (chosen, machine["id"]))
            self.audit("run", chosen, "claimed", actor,
                       {"machine_id": machine["id"], "capabilities": sorted(allowed)})
        run = self.get_run(chosen)
        if run:
            run["task"] = self.get_task(run["task_id"])
        return run

    def update_run(self, run_id: str, data: dict[str, Any], actor: str) -> dict[str, Any]:
        run = self.get_run(run_id)
        if not run:
            raise DomainError("run not found")
        status = data.get("status", run["status"])
        if status not in {"pending", "running", "completed", "failed", "blocked", "cancelled"}:
            raise DomainError("invalid run status")
        requested_failure = status == "failed"
        timed_out = str(data.get("error", "")).lower().startswith("timeout:")
        prior_failures = self._db.execute(
            "SELECT COUNT(*) FROM audit_events WHERE entity_type='run' AND entity_id=? AND event_type='automatic_retry'",
            (run_id,)).fetchone()[0]
        if requested_failure and not timed_out and prior_failures < 2:
            status = "pending"
            data = {**data, "phase": f"automatic repair/retry {prior_failures + 1}/2", "progress": 0}
        now = utc_now()
        started = run.get("started_at") or (now if status == "running" else None)
        completed = now if status in {"completed", "failed", "cancelled"} else run.get("completed_at")
        progress = int(data.get("progress", 100 if status == "completed" else run["progress"]))
        with self._lock, self._db:
            self._db.execute("""UPDATE task_runs SET status=?,phase=?,progress=?,result_json=?,error=?,
              started_at=?,completed_at=?,updated_at=? WHERE id=?""", (
                status, data.get("phase", run["phase"]), max(0, min(progress, 100)),
                json_text(data.get("result", run.get("result") or {})), data.get("error", run.get("error")),
                started, completed, now, run_id,
            ))
            self.audit("run", run_id, "run_updated", actor,
                       {"status": status, "phase": data.get("phase", run["phase"]), "progress": progress})
            if requested_failure and status == "pending":
                self.audit("run", run_id, "automatic_retry", "scheduler",
                           {"attempt": prior_failures + 1, "reason": data.get("error")})
            if status in {"completed", "failed", "blocked", "cancelled"}:
                self._db.execute("UPDATE machines SET current_run_id=NULL WHERE current_run_id=?", (run_id,))
        return self.get_run(run_id)  # type: ignore[return-value]

    def expire_stale_runs(self, benchmark_timeout_seconds: int = 1800) -> int:
        """Fail abandoned benchmark runs so a device can claim its next task."""
        now = datetime.now(timezone.utc)
        expired = []
        for run in (item for item in self.list_runs() if item["status"] == "running"):
            task = self.get_task(run["task_id"])
            if not task or task["kind"] != "benchmark" or not run.get("started_at"):
                continue
            timeout = int(task.get("manifest", {}).get("timeout_seconds") or benchmark_timeout_seconds)
            try:
                elapsed = (now - datetime.fromisoformat(run["started_at"])).total_seconds()
            except (TypeError, ValueError):
                continue
            if elapsed > timeout:
                expired.append((run, timeout, int(elapsed)))
        if not expired:
            return 0
        stamp = utc_now()
        with self._lock, self._db:
            for run, timeout, elapsed in expired:
                error = f"timeout: exceeded {timeout} seconds (observed after {elapsed} seconds)"
                self._db.execute("""UPDATE task_runs SET status='failed',phase='timed out',progress=100,
                  error=?,completed_at=?,updated_at=? WHERE id=? AND status='running'""",
                                 (error, stamp, stamp, run["id"]))
                self._db.execute("UPDATE machines SET current_run_id=NULL WHERE current_run_id=?", (run["id"],))
                self.audit("run", run["id"], "timed_out", "scheduler",
                           {"timeout_seconds": timeout, "elapsed_seconds": elapsed})
        return len(expired)

    def status(self) -> dict[str, Any]:
        counts = {row["state"]: row["n"] for row in self._db.execute("SELECT state,COUNT(*) n FROM tasks GROUP BY state")}
        return {
            "tasks": counts,
            "machines_online": self._db.execute("SELECT COUNT(*) FROM machines WHERE status='online'").fetchone()[0],
            "pending_decisions": self._db.execute("SELECT COUNT(*) FROM decisions WHERE status='pending'").fetchone()[0],
            "cared_models": self._db.execute("SELECT COUNT(*) FROM models WHERE cared=1").fetchone()[0],
        }

    def activity(self, limit: int = 40) -> dict[str, Any]:
        self.expire_stale_runs()
        now = datetime.now(timezone.utc)
        fleet = []
        for machine in self.list_machines():
            try:
                seen = datetime.fromisoformat(machine["last_seen_at"])
                age_seconds = max(0, int((now - seen).total_seconds()))
            except (TypeError, ValueError):
                age_seconds = 10**9
            effective = machine["status"]
            if effective == "online" and age_seconds > 150:
                effective = "stale"
            fleet.append({"id": machine["id"], "name": machine["name"],
                          "status": effective, "age_seconds": age_seconds,
                          "gpu": machine.get("fingerprint", {}).get("gpu", "unknown"),
                          "activity_paused": bool(machine.get("labels", {}).get("activity_paused"))})
        tasks = self.list_tasks()
        latest_observations = self.latest_observations()
        all_runs = self.list_runs()
        terminal = {"integrated", "rejected", "failed", "reverted"}
        matrix_rows = {row["model"]["id"]: row for row in self.model_matrix()["models"]}
        state_progress = {"proposed": 5, "triaged": 12, "implementing": 25,
                          "candidate_ready": 40, "validating": 55, "evaluating": 70,
                          "debating": 78, "awaiting_human": 82, "ready_to_merge": 90,
                          "integrating": 95, "integrated": 100, "observing": 100,
                          "blocked": 50, "failed": 100, "rejected": 100, "reverted": 100}
        for task in tasks:
            try:
                created = datetime.fromisoformat(task["created_at"])
                end = datetime.fromisoformat(task["updated_at"]) if task["state"] in terminal else now
                elapsed = max(0, int((end - created).total_seconds()))
            except (TypeError, ValueError):
                elapsed = 0
            progress: dict[str, Any] = {
                "percent": state_progress.get(task["state"], 0), "elapsed_seconds": elapsed,
                "phase": task["state"], "basis": "lifecycle", "devices": [],
            }
            model_id = task.get("origin", {}).get("model_id")
            model_row = matrix_rows.get(model_id)
            if task["kind"] == "benchmark":
                machine_id = task.get("origin", {}).get("machine_id")
                target_machine = next((item for item in fleet if item["id"] == machine_id), None)
                requirements = task.get("manifest", {}).get("runtimes", [])
                runtime_status = []
                for requirement in requirements:
                    result = next((item for item in latest_observations
                                   if item["model_id"] == model_id and item["machine_id"] == machine_id
                                   and item["framework"] == requirement["framework"]
                                   and item["format"] == requirement["format"]
                                   and item["backend"] == requirement["backend"]), None)
                    complete = bool(result and result.get("metrics", {}).get("prefill_tok_s") is not None
                                    and result.get("metrics", {}).get("decode_tok_s") is not None)
                    runtime_status.append({"name": f"{requirement['framework']}/{requirement['backend']}",
                                           "status": "complete" if complete else "pending"})
                completed = sum(item["status"] == "complete" for item in runtime_status)
                total = len(runtime_status)
                progress.update({"percent": round(100 * completed / total) if total else 100,
                                 "basis": "required performance measurements",
                                 "target_machine_id": machine_id,
                                 "target_machine_name": target_machine["name"] if target_machine else machine_id,
                                 "devices": [{"name": target_machine["name"] if target_machine else str(machine_id),
                                              "status": f"{completed}/{total} measurements"}],
                                 "steps": runtime_status,
                                 "passed": completed, "failed": 0, "evaluated": completed, "total": total})
            elif model_row:
                devices = [{"name": cell["machine"]["name"], "status": cell["conformance"]}
                           for cell in model_row["cells"]]
                passed = sum(item["status"] == "pass" for item in devices)
                failed = sum(item["status"] == "fail" for item in devices)
                evaluated = passed + failed
                total = len(devices)
                progress.update({"percent": round(100 * passed / total) if total else 0,
                                 "basis": "required-device conformance", "devices": devices,
                                 "steps": devices,
                                 "passed": passed, "failed": failed, "evaluated": evaluated,
                                 "total": total})
            task["activity"] = progress
            task_runs = []
            for run in (item for item in all_runs if item["task_id"] == task["id"]):
                machine = next((item for item in fleet if item["id"] == run["machine_id"]), None)
                start_value = run.get("started_at") or run["created_at"]
                end_value = run.get("completed_at") or now.isoformat()
                try:
                    run_elapsed = max(0, int((datetime.fromisoformat(end_value) - datetime.fromisoformat(start_value)).total_seconds()))
                except (TypeError, ValueError):
                    run_elapsed = 0
                run["machine_name"] = machine["name"] if machine else run["machine_id"]
                run["elapsed_seconds"] = run_elapsed
                task_runs.append(run)
            task["runs"] = task_runs
        return {
            "server_time": now.isoformat(timespec="seconds"),
            "fleet": fleet,
            "fleet_healthy": bool(fleet) and all(item["status"] == "online" for item in fleet),
            "active_tasks": [task for task in tasks if task["state"] not in terminal],
            "tasks": tasks,
            "blocked_tasks": [task for task in tasks if task["state"] in {"blocked", "awaiting_human"}],
            "events": self._all("""SELECT * FROM audit_events
                                WHERE event_type NOT IN ('heartbeat','registration_refreshed','catalog_updated')
                                  AND actor != 'documented-status-import'
                                  AND payload_json != '{}'
                                ORDER BY id DESC LIMIT ?""",
                                (max(1, min(limit, 200)),)),
        }

    def upsert_model(self, data: dict[str, Any], actor: str = "catalog") -> dict[str, Any]:
        model_id = require(data.get("id"), "id")
        now = utc_now()
        with self._lock, self._db:
            self._db.execute("""INSERT INTO models VALUES(?,?,?,?,?,?,?)
              ON CONFLICT(id) DO UPDATE SET name=excluded.name,cared=excluded.cared,
              files_json=excluded.files_json,conformance_json=excluded.conformance_json,updated_at=excluded.updated_at""", (
                model_id, require(data.get("name"), "name"), 1 if data.get("cared", True) else 0,
                json_text(data.get("files") or {}), json_text(data.get("conformance_spec") or {}), now, now,
            ))
            self.audit("model", model_id, "catalog_updated", actor, {})
        return self.get_model(model_id)  # type: ignore[return-value]

    def get_model(self, model_id: str) -> dict[str, Any] | None:
        return self._row(self._db.execute("SELECT * FROM models WHERE id=?", (model_id,)).fetchone())

    def list_models(self, cared_only: bool = False) -> list[dict[str, Any]]:
        sql = "SELECT * FROM models" + (" WHERE cared=1" if cared_only else "") + " ORDER BY name"
        return self._all(sql)

    def add_observation(self, data: dict[str, Any], actor: str) -> dict[str, Any]:
        model_id = require(data.get("model_id"), "model_id")
        machine_id = require(data.get("machine_id"), "machine_id")
        if not self.get_model(model_id) or not self.get_machine(machine_id):
            raise DomainError("model or machine not found")
        framework = require(data.get("framework"), "framework")
        if framework not in {"backpack", "llamacpp", "ort"}:
            raise DomainError("framework must be backpack, llamacpp, or ort")
        conformance = data.get("conformance", "unknown")
        if conformance not in {"pass", "fail", "unknown", "not_applicable"}:
            raise DomainError("invalid conformance status")
        observation_id = data.get("id") or f"obs-{uuid.uuid4().hex[:12]}"
        backend = data.get("backend", "webgpu")
        if framework in {"backpack", "ort"} and backend in {"d3d12", "webgpu-native", "webgpu"}:
            backend = "webgpu"
        with self._lock, self._db:
            self._db.execute("INSERT OR IGNORE INTO observations VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", (
                observation_id, model_id, machine_id, framework, data.get("format", "gguf"),
                backend, conformance,
                json_text(data.get("conformance_details") or {}), json_text(data.get("metrics") or {}),
                data.get("revision"), json_text(data.get("artifacts") or []), utc_now(),
            ))
            self.audit("model", model_id, "observation_added", actor,
                       {"machine_id": machine_id, "framework": framework, "conformance": conformance})
        return self._row(self._db.execute("SELECT * FROM observations WHERE id=?", (observation_id,)).fetchone())  # type: ignore[return-value]

    def latest_observations(self) -> list[dict[str, Any]]:
        return self._all("""SELECT o.* FROM observations o JOIN (
          SELECT model_id,machine_id,framework,format,backend,MAX(created_at) latest
          FROM observations GROUP BY model_id,machine_id,framework,format,backend
        ) x ON o.model_id=x.model_id AND o.machine_id=x.machine_id AND o.framework=x.framework
          AND o.format=x.format AND o.backend=x.backend AND o.created_at=x.latest""")

    def list_observations(self, filters: dict[str, str]) -> list[dict[str, Any]]:
        allowed = {"model_id", "machine_id", "framework", "format", "backend"}
        where, values = [], []
        for key, value in filters.items():
            if key in allowed and value:
                where.append(f"{key}=?")
                values.append(value)
        sql = "SELECT * FROM observations" + (" WHERE " + " AND ".join(where) if where else "")
        return self._all(sql + " ORDER BY created_at DESC,id DESC", tuple(values))

    def confirmed_regressions(self, threshold_percent: float = 10.0) -> list[dict[str, Any]]:
        rows = self._all("SELECT * FROM observations ORDER BY created_at,id")
        groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
        for row in rows:
            key = tuple(str(row[name]) for name in ("model_id", "machine_id", "framework", "format", "backend"))
            groups.setdefault(key, []).append(row)
        result = []
        for key, samples in groups.items():
            for metric in ("prefill_tok_s", "decode_tok_s"):
                measured = [row for row in samples if isinstance(row.get("metrics", {}).get(metric), (int, float))]
                if len(measured) < 3:
                    continue
                baseline, check1, check2 = measured[-3:]
                base = float(baseline["metrics"][metric])
                if base <= 0:
                    continue
                deltas = [(float(item["metrics"][metric]) / base - 1) * 100 for item in (check1, check2)]
                if all(delta <= -threshold_percent for delta in deltas):
                    result.append({"id": "confirmed:" + ":".join(key) + ":" + metric,
                                   "model_id": key[0], "machine_id": key[1], "framework": key[2],
                                   "format": key[3], "backend": key[4], "metric": metric,
                                   "baseline": base, "latest": check2["metrics"][metric],
                                   "delta_percent": deltas[1], "revision": check2.get("revision"),
                                   "confirmed_by": [check1["id"], check2["id"]]})
        return result

    def model_matrix(self) -> dict[str, Any]:
        models = self.list_models(cared_only=True)
        machines = self.list_machines()
        # Preserve the newest semantic result and the newest performance result
        # independently. A benchmark observation uses ``not_applicable`` for
        # conformance and must not overwrite an earlier pass/fail observation.
        def compatible_revision(metric_revision: Any, pass_revision: Any) -> bool:
            metric = str(metric_revision or "").lower().strip("-")
            passed = str(pass_revision or "").lower().strip("-")
            if not metric or not passed:
                return False
            # Benchmark-mode and date suffixes do not change runtime code.
            suffixes = ("-reuse-generator", "-reuse_generator")
            for suffix in suffixes:
                if metric.endswith(suffix):
                    metric = metric[:-len(suffix)]
                if passed.endswith(suffix):
                    passed = passed[:-len(suffix)]
            metric = re.sub(r"-(?:19|20)\d{6}$", "", metric)
            passed = re.sub(r"-(?:19|20)\d{6}$", "", passed)
            return metric == passed or metric.startswith(passed + "-") or passed.startswith(metric + "-")

        observations = []
        grouped: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = {}
        for item in self.list_observations({}):
            key = tuple(item[name] for name in ("model_id", "machine_id", "framework", "format", "backend"))
            grouped.setdefault(key, []).append(item)
        for items in grouped.values():
            passes = [item for item in items if item["conformance"] == "pass"]
            conformance = next((item for item in items if item["conformance"] in {"pass", "fail"}), None)
            metrics = next((item for item in items if item.get("metrics") and (
                item["conformance"] == "pass" or any(
                    compatible_revision(item.get("revision"), passed.get("revision")) for passed in passes
                ))), None)
            if metrics:
                metrics = dict(metrics)
                metrics["performance_validated"] = True
            observations.extend({item["id"]: item for item in (conformance, metrics) if item}.values())
        by_cell: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for item in observations:
            by_cell.setdefault((item["model_id"], item["machine_id"]), []).append(item)
        rows = []
        for model in models:
            cells = []
            for machine in machines:
                items = by_cell.get((model["id"], machine["id"]), [])
                backpack = next((x for x in items if x["framework"] == "backpack"
                                 and x["format"] == "gguf" and x["backend"] == "webgpu"), None)
                if not backpack:
                    backpack = next((x for x in items if x["framework"] == "backpack"
                                     and x["backend"] == "webgpu"), None)
                conformant = bool(backpack and backpack["conformance"] == "pass")
                cells.append({"machine": machine, "conformant": conformant,
                              "conformance": backpack["conformance"] if backpack else "unknown",
                              "results": items})
            rows.append({"model": model, "cells": cells})
        return {"models": rows, "machines": machines}

    def ensure_automatic_tasks(self) -> list[dict[str, Any]]:
        """Create conformance work first, then perf collection for passing cells."""
        created = []
        matrix = self.model_matrix()
        open_tasks = self.list_tasks()
        for row in matrix["models"]:
            model = row["model"]
            bad = [c for c in row["cells"] if not c["conformant"]]
            duplicate = any(t.get("origin", {}).get("automation_key") == f"conformance:{model['id']}"
                            and t["state"] not in {"integrated", "rejected", "failed", "reverted"} for t in open_tasks)
            if bad and not duplicate:
                failing = [c for c in bad if c["conformance"] == "fail"]
                names = [c["machine"]["name"] for c in bad]
                title = ("Fix" if failing else "Establish") + f" {model['name']} conformance on cared devices"
                task = self.create_task({
                    "title": title, "kind": "correctness",
                    "hypothesis": f"Backpack must pass {model['name']} conformance on {', '.join(names)} before performance work.",
                    "origin": {"type": "automatic", "automation_key": f"conformance:{model['id']}", "model_id": model["id"]},
                    "manifest": {"models": [model["id"]], "metrics": [], "conformance_first": True},
                    "device_policy": {"required": [{"selector": {"role": "required"}, "count": len(matrix["machines"])}]},
                }, "automation")
                created.append(task)
                open_tasks.append(task)

            for cell in row["cells"]:
                candidates = []
                if "gguf" in model.get("files", {}):
                    candidates += [("backpack", "gguf", "webgpu"), ("llamacpp", "gguf", "vulkan")]
                if "ort" in model.get("files", {}):
                    candidates += [("backpack", "ort", "webgpu"), ("ort", "ort", "webgpu")]
                missing = []
                for framework, fmt, backend in candidates:
                    matches = [item for item in cell["results"] if item["framework"] == framework
                               and item["format"] == fmt and item["backend"] == backend]
                    passed = any(item["conformance"] == "pass" for item in matches)
                    measured = next((item for item in matches
                                     if item.get("metrics", {}).get("prefill_tok_s") is not None
                                     and item.get("metrics", {}).get("decode_tok_s") is not None), None)
                    if passed and not measured:
                        missing.append({"framework": framework, "format": fmt, "backend": backend})
                if not missing:
                    continue
                machine = cell["machine"]
                for runtime in missing:
                    key = (f"performance:{model['id']}:{machine['id']}:"
                           f"{runtime['framework']}:{runtime['backend']}")
                    duplicate_perf = any(t.get("origin", {}).get("automation_key") == key
                                         and t["state"] not in {"integrated", "rejected", "failed", "reverted"}
                                         for t in open_tasks)
                    if duplicate_perf:
                        continue
                    task = self.create_task({
                        "title": (f"Collect {model['name']} {runtime['framework']}/{runtime['backend']} "
                                  f"performance on {machine['name']}"), "kind": "benchmark",
                        "hypothesis": "Prefill and decode throughput must be measured independently.",
                        "origin": {"type": "automatic", "automation_key": key, "model_id": model["id"],
                                   "machine_id": machine["id"]},
                        "manifest": {"models": [model["id"]], "metrics": ["prefill_tok_s", "decode_tok_s"],
                                     "runtimes": [runtime], "conformance_first": True},
                        "device_policy": {"machine_ids": [machine["id"]]},
                    }, "automation")
                    created.append(task)
                    open_tasks.append(task)
        return created

    def add_history(self, data: dict[str, Any], actor: str) -> dict[str, Any]:
        history_id = data.get("id") or f"history-{uuid.uuid4().hex[:10]}"
        task_id = data.get("task_id")
        if task_id and not self.get_task(task_id):
            raise DomainError("history task not found")
        gains = dict(require(data.get("gains"), "gains"))
        after = data.get("after") or {}
        performance_keys = {"prefill_tok_s", "decode_tok_s", "tokens_per_second", "time_to_first_token_ms",
                            "gpu_time_ms", "e2e_ms", "latency_ms", "throughput"}
        if task_id and not (performance_keys.intersection(after) or gains):
            raise DomainError("task history requires measured performance impact")
        if data.get("overall_gain_percent") is not None:
            gains["overall_gain_percent"] = float(data["overall_gain_percent"])
        evidence = list(data.get("evidence") or [])
        if data.get("device"):
            evidence.append({"kind": "device", "device": data["device"]})
        with self._lock, self._db:
            self._db.execute("INSERT INTO optimization_history VALUES(?,?,?,?,?,?,?,?,?,?,?)", (
                history_id, task_id, require(data.get("title"), "title"),
                require(data.get("summary"), "summary"), data.get("model_id"),
                json_text(data.get("before") or {}), json_text(after),
                json_text(gains), json_text(evidence),
                data.get("commit_sha"), utc_now(),
            ))
            self.audit("history", history_id, "created", actor, {"task_id": data.get("task_id")})
        return self._row(self._db.execute("SELECT * FROM optimization_history WHERE id=?", (history_id,)).fetchone())  # type: ignore[return-value]

    def list_history(self) -> list[dict[str, Any]]:
        records = self._all("SELECT * FROM optimization_history ORDER BY created_at DESC")
        machines = self.list_machines()
        machine_by_id = {item["id"]: item for item in machines}
        machine_names = {item["name"] for item in machines}

        def normalize_device(value: Any) -> str | None:
            text = str(value or "")
            match = re.search(r"webgfx[-_](104|103|31)(?:\D|$)", text, re.IGNORECASE)
            if match:
                candidate = f"webgfx-{match.group(1)}"
                return candidate if candidate in machine_names else None
            return text if text in machine_names else None

        def collect_devices(value: Any, found: set[str]) -> None:
            if isinstance(value, dict):
                for key, child in value.items():
                    device = normalize_device(key)
                    if device:
                        found.add(device)
                    if key in {"device", "machine", "machine_name"}:
                        device = normalize_device(child)
                        if device:
                            found.add(device)
                    collect_devices(child, found)
            elif isinstance(value, list):
                for child in value:
                    collect_devices(child, found)
            elif isinstance(value, str):
                device = normalize_device(value)
                if device:
                    found.add(device)

        def vendor_for(machine: dict[str, Any]) -> str:
            gpu = " ".join(str(machine.get("fingerprint", {}).get(key, ""))
                           for key in ("gpu_vendor", "gpu"))
            for vendor in ("NVIDIA", "AMD", "Intel"):
                if vendor.lower() in gpu.lower():
                    return vendor.lower()
            return ""

        def percent_values(value: Any, device: str, vendor: str,
                           single_device: bool, path: tuple[str, ...] = ()) -> list[tuple[str, float]]:
            result: list[tuple[str, float]] = []
            if not isinstance(value, dict):
                return result
            for key, child in value.items():
                next_path = path + (str(key),)
                path_text = ".".join(next_path).lower()
                path_device = normalize_device(path_text)
                belongs = single_device or path_device == device or (vendor and vendor in path_text)
                percent_metric = "percent" in path_text or str(key).lower() in {"decode_tps", "prefill_tps"}
                if isinstance(child, (int, float)) and percent_metric and belongs:
                    # Resource-count and dispatch reductions are useful evidence,
                    # but their sign is opposite to throughput and must not drive
                    # the performance impact classification.
                    if any(token in path_text for token in
                           ("decode", "prefill", "throughput", "overall", "impact", "latency")) or path_device:
                        result.append((" / ".join(next_path), float(child)))
                elif isinstance(child, dict):
                    result.extend(percent_values(child, device, vendor, single_device, next_path))
            return result

        def classify(value: float | None) -> dict[str, Any]:
            if value is None:
                return {"key": "unquantified", "name": "Measured outcome", "value": None, "rank": 7}
            if value >= 20:
                key, name, rank = "transformative", "Transformative improvement", 0
            elif value >= 5:
                key, name, rank = "strong", "Strong improvement", 1
            elif value >= 1:
                key, name, rank = "measured_gain", "Measured improvement", 2
            elif value > -1:
                key, name, rank = "noise", "Within measurement noise", 3
            elif value > -5:
                key, name, rank = "measured_regression", "Measured regression", 4
            elif value > -20:
                key, name, rank = "serious_regression", "Serious regression", 5
            else:
                key, name, rank = "critical_regression", "Critical regression", 6
            return {"key": key, "name": name, "value": value, "rank": rank}

        for record in records:
            devices: set[str] = set()
            collect_devices(record.get("evidence", []), devices)
            collect_devices(record.get("gains", {}), devices)
            collect_devices(record.get("id", ""), devices)
            collect_devices(record.get("title", ""), devices)
            if record.get("task_id"):
                rows = self._all("SELECT machine_id FROM evaluations WHERE task_id=?", (record["task_id"],))
                rows += self._all("SELECT machine_id FROM task_runs WHERE task_id=?", (record["task_id"],))
                devices.update(machine_by_id[row["machine_id"]]["name"] for row in rows
                               if row["machine_id"] in machine_by_id)

            impacts = []
            for device in sorted(devices):
                machine = next(item for item in machines if item["name"] == device)
                values = percent_values(record.get("gains", {}), device, vendor_for(machine), len(devices) == 1)
                impact_value = sum(value for _, value in values) / len(values) if values else None
                metrics: dict[str, Any] = {}
                for item in record.get("evidence", []):
                    if not isinstance(item, dict) or normalize_device(item.get("device")) != device:
                        continue
                    for source, target in (("prefill_tok_s", "prefill_tok_s"), ("prefill_tps", "prefill_tok_s"),
                                           ("decode_tok_s", "decode_tok_s"), ("decode_tps", "decode_tok_s"),
                                           ("candidate_prefill_tps", "prefill_tok_s"),
                                           ("candidate_decode_tps", "decode_tok_s")):
                        if item.get(source) is not None:
                            metrics[target] = item[source]
                if len(devices) == 1:
                    metrics = {**record.get("after", {}), **metrics}
                impacts.append({"machine_id": machine["id"], "device": device,
                                "gpu": machine.get("fingerprint", {}).get("gpu"),
                                "metrics": metrics,
                                "deltas": [{"metric": name, "percent": value} for name, value in values],
                                "impact": classify(impact_value)})
            record["device_impacts"] = impacts
            quantified = [item["impact"]["value"] for item in impacts if item["impact"]["value"] is not None]
            # A material device regression takes precedence in the group label;
            # otherwise use the fleet mean to describe the milestone.
            representative = min(quantified) if any(value <= -5 for value in quantified) else (
                sum(quantified) / len(quantified) if quantified else None)
            record["impact"] = classify(representative)
        return records

    def create_milestone(self, task_id: str, commit_sha: str, remote: str, remote_ref: str) -> dict[str, Any]:
        milestone_id = f"milestone-{uuid.uuid4().hex[:10]}"
        with self._lock, self._db:
            self._db.execute("INSERT INTO milestones VALUES(?,?,?,?,?,?,?,?,?)", (
                milestone_id, task_id, commit_sha, remote, remote_ref, "publishing", None, utc_now(), None,
            ))
            self.audit("milestone", milestone_id, "publishing", "integrator", {"task_id": task_id, "commit_sha": commit_sha})
        return self.get_milestone(milestone_id)  # type: ignore[return-value]

    def get_milestone(self, milestone_id: str) -> dict[str, Any] | None:
        return self._row(self._db.execute("SELECT * FROM milestones WHERE id=?", (milestone_id,)).fetchone())

    def finish_milestone(self, milestone_id: str, success: bool, error: str | None = None) -> dict[str, Any]:
        status = "active" if success else "failed"
        with self._lock, self._db:
            self._db.execute("UPDATE milestones SET status=?,error=?,published_at=? WHERE id=?",
                             (status, error, utc_now() if success else None, milestone_id))
            self.audit("milestone", milestone_id, status, "integrator", {"error": error} if error else {})
        return self.get_milestone(milestone_id)  # type: ignore[return-value]

    def current_milestone(self) -> dict[str, Any] | None:
        return self._row(self._db.execute(
            "SELECT * FROM milestones WHERE status='active' ORDER BY published_at DESC LIMIT 1"
        ).fetchone())

    def list_milestones(self) -> list[dict[str, Any]]:
        return self._all("SELECT * FROM milestones ORDER BY created_at DESC")
