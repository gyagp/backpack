from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from evolution.agent import current_base_worktree, rewrite_repo_argv
from evolution.domain import DomainError
from evolution.policy import PolicyEngine
from evolution.store import Store


class FrameworkTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.tmp.name) / "state.db")
        self.machine = self.store.register_machine({
            "name": "gpu-1", "fingerprint": {"os": "windows", "gpu_vendor": "nvidia", "backend": "d3d12"}
        })
        self.task = self.store.create_task({
            "title": "Faster decode", "hypothesis": "A candidate improves decode",
            "base_sha": "base", "candidate_sha": "candidate",
            "manifest": {"metrics": ["decode_tok_s"]},
            "device_policy": {"required": [{"selector": {"gpu_vendor": "nvidia"}, "count": 1}]},
            "decision_policy": {"thresholds": {"positive_percent": 2, "negative_percent": -2, "max_cv_percent": 5}},
        })
        self.assertEqual(1, self.task["task_number"])

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def add_pair(self, base: list[float], candidate: list[float], passed: bool = True) -> None:
        common = {"task_id": self.task["id"], "machine_id": self.machine["id"], "metric": "decode_tok_s"}
        self.store.add_evidence({**common, "variant": "base", "samples": base, "commit_sha": "base",
                                 "correctness": {"passed": True}}, "test")
        self.store.add_evidence({**common, "variant": "candidate", "samples": candidate, "commit_sha": "candidate",
                                 "correctness": {"passed": passed}}, "test")

    def test_agent_uses_synchronized_base_for_repository_commands(self) -> None:
        repo = Path(self.tmp.name) / "repo"
        base = repo / "gitignore" / "evolution" / "worktrees" / "base-abc"
        base.mkdir(parents=True)
        (base.parent / "CURRENT_BASE").write_text(f"abc\n{base}\n", encoding="utf-8")

        resolved = current_base_worktree(repo)
        rewritten = rewrite_repo_argv(
            ["python", str(repo / "evolution" / "benchmark_ort.py"),
             "--model", r"D:\models\qwen"],
            repo, resolved,
        )
        self.assertEqual(base, resolved)
        self.assertEqual(str(base / "evolution" / "benchmark_ort.py"), rewritten[1])
        self.assertEqual(r"D:\models\qwen", rewritten[3])

    def test_positive_candidate_is_accepted(self) -> None:
        self.add_pair([100, 101, 99], [110, 111, 109])
        result = PolicyEngine(self.store).evaluate(self.task["id"])
        self.assertEqual("accept", result["aggregate_verdict"])
        self.assertEqual("positive", result["evaluations"][0]["verdict"])

    def test_named_required_device_is_resolved(self) -> None:
        named = self.store.create_task({
            "title": "Named device", "hypothesis": "Named policies are evaluable",
            "base_sha": "base", "candidate_sha": "candidate",
            "manifest": {"metrics": ["decode_tok_s"]},
            "device_policy": {"required": [self.machine["name"]]},
        })
        common = {"task_id": named["id"], "machine_id": self.machine["id"],
                  "metric": "decode_tok_s", "correctness": {"passed": True}}
        self.store.add_evidence({**common, "variant": "base", "samples": [100],
                                 "commit_sha": "base"}, "test")
        self.store.add_evidence({**common, "variant": "candidate", "samples": [105],
                                 "commit_sha": "candidate"}, "test")
        result = PolicyEngine(self.store).evaluate(named["id"])
        self.assertEqual("accept", result["aggregate_verdict"])

    def test_correctness_failure_is_rejected(self) -> None:
        self.add_pair([100, 101], [120, 121], passed=False)
        result = PolicyEngine(self.store).evaluate(self.task["id"])
        self.assertEqual("reject", result["aggregate_verdict"])

    def test_missing_evidence_blocks(self) -> None:
        result = PolicyEngine(self.store).evaluate(self.task["id"])
        self.assertEqual("blocked", result["aggregate_verdict"])

    def test_protected_regression_rejects_even_when_another_metric_is_missing(self) -> None:
        task = self.store.create_task({
            "title": "Regression with ancillary gap", "kind": "optimization",
            "hypothesis": "A missing profile metric must not hide a decode regression",
            "base_sha": "base", "candidate_sha": "candidate",
            "manifest": {"metrics": ["decode_tok_s", "gpu_time_ms"]},
            "device_policy": {"required": [self.machine["id"]]},
            "decision_policy": {"protected_metrics": ["decode_tok_s"]},
        })
        common = {"task_id": task["id"], "machine_id": self.machine["id"],
                  "metric": "decode_tok_s", "correctness": {"passed": True}}
        self.store.add_evidence({**common, "variant": "base", "samples": [100, 101, 99],
                                 "commit_sha": "base"}, "test")
        self.store.add_evidence({**common, "variant": "candidate", "samples": [85, 84, 86],
                                 "commit_sha": "candidate"}, "test")

        result = PolicyEngine(self.store).evaluate(task["id"])

        self.assertEqual("reject", result["aggregate_verdict"])
        self.assertIn("protected metric regressed", result["reason"])

    def test_commit_mismatch_is_rejected(self) -> None:
        with self.assertRaises(DomainError):
            self.store.add_evidence({"task_id": self.task["id"], "machine_id": self.machine["id"],
                                     "variant": "candidate", "samples": [1, 2], "commit_sha": "wrong"}, "test")

    def test_transition_is_guarded_and_audited(self) -> None:
        self.store.transition_task(self.task["id"], "triaged", "reviewer")
        detail = self.store.task_detail(self.task["id"])
        self.assertEqual("triaged", detail["state"])
        self.assertEqual("transition", detail["audit"][-1]["event_type"])
        with self.assertRaises(DomainError):
            self.store.transition_task(self.task["id"], "integrated", "reviewer")

    def test_configured_machine_remains_offline_until_enrollment(self) -> None:
        configured = self.store.configure_machine({
            "name": "gpu-remote", "fingerprint": {"gpu_vendor": "amd"},
        })
        self.assertEqual("offline", configured["status"])
        enrolled = self.store.register_machine({
            "name": "gpu-remote", "fingerprint": {"gpu_vendor": "amd", "gpu": "RX 7900 XTX"},
        })
        self.assertEqual(configured["id"], enrolled["id"])
        self.assertEqual("online", enrolled["status"])

    def test_conformance_gates_performance_and_creates_task(self) -> None:
        model = self.store.upsert_model({"id": "model-a", "name": "Model A", "files": {"gguf": {}}})
        self.store.add_observation({
            "model_id": model["id"], "machine_id": self.machine["id"], "framework": "backpack",
            "format": "gguf", "backend": "d3d12", "conformance": "fail",
            "metrics": {"decode_tok_s": 999}, "revision": "bad",
        }, "test")
        matrix = self.store.model_matrix()
        cell = next(row for row in matrix["models"] if row["model"]["id"] == "model-a")["cells"][0]
        self.assertFalse(cell["conformant"])
        self.assertEqual({}, cell["results"][0]["metrics"] if not cell["results"] else {})
        created = self.store.ensure_automatic_tasks()
        self.assertTrue(any(task["origin"].get("automation_key") == "conformance:model-a" for task in created))
        self.assertEqual([], self.store.ensure_automatic_tasks())

    def test_matrix_uses_latest_metric_with_matching_conformance_revision(self) -> None:
        model = self.store.upsert_model({"id": "model-valid", "name": "Valid", "files": {"gguf": {}}})
        common = {"model_id": model["id"], "machine_id": self.machine["id"],
                  "framework": "llamacpp", "format": "gguf", "backend": "vulkan"}
        self.store.add_observation({**common, "conformance": "pass", "revision": "b10069-20260720"}, "test")
        self.store.add_observation({**common, "conformance": "not_applicable", "revision": "b10069",
                                    "metrics": {"prefill_tok_s": 100, "decode_tok_s": 20}}, "test")
        self.store.add_observation({**common, "conformance": "not_applicable", "revision": "unverified-newer",
                                    "metrics": {"prefill_tok_s": 999, "decode_tok_s": 999}}, "test")
        cell = next(row for row in self.store.model_matrix()["models"]
                    if row["model"]["id"] == model["id"])["cells"][0]
        metric = next(item for item in cell["results"] if item.get("performance_validated"))
        self.assertEqual("b10069", metric["revision"])
        self.assertEqual(20, metric["metrics"]["decode_tok_s"])

    def test_valid_latest_measurement_closes_automatic_task(self) -> None:
        model = self.store.upsert_model({"id": "measured", "name": "Measured", "files": {"gguf": {}}})
        task = self.store.create_task({
            "title": "Collect measured performance", "kind": "benchmark",
            "hypothesis": "A valid measurement completes collection",
            "origin": {"type": "automatic", "model_id": model["id"],
                       "machine_id": self.machine["id"]},
            "manifest": {"metrics": ["prefill_tok_s", "decode_tok_s"],
                         "runtimes": [{"framework": "llamacpp", "format": "gguf",
                                       "backend": "vulkan"}]},
        }, "test")
        self.store.add_observation({
            "model_id": model["id"], "machine_id": self.machine["id"],
            "framework": "llamacpp", "format": "gguf", "backend": "vulkan",
            "conformance": "pass", "revision": "b1",
            "metrics": {"prefill_tok_s": 100, "decode_tok_s": 20},
        }, "test")
        self.assertEqual(1, self.store.reconcile_completed_tasks())
        self.assertEqual("integrated", self.store.get_task(task["id"])["state"])

    def test_valid_latest_measurement_closes_validation_benchmark(self) -> None:
        model = self.store.upsert_model({"id": "validated", "name": "Validated", "files": {"ort": {}}})
        task = self.store.create_task({
            "title": "Validate rebuilt runtime", "kind": "benchmark",
            "hypothesis": "The rebuilt runtime remains conformant",
            "origin": {"type": "validation", "model_id": model["id"],
                       "machine_id": self.machine["id"]},
            "manifest": {"adapter": "argv", "argv": ["benchmark"],
                         "metrics": ["prefill_tok_s", "decode_tok_s"],
                         "runtimes": [{"framework": "ort", "format": "ort",
                                       "backend": "webgpu"}]},
            "device_policy": {"machine_ids": [self.machine["id"]]},
        }, "test")
        self.store.ensure_task_runs()
        self.store.add_observation({
            "model_id": model["id"], "machine_id": self.machine["id"],
            "framework": "ort", "format": "ort", "backend": "webgpu",
            "conformance": "pass", "revision": "source-build",
            "metrics": {"prefill_tok_s": 500, "decode_tok_s": 25},
        }, "test")

        self.assertEqual(0, self.store.reconcile_completed_tasks())
        self.store.add_observation({
            "model_id": model["id"], "machine_id": self.machine["id"],
            "framework": "ort", "format": "ort", "backend": "webgpu",
            "conformance": "pass", "revision": "source-build",
            "conformance_details": {"source": f"benchmark task {task['id']}"},
            "metrics": {"prefill_tok_s": 505, "decode_tok_s": 26},
        }, "test")
        self.assertEqual("proposed", self.store.get_task(task["id"])["state"])
        run = self.store.list_runs(task["id"])[0]
        self.store.update_run(run["id"], {
            "status": "completed", "progress": 100, "result": {"exit_code": 0},
        }, "agent")

        self.assertEqual(0, self.store.reconcile_completed_tasks())
        self.assertEqual("integrated", self.store.get_task(task["id"])["state"])

    def test_fleet_conformance_closes_diagnostic_correctness_task(self) -> None:
        model = self.store.upsert_model({"id": "diagnostic", "name": "Diagnostic",
                                         "files": {"ort": {}}})
        task = self.store.create_task({
            "title": "Support packed diagnostic input", "kind": "correctness",
            "hypothesis": "Fleet conformance proves the implementation is complete",
            "origin": {"type": "diagnostic", "model_id": model["id"]},
            "manifest": {"runtime": "backpack", "format": "ort", "backend": "webgpu"},
        }, "test")
        self.store.add_observation({
            "model_id": model["id"], "machine_id": self.machine["id"],
            "framework": "backpack", "format": "ort", "backend": "webgpu",
            "conformance": "pass", "revision": "done",
        }, "test")
        self.assertEqual(1, self.store.reconcile_completed_tasks())
        closed = self.store.get_task(task["id"])
        self.assertEqual("integrated", closed["state"])
        self.assertEqual("accepted", closed["aggregate_verdict"])

    def test_successful_profiling_run_closes_task(self) -> None:
        task = self.store.create_task({
            "title": "Profile cared model", "kind": "profiling",
            "hypothesis": "A completed profile identifies the next bottleneck",
            "origin": {"type": "profiling"},
            "manifest": {"adapter": "argv", "argv": ["profile"]},
            "device_policy": {"machine_ids": [self.machine["id"]]},
        }, "test")
        self.store.ensure_task_runs()
        run = self.store.list_runs(task["id"])[0]
        self.store.update_run(run["id"], {
            "status": "completed", "progress": 100,
            "result": {"exit_code": 0},
        }, "agent")

        self.assertEqual(0, self.store.reconcile_completed_tasks())
        closed = self.store.get_task(task["id"])
        self.assertEqual("integrated", closed["state"])
        self.assertEqual("accepted", closed["aggregate_verdict"])

    def test_cross_device_optimization_history_closes_stale_task(self) -> None:
        second = self.store.register_machine({"name": "gpu-2"})
        task = self.store.create_task({
            "title": "Fuse measured graph operations", "kind": "optimization",
            "hypothesis": "Fewer dispatches improve decode throughput",
            "device_policy": {"required": [self.machine["name"], second["name"]]},
        }, "test")
        self.store.add_history({
            "task_id": task["id"], "title": "Measured fusion",
            "summary": "Passed conformance and improved decode on both devices",
            "commit_sha": "abc123", "gains": {"decode_percent": 2.5},
            "evidence": [{"kind": "cross-device", "conformance": "pass",
                          "devices": [self.machine["name"], second["name"]]}],
        }, "test")

        self.assertEqual(1, self.store.reconcile_completed_tasks())
        closed = self.store.get_task(task["id"])
        self.assertEqual("integrated", closed["state"])
        self.assertEqual("accepted", closed["aggregate_verdict"])

    def test_regressed_history_does_not_close_optimization(self) -> None:
        task = self.store.create_task({
            "title": "Regressed fusion", "kind": "optimization",
            "hypothesis": "This candidate should remain rejected",
            "device_policy": {"required": [self.machine["name"]]},
        }, "test")
        self.store.add_history({
            "task_id": task["id"], "title": "Regressed measurement",
            "summary": "Conformance passed but decode regressed",
            "commit_sha": "bad123", "gains": {"decode_percent": -8.0},
            "evidence": [{"conformance": "pass", "devices": [self.machine["name"]]}],
        }, "test")

        self.assertEqual(0, self.store.reconcile_completed_tasks())
        self.assertEqual("proposed", self.store.get_task(task["id"])["state"])

    def test_history_keeps_detailed_gain(self) -> None:
        item = self.store.add_history({
            "title": "Fused projection", "summary": "Removed two dispatches",
            "gains": {"decode_tok_s": 8.5}, "before": {"decode_tok_s": 100},
            "after": {"decode_tok_s": 108.5}, "evidence": ["ev-1"],
        }, "test")
        self.assertEqual(8.5, item["gains"]["decode_tok_s"])

    def test_legacy_history_is_grouped_into_numbered_done_task(self) -> None:
        for device in ("gpu-1", "gpu-2"):
            self.store.add_history({
                "title": "Fused projection", "summary": "Measured improvement",
                "commit_sha": "abc123", "gains": {device: {"decode_gain_percent": 4.0}},
            }, "test")
        path = self.store.path
        self.store.close()
        self.store = Store(path)
        rows = self.store.list_history()
        self.assertEqual(1, len({row["task_id"] for row in rows}))
        task = self.store.get_task(rows[0]["task_id"])
        self.assertEqual(2, task["task_number"])
        self.assertEqual("integrated", task["state"])
        self.assertEqual("history-import", task["origin"]["type"])

    def test_history_resolves_devices_and_classifies_impact(self) -> None:
        self.store.add_history({
            "title": "Faster projection", "summary": "Measured on the enrolled GPU",
            "gains": {"gpu-1": {"decode_gain_percent": 8.5}},
            "evidence": [{"kind": "benchmark", "device": "gpu-1", "decode_tps": 108.5}],
        }, "test")
        item = self.store.list_history()[0]
        self.assertEqual("gpu-1", item["device_impacts"][0]["device"])
        self.assertEqual(108.5, item["device_impacts"][0]["metrics"]["decode_tok_s"])
        self.assertEqual("Strong improvement", item["device_impacts"][0]["impact"]["name"])

    def test_failed_milestone_does_not_replace_active_base(self) -> None:
        first = self.store.create_milestone(self.task["id"], "sha-one", "origin", "refs/heads/evolution/base")
        self.store.finish_milestone(first["id"], True)
        second = self.store.create_milestone(self.task["id"], "sha-two", "origin", "refs/heads/evolution/base")
        self.store.finish_milestone(second["id"], False, "push failed")
        self.assertEqual("sha-one", self.store.current_milestone()["commit_sha"])

    def test_activity_reports_elapsed_and_device_progress(self) -> None:
        self.store.upsert_model({"id": "progress-model", "name": "Progress", "files": {"gguf": {}}})
        task = self.store.create_task({
            "title": "Conformance", "hypothesis": "Pass everywhere",
            "origin": {"type": "automatic", "model_id": "progress-model"},
        })
        self.store.add_observation({"model_id": "progress-model", "machine_id": self.machine["id"],
                                    "framework": "backpack", "format": "gguf", "backend": "d3d12",
                                    "conformance": "pass", "metrics": {}, "revision": "test"}, "test")
        active = next(item for item in self.store.activity()["active_tasks"] if item["id"] == task["id"])
        self.assertEqual(100, active["activity"]["percent"])
        self.assertEqual(1, active["activity"]["evaluated"])
        self.assertGreaterEqual(active["activity"]["elapsed_seconds"], 0)

    def test_optimization_is_blocked_behind_conformance(self) -> None:
        self.store.upsert_model({"id": "gated-model", "name": "Gated", "files": {"gguf": {}}})
        optimization = self.store.create_task({"title": "Tune kernel", "kind": "optimization",
                                               "hypothesis": "A faster tile helps"})
        self.assertEqual("blocked", optimization["state"])
        with self.assertRaises(DomainError):
            self.store.transition_task(optimization["id"], "implementing", "test")

    def test_regressed_optimization_cannot_advance_to_merge(self) -> None:
        task = self.store.create_task({
            "title": "Regressed candidate", "kind": "optimization",
            "hypothesis": "The candidate might be faster",
            "manifest": {"metrics": ["decode_tok_s"]},
            "device_policy": {"required": [self.machine["id"]]},
        })
        with self.store._db:
            self.store._db.execute(
                "UPDATE tasks SET state='evaluating',aggregate_verdict='accept' WHERE id=?",
                (task["id"],),
            )
            self.store._db.execute(
                "INSERT INTO evaluations VALUES(?,?,?,?,?,?,?,?,?,?)",
                ("eval-regression", task["id"], self.machine["id"], "decode_tok_s",
                 "negative", 100.0, 70.0, -30.0, "{}", "2026-07-22T00:00:00+00:00"),
            )
        with self.assertRaisesRegex(DomainError, "regressed or inconclusive"):
            self.store.transition_task(task["id"], "ready_to_merge", "test")

    def test_passing_conformance_schedules_missing_performance(self) -> None:
        self.store.upsert_model({"id": "perf-model", "name": "Perf", "files": {"gguf": {}}})
        self.store.add_observation({"model_id": "perf-model", "machine_id": self.machine["id"],
                                    "framework": "backpack", "format": "gguf", "backend": "d3d12",
                                    "conformance": "pass", "metrics": {}, "revision": "test"}, "test")
        self.store.add_observation({"model_id": "perf-model", "machine_id": self.machine["id"],
                                    "framework": "llamacpp", "format": "gguf", "backend": "vulkan",
                                    "conformance": "pass", "metrics": {}, "revision": "test"}, "test")
        tasks = self.store.ensure_automatic_tasks()
        perf = [task for task in tasks if task["kind"] == "benchmark"]
        runtimes = [runtime for task in perf for runtime in task["manifest"]["runtimes"]]
        self.assertTrue(any(item["framework"] == "backpack" for item in runtimes))
        self.assertTrue(any(item["framework"] == "llamacpp" for item in runtimes))
        self.assertTrue(all(len(task["manifest"]["runtimes"]) == 1 for task in perf))

    def test_device_runs_track_real_execution_state_and_result(self) -> None:
        self.store.ensure_task_runs()
        run = next(item for item in self.store.list_runs() if item["task_id"] == self.task["id"])
        self.assertEqual("pending", run["status"])
        running = self.store.update_run(run["id"], {"status": "running", "phase": "benchmark", "progress": 25}, "agent")
        self.assertIsNotNone(running["started_at"])
        completed = self.store.update_run(run["id"], {"status": "completed", "progress": 100,
                                                        "result": {"decode_tok_s": 42}}, "agent")
        self.assertEqual(42, completed["result"]["decode_tok_s"])
        self.assertIsNotNone(completed["completed_at"])

    def test_stale_benchmark_run_times_out_without_retry(self) -> None:
        task = self.store.create_task({"title": "Timed benchmark", "kind": "benchmark",
                                       "hypothesis": "Measure throughput",
                                       "manifest": {"adapter": "argv", "argv": ["benchmark"],
                                                    "timeout_seconds": 1},
                                       "device_policy": {"machine_ids": [self.machine["id"]]}})
        self.store.ensure_task_runs()
        run = next(item for item in self.store.list_runs(task["id"]) if item["machine_id"] == self.machine["id"])
        self.store.update_run(run["id"], {"status": "running"}, "agent")
        with self.store._db:
            self.store._db.execute("UPDATE task_runs SET started_at=? WHERE id=?",
                                   ("2020-01-01T00:00:00+00:00", run["id"]))
        self.assertEqual(1, self.store.expire_stale_runs())
        expired = self.store.get_run(run["id"])
        self.assertEqual("failed", expired["status"])
        self.assertEqual("timed out", expired["phase"])
        self.assertIn("timeout:", expired["error"])

    def test_worker_claim_is_capability_filtered_and_atomic(self) -> None:
        executable = self.store.create_task({
            "title": "Typed diagnostic", "kind": "correctness", "hypothesis": "Run a diagnostic",
            "manifest": {"adapter": "argv", "argv": ["python", "--version"]},
            "device_policy": {"machine_ids": [self.machine["id"]]},
        })
        self.store.ensure_task_runs()
        self.assertIsNone(self.store.claim_run("gpu-1", ["unknown"], "gpu-1"))
        claimed = self.store.claim_run("gpu-1", ["argv"], "gpu-1")
        self.assertIsNotNone(claimed)
        self.assertEqual(executable["id"], claimed["task_id"])
        self.assertEqual("running", claimed["status"])
        self.assertIsNone(self.store.claim_run("gpu-1", ["argv"], "gpu-1"))

    def test_terminal_task_cancels_pending_runs_and_cannot_be_claimed(self) -> None:
        task = self.store.create_task({
            "title": "Discarded diagnostic", "kind": "correctness", "hypothesis": "Run",
            "manifest": {"adapter": "argv", "argv": ["python", "--version"]},
            "device_policy": {"machine_ids": [self.machine["id"]]},
        })
        self.store.ensure_task_runs()
        run = self.store.list_runs(task["id"])[0]
        self.assertEqual("pending", run["status"])
        self.store.transition_task(task["id"], "rejected", "test", "superseded")
        self.assertEqual("cancelled", self.store.get_run(run["id"])["status"])
        self.assertIsNotNone(self.store.get_run(run["id"])["completed_at"])
        self.assertIsNone(self.store.claim_run("gpu-1", ["argv"], "gpu-1"))

    def test_activity_hides_catalog_and_seed_import_noise(self) -> None:
        self.store.upsert_model({"id": "noise-model", "name": "Noise", "files": {}})
        self.store.audit("model", "noise-model", "observation_added", "documented-status-import", {})
        self.store.audit("task", self.task["id"], "meaningful_event", "worker", {"phase": "running"})
        events = self.store.activity()["events"]
        event_types = [event["event_type"] for event in events]
        self.assertIn("meaningful_event", event_types)
        self.assertNotIn("catalog_updated", event_types)
        self.assertFalse(any(event["actor"] == "documented-status-import" for event in events))

    def test_device_activity_pause_survives_heartbeat_and_blocks_claims(self) -> None:
        task = self.store.create_task({
            "title": "Runnable", "kind": "correctness", "hypothesis": "Run",
            "manifest": {"adapter": "argv", "argv": ["python", "--version"]},
            "device_policy": {"machine_ids": [self.machine["id"]]},
        })
        self.store.ensure_task_runs()
        self.store.set_machine_activity(self.machine["id"], True, "operator")
        self.store.register_machine({"name": "gpu-1", "fingerprint": {"gpu_vendor": "nvidia"},
                                     "labels": {"role": "required"}})
        self.assertTrue(self.store.get_machine(self.machine["id"])["labels"]["activity_paused"])
        self.assertIsNone(self.store.claim_run("gpu-1", ["argv"], "gpu-1"))
        self.store.set_machine_activity(self.machine["id"], False, "operator")
        claimed = self.store.claim_run("gpu-1", ["argv"], "gpu-1")
        self.assertEqual(task["id"], claimed["task_id"])


if __name__ == "__main__":
    unittest.main()
