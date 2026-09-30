from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evolution.agent import main as agent_main
from evolution.domain import DomainError
from evolution.store import Store
from evolution.policy import PolicyEngine


class DeviceScopeTest(unittest.TestCase):
    def test_merge_guard_resolves_device_names_and_selectors_like_evaluator(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(Path(directory) / "state.db", machine_names=("webgfx-104",))
            try:
                machine = store.register_machine({"name": "webgfx-104", "fingerprint": {"gpu_vendor": "nvidia"}})
                for required in [["webgfx-104"], [{"selector": {"gpu_vendor": "nvidia"}, "count": 1}]]:
                    task = store.create_task({"title": "Local optimization", "hypothesis": "Measured gain",
                                              "device_policy": {"required": required},
                                              "manifest": {"metrics": ["prefill_tok_s", "decode_tok_s"]}})
                    store.set_candidate(task["id"], "base", "candidate", "test")
                    for state in ["triaged", "implementing", "candidate_ready", "validating", "evaluating"]:
                        store.transition_task(task["id"], state, "test")
                    for metric in ["prefill_tok_s", "decode_tok_s"]:
                        for variant in ["base", "candidate"]:
                            store.add_evidence({"task_id": task["id"], "machine_id": machine["id"],
                                                "variant": variant, "metric": metric, "commit_sha": variant,
                                                "samples": [100 if variant == "base" else 120] * 5,
                                                "correctness": {"passed": True}}, "test")
                    self.assertEqual("accept", PolicyEngine(store).evaluate(task["id"])["aggregate_verdict"])
                    self.assertEqual("ready_to_merge", store.transition_task(task["id"], "ready_to_merge", "test")["state"])
            finally:
                store.close()

    def test_unresolved_identity_does_not_idle_validated_available_models(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(Path(directory) / "state.db", machine_names=("webgfx-104",))
            try:
                machine = store.register_machine({"name": "webgfx-104"})
                store.upsert_model({"id": "ready", "name": "Ready", "files": {"gguf": {"path": "ready.gguf"}}})
                store.add_observation({"model_id": "ready", "machine_id": machine["id"],
                                       "framework": "backpack", "format": "gguf", "conformance": "pass"}, "test")
                pending = {"id": "requested", "name": "Requested", "files": {},
                           "conformance_spec": {"status": "artifact_identity_pending"}}
                store.upsert_model(pending)
                self.assertTrue(store.has_conformance_gaps())
                self.assertFalse(store.has_conformance_gaps(assignable_only=True))
                self.assertEqual(2, len(store.model_matrix()["models"]))
                # Once an artifact exists, stale identity metadata cannot
                # bypass its conformance gate.
                pending["files"] = {"gguf": {"path": "requested.gguf"}}
                store.upsert_model(pending)
                self.assertTrue(store.has_conformance_gaps(assignable_only=True))
            finally:
                store.close()

    def test_existing_fleet_cannot_receive_work_outside_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.db"
            store = Store(path)
            local = store.register_machine({"name": "webgfx-104"})
            remote = store.register_machine({"name": "webgfx-103"})
            task = store.create_task({"title": "Scope check", "kind": "conformance",
                                      "hypothesis": "Only the selected device receives work",
                                      "manifest": {"adapter": "argv", "argv": ["python", "--version"]}})
            store.ensure_task_runs()
            store.close()
            store = Store(path, machine_names=("webgfx-104",))
            try:
                self.assertEqual([local["id"]], [m["id"] for m in store.list_machines()])
                self.assertIsNone(store.claim_run("webgfx-103", ["argv"], "test"))
                remote_run = next(r for r in store.list_runs(task["id"]) if r["machine_id"] == remote["id"])
                self.assertEqual("cancelled", remote_run["status"])
                run = store.claim_run("WEBGFX-104", ["argv"], "test")
                self.assertEqual(local["id"], run["machine_id"])
                store.ensure_task_runs()
                self.assertEqual(2, len(store.list_runs(task["id"])))
                with self.assertRaisesRegex(DomainError, "outside the active goal scope"):
                    store.register_machine({"name": "webgfx-31"})
                with self.assertRaisesRegex(DomainError, "outside the active goal scope"):
                    store.configure_machine({"name": "webgfx-31"})
                store.upsert_model({"id": "model", "name": "Model"})
                self.assertEqual([local["id"]], [m["id"] for m in store.model_matrix()["machines"]])
            finally:
                store.close()

    def test_remote_agent_cannot_sync_even_with_local_name_override(self):
        with patch("evolution.agent.socket.gethostname", return_value="webgfx-103"), \
                patch("evolution.agent.request_json") as request:
            with self.assertRaises(SystemExit) as result:
                agent_main(["--name", "webgfx-104", "sync-base"])
            self.assertEqual(2, result.exception.code)
            request.assert_not_called()


if __name__ == "__main__":
    unittest.main()
