from __future__ import annotations

import tempfile
import unittest
import sqlite3
from pathlib import Path

from evolution.agent import (argv_option, backpack_conformance_argv,
                             conformance_passed, current_base_worktree,
                             extract_backpack_output, rewrite_python_argv,
                             rewrite_repo_argv)
from evolution.domain import DomainError, Thresholds
from evolution.policy import PolicyEngine
from evolution.server import read_goal, write_goal
from evolution.store import Store, latest_backpack_executable
from evolution.benchmark_llamacpp import (conformance_passed as llamacpp_conformance_passed,
                                          final_answer as llamacpp_final_answer)


class FrameworkTest(unittest.TestCase):
    def test_thresholds_support_legacy_root_values_and_nested_overrides(self) -> None:
        legacy = Thresholds.from_policy({
            "positive_percent": 3, "negative_percent": -1, "max_cv_percent": 4})
        self.assertEqual((3, -1, 4), (legacy.positive_percent,
                                     legacy.negative_percent,
                                     legacy.max_cv_percent))
        nested = Thresholds.from_policy({
            "negative_percent": -1,
            "thresholds": {"negative_percent": -0.5}})
        self.assertEqual(-0.5, nested.negative_percent)

    def test_llamacpp_exact_conformance_rejects_extra_text(self) -> None:
        self.assertTrue(llamacpp_conformance_passed("4", "4", "4"))
        self.assertFalse(llamacpp_conformance_passed("The answer is 4", "4", "4"))

    def test_llamacpp_extracts_answer_after_reasoning(self) -> None:
        output = "Thinking Process:\n2 + 2 = 4.\n</think>\n\n4 [end of text]\n"
        self.assertEqual("4", llamacpp_final_answer(output))

    def test_backpack_benchmark_builds_same_artifact_chat_validation(self) -> None:
        benchmark = [r"D:\backup\x64\backpack\abc-20260724\backpack_llm.exe",
                     "--model", r"D:\models\qwen.gguf", "--benchmark",
                     "--bench-prompt-len", "512", "--bench-gen-tokens", "128"]
        validation = backpack_conformance_argv(
            benchmark, {"prompt": "What is 2 + 2?", "temperature": 0, "max_tokens": 8})
        self.assertEqual(benchmark[0], validation[0])
        self.assertEqual(r"D:\models\qwen.gguf", argv_option(validation, "--model"))
        self.assertNotIn("--benchmark", validation)
        self.assertNotIn("--bench-gen-tokens", validation)
        self.assertEqual("What is 2 + 2?", argv_option(validation, "--chat"))
        self.assertEqual("8", argv_option(validation, "--max-tokens"))

    def test_backpack_output_extraction_excludes_prompt_and_performance(self) -> None:
        stderr = "Prompt: 12 tokens\n--- Output ---\n4\n\n--- Performance ---\nGenerate: 1"
        self.assertEqual("4", extract_backpack_output("", stderr))

    def test_exact_conformance_rejects_factually_correct_extra_text(self) -> None:
        spec = {"required_fact": "4", "expected_output": "4"}
        self.assertTrue(conformance_passed(spec, "4"))
        self.assertFalse(conformance_passed(spec, "The result is 4."))

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

    def test_goal_document_round_trip_and_validation(self) -> None:
        path = Path(self.tmp.name) / "goal.md"
        saved = write_goal("# Goal\r\n\r\n- Stay conformant", path)
        self.assertEqual("# Goal\n\n- Stay conformant\n", path.read_text(encoding="utf-8"))
        self.assertEqual(saved, read_goal(path))
        self.assertEqual(12, len(saved["revision"]))
        with self.assertRaises(DomainError):
            write_goal("   ", path)

    def test_memory_is_deduplicated_and_task_context_is_bounded(self) -> None:
        first = self.store.upsert_memory({
            "scope": "project", "scope_id": "backpack", "kind": "constraint",
            "title": "Conformance first", "content": "Do not publish performance before correctness.",
            "importance": 90, "confidence": 1,
        }, "test")
        second = self.store.upsert_memory({
            "scope": "project", "scope_id": "backpack", "kind": "constraint",
            "title": "Conformance first", "content": "Do not publish performance before correctness.",
            "importance": 95, "confidence": 1,
        }, "test")
        self.assertEqual(first["id"], second["id"])
        self.assertEqual(1, len(self.store.list_memory()))
        packet = self.store.task_context(self.task["id"], max_tokens=1000)
        again = self.store.task_context(self.task["id"], max_tokens=1000)
        self.assertLessEqual(packet["estimated_tokens"], 1000)
        self.assertEqual(packet["digest"], again["digest"])
        self.assertEqual([first["id"]], packet["memory_ids"])
        oversized = self.store.task_context(self.task["id"], failure="x" * 50_000, max_tokens=1000)
        self.assertLessEqual(oversized["estimated_tokens"], 1000)
        self.assertIn("[truncated]", oversized["packet"]["current_failure"])

    def test_agent_session_records_lineage_budget_and_memory_feedback(self) -> None:
        memory = self.store.upsert_memory({
            "scope": "project", "scope_id": "backpack", "kind": "procedure",
            "title": "Paired benchmark", "content": "Benchmark base and candidate with 512/128.",
            "importance": 90, "confidence": 1,
        }, "test")
        started = self.store.start_agent_session({
            "task_id": self.task["id"], "role": "task_worker", "context_budget": 2000,
            "parent_session_id": "director-1", "objective": "Implement one atomic experiment",
        }, "test")
        self.assertEqual("running", started["status"])
        self.assertEqual("director-1", started["parent_session_id"])
        self.assertLessEqual(started["context_tokens"], started["context_budget"])
        finished = self.store.finish_agent_session(started["id"], {
            "status": "completed", "result_summary": "Validated exact output and benchmarked.",
            "artifact_path": "gitignore/logs/session.jsonl",
        }, "test")
        self.assertEqual("completed", finished["status"])
        self.assertEqual(1, self.store.get_memory(memory["id"])["success_count"])

    def test_terminal_task_promotes_outcome_and_compacts_low_value_working_memory(self) -> None:
        self.store.upsert_memory({
            "scope": "task", "scope_id": self.task["id"], "kind": "note",
            "title": "Scratch", "content": "Transient exploration", "importance": 10,
            "source_task_id": self.task["id"],
        }, "test")
        self.store.transition_task(self.task["id"], "rejected", "test", "Candidate regressed decode by 4%")
        records = self.store.list_memory({"scope_id": self.task["id"]})
        outcome = next(item for item in records if item["kind"] == "outcome")
        scratch = next(item for item in records if item["kind"] == "note")
        self.assertIn("regressed decode", outcome["content"])
        self.assertEqual("archived", scratch["state"])

    def test_learning_study_updates_cursor_and_creates_unverified_hypothesis_memory(self) -> None:
        study = self.store.add_learning_study({
            "id": "study-ort-1", "source": "ORT WebGPU", "title": "Study packed matmul",
            "revision": "abc123", "status": "completed", "summary": "Found a packing strategy",
            "findings": ["Prepack immutable weights by output tile."],
            "references": ["onnxruntime/core/providers/webgpu/matmul.cc:42"],
            "task_proposals": [{"title": "Test immutable weight prepacking", "kind": "correctness",
                                "hypothesis": "Prepacking preserves exact output."}],
        }, "test")
        cursor = next(item for item in self.store.list_learning_cursors()
                      if item["source"] == "ORT WebGPU")
        memory = next(item for item in self.store.list_memory({"scope": "upstream"})
                      if item["scope_id"] == "ORT WebGPU")
        self.assertEqual(study["id"], cursor["study_id"])
        self.assertEqual("abc123", cursor["revision"])
        self.assertEqual("hypothesis", memory["kind"])
        self.assertEqual(0.4, memory["confidence"])
        generated = study["generated_tasks"][0]
        for state in ("triaged", "implementing", "candidate_ready", "validating", "evaluating",
                      "ready_to_merge", "integrating", "integrated"):
            self.store.transition_task(generated, state, "test", "Exact output passed")
        feedback = [item for item in self.store.list_memory({"scope": "upstream"})
                    if item["kind"] == "validated_finding"]
        self.assertEqual(1, len(feedback))
        self.assertEqual(generated, feedback[0]["source_task_id"])

    def test_director_delegates_one_bounded_leaf_agent(self) -> None:
        delegated = self.store.delegate_task(self.task["id"], {
            "role": "task_worker", "machine_id": self.machine["id"],
            "context_budget": 4000, "output_budget": 1000,
        }, "director")
        self.assertEqual("codex", delegated["manifest"]["adapter"])
        self.assertEqual("task_worker", delegated["manifest"]["agent_role"])
        self.assertEqual([self.machine["id"]], delegated["device_policy"]["machine_ids"])
        self.assertEqual(1, len(delegated["runs"]))

    def test_existing_learning_studies_are_backfilled_idempotently(self) -> None:
        path = Path(self.tmp.name) / "legacy.db"
        legacy = Store(path)
        legacy.add_learning_study({
            "id": "legacy-study", "source": "llama.cpp", "title": "Legacy study",
            "revision": "r1", "status": "completed", "summary": "A finding",
            "findings": ["Reuse quantized activations."], "references": ["src/mmq.cu:1"],
        }, "test")
        with legacy._db:
            legacy._db.execute("DELETE FROM learning_cursors")
            legacy._db.execute("DELETE FROM memory_records")
        legacy.close()
        migrated = Store(path)
        self.assertEqual(1, len(migrated.list_learning_cursors()))
        self.assertEqual(1, len(migrated.list_memory({"scope": "upstream"})))
        migrated.close()
        reopened = Store(path)
        self.assertEqual(1, len(reopened.list_learning_cursors()))
        self.assertEqual(1, len(reopened.list_memory({"scope": "upstream"})))
        reopened.close()

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

    def test_agent_uses_its_local_python_for_python_adapter(self) -> None:
        rewritten = rewrite_python_argv([
            r"C:\Users\server\Python312\python.exe",
            r"D:\workspace\project\backpack\evolution\benchmark_ort.py",
            "--model", r"D:\models\qwen",
        ])
        self.assertEqual(Path(rewritten[0]).resolve(), Path(__import__("sys").executable).resolve())
        self.assertEqual("--model", rewritten[2])

    def test_agent_preserves_non_python_adapter(self) -> None:
        argv = [r"D:\tools\llama-bench.exe", "--model", r"D:\models\qwen.gguf"]
        self.assertEqual(argv, rewrite_python_argv(argv))

    def test_latest_backpack_executable_uses_revisioned_backup(self) -> None:
        root = Path(self.tmp.name) / "backups"
        older = root / "aaaaaaa-20260723" / "backpack_llm.exe"
        latest = root / "bbbbbbb-20260724" / "backpack_llm.exe"
        older.parent.mkdir(parents=True)
        latest.parent.mkdir(parents=True)
        older.touch()
        latest.touch()
        self.assertEqual(latest, latest_backpack_executable(root))

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

    def test_missing_held_back_correctness_blocks_candidate(self) -> None:
        task = self.store.create_task({
            "title": "Layered validation", "hypothesis": "The candidate is lossless",
            "base_sha": "base", "candidate_sha": "candidate",
            "manifest": {"metrics": ["decode_tok_s"],
                         "checks": ["exact-output", "held-back-prompt", "held-back-shape"]},
            "device_policy": {"required": [self.machine["id"]]},
        })
        common = {"task_id": task["id"], "machine_id": self.machine["id"],
                  "metric": "decode_tok_s"}
        self.store.add_evidence({**common, "variant": "base", "samples": [100],
                                 "commit_sha": "base", "correctness": {"passed": True}}, "test")
        self.store.add_evidence({**common, "variant": "candidate", "samples": [110],
                                 "commit_sha": "candidate",
                                 "correctness": {"passed": True, "checks": {
                                     "exact-output": {"passed": True}}}}, "test")

        result = PolicyEngine(self.store).evaluate(task["id"])

        self.assertEqual("blocked", result["aggregate_verdict"])
        self.assertEqual(["held-back-prompt", "held-back-shape"],
                         result["evaluations"][0]["details"]["missing_checks"])

    def test_failed_layer_parity_rejects_fast_candidate(self) -> None:
        task = self.store.create_task({
            "title": "Layer parity", "hypothesis": "The candidate is lossless",
            "base_sha": "base", "candidate_sha": "candidate",
            "manifest": {"metrics": ["decode_tok_s"], "checks": ["layer-parity"]},
            "device_policy": {"required": [self.machine["id"]]},
        })
        common = {"task_id": task["id"], "machine_id": self.machine["id"],
                  "metric": "decode_tok_s"}
        self.store.add_evidence({**common, "variant": "base", "samples": [100],
                                 "commit_sha": "base", "correctness": {"passed": True}}, "test")
        self.store.add_evidence({**common, "variant": "candidate", "samples": [140],
                                 "commit_sha": "candidate", "correctness": {"passed": True,
                                     "checks": {"layer-parity": "failed"}}}, "test")

        result = PolicyEngine(self.store).evaluate(task["id"])

        self.assertEqual("reject", result["aggregate_verdict"])
        self.assertEqual(["layer-parity"], result["evaluations"][0]["details"]["failed_checks"])

    def test_layered_correctness_accepts_complete_lossless_evidence(self) -> None:
        task = self.store.create_task({
            "title": "Held-back pass", "hypothesis": "The candidate is lossless",
            "base_sha": "base", "candidate_sha": "candidate",
            "manifest": {"metrics": ["decode_tok_s"],
                         "correctness_checks": ["exact-output", "held-back-prompt"]},
            "device_policy": {"required": [self.machine["id"]]},
        })
        common = {"task_id": task["id"], "machine_id": self.machine["id"],
                  "metric": "decode_tok_s"}
        self.store.add_evidence({**common, "variant": "base", "samples": [100],
                                 "commit_sha": "base", "correctness": {"passed": True}}, "test")
        self.store.add_evidence({**common, "variant": "candidate", "samples": [110],
                                 "commit_sha": "candidate", "correctness": {"passed": True,
                                     "checks": {"exact-output": True,
                                                "held-back-prompt": {"status": "pass"}}}}, "test")

        result = PolicyEngine(self.store).evaluate(task["id"])

        self.assertEqual("accept", result["aggregate_verdict"])

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

    def test_unprotected_diagnostic_regression_does_not_block_accepted_transition(self) -> None:
        task = self.store.create_task({
            "title": "Throughput gain with profiling overhead",
            "kind": "optimization",
            "hypothesis": "An unprotected diagnostic must not contradict the aggregate policy.",
            "origin": {"type": "test"},
            "decision_policy": {"protected_metrics": ["decode_tok_s"]},
        }, "test")
        with self.store._db:
            self.store._db.execute(
                "UPDATE tasks SET state='evaluating',aggregate_verdict='accept' WHERE id=?",
                (task["id"],))
            self.store._db.execute(
                "INSERT INTO evaluations VALUES(?,?,?,?,?,?,?,?,?,?)",
                ("eval-positive", task["id"], self.machine["id"], "decode_tok_s",
                 "positive", 60.0, 66.0, 10.0, "{}", "2026-01-01T00:00:00+00:00"))
            self.store._db.execute(
                "INSERT INTO evaluations VALUES(?,?,?,?,?,?,?,?,?,?)",
                ("eval-diagnostic", task["id"], self.machine["id"], "profile_gpu_ms",
                 "negative", 4.0, 5.0, -20.0, "{}", "2026-01-01T00:00:00+00:00"))

        transitioned = self.store.transition_task(task["id"], "ready_to_merge", "test")
        self.assertEqual("ready_to_merge", transitioned["state"])

    def test_unprotected_diagnostic_gain_cannot_authorize_neutral_throughput(self) -> None:
        task = self.store.create_task({
            "title": "Fewer dispatches without throughput gain", "kind": "optimization",
            "hypothesis": "A diagnostic gain is useful evidence but not a product gain.",
            "base_sha": "base", "candidate_sha": "candidate",
            "manifest": {"metrics": ["decode_tok_s", "captured_dispatches"]},
            "device_policy": {"required": [self.machine["id"]]},
            "decision_policy": {"protected_metrics": ["decode_tok_s"]},
        })
        common = {"task_id": task["id"], "machine_id": self.machine["id"],
                  "correctness": {"passed": True}}
        for metric, base, candidate in (
            ("decode_tok_s", [100, 100], [100.5, 100.5]),
            ("captured_dispatches", [956, 956], [892, 892]),
        ):
            self.store.add_evidence({**common, "metric": metric, "variant": "base",
                                     "samples": base, "commit_sha": "base"}, "test")
            self.store.add_evidence({**common, "metric": metric, "variant": "candidate",
                                     "samples": candidate, "commit_sha": "candidate"}, "test")

        result = PolicyEngine(self.store).evaluate(task["id"])

        self.assertEqual("debate", result["aggregate_verdict"])
        self.assertEqual("all protected results are neutral", result["reason"])

    def test_merge_requires_every_protected_metric_on_every_required_device(self) -> None:
        second = self.store.register_machine({
            "name": "second-required", "fingerprint": {"gpu_vendor": "amd"},
        })
        task = self.store.create_task({
            "title": "Incomplete device matrix", "kind": "optimization",
            "hypothesis": "Partial evidence must never authorize a merge.",
            "origin": {"type": "test"},
            "device_policy": {"required": [self.machine["id"], second["id"]]},
            "decision_policy": {
                "protected_metrics": ["prefill_tok_s", "decode_tok_s"],
            },
        }, "test")
        with self.store._db:
            self.store._db.execute(
                "UPDATE tasks SET state='evaluating',aggregate_verdict='accept' WHERE id=?",
                (task["id"],))
            for machine_id, metric in (
                (self.machine["id"], "prefill_tok_s"),
                (self.machine["id"], "decode_tok_s"),
                (second["id"], "decode_tok_s"),
            ):
                self.store._db.execute(
                    "INSERT INTO evaluations VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (f"eval-{machine_id}-{metric}", task["id"], machine_id, metric,
                     "neutral", 100.0, 100.0, 0.0, "{}",
                     "2026-01-01T00:00:00+00:00"))

        with self.assertRaisesRegex(DomainError, "second-required|prefill_tok_s"):
            self.store.transition_task(task["id"], "ready_to_merge", "test")

    def test_non_overlapping_regression_rejects_before_cv_gate(self) -> None:
        for metric, base, candidate in (
            ("prefill_tok_s", [303.7, 303.5], [276.2, 235.3]),
            ("gpu_time_ms", [10.0, 10.1], [12.0, 16.0]),
        ):
            task = self.store.create_task({
                "title": f"Separated {metric}", "kind": "optimization",
                "hypothesis": "A fully separated regression band is unsafe",
                "base_sha": "base", "candidate_sha": "candidate",
                "manifest": {"metrics": [metric]},
                "device_policy": {"required": [self.machine["id"]]},
                "decision_policy": {"protected_metrics": [metric]},
            })
            common = {"task_id": task["id"], "machine_id": self.machine["id"],
                      "metric": metric, "correctness": {"passed": True}}
            self.store.add_evidence({**common, "variant": "base", "samples": base,
                                     "commit_sha": "base"}, "test")
            self.store.add_evidence({**common, "variant": "candidate", "samples": candidate,
                                     "commit_sha": "candidate"}, "test")

            result = PolicyEngine(self.store).evaluate(task["id"])

            row = result["evaluations"][0]
            self.assertEqual("reject", result["aggregate_verdict"])
            self.assertEqual("negative", row["verdict"])
            self.assertTrue(row["details"]["non_overlapping_regression"])
            self.assertGreater(row["details"]["max_cv_percent"], 5)

    def test_lower_gpu_time_is_an_improvement(self) -> None:
        task = self.store.create_task({
            "title": "Reduce GPU time", "kind": "optimization",
            "hypothesis": "Less GPU time is faster",
            "base_sha": "base", "candidate_sha": "candidate",
            "manifest": {"metrics": ["gpu_time_ms"]},
            "device_policy": {"required": [self.machine["id"]]},
        })
        common = {"task_id": task["id"], "machine_id": self.machine["id"],
                  "metric": "gpu_time_ms", "correctness": {"passed": True}}
        self.store.add_evidence({**common, "variant": "base", "samples": [21.5, 21.6, 21.4],
                                 "commit_sha": "base"}, "test")
        self.store.add_evidence({**common, "variant": "candidate", "samples": [20.7, 20.8, 20.6],
                                 "commit_sha": "candidate"}, "test")

        result = PolicyEngine(self.store).evaluate(task["id"])

        self.assertEqual("accept", result["aggregate_verdict"])
        self.assertGreater(result["evaluations"][0]["delta_percent"], 0)
        self.assertEqual("lower_is_better", result["evaluations"][0]["details"]["direction"])

    def test_lower_operation_count_is_an_improvement(self) -> None:
        task = self.store.create_task({
            "title": "Reduce replay calls", "kind": "optimization",
            "hypothesis": "Fewer host API calls reduce replay overhead",
            "base_sha": "base", "candidate_sha": "candidate",
            "manifest": {"metrics": ["replay_write_calls"]},
            "device_policy": {"required": [self.machine["id"]]},
            "decision_policy": {"protected_metrics": ["replay_write_calls"]},
        })
        common = {"task_id": task["id"], "machine_id": self.machine["id"],
                  "metric": "replay_write_calls", "correctness": {"passed": True}}
        self.store.add_evidence({**common, "variant": "base", "samples": [1464, 1464],
                                 "commit_sha": "base"}, "test")
        self.store.add_evidence({**common, "variant": "candidate", "samples": [56, 56],
                                 "commit_sha": "candidate"}, "test")
        result = PolicyEngine(self.store).evaluate(task["id"])
        self.assertEqual("accept", result["aggregate_verdict"])
        self.assertEqual("lower_is_better", result["evaluations"][0]["details"]["direction"])

    def test_commit_mismatch_is_rejected(self) -> None:
        with self.assertRaises(DomainError):
            self.store.add_evidence({"task_id": self.task["id"], "machine_id": self.machine["id"],
                                     "variant": "candidate", "samples": [1, 2], "commit_sha": "wrong"}, "test")

    def test_candidate_revision_change_invalidates_existing_evidence(self) -> None:
        self.add_pair([100, 101, 99], [110, 111, 109])
        self.store.set_candidate(self.task["id"], "base", "candidate-v2", "test")

        result = PolicyEngine(self.store).evaluate(self.task["id"])

        self.assertEqual("blocked", result["aggregate_verdict"])
        self.assertEqual("inconclusive", result["evaluations"][0]["verdict"])
        self.assertEqual(
            "evidence revision does not match frozen candidate",
            result["evaluations"][0]["details"]["reason"],
        )

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
                                    "metrics": {"prefill_tok_s": 100, "decode_tok_s": 20,
                                                "prompt_tokens": 512, "generated_tokens": 128}}, "test")
        self.store.add_observation({**common, "conformance": "not_applicable",
                                    "revision": "b10069-experimental-kernel",
                                    "metrics": {"prefill_tok_s": 200, "decode_tok_s": 40,
                                                "prompt_tokens": 512, "generated_tokens": 128}}, "test")
        self.store.add_observation({**common, "conformance": "not_applicable", "revision": "b10069",
                                    "metrics": {"prefill_tok_s": 777, "decode_tok_s": 777,
                                                "prompt_tokens": 128, "generated_tokens": 64}}, "test")
        self.store.add_observation({**common, "conformance": "not_applicable", "revision": "unverified-newer",
                                    "metrics": {"prefill_tok_s": 999, "decode_tok_s": 999}}, "test")
        cell = next(row for row in self.store.model_matrix()["models"]
                    if row["model"]["id"] == model["id"])["cells"][0]
        metric = next(item for item in cell["results"] if item.get("performance_validated"))
        self.assertEqual("b10069", metric["revision"])
        self.assertEqual(20, metric["metrics"]["decode_tok_s"])
        self.assertEqual({"prompt_tokens": 512, "generated_tokens": 128},
                         self.store.model_matrix()["performance_profile"])

    def test_confirmed_regression_requires_comparable_passing_samples(self) -> None:
        model = self.store.upsert_model({"id": "regression-model", "name": "Regression",
                                         "files": {"gguf": {}}})
        common = {"model_id": model["id"], "machine_id": self.machine["id"],
                  "framework": "backpack", "format": "gguf", "backend": "webgpu",
                  "conformance": "pass"}

        sequence = 0
        def observe(revision: str, rate: float, prompt: int, generated: int,
                    conformance: str = "pass") -> None:
            nonlocal sequence
            sequence += 1
            self.store.add_observation({**common, "id": f"obs-{sequence:02d}",
                "conformance": conformance,
                "revision": revision, "metrics": {"prefill_tok_s": rate,
                    "prompt_tokens": prompt, "generated_tokens": generated}}, "test")

        observe("base", 100, 128, 64)
        observe("different-shape-1", 50, 32, 128)
        observe("different-shape-2", 45, 32, 128)
        observe("failed-sample", 10, 128, 64, "fail")
        self.assertEqual([], self.store.confirmed_regressions())

        observe("comparable-1", 80, 128, 64)
        observe("comparable-2", 79, 128, 64)
        regressions = self.store.confirmed_regressions()
        self.assertEqual(1, len(regressions))
        self.assertEqual({"prompt_tokens": 128, "generated_tokens": 64,
                          "graph_capture": "not_applicable"},
                         regressions[0]["benchmark_signature"])

    def test_observation_regression_is_quarantined_and_does_not_replace_latest(self) -> None:
        model = self.store.upsert_model({"id": "guarded", "name": "Guarded", "files": {"gguf": {}}})
        common = {"model_id": model["id"], "machine_id": self.machine["id"],
                  "framework": "backpack", "format": "gguf", "backend": "webgpu",
                  "conformance": "pass"}
        baseline = self.store.add_observation({**common, "id": "guard-base", "revision": "base",
            "metrics": {"prefill_tok_s": 100, "decode_tok_s": 20,
                        "prompt_tokens": 512, "generated_tokens": 128}}, "test")
        regressed = self.store.add_observation({**common, "id": "guard-drop", "revision": "drop",
            "metrics": {"prefill_tok_s": 97.9, "decode_tok_s": 20,
                        "prompt_tokens": 512, "generated_tokens": 128}}, "test")
        self.assertEqual("valid", baseline["validity"])
        self.assertEqual("quarantined", regressed["validity"])
        self.assertIn("prefill_tok_s", regressed["validity_reason"])
        self.assertEqual("guard-base", self.store.latest_observations()[0]["id"])
        self.assertEqual(["guard-base"], [row["id"] for row in self.store.list_observations({})])
        self.assertEqual({"guard-base", "guard-drop"},
                         {row["id"] for row in self.store.list_observations({"include_invalid": "true"})})

    def test_regression_guard_requires_same_workload_and_graph_capture(self) -> None:
        model = self.store.upsert_model({"id": "ort-guard", "name": "ORT Guard", "files": {"ort": {}}})
        common = {"model_id": model["id"], "machine_id": self.machine["id"],
                  "framework": "ort", "format": "onnx", "backend": "webgpu",
                  "conformance": "pass"}
        self.store.add_observation({**common, "id": "ort-base", "metrics": {
            "prefill_tok_s": 100, "decode_tok_s": 20, "prompt_tokens": 512,
            "generated_tokens": 128, "graph_capture": True}}, "test")
        shape = self.store.add_observation({**common, "id": "ort-shape", "metrics": {
            "prefill_tok_s": 20, "decode_tok_s": 4, "prompt_tokens": 256,
            "generated_tokens": 128, "graph_capture": True}}, "test")
        capture = self.store.add_observation({**common, "id": "ort-capture", "metrics": {
            "prefill_tok_s": 20, "decode_tok_s": 4, "prompt_tokens": 512,
            "generated_tokens": 128, "graph_capture": False}}, "test")
        self.assertEqual("valid", shape["validity"])
        self.assertEqual("valid", capture["validity"])

    def test_backpack_ort_regression_guard_does_not_require_graph_capture(self) -> None:
        model = self.store.upsert_model({"id": "backpack-ort-guard", "name": "Backpack ORT Guard",
                                         "files": {"ort": {}}})
        common = {"model_id": model["id"], "machine_id": self.machine["id"],
                  "framework": "backpack", "format": "ort", "backend": "webgpu",
                  "conformance": "pass"}
        self.store.add_observation({**common, "id": "backpack-ort-base", "metrics": {
            "prefill_tok_s": 100, "decode_tok_s": 20, "prompt_tokens": 512,
            "generated_tokens": 128}}, "test")
        result = self.store.add_observation({**common, "id": "backpack-ort-drop", "metrics": {
            "prefill_tok_s": 100, "decode_tok_s": 19.5, "prompt_tokens": 512,
            "generated_tokens": 128, "graph_capture": "not_applicable"}}, "test")
        self.assertEqual("quarantined", result["validity"])
        self.assertIn("decode_tok_s", result["validity_reason"])

    def test_split_performance_observation_cannot_bypass_regression_guard(self) -> None:
        model = self.store.upsert_model({"id": "split-guard", "name": "Split Guard",
                                         "files": {"gguf": {}}})
        common = {"model_id": model["id"], "machine_id": self.machine["id"],
                  "framework": "backpack", "format": "gguf", "backend": "webgpu"}
        self.store.add_observation({**common, "id": "split-pass", "revision": "base-20260726",
                                    "conformance": "pass", "metrics": {}}, "test")
        self.store.add_observation({**common, "id": "split-base", "revision": "base",
                                    "conformance": "not_applicable", "metrics": {
            "prefill_tok_s": 100, "decode_tok_s": 20, "prompt_tokens": 512,
            "generated_tokens": 128}}, "test")
        result = self.store.add_observation({**common, "id": "split-drop", "revision": "candidate",
                                             "conformance": "not_applicable", "metrics": {
            "prefill_tok_s": 90, "decode_tok_s": 18, "prompt_tokens": 512,
            "generated_tokens": 128}}, "test")
        self.assertEqual("quarantined", result["validity"])
        self.assertIn("split-base", result["validity_reason"])

    def test_confirmed_regression_remains_auditable_but_not_in_status(self) -> None:
        model = self.store.upsert_model({"id": "confirmed-guard", "name": "Confirmed", "files": {"gguf": {}}})
        common = {"model_id": model["id"], "machine_id": self.machine["id"],
                  "framework": "llamacpp", "format": "gguf", "backend": "vulkan",
                  "conformance": "pass"}
        self.store.add_observation({**common, "id": "confirmed-base", "metrics": {
            "prefill_tok_s": 100, "decode_tok_s": 20, "prompt_tokens": 512,
            "generated_tokens": 128}}, "test")
        result = self.store.add_observation({**common, "id": "confirmed-drop",
            "confirmed_regression_evidence": {"repetitions": 5, "artifact": "paired.json"},
            "metrics": {"prefill_tok_s": 90, "decode_tok_s": 18, "prompt_tokens": 512,
                        "generated_tokens": 128}}, "test")
        self.assertEqual("quarantined", result["validity"])
        self.assertIn("confirmed protected regression", result["validity_reason"])
        self.assertEqual("confirmed-base", self.store.latest_observations()[0]["id"])

    def test_regression_guard_prevents_cumulative_ratcheting(self) -> None:
        model = self.store.upsert_model({"id": "ratchet-guard", "name": "Ratchet", "files": {"gguf": {}}})
        common = {"model_id": model["id"], "machine_id": self.machine["id"],
                  "framework": "backpack", "format": "gguf", "backend": "webgpu",
                  "conformance": "pass"}
        for observation_id, rate in (("ratchet-base", 100), ("ratchet-small", 98.5)):
            result = self.store.add_observation({**common, "id": observation_id, "metrics": {
                "prefill_tok_s": rate, "decode_tok_s": 20, "prompt_tokens": 512,
                "generated_tokens": 128}}, "test")
            self.assertEqual("valid", result["validity"])
        result = self.store.add_observation({**common, "id": "ratchet-cumulative", "metrics": {
            "prefill_tok_s": 97, "decode_tok_s": 20, "prompt_tokens": 512,
            "generated_tokens": 128}}, "test")
        self.assertEqual("quarantined", result["validity"])
        self.assertIn("ratchet-base", result["validity_reason"])

    def test_invalidated_observation_is_auditable_but_not_latest(self) -> None:
        model = self.store.upsert_model({"id": "invalidated", "name": "Invalidated", "files": {"gguf": {}}})
        common = {"model_id": model["id"], "machine_id": self.machine["id"],
                  "framework": "llamacpp", "format": "gguf", "backend": "vulkan",
                  "conformance": "pass"}
        self.store.add_observation({**common, "id": "keep", "metrics": {}}, "test")
        self.store.add_observation({**common, "id": "discard", "metrics": {}}, "test")
        invalid = self.store.invalidate_observation("discard", "wrong executable", "reviewer")
        self.assertEqual("invalid", invalid["validity"])
        self.assertEqual("wrong executable", invalid["validity_reason"])
        self.assertEqual("reviewer", invalid["invalidated_by"])
        self.assertEqual("keep", self.store.latest_observations()[0]["id"])
        audit = self.store.list_observations({"include_invalid": "1"})
        self.assertEqual(2, len(audit))
        self.assertEqual("invalid", next(row for row in audit if row["id"] == "discard")["validity"])

    def test_existing_observation_schema_is_migrated_in_place(self) -> None:
        path = Path(self.tmp.name) / "legacy.db"
        db = sqlite3.connect(path)
        db.execute("""CREATE TABLE observations (
          id TEXT PRIMARY KEY, model_id TEXT NOT NULL, machine_id TEXT NOT NULL,
          framework TEXT NOT NULL, format TEXT NOT NULL, backend TEXT NOT NULL,
          conformance TEXT NOT NULL, conformance_details_json TEXT NOT NULL,
          metrics_json TEXT NOT NULL, revision TEXT, artifacts_json TEXT NOT NULL,
          created_at TEXT NOT NULL)""")
        db.execute("INSERT INTO observations VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", (
            "legacy", "model", "machine", "llamacpp", "gguf", "vulkan", "pass",
            "{}", "{}", "legacy", "[]", "2026-01-01T00:00:00+00:00"))
        db.commit()
        db.close()
        migrated = Store(path)
        row = migrated.list_observations({"include_invalid": "true"})[0]
        self.assertEqual("valid", row["validity"])
        self.assertIsNone(row["validity_reason"])
        migrated.close()

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
            "conformance": "pass", "revision": "old-shape",
            "metrics": {"prefill_tok_s": 999, "decode_tok_s": 999,
                        "prompt_tokens": 128, "generated_tokens": 64},
        }, "test")
        self.assertEqual(0, self.store.reconcile_completed_tasks())
        self.store.add_observation({
            "model_id": model["id"], "machine_id": self.machine["id"],
            "framework": "llamacpp", "format": "gguf", "backend": "vulkan",
            "conformance": "pass", "revision": "b1",
            "metrics": {"prefill_tok_s": 100, "decode_tok_s": 20,
                        "prompt_tokens": 512, "generated_tokens": 128},
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
            "metrics": {"prefill_tok_s": 500, "decode_tok_s": 25,
                        "prompt_tokens": 512, "generated_tokens": 128},
        }, "test")

        self.assertEqual(0, self.store.reconcile_completed_tasks())
        self.store.add_observation({
            "model_id": model["id"], "machine_id": self.machine["id"],
            "framework": "ort", "format": "ort", "backend": "webgpu",
            "conformance": "pass", "revision": "source-build",
            "conformance_details": {"source": f"benchmark task {task['id']}"},
            "metrics": {"prefill_tok_s": 505, "decode_tok_s": 26,
                        "prompt_tokens": 512, "generated_tokens": 128},
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

    def test_successful_profiling_origin_does_not_auto_integrate_optimization(self) -> None:
        task = self.store.create_task({
            "title": "Retain experimental weight layout", "kind": "optimization",
            "hypothesis": "The layout may improve prefill",
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
        current = self.store.get_task(task["id"])
        self.assertNotEqual("integrated", current["state"])
        self.assertIsNone(current["aggregate_verdict"])

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

    def test_history_does_not_average_away_a_protected_regression(self) -> None:
        task = self.store.create_task({
            "title": "Mixed performance result",
            "kind": "optimization",
            "hypothesis": "A gain in one metric must not hide a regression in another.",
            "origin": {"type": "test"},
            "device_policy": {"required": [self.machine["id"]]},
        }, "test")
        self.store.add_history({
            "task_id": task["id"],
            "title": "Mixed result",
            "summary": "Prefill improved but decode regressed.",
            "commit_sha": "deadbeef",
            "gains": {self.machine["name"].replace("-", "_"): {
                "prefill_percent": 27.3,
                "decode_percent": -10.0,
            }},
            "evidence": [{"device": self.machine["name"]}],
        }, "test")

        item = next(row for row in self.store.list_history()
                    if row["task_id"] == task["id"])
        self.assertEqual("Serious regression", item["impact"]["name"])
        self.assertEqual(-10.0, item["impact"]["value"])
        self.assertEqual("Serious regression",
                         item["device_impacts"][0]["impact"]["name"])

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
        self.store.upsert_model({"id": "perf-model", "name": "Perf",
                                 "files": {"gguf": {"path": r"D:\models\model.gguf"}}})
        self.store.add_observation({"model_id": "perf-model", "machine_id": self.machine["id"],
                                    "framework": "backpack", "format": "gguf", "backend": "d3d12",
                                    "conformance": "pass", "metrics": {"prefill_tok_s": 999,
                                        "decode_tok_s": 999, "prompt_tokens": 128,
                                        "generated_tokens": 64}, "revision": "test"}, "test")
        self.store.add_observation({"model_id": "perf-model", "machine_id": self.machine["id"],
                                    "framework": "llamacpp", "format": "gguf", "backend": "vulkan",
                                    "conformance": "pass", "metrics": {}, "revision": "test"}, "test")
        tasks = self.store.ensure_automatic_tasks()
        perf = [task for task in tasks if task["kind"] == "benchmark"]
        runtimes = [runtime for task in perf for runtime in task["manifest"]["runtimes"]]
        self.assertTrue(any(item["framework"] == "backpack" for item in runtimes))
        self.assertTrue(any(item["framework"] == "llamacpp" for item in runtimes))
        self.assertTrue(all(len(task["manifest"]["runtimes"]) == 1 for task in perf))
        self.assertTrue(all(task["origin"]["automation_key"].endswith(
            ":" + task["manifest"]["runtimes"][0]["format"]) for task in perf))
        self.assertTrue(all(task["manifest"]["prompt_tokens"] == 512 for task in perf))
        self.assertTrue(all(task["manifest"]["generated_tokens"] == 128 for task in perf))
        self.store.ensure_runnable_automatic_tasks()
        refreshed = [self.store.get_task(task["id"]) for task in perf]
        llama = next(task for task in refreshed
                     if task["manifest"]["runtimes"][0]["framework"] == "llamacpp")
        self.assertIn("--prompt-tokens", llama["manifest"]["argv"])
        self.assertIn("512", llama["manifest"]["argv"])
        self.assertIn("--required-fact", llama["manifest"]["argv"])
        self.assertIn("--prompt", llama["manifest"]["argv"])

    def test_ort_benchmark_routes_to_ort_adapter_and_onnx_artifact(self) -> None:
        self.store.upsert_model({"id": "ort-perf", "name": "ORT Perf",
                                 "files": {"ort": {"path": r"D:\models\ort-perf"}},
                                 "conformance_spec": {"prompt": "2+2?", "required_fact": "4"}})
        common = {"model_id": "ort-perf", "machine_id": self.machine["id"],
                  "format": "ort", "backend": "webgpu", "conformance": "pass",
                  "metrics": {}, "revision": "test"}
        self.store.add_observation({**common, "framework": "backpack"}, "test")
        self.store.add_observation({**common, "framework": "ort"}, "test")
        tasks = [task for task in self.store.ensure_automatic_tasks()
                 if task["kind"] == "benchmark"]
        self.store.ensure_runnable_automatic_tasks()
        ort = next(self.store.get_task(task["id"]) for task in tasks
                   if task["manifest"]["runtimes"][0]["framework"] == "ort")
        argv = ort["manifest"]["argv"]
        self.assertTrue(str(argv[1]).endswith("benchmark_ort.py"))
        self.assertEqual(r"D:\models\ort-perf", argv[argv.index("--model") + 1])
        self.assertEqual("512", argv[argv.index("--prompt-tokens") + 1])
        self.assertEqual("128", argv[argv.index("--generation-tokens") + 1])

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
        retried = self.store.update_run(run["id"], {"status": "pending", "progress": 0}, "scheduler")
        self.assertIsNone(retried["started_at"])
        self.assertIsNone(retried["completed_at"])

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
