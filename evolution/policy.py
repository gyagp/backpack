from __future__ import annotations
import math
import statistics
from collections import defaultdict
from typing import Any

from .domain import DomainError, Thresholds
from .store import Store


def _cv_percent(samples: list[float]) -> float:
    mean = statistics.fmean(samples)
    if len(samples) < 2 or mean == 0:
        return 0.0
    return abs(statistics.stdev(samples) / mean * 100.0)


def _lower_is_better(metric: str) -> bool:
    name = metric.lower()
    return (name.endswith("_ms") or name.endswith("_latency") or
            "latency" in name or name.startswith("time_to_") or
            name in {"gpu_time", "cpu_time", "memory_bytes", "peak_memory_bytes"})


def _required_correctness_checks(task: dict[str, Any]) -> list[str]:
    """Return explicitly applicable lossless gates for this experiment.

    ``manifest.checks`` also contains non-conformance checks in older tasks, so
    only names from the layered validation vocabulary are inferred there.
    Tasks may use ``decision_policy.required_correctness_checks`` to require a
    project-specific check without changing that vocabulary.
    """
    configured = task.get("decision_policy", {}).get("required_correctness_checks")
    if configured is None:
        configured = task.get("manifest", {}).get("correctness_checks")
    if configured is None:
        known = {
            "exact-output", "intermediate-parity", "layer-parity",
            "held-back-prompt", "held-back-shape", "fallback-parity",
            "recurrent-state-parity", "draft-logit-parity",
            "accepted-token-parity",
        }
        configured = [item for item in task.get("manifest", {}).get("checks", [])
                      if item in known]
    return list(dict.fromkeys(str(item) for item in (configured or [])))


def _check_result(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, dict):
        if "passed" in value:
            return bool(value["passed"])
        value = value.get("status")
    if isinstance(value, str):
        if value.lower() in {"pass", "passed", "ok"}:
            return True
        if value.lower() in {"fail", "failed", "error"}:
            return False
    return None


def _layered_correctness(task: dict[str, Any], correctness: dict[str, Any]) -> tuple[list[str], list[str]]:
    checks = correctness.get("checks") if isinstance(correctness.get("checks"), dict) else {}
    missing: list[str] = []
    failed: list[str] = []
    for name in _required_correctness_checks(task):
        result = _check_result(checks.get(name, correctness.get(name)))
        if result is None:
            missing.append(name)
        elif not result:
            failed.append(name)
    return missing, failed


def machine_matches(machine: dict[str, Any], selector: dict[str, Any]) -> bool:
    values = {**machine.get("fingerprint", {}), **machine.get("labels", {})}
    return all(values.get(key) == value for key, value in selector.items())


def required_machine_ids(task: dict[str, Any], machines: list[dict[str, Any]]) -> tuple[set[str], list[str]]:
    required: set[str] = set()
    missing: list[str] = []
    for item in task.get("device_policy", {}).get("required", []):
        if isinstance(item, str):
            match = next((machine for machine in machines
                          if machine["id"] == item or machine.get("name") == item), None)
            if match:
                required.add(match["id"])
            else:
                missing.append(item)
            continue
        selector = item.get("selector", {})
        count = int(item.get("count", 1))
        matches = [m["id"] for m in machines if machine_matches(m, selector)]
        if len(matches) < count:
            missing.append(str(selector))
        required.update(matches[:count])
    return required, missing


class PolicyEngine:
    def __init__(self, store: Store):
        self.store = store

    def evaluate(self, task_id: str) -> dict[str, Any]:
        task = self.store.get_task(task_id)
        if not task:
            raise DomainError("task not found")
        evidence = self.store.list_evidence(task_id)
        machines = self.store.list_machines()
        required_ids, missing_selectors = required_machine_ids(task, machines)
        if missing_selectors:
            reason = "required device selectors unavailable: " + ", ".join(missing_selectors)
            self.store.replace_evaluations(task_id, [], "blocked", reason)
            return {"aggregate_verdict": "blocked", "reason": reason, "evaluations": []}
        if not required_ids:
            reason = "device policy has no resolved required machines"
            self.store.replace_evaluations(task_id, [], "blocked", reason)
            return {"aggregate_verdict": "blocked", "reason": reason, "evaluations": []}

        grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
        for item in evidence:
            grouped[(item["machine_id"], item["metric"])][item["variant"]] = item

        thresholds = Thresholds.from_policy(task.get("decision_policy"))
        metrics = task.get("manifest", {}).get("metrics") or ["decode_tok_s"]
        protected = set(task.get("decision_policy", {}).get("protected_metrics", metrics))
        rows: list[dict[str, Any]] = []
        for machine_id in required_ids:
            for metric in metrics:
                pair = grouped.get((machine_id, metric), {})
                if "base" not in pair or "candidate" not in pair:
                    rows.append({"machine_id": machine_id, "metric": metric, "verdict": "inconclusive",
                                 "details": {"reason": "missing base or candidate evidence"}})
                    continue
                base, candidate = pair["base"], pair["candidate"]
                expected_base = task.get("base_sha")
                expected_candidate = task.get("candidate_sha")
                if ((expected_base and base.get("commit_sha") != expected_base) or
                        (expected_candidate and candidate.get("commit_sha") != expected_candidate)):
                    rows.append({
                        "machine_id": machine_id, "metric": metric,
                        "verdict": "inconclusive",
                        "details": {
                            "reason": "evidence revision does not match frozen candidate",
                            "expected_base_sha": expected_base,
                            "observed_base_sha": base.get("commit_sha"),
                            "expected_candidate_sha": expected_candidate,
                            "observed_candidate_sha": candidate.get("commit_sha"),
                        },
                    })
                    continue
                correctness = candidate.get("correctness", {})
                if not correctness.get("passed", False):
                    rows.append({"machine_id": machine_id, "metric": metric, "verdict": "negative",
                                 "details": {"reason": "candidate correctness failed", "correctness": correctness,
                                             "correctness_failed": True}})
                    continue
                missing_checks, failed_checks = _layered_correctness(task, correctness)
                if failed_checks:
                    rows.append({"machine_id": machine_id, "metric": metric, "verdict": "negative",
                                 "details": {"reason": "required correctness check failed",
                                             "failed_checks": failed_checks,
                                             "correctness": correctness, "correctness_failed": True}})
                    continue
                if missing_checks:
                    rows.append({"machine_id": machine_id, "metric": metric, "verdict": "inconclusive",
                                 "details": {"reason": "required correctness checks are missing",
                                             "missing_checks": missing_checks,
                                             "correctness": correctness}})
                    continue
                base_samples = [float(v) for v in base["samples"]]
                candidate_samples = [float(v) for v in candidate["samples"]]
                base_median = statistics.median(base_samples)
                candidate_median = statistics.median(candidate_samples)
                lower_is_better = _lower_is_better(metric)
                if lower_is_better:
                    delta = math.inf if candidate_median == 0 else (base_median / candidate_median - 1.0) * 100.0
                else:
                    delta = math.inf if base_median == 0 else (candidate_median / base_median - 1.0) * 100.0
                cv = max(_cv_percent(base_samples), _cv_percent(candidate_samples))
                # Variability must not hide an unambiguous protected
                # regression. If even the best candidate sample is worse than
                # the worst base sample by the rejection threshold, the two
                # measured bands do not overlap and more repetitions cannot
                # turn this candidate into a safe milestone.
                if lower_is_better:
                    separated_delta = (math.inf if min(candidate_samples) == 0 else
                                       (max(base_samples) / min(candidate_samples) - 1.0) * 100.0)
                else:
                    separated_delta = (math.inf if min(base_samples) == 0 else
                                       (max(candidate_samples) / min(base_samples) - 1.0) * 100.0)
                separated_regression = (metric in protected and
                                        separated_delta < thresholds.negative_percent)
                if separated_regression:
                    verdict, why = "negative", "non-overlapping regression band exceeds threshold"
                elif cv > thresholds.max_cv_percent:
                    verdict = "inconclusive"
                    why = "sample variability exceeds policy"
                elif delta > thresholds.positive_percent:
                    verdict, why = "positive", "improvement exceeds threshold"
                elif delta < thresholds.negative_percent:
                    verdict, why = "negative", "regression exceeds threshold"
                else:
                    verdict, why = "neutral", "delta is within equivalence band"
                rows.append({
                    "machine_id": machine_id, "metric": metric, "verdict": verdict,
                    "base_median": base_median, "candidate_median": candidate_median,
                    "delta_percent": delta,
                    "details": {"reason": why, "max_cv_percent": cv,
                                "separated_delta_percent": separated_delta,
                                "non_overlapping_regression": separated_regression,
                                "protected": metric in protected,
                                "direction": "lower_is_better" if lower_is_better else "higher_is_better"},
                })

        verdicts = {row["verdict"] for row in rows}
        protected_negative = any(r["verdict"] == "negative" and r["metric"] in protected for r in rows)
        correctness_negative = any(r["details"].get("correctness_failed") for r in rows)
        if correctness_negative:
            aggregate, reason = "reject", "correctness failed on a required device"
        elif protected_negative:
            aggregate, reason = "reject", "a protected metric regressed on a required device"
        elif "inconclusive" in verdicts:
            aggregate, reason = "blocked", "required evidence is missing or too noisy"
        elif "positive" in verdicts:
            aggregate, reason = "accept", "all required results are positive or neutral"
        else:
            aggregate, reason = "debate", "all required results are neutral"

        self.store.replace_evaluations(task_id, rows, aggregate, reason)
        if aggregate == "debate" and not any(d["status"] == "pending" for d in self.store.list_decisions() if d["task_id"] == task_id):
            self.store.create_decision(task_id, "Should this mixed/neutral candidate be integrated?",
                                       ["approve", "reject", "request_more_evidence"], "request_more_evidence")
        return {"aggregate_verdict": aggregate, "reason": reason, "evaluations": rows}
