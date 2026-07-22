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
                correctness = candidate.get("correctness", {})
                if not correctness.get("passed", False):
                    rows.append({"machine_id": machine_id, "metric": metric, "verdict": "negative",
                                 "details": {"reason": "candidate correctness failed", "correctness": correctness}})
                    continue
                base_samples = [float(v) for v in base["samples"]]
                candidate_samples = [float(v) for v in candidate["samples"]]
                base_median = statistics.median(base_samples)
                candidate_median = statistics.median(candidate_samples)
                delta = math.inf if base_median == 0 else (candidate_median / base_median - 1.0) * 100.0
                cv = max(_cv_percent(base_samples), _cv_percent(candidate_samples))
                if cv > thresholds.max_cv_percent:
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
                    "details": {"reason": why, "max_cv_percent": cv, "protected": metric in protected},
                })

        verdicts = {row["verdict"] for row in rows}
        protected_negative = any(r["verdict"] == "negative" and r["metric"] in protected for r in rows)
        correctness_negative = any(r["details"].get("reason") == "candidate correctness failed" for r in rows)
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
