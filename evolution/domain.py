from __future__ import annotations
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


TASK_STATES = {
    "proposed", "triaged", "implementing", "candidate_ready", "validating",
    "evaluating", "debating", "awaiting_human", "ready_to_merge",
    "integrating", "integrated", "observing", "rejected", "blocked",
    "failed", "reverted",
}

TRANSITIONS = {
    "proposed": {"triaged", "rejected"},
    "triaged": {"implementing", "candidate_ready", "blocked", "rejected"},
    "implementing": {"candidate_ready", "blocked", "failed"},
    "candidate_ready": {"validating", "failed"},
    "validating": {"evaluating", "blocked", "failed"},
    "evaluating": {"validating", "debating", "awaiting_human", "ready_to_merge", "rejected", "blocked"},
    "debating": {"validating", "awaiting_human", "ready_to_merge", "rejected"},
    "awaiting_human": {"validating", "ready_to_merge", "rejected"},
    "ready_to_merge": {"integrating", "blocked"},
    "integrating": {"integrated", "blocked", "failed"},
    "integrated": {"observing", "reverted"},
    "observing": {"reverted"},
    "blocked": {"triaged", "implementing", "candidate_ready", "validating", "evaluating", "rejected"},
    "failed": {"implementing", "candidate_ready", "rejected"},
    "rejected": set(),
    "reverted": set(),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def json_text(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def parse_json(value: str | None, default: Any = None) -> Any:
    if not value:
        return default
    return json.loads(value)


class DomainError(ValueError):
    pass


def require(value: Any, name: str) -> Any:
    if value is None or value == "" or value == []:
        raise DomainError(f"{name} is required")
    return value


def validate_transition(current: str, target: str) -> None:
    if target not in TASK_STATES:
        raise DomainError(f"unknown task state: {target}")
    if target not in TRANSITIONS.get(current, set()):
        raise DomainError(f"invalid task transition: {current} -> {target}")


@dataclass(frozen=True)
class Thresholds:
    positive_percent: float = 2.0
    negative_percent: float = -2.0
    max_cv_percent: float = 5.0

    @classmethod
    def from_policy(cls, policy: dict[str, Any] | None) -> "Thresholds":
        values = (policy or {}).get("thresholds", {})
        return cls(
            positive_percent=float(values.get("positive_percent", 2.0)),
            negative_percent=float(values.get("negative_percent", -2.0)),
            max_cv_percent=float(values.get("max_cv_percent", 5.0)),
        )
