from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from .domain import DomainError
from .store import Store


class MilestonePublisher:
    """Publishes an accepted candidate SHA without touching the working tree."""

    def __init__(self, store: Store, repo: Path, remote: str = "origin",
                 remote_ref: str = "refs/heads/evolution/base"):
        self.store = store
        self.repo = repo
        self.remote = remote
        self.remote_ref = remote_ref

    def _git(self, *args: str, timeout: int = 120) -> str:
        result = subprocess.run(["git", "-C", str(self.repo), *args], text=True,
                                capture_output=True, timeout=timeout, shell=False)
        if result.returncode:
            raise DomainError((result.stderr or result.stdout).strip())
        return result.stdout.strip()

    def validate(self, task_id: str) -> dict[str, Any]:
        task = self.store.get_task(task_id)
        if not task:
            raise DomainError("task not found")
        if task["state"] != "integrating":
            raise DomainError("task must be in integrating state")
        if task.get("aggregate_verdict") != "accept":
            raise DomainError("only an accepted required-device verdict can become a milestone")
        sha = task.get("candidate_sha")
        if not sha:
            raise DomainError("candidate SHA is not frozen")
        resolved = self._git("rev-parse", f"{sha}^{{commit}}")
        if resolved != sha and not resolved.startswith(sha):
            raise DomainError("candidate SHA does not resolve to the declared commit")
        return task

    def publish(self, task_id: str) -> dict[str, Any]:
        task = self.validate(task_id)
        sha = task["candidate_sha"]
        milestone = self.store.create_milestone(task_id, sha, self.remote, self.remote_ref)
        try:
            # Force-with-lease prevents silently overwriting a base advanced by another publisher.
            current = self._git("ls-remote", self.remote, self.remote_ref)
            lease = current.split()[0] if current else ""
            lease_arg = f"--force-with-lease={self.remote_ref}:{lease}"
            self._git("push", lease_arg, self.remote, f"{sha}:{self.remote_ref}", timeout=300)
            return self.store.finish_milestone(milestone["id"], True)
        except Exception as exc:
            self.store.finish_milestone(milestone["id"], False, str(exc))
            raise
