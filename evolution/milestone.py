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

    def _is_ancestor(self, older: str, newer: str) -> bool:
        result = subprocess.run(
            ["git", "-C", str(self.repo), "merge-base", "--is-ancestor", older, newer],
            text=True, capture_output=True, timeout=120, shell=False)
        return result.returncode == 0

    def _contains_candidate_patch(self, head: str, candidate: str) -> bool:
        if self._is_ancestor(candidate, head):
            return True
        # Experiments are commonly cherry-picked onto a base that advanced
        # while validation was running. ``git cherry`` recognizes the same
        # patch without pretending that the unmeasured integration SHA was the
        # measured candidate revision.
        rows = self._git("cherry", head, candidate).splitlines()
        return any(row.startswith("- ") and candidate.startswith(row[2:].strip())
                   or row.startswith("- ") and row[2:].strip().startswith(candidate)
                   for row in rows)

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
        candidate = task["candidate_sha"]
        try:
            # Force-with-lease prevents silently overwriting a base advanced by another publisher.
            current = self._git("ls-remote", self.remote, self.remote_ref)
            lease = current.split()[0] if current else ""
            head = self._git("rev-parse", "HEAD")
            if lease and self._is_ancestor(lease, head) and \
                    self._contains_candidate_patch(head, candidate):
                publish_sha = head
            elif (not lease or self._is_ancestor(lease, candidate)):
                publish_sha = candidate
            else:
                raise DomainError(
                    "milestone base advanced and the current branch does not contain the accepted candidate patch")
            milestone = self.store.create_milestone(
                task_id, publish_sha, self.remote, self.remote_ref)
            lease_arg = f"--force-with-lease={self.remote_ref}:{lease}"
            self._git("push", lease_arg, self.remote,
                      f"{publish_sha}:{self.remote_ref}", timeout=300)
            return self.store.finish_milestone(milestone["id"], True)
        except Exception as exc:
            if "milestone" in locals():
                self.store.finish_milestone(milestone["id"], False, str(exc))
            raise
