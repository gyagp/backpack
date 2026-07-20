from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from .domain import DomainError


ROOT = Path(__file__).resolve().parents[1]


def provision_winrm(computer_name: str, server_url: str) -> dict[str, Any]:
    script = Path(__file__).with_name("provision_winrm.ps1")
    command = [
        "powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script),
        "-ComputerName", computer_name, "-ServerUrl", server_url,
    ]
    try:
        result = subprocess.run(command, text=True, capture_output=True, timeout=7200, shell=False)
    except subprocess.TimeoutExpired as exc:
        raise DomainError("WinRM provisioning timed out after two hours") from exc
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()
        if "implicit credentials" in detail or "TrustedHosts" in detail:
            detail += "\nConfigure WinRM and TrustedHosts on webgfx-104 as Administrator."
        raise DomainError(f"WinRM provisioning failed: {detail[-4000:]}")
    return {"transport": "winrm", "target": computer_name, "output": result.stdout.strip()[-4000:]}
