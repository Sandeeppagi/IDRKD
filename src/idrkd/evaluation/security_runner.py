"""Executable security evidence for model promotion."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import UTC, datetime
import shlex
import subprocess
import sys
import time
from typing import Any


DEFAULT_SECURITY_TESTS = (
    "tests/unit/test_week5_security_hardening.py",
    "tests/unit/test_week6_mcp_a2a_security.py",
)

RunCommand = Callable[..., subprocess.CompletedProcess[str]]


def run_security_suite(
    *,
    tests: Sequence[str] = DEFAULT_SECURITY_TESTS,
    cwd: str = ".",
    runner: RunCommand = subprocess.run,
) -> dict[str, Any]:
    command = [sys.executable, "-m", "pytest", "-q", *tests]
    started = time.perf_counter()
    completed = runner(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "suite": "tenant-and-agent-security",
        "tests": list(tests),
        "command": shlex.join(command),
        "duration_seconds": time.perf_counter() - started,
        "returncode": completed.returncode,
        "passed": completed.returncode == 0,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
