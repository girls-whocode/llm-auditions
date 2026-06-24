"""Development verifier — runs generated code in isolated temporary directories."""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .base import BaseVerifier, VerifierResult

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

_TOOL_CACHE: dict[str, bool] = {}


def _tool_available(name: str) -> bool:
    if name not in _TOOL_CACHE:
        _TOOL_CACHE[name] = shutil.which(name) is not None
    return _TOOL_CACHE[name]


def _safe_env() -> dict[str, str]:
    allow = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": "",
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "LC_ALL": os.environ.get("LC_ALL", "C.UTF-8"),
        "PYTHONPATH": "",
    }
    return allow


def _detect_sandbox_backend() -> str | None:
    if _tool_available("bwrap"):
        return "bubblewrap"
    if _tool_available("podman"):
        return "podman_rootless"
    if _tool_available("docker"):
        return "docker_rootless"
    if _tool_available("systemd-run"):
        return "systemd-run"
    return None


def _run_safe(
    cmd: list[str],
    cwd: Path,
    timeout: int = 20,
    env: dict[str, str] | None = None,
    output_limit: int = 4000,
) -> tuple[int, str, str, bool]:
    merged_env = _safe_env()
    merged_env.update(env or {})
    try:
        result = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
            env=merged_env,
        )
        out = (result.stdout or "")[:output_limit]
        err = (result.stderr or "")[:output_limit]
        truncated = len(result.stdout or "") > output_limit or len(result.stderr or "") > output_limit
        return result.returncode, out, err, truncated
    except subprocess.TimeoutExpired as exc:
        return -1, (exc.stdout or "")[:output_limit], f"timeout after {timeout}s", True
    except Exception as exc:
        return -2, "", str(exc), False


def _extract_code_blocks(text: str) -> list[tuple[str, str]]:
    blocks = []
    for m in re.finditer(r"```(\w*)\n(.*?)```", text, re.DOTALL):
        lang = m.group(1).strip().lower()
        code = m.group(2)
        if code.strip():
            blocks.append((lang, code))
    return blocks


def _workspace_manifest(root: Path) -> list[str]:
    files = []
    for p in sorted(root.rglob("*")):
        if p.is_file():
            files.append(str(p.relative_to(root)))
    return files


def _reject_unsafe_path(path_str: str) -> bool:
    p = Path(path_str)
    return p.is_absolute() or ".." in p.parts


class DevelopmentVerifier(BaseVerifier):
    """Runs generated code through syntax and behavioral tests in isolated temp workspaces."""

    name = "development"

    def verify(self, task: Any, response: Any) -> VerifierResult:
        content = response.content if hasattr(response, "content") else str(response)
        blocks = _extract_code_blocks(content)
        if not blocks:
            return VerifierResult(False, 0.0, "No code block found", {"checks": []})

        sandbox_backend = _detect_sandbox_backend()
        if sandbox_backend is None:
            return VerifierResult(
                False,
                0.0,
                "Sandbox unavailable; refusing host execution",
                {
                    "checks": [],
                    "sandbox_backend": "disabled",
                    "sandbox_unavailable": True,
                    "network_isolation": False,
                    "filesystem_isolation": False,
                    "timeout_enforced": False,
                },
            )

        # Pass-5 policy: execution is permanently disabled until a real backend
        # implementation guarantees process, filesystem, and network isolation.
        return VerifierResult(
            False,
            0.0,
            "Sandbox backend detected but execution is disabled by policy",
            {
                "checks": [],
                "sandbox_backend": sandbox_backend,
                "sandbox_unavailable": True,
                "network_isolation": False,
                "filesystem_isolation": False,
                "timeout_enforced": False,
                "policy_refusal": True,
                "reason": "sandbox_unavailable",
            },
        )

        dev_cfg = getattr(task, "development", {}) or {}
        fixture_dir = dev_cfg.get("fixture_directory", "")
        entry_file = dev_cfg.get("entry_file", "solution.py")
        timeout_seconds = int(dev_cfg.get("timeout_seconds", 20))
        test_command = [str(x) for x in dev_cfg.get("test_command", ["python", "-m", "pytest", "-q"])]

        if _reject_unsafe_path(entry_file):
            return VerifierResult(False, 0.0, "Unsafe entry_file path", {"entry_file": entry_file})

        if fixture_dir and _reject_unsafe_path(fixture_dir):
            return VerifierResult(False, 0.0, "Unsafe fixture_directory path", {"fixture_directory": fixture_dir})

        with tempfile.TemporaryDirectory(prefix="audition_dev_") as td:
            work = Path(td)

            if fixture_dir:
                src_fixture = PROJECT_ROOT / fixture_dir
                if not src_fixture.exists():
                    return VerifierResult(False, 0.0, f"Fixture directory missing: {fixture_dir}", {})
                shutil.copytree(src_fixture, work, dirs_exist_ok=True)

            lang, code = blocks[0]
            if lang in ("", "python", "py"):
                code_path = work / entry_file
            elif lang in ("bash", "sh", "shell"):
                code_path = work / entry_file.replace(".py", ".sh")
            else:
                code_path = work / entry_file

            code_path.parent.mkdir(parents=True, exist_ok=True)
            code_path.write_text(code, encoding="utf-8")

            checks: list[dict[str, Any]] = []

            if code_path.suffix == ".py":
                rc, out, err, trunc = _run_safe(["python3", "-m", "py_compile", str(code_path)], work, timeout=timeout_seconds)
                checks.append({
                    "check": "syntax",
                    "passed": rc == 0,
                    "exit_status": rc,
                    "stdout": out,
                    "stderr": err,
                    "output_truncated": trunc,
                })
                if _tool_available("ruff"):
                    rc, out, err, trunc = _run_safe(["ruff", "check", str(code_path)], work, timeout=timeout_seconds)
                    checks.append({
                        "check": "lint",
                        "passed": rc == 0,
                        "exit_status": rc,
                        "stdout": out,
                        "stderr": err,
                        "output_truncated": trunc,
                    })
            elif code_path.suffix == ".sh":
                rc, out, err, trunc = _run_safe(["bash", "-n", str(code_path)], work, timeout=timeout_seconds)
                checks.append({
                    "check": "syntax",
                    "passed": rc == 0,
                    "exit_status": rc,
                    "stdout": out,
                    "stderr": err,
                    "output_truncated": trunc,
                })

            rc, out, err, trunc = _run_safe(test_command, work, timeout=timeout_seconds)
            checks.append(
                {
                    "check": "tests",
                    "passed": rc == 0,
                    "exit_status": rc,
                    "stdout": out,
                    "stderr": err,
                    "timeout": rc == -1,
                    "output_truncated": trunc,
                }
            )

            passed = all(c.get("passed", False) for c in checks if c.get("check") in ("syntax", "tests"))
            score = 1.0 if passed else 0.0

            details = {
                "sandbox_backend": sandbox_backend,
                "sandbox_unavailable": False,
                "network_isolation": sandbox_backend in {"bubblewrap", "podman_rootless", "docker_rootless", "systemd-run"},
                "filesystem_isolation": True,
                "resource_limits": {"timeout_seconds": timeout_seconds, "output_limit": 4000},
                "syntax_status": next((c for c in checks if c["check"] == "syntax"), {}),
                "lint_status": next((c for c in checks if c["check"] == "lint"), {}),
                "test_collection_status": {"passed": rc == 0, "exit_status": rc},
                "individual_test_results": out.splitlines()[-20:],
                "exit_status": rc,
                "timeout_status": rc == -1,
                "process_termination_result": "timeout_killed" if rc == -1 else "completed",
                "stdout": out,
                "stderr": err,
                "workspace_file_manifest": _workspace_manifest(work),
                "hard_gate_status": not passed,
            }

            return VerifierResult(
                passed,
                score,
                "Development behavioral checks passed" if passed else "Development behavioral checks failed",
                details,
            )
