from __future__ import annotations

import hashlib
import json
from pathlib import Path

ENGINE_VERSION = "0.10.0"
TASK_SUITE_VERSION = "2"
SCORING_VERSION = "2"
VERIFIER_VERSION = "2"
REPORT_VERSION = "2"


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _hash_files(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for p in sorted(paths):
        digest.update(p.as_posix().encode())
        digest.update(_hash_file(p).encode())
    return digest.hexdigest()[:16]


def execution_source_hashes(project_root: Path) -> dict[str, str]:
    src = project_root / "src" / "llm_auditions"
    verifier_paths = list((src / "verifiers").glob("*.py"))
    payload = {
        "engine": _hash_files([src / "runner.py", src / "ollama_client.py", src / "models.py"]),
        "scoring": _hash_files([src / "scoring.py", src / "models.py"]),
        "verifier": _hash_files(verifier_paths + [src / "models.py"]),
        "report": _hash_files([src / "reporting.py", src / "models.py"]),
    }
    return payload


def version_payload() -> dict[str, str]:
    return {
        "engine_version": ENGINE_VERSION,
        "task_suite_version": TASK_SUITE_VERSION,
        "scoring_version": SCORING_VERSION,
        "verifier_version": VERIFIER_VERSION,
        "report_version": REPORT_VERSION,
    }


def short_version_tuple() -> tuple[str, str, str, str, str]:
    return ENGINE_VERSION, TASK_SUITE_VERSION, SCORING_VERSION, VERIFIER_VERSION, REPORT_VERSION
