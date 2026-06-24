from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from llm_auditions.configuration import Configuration
from llm_auditions.models import ModelInfo
from llm_auditions.runner import AuditionRunner
from llm_auditions.versioning import execution_source_hashes

PROJECT_ROOT = Path(__file__).parent.parent


def _cfg() -> Configuration:
    c = Configuration(project_root=PROJECT_ROOT)
    c.load()
    return c


def _write_manifest(run: Path, source_hashes: dict[str, str]) -> None:
    task_manifest = {"run_id": "resume-source-hash", "profile": "smoke", "requests": []}
    task_manifest_hash = hashlib.sha256(json.dumps(task_manifest, sort_keys=True).encode()).hexdigest()[:16]
    execution_plan_hash = hashlib.sha256(json.dumps(task_manifest["requests"], sort_keys=True).encode()).hexdigest()[:16]
    payload = {
        "run_id": "resume-source-hash",
        "created_at": "2026-01-01T00:00:00Z",
        "profile": "smoke",
        "engine_version": "0.10.0",
        "task_suite_version": "2",
        "scoring_version": "2",
        "verifier_version": "2",
        "report_version": "2",
        "ollama_version": "test",
        "ollama_base_url": "http://localhost:11434",
        "models": [],
        "model_digests": {"gemma4:12b": "sha256:match"},
        "environment": {},
        "task_count": 1,
        "candidate_count": 1,
        "model_count": 1,
        "request_count": 1,
        "teams_included": ["baseline"],
        "roles_included": ["baseline.baseline_all"],
        "models_included": ["gemma4:12b"],
        "requested_think_modes": ["false"],
        "structured_output_modes": ["prompt_only"],
        "task_manifest_hash": task_manifest_hash,
        "config_hashes": {},
        "schema_hashes": {},
        "fixture_hashes": {},
        "execution_source_hashes": source_hashes,
        "execution_plan_hash": execution_plan_hash,
        "git_commit": "unknown",
        "git_dirty": False,
    }
    (run / "run_manifest.json").write_text(json.dumps(payload))
    (run / "task_manifest.json").write_text(json.dumps(task_manifest))


def test_resume_matching_source_hashes_allows(monkeypatch, tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    _write_manifest(run, execution_source_hashes(PROJECT_ROOT))

    runner = AuditionRunner(_cfg(), run, profile="smoke")
    monkeypatch.setattr(
        runner.client,
        "list_models",
        lambda: [ModelInfo(name="gemma4:12b", id="sha256:matc", full_digest="sha256:match", size="1", modified="")],
    )

    runner.load_existing_run()


def test_resume_source_hash_mismatch_refuses(monkeypatch, tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    hashes = execution_source_hashes(PROJECT_ROOT)
    hashes["engine"] = "0000000000000000"
    _write_manifest(run, hashes)

    runner = AuditionRunner(_cfg(), run, profile="smoke")
    monkeypatch.setattr(
        runner.client,
        "list_models",
        lambda: [ModelInfo(name="gemma4:12b", id="sha256:matc", full_digest="sha256:match", size="1", modified="")],
    )

    with pytest.raises(RuntimeError) as exc:
        runner.load_existing_run()
    assert "execution source hash mismatch" in str(exc.value)
