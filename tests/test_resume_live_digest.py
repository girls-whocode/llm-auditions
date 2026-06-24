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


def _write_manifest(run: Path, model_digest: str = "sha256:abc", scoring_version: str = "2", verifier_version: str = "2") -> None:
    task_manifest = {"run_id": "resume-live-digest", "profile": "smoke", "requests": []}
    task_manifest_hash = hashlib.sha256(json.dumps(task_manifest, sort_keys=True).encode()).hexdigest()[:16]
    execution_plan_hash = hashlib.sha256(json.dumps(task_manifest["requests"], sort_keys=True).encode()).hexdigest()[:16]
    payload = {
        "run_id": "resume-live-digest",
        "created_at": "2026-01-01T00:00:00Z",
        "profile": "smoke",
        "engine_version": "0.10.0",
        "task_suite_version": "2",
        "scoring_version": scoring_version,
        "verifier_version": verifier_version,
        "report_version": "2",
        "ollama_version": "test",
        "ollama_base_url": "http://localhost:11434",
        "models": [],
        "model_digests": {"gemma4:12b": model_digest},
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
        "execution_source_hashes": execution_source_hashes(PROJECT_ROOT),
        "execution_plan_hash": execution_plan_hash,
        "git_commit": "unknown",
        "git_dirty": False,
    }
    (run / "run_manifest.json").write_text(json.dumps(payload))
    (run / "task_manifest.json").write_text(json.dumps(task_manifest))


def _cfg() -> Configuration:
    c = Configuration(project_root=PROJECT_ROOT)
    c.load()
    return c


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_resume_matching_live_digest_allows(monkeypatch, tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    _write_manifest(run, model_digest="sha256:match")

    runner = AuditionRunner(_cfg(), run, profile="smoke")
    monkeypatch.setattr(runner.client, "list_models", lambda: [ModelInfo(name="gemma4:12b", id="sha256:matc", full_digest="sha256:match", size="1", modified="")])

    before_manifest = _hash(run / "run_manifest.json")
    before_task_manifest = _hash(run / "task_manifest.json")
    runner.load_existing_run()

    assert _hash(run / "run_manifest.json") == before_manifest
    assert _hash(run / "task_manifest.json") == before_task_manifest


def test_resume_changed_live_digest_refuses(monkeypatch, tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    _write_manifest(run, model_digest="sha256:old")

    runner = AuditionRunner(_cfg(), run, profile="smoke")
    monkeypatch.setattr(runner.client, "list_models", lambda: [ModelInfo(name="gemma4:12b", id="sha256:new", full_digest="sha256:new", size="1", modified="")])

    with pytest.raises(RuntimeError) as exc:
        runner.load_existing_run()
    msg = str(exc.value)
    assert "Resume refused:" in msg
    assert "model: gemma4:12b" in msg
    assert "stored digest: sha256:old" in msg
    assert "current digest: sha256:new" in msg


def test_resume_missing_live_model_refuses(monkeypatch, tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    _write_manifest(run, model_digest="sha256:old")

    runner = AuditionRunner(_cfg(), run, profile="smoke")
    monkeypatch.setattr(runner.client, "list_models", lambda: [])

    with pytest.raises(RuntimeError) as exc:
        runner.load_existing_run()
    assert "model: gemma4:12b" in str(exc.value)


def test_resume_digest_failure_makes_zero_chat_calls(monkeypatch, tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    _write_manifest(run, model_digest="sha256:old")

    runner = AuditionRunner(_cfg(), run, profile="smoke")

    calls = {"chat": 0}

    def _chat(**kwargs):
        calls["chat"] += 1
        raise AssertionError("chat must not be called")

    monkeypatch.setattr(runner.client, "chat", _chat)
    monkeypatch.setattr(runner.client, "list_models", lambda: [ModelInfo(name="gemma4:12b", id="sha256:new", full_digest="sha256:new", size="1", modified="")])

    with pytest.raises(RuntimeError):
        runner.load_existing_run()
    assert calls["chat"] == 0


def test_resume_scoring_version_mismatch_refuses(monkeypatch, tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    _write_manifest(run, model_digest="sha256:match", scoring_version="999")

    runner = AuditionRunner(_cfg(), run, profile="smoke")
    monkeypatch.setattr(runner.client, "list_models", lambda: [ModelInfo(name="gemma4:12b", id="sha256:matc", full_digest="sha256:match", size="1", modified="")])

    with pytest.raises(RuntimeError) as exc:
        runner.load_existing_run()
    assert "scoring_version" in str(exc.value)


def test_resume_verifier_version_mismatch_refuses(monkeypatch, tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    _write_manifest(run, model_digest="sha256:match", verifier_version="999")

    runner = AuditionRunner(_cfg(), run, profile="smoke")
    monkeypatch.setattr(runner.client, "list_models", lambda: [ModelInfo(name="gemma4:12b", id="sha256:matc", full_digest="sha256:match", size="1", modified="")])

    with pytest.raises(RuntimeError) as exc:
        runner.load_existing_run()
    assert "verifier_version" in str(exc.value)
