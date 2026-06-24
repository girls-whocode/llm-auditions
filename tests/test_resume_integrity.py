from __future__ import annotations

import json
from pathlib import Path

from llm_auditions.configuration import Configuration
from llm_auditions.runner import AuditionRunner

PROJECT_ROOT = Path(__file__).parent.parent


def test_resume_digest_mismatch_refuses(tmp_path: Path):
    cfg = Configuration(project_root=PROJECT_ROOT)
    cfg.load()
    run = tmp_path / "run"
    run.mkdir()
    (run / "run_manifest.json").write_text(json.dumps({
        "run_id": "r1",
        "created_at": "2026-01-01T00:00:00Z",
        "profile": "smoke",
        "engine_version": "0.9.0",
        "task_suite_version": "1",
        "scoring_version": "1",
        "verifier_version": "1",
        "report_version": "1",
        "ollama_version": "test",
        "ollama_base_url": "http://localhost:11434",
        "models": [],
        "model_digests": {"gemma4:12b": "old_digest"},
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
        "task_manifest_hash": "x",
        "config_hashes": {},
        "schema_hashes": {},
        "fixture_hashes": {},
        "execution_plan_hash": "y",
        "git_commit": "unknown",
        "git_dirty": False,
    }))
    (run / "task_manifest.json").write_text(json.dumps({"run_id": "r1", "profile": "smoke", "requests": []}))
    runner = AuditionRunner(cfg, run, profile="smoke")
    try:
        runner.load_existing_run()
    except RuntimeError as exc:
        assert "Resume refused" in str(exc)
    else:
        raise AssertionError("expected resume mismatch refusal")
