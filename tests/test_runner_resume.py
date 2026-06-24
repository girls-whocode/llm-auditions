from __future__ import annotations

import hashlib
import json
from pathlib import Path

from llm_auditions.configuration import Configuration
from llm_auditions.models import TaskDefinition
from llm_auditions.runner import AuditionRunner


PROJECT_ROOT = Path(__file__).parent.parent


class _StubClient:
    def __init__(self):
        self.calls = 0

    def chat(self, **kwargs):
        self.calls += 1
        raise AssertionError("chat must not be called for completed identity")


def _task() -> TaskDefinition:
    return TaskDefinition(
        id="resume_task_001",
        team="baseline",
        role="baseline_all",
        prompt="Say hello",
        system_prompt="",
        think_modes=["false"],
        verifier="structure",
        required_json_schema="",
        structured_output_mode="prompt_only",
    )


def test_resume_completed_result_skips_without_ollama(tmp_path: Path):
    cfg = Configuration(project_root=PROJECT_ROOT)
    cfg.load()
    task = _task()

    runner = AuditionRunner(cfg, tmp_path / "run", profile="smoke")
    model_name = cfg.get_role_candidates("baseline", "baseline_all")[0]
    pre = runner._make_identity(task, model_name, "false", "false", None, None)
    key = pre.pre_request_key()

    (tmp_path / "run").mkdir(parents=True)
    task_manifest = {"run_id": "rid", "profile": "smoke", "requests": []}
    task_manifest_hash = hashlib.sha256(json.dumps(task_manifest, sort_keys=True).encode()).hexdigest()[:16]
    execution_plan_hash = hashlib.sha256(json.dumps(task_manifest["requests"], sort_keys=True).encode()).hexdigest()[:16]
    (tmp_path / "run" / "run_manifest.json").write_text(
        json.dumps(
            {
                "run_id": "rid",
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
                "model_digests": {},
                "environment": {},
                "task_count": 1,
                "candidate_count": 1,
                "model_count": 1,
                "request_count": 1,
                "teams_included": ["baseline"],
                "roles_included": ["baseline.baseline_all"],
                    "models_included": [model_name],
                "requested_think_modes": ["false"],
                "structured_output_modes": ["prompt_only"],
                "task_manifest_hash": task_manifest_hash,
                "config_hashes": {},
                "schema_hashes": {},
                "fixture_hashes": {},
                    "execution_source_hashes": {},
                "execution_plan_hash": execution_plan_hash,
                "git_commit": "unknown",
                "git_dirty": False,
            }
        )
    )
    (tmp_path / "run" / "task_manifest.json").write_text(json.dumps(task_manifest))
    (tmp_path / "run" / "run_state.json").write_text(
        json.dumps(
            {
                "run_id": "rid",
                "profile": "smoke",
                "started_at": "2026-01-01T00:00:00Z",
                "updated_at": "2026-01-01T00:00:00Z",
                "status": "running",
                "completed_identity_keys": [key],
                "completed_count": 1,
                "error_count": 0,
                "unsupported_mode_count": 0,
            }
        )
    )

    runner.load_existing_run()
    live_pre = runner._make_identity(task, model_name, "false", "false", None, None)
    runner._completed_keys = {live_pre.pre_request_key()}
    runner._get_candidates = lambda _: [model_name]
    runner.client = _StubClient()

    out = list(runner.run_tasks([task]))
    assert out == []
    assert runner.client.calls == 0
