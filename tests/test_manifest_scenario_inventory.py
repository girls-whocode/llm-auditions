from __future__ import annotations

import json
from pathlib import Path

from llm_auditions.configuration import Configuration
from llm_auditions.models import TaskDefinition
from llm_auditions.runner import AuditionRunner


class _FakeClient:
    def get_version(self) -> str:
        return "test"

    def discover_model_capabilities(self):
        return []

    def list_models(self):
        return []


def _task() -> TaskDefinition:
    return TaskDefinition(
        id="manifest_cmp",
        team="linux_infrastructure",
        role="fast_worker",
        prompt="p",
        verification_classification="rubric_assisted",
        rubric_finalization="deterministic",
        rubric_rules=[
            {
                "rule_id": "r1",
                "description": "required",
                "type": "required",
                "weight": 1.0,
                "matcher": {"type": "phrase_aliases", "phrases": ["safe"]},
            }
        ],
        comparison_id="linux_ops_safety_001",
        comparison_track="independent",
        worker_class="fast",
        comparison_scenario_ref="fixtures/comparisons/linux_ops_safety_001.json",
    )


def test_manifest_includes_comparison_scenario_fixture_hash(tmp_path: Path):
    cfg = Configuration()
    cfg.load()
    run = tmp_path / "run"
    runner = AuditionRunner(cfg, run, profile="smoke")
    runner.client = _FakeClient()

    task = _task()
    row = {
        "plan_row_id": "p1",
        "team": task.team,
        "role": task.role,
        "task_id": task.id,
        "task_version": task.task_version,
        "model": "m",
        "full_model_digest": "d",
        "requested_think_mode": "false",
        "structured_output_mode": "prompt_only",
        "temperature": 0.0,
        "num_ctx": 1024,
        "num_predict": 128,
        "schema_hash": "",
        "fixture_hashes": task.fixture_hashes(Path(__file__).parent.parent),
        "comparison_id": task.comparison_id,
        "comparison_track": task.comparison_track,
        "comparison_scenario_ref": task.comparison_scenario_ref,
        "scenario_version": "1",
        "scenario_content_hash": "",
        "comparison_information_mode": "symmetric",
        "fast_plan_row_id": "",
    }

    runner.setup_new_run(tasks=[task], plan_rows=[row])
    manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    assert "fixtures/comparisons/linux_ops_safety_001.json" in manifest["fixture_hashes"]
