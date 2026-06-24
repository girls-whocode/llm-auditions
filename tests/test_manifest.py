from __future__ import annotations

import json
from pathlib import Path

from llm_auditions.configuration import Configuration
from llm_auditions.models import TaskDefinition
from llm_auditions.runner import AuditionRunner


PROJECT_ROOT = Path(__file__).parent.parent


def test_manifest_counts_for_targeted_plan(tmp_path: Path, monkeypatch):
    cfg = Configuration(project_root=PROJECT_ROOT)
    cfg.load()

    class _Client:
        def get_version(self):
            return "test"

        def discover_model_capabilities(self):
            return []

    runner = AuditionRunner(cfg, tmp_path / "run", profile="smoke")
    runner.client = _Client()

    tasks = [
        TaskDefinition(id=f"t{i}", team="baseline", role="baseline_all", prompt="p", think_modes=["false"])
        for i in range(4)
    ]
    plan = [
        {
            "team": "baseline",
            "role": "baseline_all",
            "task_id": f"t{i}",
            "task_version": "v1",
            "model": "gemma4:12b",
            "requested_think_mode": "false",
            "structured_output_mode": "prompt_only",
        }
        for i in range(4)
    ]
    m = runner.setup_new_run(tasks, plan)
    assert m.task_count == 4
    assert m.model_count == 1
    assert m.request_count == 4

    manifest = json.loads((tmp_path / "run" / "run_manifest.json").read_text())
    assert manifest["task_count"] == 4
    assert manifest["model_count"] == 1
    assert manifest["request_count"] == 4
