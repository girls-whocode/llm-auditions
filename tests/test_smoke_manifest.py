from __future__ import annotations

import json
from pathlib import Path

from llm_auditions.cli import _build_execution_plan
from llm_auditions.configuration import Configuration
from llm_auditions.runner import AuditionRunner

PROJECT_ROOT = Path(__file__).parent.parent


def test_smoke_manifest_counts_match_plan(tmp_path: Path):
    cfg = Configuration(project_root=PROJECT_ROOT)
    cfg.load()
    tasks, rows, _ = _build_execution_plan(cfg, "smoke")
    runner = AuditionRunner(cfg, tmp_path / "run", profile="smoke")

    class _Client:
        def get_version(self): return "test"
        def discover_model_capabilities(self): return []

    runner.client = _Client()
    manifest = runner.setup_new_run(tasks, rows)
    assert manifest.task_count == len({(r['team'], r['role'], r['task_id'], r['task_version']) for r in rows})
    assert manifest.request_count == len(rows)
    assert manifest.model_count == len({r['model'] for r in rows})
    assert manifest.candidate_count == len({(r['team'], r['role'], r['model']) for r in rows})
