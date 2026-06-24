from __future__ import annotations

from pathlib import Path

from llm_auditions.cli import _build_execution_plan
from llm_auditions.configuration import Configuration

PROJECT_ROOT = Path(__file__).parent.parent


def test_single_task_single_model_single_row():
    cfg = Configuration(project_root=PROJECT_ROOT)
    cfg.load()
    _, rows, _ = _build_execution_plan(
        cfg,
        "smoke",
        team_filter="baseline",
        role_filter="baseline_all",
        model_filter="gemma4:12b",
        task_filter="baseline_json_schema_adherence",
        think_mode_filter="false",
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["team"] == "baseline"
    assert row["role"] == "baseline_all"
    assert row["task_id"] == "baseline_json_schema_adherence"
    assert row["model"] == "gemma4:12b"
    assert row["requested_think_mode"] == "false"
