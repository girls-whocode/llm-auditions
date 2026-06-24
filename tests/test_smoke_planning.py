from __future__ import annotations

from pathlib import Path

from llm_auditions.cli import _build_execution_plan
from llm_auditions.configuration import Configuration


PROJECT_ROOT = Path(__file__).parent.parent


def test_smoke_plan_small_and_false_only():
    cfg = Configuration(project_root=PROJECT_ROOT)
    cfg.load()
    tasks, rows, md = _build_execution_plan(cfg, "smoke")
    assert rows
    assert md["request_count"] <= md["smoke_ceiling"]
    assert md["think_modes"] == ["false"]


def test_filters_reduce_smoke_plan():
    cfg = Configuration(project_root=PROJECT_ROOT)
    cfg.load()
    _, rows_all, _ = _build_execution_plan(cfg, "smoke")
    _, rows_team, _ = _build_execution_plan(cfg, "smoke", team_filter="baseline")
    assert len(rows_team) <= len(rows_all)
