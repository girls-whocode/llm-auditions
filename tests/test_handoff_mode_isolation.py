from __future__ import annotations

from llm_auditions.cli import _build_execution_plan
from llm_auditions.configuration import Configuration
from llm_auditions.models import TaskDefinition


def _task(role: str, worker_class: str) -> TaskDefinition:
    return TaskDefinition(
        id=f"mode_iso_{worker_class}",
        team="linux_infrastructure",
        role=role,
        prompt="role instruction",
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
        comparison_id="linux_firewall_change_002",
        comparison_track="handoff",
        worker_class=worker_class,
        comparison_scenario_ref="fixtures/comparisons/linux_firewall_change_002.json",
    )


def test_handoff_multi_mode_isolation(monkeypatch):
    cfg = Configuration()
    cfg.load()
    tasks = [_task("fast_worker", "fast"), _task("escalation", "heavy")]

    monkeypatch.setattr("llm_auditions.cli._load_tasks", lambda: tasks)
    monkeypatch.setattr(cfg, "get_role_candidates", lambda team, role: ["fast-a", "fast-b"] if role == "fast_worker" else ["heavy-a", "heavy-b", "heavy-c"])

    _, rows, md = _build_execution_plan(config=cfg, profile="standard")

    fast_rows = [row for row in rows if row["role"] == "fast_worker"]
    heavy_rows = [row for row in rows if row["role"] == "escalation"]
    assert len(fast_rows) == 4
    assert len(heavy_rows) == 12
    assert md["request_count"] == 16

    fast_by_id = {row["plan_row_id"]: row for row in fast_rows}
    cross_mode = 0
    for heavy in heavy_rows:
        fast = fast_by_id[heavy["fast_plan_row_id"]]
        if heavy["requested_think_mode"] != fast["requested_think_mode"]:
            cross_mode += 1
    assert cross_mode == 0
