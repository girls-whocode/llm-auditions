from __future__ import annotations

from llm_auditions.cli import _build_execution_plan
from llm_auditions.configuration import Configuration
from llm_auditions.models import TaskDefinition


def _task(role: str, worker_class: str, scenario_ref: str) -> TaskDefinition:
    return TaskDefinition(
        id=f"scenario_iso_{worker_class}",
        team="architecture",
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
        comparison_id="arch_failure_domains_002",
        comparison_track="handoff",
        worker_class=worker_class,
        comparison_scenario_ref=scenario_ref,
    )


def test_handoff_scenario_and_information_mode_isolation(monkeypatch):
    cfg = Configuration()
    cfg.load()

    tasks = [
        _task("worker", "fast", "fixtures/comparisons/arch_failure_domains_002.json"),
        _task("reviewer", "heavy", "fixtures/comparisons/sec_incident_response_002.json"),
    ]

    monkeypatch.setattr("llm_auditions.cli._load_tasks", lambda: tasks)
    monkeypatch.setattr(cfg, "get_role_candidates", lambda team, role: ["m1"])

    _, rows, _ = _build_execution_plan(config=cfg, profile="standard", think_mode_filter="false")
    fast_rows = [row for row in rows if row["worker_class"] == "fast"]
    heavy_rows = [row for row in rows if row["worker_class"] == "heavy"]

    assert len(fast_rows) == 1
    assert len(heavy_rows) == 0
