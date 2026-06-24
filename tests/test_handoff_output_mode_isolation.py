from __future__ import annotations

from llm_auditions.cli import _build_execution_plan
from llm_auditions.configuration import Configuration
from llm_auditions.models import TaskDefinition


def _task(role: str, worker_class: str, output_mode: str) -> TaskDefinition:
    return TaskDefinition(
        id=f"output_iso_{worker_class}",
        team="security",
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
        comparison_id="sec_incident_response_002",
        comparison_track="handoff",
        worker_class=worker_class,
        structured_output_mode=output_mode,
        comparison_scenario_ref="fixtures/comparisons/sec_incident_response_002.json",
    )


def test_handoff_output_mode_isolation(monkeypatch):
    cfg = Configuration()
    cfg.load()
    tasks = [
        _task("fast_worker", "fast", "prompt_only"),
        _task("worker", "heavy", "ollama_schema"),
    ]

    monkeypatch.setattr("llm_auditions.cli._load_tasks", lambda: tasks)
    monkeypatch.setattr(cfg, "get_role_candidates", lambda team, role: ["fast-a"] if role == "fast_worker" else ["heavy-a"])

    _, rows, _ = _build_execution_plan(config=cfg, profile="standard", think_mode_filter="false")
    fast_rows = [row for row in rows if row["worker_class"] == "fast"]
    heavy_rows = [row for row in rows if row["worker_class"] == "heavy"]
    assert len(fast_rows) == 1
    assert len(heavy_rows) == 0
