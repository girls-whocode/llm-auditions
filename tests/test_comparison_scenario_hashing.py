from __future__ import annotations

import json
from pathlib import Path

from llm_auditions.configuration import Configuration
from llm_auditions.models import TaskDefinition
from llm_auditions.runner import AuditionRunner


PROJECT_ROOT = Path(__file__).parent.parent


def _cfg() -> Configuration:
    cfg = Configuration(project_root=PROJECT_ROOT)
    cfg.load()
    return cfg


def _task(ref: str) -> TaskDefinition:
    return TaskDefinition(
        id="comparison_hash_task",
        team="linux_infrastructure",
        role="fast_worker",
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
        comparison_id="cmp_hash_001",
        comparison_track="independent",
        worker_class="fast",
        comparison_scenario_ref=ref,
    )


def test_scenario_hash_changes_when_content_changes(monkeypatch, tmp_path: Path):
    from llm_auditions import runner as runner_mod

    monkeypatch.setattr(runner_mod, "PROJECT_ROOT", tmp_path)

    scenario_rel = "fixtures/comparisons/cmp_hash_001.json"
    scenario_path = tmp_path / scenario_rel
    scenario_path.parent.mkdir(parents=True, exist_ok=True)
    scenario_path.write_text(
        json.dumps(
            {
                "comparison_id": "cmp_hash_001",
                "scenario_version": "1",
                "title": "Storage latency",
                "scenario": "Latency rose after migration.",
                "constraints": ["No destructive change"],
                "required_facts": ["Collect per-device latency"],
            }
        ),
        encoding="utf-8",
    )

    runner = AuditionRunner(_cfg(), tmp_path / "run")
    task = _task(scenario_rel)
    first = runner._comparison_scenario_hash(task)

    scenario_path.write_text(
        json.dumps(
            {
                "comparison_id": "cmp_hash_001",
                "scenario_version": "1",
                "title": "Storage latency",
                "scenario": "Latency rose sharply after migration.",
                "constraints": ["No destructive change"],
                "required_facts": ["Collect per-device latency"],
            }
        ),
        encoding="utf-8",
    )
    second = runner._comparison_scenario_hash(task)

    assert second != first


def test_scenario_hash_uses_content_not_file_path(monkeypatch, tmp_path: Path):
    from llm_auditions import runner as runner_mod

    monkeypatch.setattr(runner_mod, "PROJECT_ROOT", tmp_path)

    payload = {
        "comparison_id": "cmp_hash_001",
        "scenario_version": "1",
        "title": "Storage latency",
        "scenario": "Latency rose after migration.",
        "constraints": ["No destructive change"],
        "required_facts": ["Collect per-device latency"],
    }

    rel_a = "fixtures/comparisons/a.json"
    rel_b = "fixtures/comparisons/b.json"
    path_a = tmp_path / rel_a
    path_b = tmp_path / rel_b
    path_a.parent.mkdir(parents=True, exist_ok=True)
    path_a.write_text(json.dumps(payload), encoding="utf-8")
    path_b.write_text(json.dumps(payload), encoding="utf-8")

    runner = AuditionRunner(_cfg(), tmp_path / "run")
    hash_a = runner._comparison_scenario_hash(_task(rel_a))
    hash_b = runner._comparison_scenario_hash(_task(rel_b))

    assert hash_a == hash_b
