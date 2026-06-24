from __future__ import annotations

import json
from pathlib import Path

from llm_auditions.task_loader import load_tasks_from_dir


PROJECT_ROOT = Path(__file__).parent.parent


def test_all_comparison_tasks_link_shared_rubrics_and_role_rules():
    tasks = [task for task in load_tasks_from_dir(PROJECT_ROOT / "fixtures") if task.comparison_id]
    assert len(tasks) == 24
    for task in tasks:
        assert task.comparison_scenario_ref.startswith("fixtures/comparisons/")
        dumped = task.model_dump(mode="json")
        assert dumped.get("use_shared_scenario_rubric") is True
        role_rules = dumped.get("role_rubric_rules")
        assert isinstance(role_rules, list)
        assert role_rules


def test_effective_rubrics_cover_shared_required_facts():
    tasks = [task for task in load_tasks_from_dir(PROJECT_ROOT / "fixtures") if task.comparison_id]
    by_cmp: dict[str, list] = {}
    for task in tasks:
        by_cmp.setdefault(task.comparison_id, []).append(task)

    for comparison_id, pair in by_cmp.items():
        assert len(pair) == 2
        scenario_ref = pair[0].comparison_scenario_ref
        payload = json.loads((PROJECT_ROOT / scenario_ref).read_text(encoding="utf-8"))
        fact_ids = {fact["fact_id"] for fact in payload["required_facts"]}

        shared_rule_ids = {rule["rule_id"] for rule in payload["shared_rubric_rules"]}
        for task in pair:
            effective_rule_ids = {rule.rule_id for rule in task.rubric_rules}
            assert shared_rule_ids.issubset(effective_rule_ids)

            covered_fact_ids: set[str] = set()
            for rule in task.rubric_rules:
                for source in getattr(rule, "source_fact_ids", []) or []:
                    covered_fact_ids.add(source)
            assert fact_ids.issubset(covered_fact_ids), f"comparison {comparison_id} task {task.id} missing fact coverage"
