from __future__ import annotations

import json
from pathlib import Path

from llm_auditions.task_loader import load_tasks_from_dir


PROJECT_ROOT = Path(__file__).parent.parent
REQUIRED_SCENARIO_FIELDS = {
    "comparison_id",
    "scenario_version",
    "title",
    "scenario",
    "constraints",
    "required_facts",
}


def test_comparison_scenarios_are_authoritative_shared_problems():
    scenario_dir = PROJECT_ROOT / "fixtures" / "comparisons"
    scenario_files = sorted(scenario_dir.glob("*.json"))
    assert len(scenario_files) == 12

    for path in scenario_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert REQUIRED_SCENARIO_FIELDS.issubset(payload.keys())
        assert payload["comparison_id"]
        assert payload["scenario_version"]
        assert payload["title"].strip()
        assert payload["scenario"].strip()
        assert isinstance(payload["constraints"], list)
        assert isinstance(payload["required_facts"], list)
        assert payload["constraints"]
        assert payload["required_facts"]
        assert "compare" not in payload["scenario"].strip().lower()


def test_fast_and_heavy_share_identical_scenario_content_in_fixtures():
    tasks = load_tasks_from_dir(PROJECT_ROOT / "fixtures")
    comparison_tasks = [t for t in tasks if t.comparison_id]
    assert comparison_tasks

    grouped: dict[tuple[str, str], list] = {}
    for task in comparison_tasks:
        grouped.setdefault((task.comparison_id, task.comparison_track), []).append(task)

    for pair_key, pair_tasks in grouped.items():
        refs = {task.comparison_scenario_ref for task in pair_tasks}
        assert len(refs) == 1, f"mismatched scenario refs for {pair_key}"
        ref = next(iter(refs))
        payload = json.loads((PROJECT_ROOT / ref).read_text(encoding="utf-8"))
        assert payload["comparison_id"] == pair_key[0]
