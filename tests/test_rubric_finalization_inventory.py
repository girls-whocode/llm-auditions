from __future__ import annotations

from pathlib import Path

from llm_auditions.cli import _load_raw_task_entries, _rubric_finalization_inventory


PROJECT_ROOT = Path(__file__).parent.parent


def test_rubric_finalization_inventory_has_no_missing_entries():
    raw = _load_raw_task_entries(PROJECT_ROOT / "fixtures")
    inventory = _rubric_finalization_inventory(raw)

    assert inventory
    for team_role, counts in inventory.items():
        assert counts.get("missing", 0) == 0, f"missing rubric_finalization entries in {team_role}"

    total_deterministic = sum(v.get("deterministic", 0) for v in inventory.values())
    total_mixed = sum(v.get("mixed", 0) for v in inventory.values())
    total_human_review = sum(v.get("human_review", 0) for v in inventory.values())
    assert total_deterministic > 0
    assert total_mixed + total_human_review > 0
