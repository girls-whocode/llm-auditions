from __future__ import annotations

import json
from pathlib import Path

from llm_auditions.task_loader import load_tasks_from_dir


def test_comparison_tasks_use_shared_scenario_refs():
    project_root = Path(__file__).parent.parent
    tasks = load_tasks_from_dir(project_root / "fixtures")

    comparisons = [t for t in tasks if t.comparison_id]
    assert comparisons, "expected at least one comparison task"

    grouped: dict[tuple[str, str], list] = {}
    for t in comparisons:
        grouped.setdefault((t.comparison_id, t.comparison_track), []).append(t)

    for pair_key, items in grouped.items():
        refs = {getattr(t, "comparison_scenario_ref", "") for t in items}
        assert refs == {next(iter(refs))}
        ref = next(iter(refs))
        assert ref

        scenario_path = project_root / ref
        assert scenario_path.exists(), f"missing shared scenario fixture for {pair_key}: {ref}"
        payload = json.loads(scenario_path.read_text(encoding="utf-8"))
        assert payload.get("comparison_id") == pair_key[0]
        assert payload.get("track") == pair_key[1]
        assert payload.get("prompt")
