from __future__ import annotations

import csv
import json
from pathlib import Path

from llm_auditions.reporting import generate_reports


def test_escalation_comparison_id_pairs_fast_heavy(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "events.jsonl").write_text("")
    (run / "run_manifest.json").write_text(json.dumps({"profile": "smoke", "task_suite_version": "1"}))

    team_dir = run / "general_knowledge" / "fast_worker"
    team_dir.mkdir(parents=True)
    base_fast = {
        "task": {"id": "task1", "team": "general_knowledge", "role": "fast_worker", "comparison_id": "cmp1", "comparison_track": "independent", "worker_class": "fast"},
        "identity": {"model_name": "gemma4:12b", "task_version": "v1", "requested_think_mode": "false", "effective_think_mode": "false"},
        "scores": {"correctness_score": 0.5, "completeness_score": 0.5, "safety_score": 1.0, "latency_score": 1.0, "weighted_total": 0.6},
        "score_status": "final",
    }
    (team_dir / "a.result.json").write_text(json.dumps(base_fast))

    team_dir_h = run / "general_knowledge" / "heavy_worker"
    team_dir_h.mkdir(parents=True)
    base_heavy = {
        "task": {"id": "task1", "team": "general_knowledge", "role": "heavy_worker", "comparison_id": "cmp1", "comparison_track": "independent", "worker_class": "heavy"},
        "identity": {"model_name": "gemma4:26b", "task_version": "v1", "requested_think_mode": "false", "effective_think_mode": "false"},
        "scores": {"correctness_score": 0.8, "completeness_score": 0.8, "safety_score": 1.0, "latency_score": 0.8, "weighted_total": 0.85},
        "score_status": "final",
    }
    (team_dir_h / "b.result.json").write_text(json.dumps(base_heavy))

    generate_reports(run)
    rows = list(csv.DictReader((run / "escalation_value.csv").read_text().splitlines()))
    assert rows
    assert rows[0]["comparison_id"] == "cmp1"
