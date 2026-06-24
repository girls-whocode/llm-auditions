from __future__ import annotations

import csv
import json
from pathlib import Path

from llm_auditions.reporting import generate_reports


def test_reporting_outputs_mode_columns(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "run_manifest.json").write_text(json.dumps({"profile": "smoke", "task_suite_version": "1"}))
    event = {
        "team": "baseline",
        "role": "baseline_all",
        "task_id": "t1",
        "model": "gemma4:12b",
        "requested_think_mode": "false",
        "effective_think_mode": "false",
        "structured_output_mode": "prompt_only",
        "status": "completed",
        "score_status": "final",
        "ranking_eligible": True,
        "weighted_total": 0.9,
        "wall_clock_s": 1.0,
    }
    (run / "events.jsonl").write_text(json.dumps(event) + "\n")
    generate_reports(run)

    rows = list(csv.DictReader((run / "leaderboard_by_role.csv").read_text().splitlines()))
    assert rows
    assert "requested_think_mode" in rows[0]
    assert "structured_output_mode" in rows[0]
