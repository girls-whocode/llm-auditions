from __future__ import annotations

import csv
import json
from pathlib import Path

from llm_auditions.models import OllamaMetrics
from llm_auditions.reporting import generate_reports


def test_timing_conversion_properties_are_exact():
    metrics = OllamaMetrics(
        load_duration_ns=2_000_000_000,
        prompt_eval_duration_ns=3_000_000_000,
        eval_duration_ns=5_000_000_000,
        total_duration_ns=10_000_000_000,
        wall_clock_seconds_local=11.5,
    )

    assert metrics.load_seconds == 2.0
    assert metrics.prompt_eval_seconds == 3.0
    assert metrics.generation_seconds == 5.0
    assert metrics.ollama_total_seconds == 10.0
    assert metrics.wall_clock_seconds == 11.5
    assert metrics.overhead_seconds == 1.5


def test_reporting_summary_includes_timing_medians(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "run_manifest.json").write_text(json.dumps({"profile": "smoke", "task_suite_version": "1"}))
    (run / "events.jsonl").write_text("")

    role_dir = run / "baseline" / "baseline_all"
    role_dir.mkdir(parents=True)
    result = {
        "status": "completed",
        "task": {"id": "t1", "team": "baseline", "role": "baseline_all", "verification_classification": "rubric_assisted", "weight": 1.0},
        "identity": {
            "model_name": "gemma4:12b",
            "model_digest": "d1",
            "requested_think_mode": "false",
            "effective_think_mode": "false",
            "structured_output_mode": "prompt_only",
            "task_suite_version": "1",
        },
        "response": {
            "metrics": {
                "load_duration_ns": 2_000_000_000,
                "prompt_eval_duration_ns": 3_000_000_000,
                "eval_duration_ns": 5_000_000_000,
                "total_duration_ns": 10_000_000_000,
                "load_seconds": 2.0,
                "prompt_eval_seconds": 3.0,
                "generation_seconds": 5.0,
                "ollama_total_seconds": 10.0,
                "wall_clock_seconds": 11.5,
                "overhead_seconds": 1.5,
                "generation_rate": 40.0,
            },
            "truncated_length_stop": False,
            "empty_final_content": False,
        },
        "scores": {"weighted_total": 0.9},
        "score_status": "final",
        "ranking_eligible": True,
        "schema_errors": [],
        "safety_flags": [],
        "verifier_output": {"rubric_rules": []},
        "deterministic_results": [],
        "hard_fail": False,
    }
    (role_dir / "r.result.json").write_text(json.dumps(result))

    generate_reports(run)
    summary = json.loads((run / "summary.json").read_text())

    assert summary["median_load_seconds"] == 2.0
    assert summary["median_prompt_eval_seconds"] == 3.0
    assert summary["median_generation_seconds"] == 5.0
    assert summary["median_ollama_total_seconds"] == 10.0
    assert summary["median_wall_clock_seconds"] == 11.5
    assert summary["median_overhead_seconds"] == 1.5
    assert summary["median_generation_rate"] == 40.0

    rows = list(csv.DictReader((run / "leaderboard_by_role.csv").read_text().splitlines()))
    assert len(rows) == 1
    assert float(rows[0]["median_load_seconds"]) == 2.0
    assert float(rows[0]["median_prompt_eval_seconds"]) == 3.0
    assert float(rows[0]["median_generation_seconds"]) == 5.0
    assert float(rows[0]["median_ollama_total_seconds"]) == 10.0
    assert float(rows[0]["median_wall_clock_seconds"]) == 11.5
    assert float(rows[0]["median_overhead_seconds"]) == 1.5
    assert float(rows[0]["median_generation_rate"]) == 40.0
