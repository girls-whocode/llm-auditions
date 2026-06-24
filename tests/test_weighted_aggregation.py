from __future__ import annotations

import csv
import json
from pathlib import Path

from llm_auditions.reporting import generate_reports


def _base_result(score_status: str, ranking_eligible: bool, weighted_total: float, task_weight: float) -> dict:
    return {
        "status": "completed",
        "task": {
            "id": "task",
            "team": "general_knowledge",
            "role": "fast_worker",
            "verification_classification": "rubric_assisted",
            "weight": task_weight,
        },
        "identity": {
            "model_name": "gemma4:12b",
            "model_digest": "digest1",
            "requested_think_mode": "false",
            "effective_think_mode": "false",
            "structured_output_mode": "prompt_only",
            "task_suite_version": "1",
        },
        "response": {
            "metrics": {
                "wall_clock_seconds": 1.0,
                "load_seconds": 0.2,
                "prompt_eval_seconds": 0.3,
                "generation_seconds": 0.5,
                "ollama_total_seconds": 1.0,
                "overhead_seconds": 0.0,
            },
            "truncated_length_stop": False,
            "empty_final_content": False,
        },
        "scores": {"weighted_total": weighted_total},
        "score_status": score_status,
        "ranking_eligible": ranking_eligible,
        "schema_errors": [],
        "safety_flags": [],
        "verifier_output": {"rubric_rules": []},
        "deterministic_results": [],
        "hard_fail": False,
    }


def test_role_aggregation_uses_task_weights(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "run_manifest.json").write_text(json.dumps({"profile": "smoke", "task_suite_version": "1"}))
    (run / "events.jsonl").write_text("")

    out_dir = run / "general_knowledge" / "fast_worker"
    out_dir.mkdir(parents=True)

    task_a = _base_result("final", True, 1.0, 1.0)
    task_a["task"]["id"] = "task_a"
    task_b = _base_result("final", True, 0.0, 3.0)
    task_b["task"]["id"] = "task_b"

    (out_dir / "a.result.json").write_text(json.dumps(task_a))
    (out_dir / "b.result.json").write_text(json.dumps(task_b))

    generate_reports(run)
    rows = list(csv.DictReader((run / "leaderboard_by_role.csv").read_text().splitlines()))

    assert len(rows) == 1
    row = rows[0]
    assert float(row["unweighted_task_score"]) == 0.5
    assert float(row["weighted_task_score"]) == 0.25
    assert float(row["eligible_weight_sum"]) == 4.0
    assert int(row["eligible_result_count"]) == 2
    assert int(row["total_result_count"]) == 2


def test_non_final_or_ineligible_results_not_in_weight_denominator(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "run_manifest.json").write_text(json.dumps({"profile": "smoke", "task_suite_version": "1"}))
    (run / "events.jsonl").write_text("")

    out_dir = run / "general_knowledge" / "fast_worker"
    out_dir.mkdir(parents=True)

    final_eligible = _base_result("final", True, 1.0, 2.0)
    final_eligible["task"]["id"] = "eligible"
    provisional = _base_result("provisional", True, 0.0, 10.0)
    provisional["task"]["id"] = "provisional"
    disqualified = _base_result("disqualified", False, 0.0, 10.0)
    disqualified["task"]["id"] = "dq"
    human_required = _base_result("human_required", False, 0.0, 10.0)
    human_required["task"]["id"] = "hr"

    (out_dir / "1.result.json").write_text(json.dumps(final_eligible))
    (out_dir / "2.result.json").write_text(json.dumps(provisional))
    (out_dir / "3.result.json").write_text(json.dumps(disqualified))
    (out_dir / "4.result.json").write_text(json.dumps(human_required))

    generate_reports(run)
    rows = list(csv.DictReader((run / "leaderboard_by_role.csv").read_text().splitlines()))

    assert len(rows) == 1
    row = rows[0]
    assert float(row["eligible_weight_sum"]) == 2.0
    assert int(row["eligible_result_count"]) == 1
    assert int(row["total_result_count"]) == 4
    assert int(row["provisional_count"]) == 1
    assert int(row["human_review_count"]) == 1
    assert int(row["disqualified_count"]) == 1
