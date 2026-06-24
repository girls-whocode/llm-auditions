from __future__ import annotations

import csv
import json
from pathlib import Path

from llm_auditions.reporting import generate_reports


def test_report_metrics_derived_from_artifacts(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "run_manifest.json").write_text(json.dumps({"profile": "smoke", "task_suite_version": "1"}))
    (run / "events.jsonl").write_text(json.dumps({
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
        "wall_clock_s": 1.2,
        "truncated_length_stop": False,
        "empty_final_content": False,
        "safety_flags": [],
    }) + "\n")
    generate_reports(run)
    summary = json.loads((run / "summary.json").read_text())
    assert summary["total_requests"] == 1
    assert summary["final_count"] == 1
    assert summary["sample_size"] == 1


def test_leaderboard_excludes_non_final_or_ineligible(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "run_manifest.json").write_text(json.dumps({"profile": "smoke", "task_suite_version": "1"}))
    (run / "events.jsonl").write_text("")

    role_dir = run / "general_knowledge" / "fast_worker"
    role_dir.mkdir(parents=True)

    base = {
        "status": "completed",
        "task": {
            "id": "gk",
            "team": "general_knowledge",
            "role": "fast_worker",
            "verification_classification": "rubric_assisted",
        },
        "identity": {
            "model_name": "gemma4:12b",
            "model_digest": "d1",
            "requested_think_mode": "false",
            "effective_think_mode": "false",
            "structured_output_mode": "prompt_only",
            "task_suite_version": "1",
        },
        "response": {"metrics": {"wall_clock_seconds": 1.0, "net_generation_seconds": 1.0}, "truncated_length_stop": False, "empty_final_content": False},
        "scores": {"weighted_total": 0.0},
        "schema_errors": [],
        "safety_flags": [],
        "verifier_output": {"rubric_rules": []},
        "deterministic_results": [],
        "hard_fail": False,
    }

    eligible_final = dict(base)
    eligible_final.update({"score_status": "final", "ranking_eligible": True, "scores": {"weighted_total": 0.9}})
    ineligible_final = dict(base)
    ineligible_final.update({"score_status": "final", "ranking_eligible": False, "scores": {"weighted_total": 0.1}})
    provisional = dict(base)
    provisional.update({"score_status": "provisional", "ranking_eligible": True, "scores": {"weighted_total": 0.2}})
    human_required = dict(base)
    human_required.update({"score_status": "human_required", "ranking_eligible": False, "scores": {"weighted_total": 0.3}})

    (role_dir / "a.result.json").write_text(json.dumps(eligible_final))
    (role_dir / "b.result.json").write_text(json.dumps(ineligible_final))
    (role_dir / "c.result.json").write_text(json.dumps(provisional))
    (role_dir / "d.result.json").write_text(json.dumps(human_required))

    generate_reports(run)

    rows = list(csv.DictReader((run / "leaderboard_by_role.csv").read_text().splitlines()))
    assert len(rows) == 1
    assert rows[0]["sample_size"] == "1"
    assert rows[0]["eligible_count"] == "1"
    assert float(rows[0]["avg_score"]) == 0.9
