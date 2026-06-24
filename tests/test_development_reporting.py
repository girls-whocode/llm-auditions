from __future__ import annotations

import json
from pathlib import Path

from llm_auditions.reporting import generate_reports


def test_summary_reports_development_exclusions(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "run_manifest.json").write_text(json.dumps({"profile": "smoke", "task_suite_version": "2"}))
    (run / "events.jsonl").write_text("")

    dev_dir = run / "development" / "developer"
    dev_dir.mkdir(parents=True)
    dev_result = {
        "status": "completed",
        "task": {
            "id": "dev_task",
            "team": "development",
            "role": "developer",
            "verification_classification": "rubric_assisted",
            "weight": 1.0,
        },
        "identity": {
            "model_name": "gemma4:12b",
            "model_digest": "d1",
            "requested_think_mode": "false",
            "effective_think_mode": "false",
            "structured_output_mode": "prompt_only",
            "task_suite_version": "2",
        },
        "scores": {"weighted_total": 0.0},
        "score_status": "human_required",
        "ranking_eligible": False,
        "human_review_required": True,
        "hard_fail": False,
        "verifier_output": {"policy_refusal": True, "reason": "sandbox_unavailable"},
        "response": {"metrics": {"generation_seconds": 0.0, "wall_clock_seconds": 0.1}},
    }
    (dev_dir / "dev.result.json").write_text(json.dumps(dev_result))

    base_dir = run / "baseline" / "baseline_all"
    base_dir.mkdir(parents=True)
    base_result = {
        "status": "completed",
        "task": {
            "id": "base_task",
            "team": "baseline",
            "role": "baseline_all",
            "verification_classification": "rubric_assisted",
            "weight": 1.0,
        },
        "identity": {
            "model_name": "gemma4:12b",
            "model_digest": "d1",
            "requested_think_mode": "false",
            "effective_think_mode": "false",
            "structured_output_mode": "prompt_only",
            "task_suite_version": "2",
        },
        "scores": {"weighted_total": 1.0},
        "score_status": "final",
        "ranking_eligible": True,
        "hard_fail": False,
        "verifier_output": {},
        "response": {"metrics": {"generation_seconds": 1.0, "wall_clock_seconds": 1.2}},
    }
    (base_dir / "base.result.json").write_text(json.dumps(base_result))

    generate_reports(run)
    summary = json.loads((run / "summary.json").read_text())

    assert summary["development_excluded_count"] == 1
    assert summary["development_excluded_reasons"]["sandbox_unavailable"] == 1
