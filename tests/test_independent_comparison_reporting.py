from __future__ import annotations

import csv
import json
from pathlib import Path

from llm_auditions.reporting import generate_reports


def test_independent_report_pairs_only_matching_scenario_mode_contract(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "events.jsonl").write_text("", encoding="utf-8")
    (run / "run_manifest.json").write_text(json.dumps({"profile": "smoke", "task_suite_version": "2"}), encoding="utf-8")

    fast_dir = run / "general_knowledge" / "fast_worker"
    heavy_dir = run / "general_knowledge" / "heavy_worker"
    fast_dir.mkdir(parents=True)
    heavy_dir.mkdir(parents=True)

    fast = {
        "task": {"id": "gk_fast", "team": "general_knowledge", "role": "fast_worker", "comparison_id": "gk_reasoning_precision_002", "comparison_track": "independent", "worker_class": "fast"},
        "identity": {
            "team": "general_knowledge", "role": "fast_worker", "task_id": "gk_fast", "task_version": "v1",
            "model_name": "fast-model", "model_digest": "d-fast", "requested_think_mode": "false", "effective_think_mode": "false",
            "structured_output_mode": "prompt_only", "scenario_content_hash": "scenario-1", "comparison_scenario_hash": "scenario-1", "task_suite_version": "2"
        },
        "response": {"content": "fast", "metrics": {}, "effective_prompt": {"prompt_components": {"scenario_version": "1", "comparison_information_mode": "symmetric"}}},
        "scores": {"weighted_total": 0.5, "correctness_score": 0.4, "completeness_score": 0.4, "safety_score": 1.0},
        "score_status": "final",
        "ranking_eligible": True,
    }
    heavy_match = {
        "task": {"id": "gk_heavy", "team": "general_knowledge", "role": "heavy_worker", "comparison_id": "gk_reasoning_precision_002", "comparison_track": "independent", "worker_class": "heavy"},
        "identity": {
            "team": "general_knowledge", "role": "heavy_worker", "task_id": "gk_heavy", "task_version": "v1",
            "model_name": "heavy-model", "model_digest": "d-heavy", "requested_think_mode": "false", "effective_think_mode": "false",
            "structured_output_mode": "prompt_only", "scenario_content_hash": "scenario-1", "comparison_scenario_hash": "scenario-1", "task_suite_version": "2"
        },
        "response": {"content": "heavy", "metrics": {}, "effective_prompt": {"prompt_components": {"scenario_version": "1", "comparison_information_mode": "symmetric"}}},
        "scores": {"weighted_total": 0.8, "correctness_score": 0.8, "completeness_score": 0.8, "safety_score": 1.0},
        "score_status": "final",
        "ranking_eligible": True,
    }
    heavy_mismatch = {
        "task": {"id": "gk_heavy_other", "team": "general_knowledge", "role": "heavy_worker", "comparison_id": "gk_reasoning_precision_002", "comparison_track": "independent", "worker_class": "heavy"},
        "identity": {
            "team": "general_knowledge", "role": "heavy_worker", "task_id": "gk_heavy_other", "task_version": "v1",
            "model_name": "heavy-model-2", "model_digest": "d-heavy2", "requested_think_mode": "false", "effective_think_mode": "false",
            "structured_output_mode": "prompt_only", "scenario_content_hash": "scenario-2", "comparison_scenario_hash": "scenario-2", "task_suite_version": "2"
        },
        "response": {"content": "heavy2", "metrics": {}, "effective_prompt": {"prompt_components": {"scenario_version": "1", "comparison_information_mode": "symmetric"}}},
        "scores": {"weighted_total": 0.9, "correctness_score": 0.9, "completeness_score": 0.9, "safety_score": 1.0},
        "score_status": "final",
        "ranking_eligible": True,
    }

    (fast_dir / "fast.result.json").write_text(json.dumps(fast), encoding="utf-8")
    (heavy_dir / "heavy.result.json").write_text(json.dumps(heavy_match), encoding="utf-8")
    (heavy_dir / "heavy2.result.json").write_text(json.dumps(heavy_mismatch), encoding="utf-8")

    generate_reports(run)
    rows = list(csv.DictReader((run / "escalation_value.csv").read_text(encoding="utf-8").splitlines()))
    independent = [r for r in rows if r["track"] == "independent"]

    assert len(independent) == 1
    assert independent[0]["scenario_content_hash"] == "scenario-1"
    assert independent[0]["fast_model"] == "fast-model"
    assert independent[0]["heavy_model"] == "heavy-model"
