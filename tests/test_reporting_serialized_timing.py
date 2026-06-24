from __future__ import annotations

import csv
import json
from pathlib import Path

from llm_auditions.models import (
    ModelResponse,
    OllamaMetrics,
    ResultIdentity,
    ScoreBreakdown,
    TaskDefinition,
    TaskResult,
)
from llm_auditions.reporting import generate_reports


def test_reporting_reconstructs_timing_from_serialized_raw_fields(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "run_manifest.json").write_text(json.dumps({"profile": "smoke", "task_suite_version": "2"}), encoding="utf-8")
    (run / "events.jsonl").write_text("", encoding="utf-8")

    task = TaskDefinition(
        id="timing_task",
        team="baseline",
        role="baseline_all",
        prompt="p",
        verification_classification="rubric_assisted",
        rubric_finalization="deterministic",
        rubric_rules=[
            {
                "rule_id": "r1",
                "description": "required",
                "type": "required",
                "weight": 1.0,
                "matcher": {"type": "phrase_aliases", "phrases": ["ok"]},
            }
        ],
    )
    identity = ResultIdentity(
        team="baseline",
        role="baseline_all",
        task_id="timing_task",
        task_version="v1",
        model_name="gemma4:12b",
        model_digest="d1",
        requested_think_mode="false",
        effective_think_mode="false",
        structured_output_mode="prompt_only",
        temperature=0.0,
        num_ctx=8192,
        num_predict=256,
        system_prompt_hash="a",
        user_prompt_hash="b",
        task_suite_version="2",
        verifier_version="2",
        scoring_version="2",
        engine_version="0.10.0",
    )
    metrics = OllamaMetrics(
        load_duration_ns=2_000_000_000,
        prompt_eval_duration_ns=3_000_000_000,
        eval_duration_ns=5_000_000_000,
        total_duration_ns=10_000_000_000,
        wall_clock_seconds_local=11.5,
        eval_count=100,
    )
    response = ModelResponse(
        model="gemma4:12b",
        requested_think_mode="false",
        effective_think_mode="false",
        content="ok",
        metrics=metrics,
    )
    result = TaskResult(
        identity=identity,
        task=task,
        response=response,
        status="completed",
        score_status="final",
        ranking_eligible=True,
        scores=ScoreBreakdown(weighted_total=0.9, score_status="final", ranking_eligible=True),
    )

    out_dir = run / "baseline" / "baseline_all"
    out_dir.mkdir(parents=True)
    (out_dir / "timing.result.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")

    generate_reports(run)

    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    assert summary["median_load_seconds"] == 2.0
    assert summary["median_prompt_eval_seconds"] == 3.0
    assert summary["median_generation_seconds"] == 5.0
    assert summary["median_ollama_total_seconds"] == 10.0
    assert summary["median_wall_clock_seconds"] == 11.5
    assert summary["median_overhead_seconds"] == 1.5
    assert summary["median_generation_rate"] == 20.0

    rows = list(csv.DictReader((run / "leaderboard_by_role.csv").read_text(encoding="utf-8").splitlines()))
    assert len(rows) == 1
    row = rows[0]
    assert float(row["median_load_seconds"]) == 2.0
    assert float(row["median_prompt_eval_seconds"]) == 3.0
    assert float(row["median_generation_seconds"]) == 5.0
    assert float(row["median_ollama_total_seconds"]) == 10.0
    assert float(row["median_wall_clock_seconds"]) == 11.5
    assert float(row["median_overhead_seconds"]) == 1.5
    assert float(row["median_generation_rate"]) == 20.0
