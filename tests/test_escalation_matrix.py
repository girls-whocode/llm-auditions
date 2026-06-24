from __future__ import annotations

import csv
import json
from pathlib import Path

from llm_auditions.reporting import generate_reports
from llm_auditions.task_loader import load_tasks_from_dir


def test_escalation_output_uses_comparison_ids(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "run_manifest.json").write_text(json.dumps({"profile": "smoke", "task_suite_version": "1"}))
    fast = {
        "task": {"id": "x", "team": "general_knowledge", "role": "fast_worker", "comparison_id": "cmp1", "comparison_track": "independent", "worker_class": "fast"},
        "identity": {"model_name": "gemma4:12b", "task_version": "v1"},
        "scores": {"correctness_score": 0.4, "completeness_score": 0.4, "safety_score": 1.0, "latency_score": 1.0, "weighted_total": 0.5},
        "score_status": "final",
        "ranking_eligible": True,
        "hard_fail": False,
    }
    heavy = {
        "task": {"id": "x", "team": "general_knowledge", "role": "heavy_worker", "comparison_id": "cmp1", "comparison_track": "independent", "worker_class": "heavy"},
        "identity": {"model_name": "gemma4:26b", "task_version": "v1"},
        "scores": {"correctness_score": 0.8, "completeness_score": 0.8, "safety_score": 1.0, "latency_score": 0.8, "weighted_total": 0.8},
        "score_status": "final",
        "ranking_eligible": True,
        "hard_fail": False,
    }
    (run / "events.jsonl").write_text("")
    (run / "general_knowledge" / "fast_worker").mkdir(parents=True)
    (run / "general_knowledge" / "heavy_worker").mkdir(parents=True)
    (run / "general_knowledge" / "fast_worker" / "a.result.json").write_text(json.dumps(fast))
    (run / "general_knowledge" / "heavy_worker" / "b.result.json").write_text(json.dumps(heavy))
    generate_reports(run)
    rows = list(csv.DictReader((run / "escalation_value.csv").read_text().splitlines()))
    assert rows and rows[0]["comparison_id"] == "cmp1"


def test_escalation_generates_candidate_cross_product(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "run_manifest.json").write_text(json.dumps({"profile": "smoke", "task_suite_version": "1"}))
    (run / "events.jsonl").write_text("")
    (run / "general_knowledge" / "fast_worker").mkdir(parents=True)
    (run / "general_knowledge" / "heavy_worker").mkdir(parents=True)

    fast_models = ["gemma4:12b", "qwen3.5:9b"]
    heavy_models = ["gemma4:26b", "qwen3-coder:30b", "phi4-reasoning:14b"]

    idx = 0
    for fm in fast_models:
        payload = {
            "task": {"id": "pair", "team": "general_knowledge", "role": "fast_worker", "comparison_id": "cmp_cross", "comparison_track": "independent", "worker_class": "fast"},
            "identity": {"model_name": fm, "model_digest": f"d-{fm}", "task_version": "v1", "requested_think_mode": "false", "structured_output_mode": "prompt_only", "task_suite_version": "1"},
            "scores": {"correctness_score": 0.5, "completeness_score": 0.5, "safety_score": 1.0, "latency_score": 1.0, "weighted_total": 0.5},
            "score_status": "final",
            "ranking_eligible": True,
            "hard_fail": False,
            "response": {"metrics": {"generation_seconds": 1.0, "wall_clock_seconds": 1.2, "eval_count": 100}},
        }
        (run / "general_knowledge" / "fast_worker" / f"f{idx}.result.json").write_text(json.dumps(payload))
        idx += 1

    idx = 0
    for hm in heavy_models:
        payload = {
            "task": {"id": "pair", "team": "general_knowledge", "role": "heavy_worker", "comparison_id": "cmp_cross", "comparison_track": "independent", "worker_class": "heavy"},
            "identity": {"model_name": hm, "model_digest": f"d-{hm}", "task_version": "v1", "requested_think_mode": "false", "structured_output_mode": "prompt_only", "task_suite_version": "1"},
            "scores": {"correctness_score": 0.8, "completeness_score": 0.8, "safety_score": 1.0, "latency_score": 0.8, "weighted_total": 0.8},
            "score_status": "final",
            "ranking_eligible": True,
            "hard_fail": False,
            "response": {"metrics": {"generation_seconds": 2.0, "wall_clock_seconds": 2.3, "eval_count": 220}},
        }
        (run / "general_knowledge" / "heavy_worker" / f"h{idx}.result.json").write_text(json.dumps(payload))
        idx += 1

    generate_reports(run)
    rows = list(csv.DictReader((run / "escalation_value.csv").read_text().splitlines()))
    assert len(rows) == 6


def test_escalation_does_not_cross_pair_think_or_output_modes(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "run_manifest.json").write_text(json.dumps({"profile": "smoke", "task_suite_version": "1"}))
    (run / "events.jsonl").write_text("")
    (run / "general_knowledge" / "fast_worker").mkdir(parents=True)
    (run / "general_knowledge" / "heavy_worker").mkdir(parents=True)

    fast_false = {
        "task": {"id": "pair", "team": "general_knowledge", "role": "fast_worker", "comparison_id": "cmp_mode", "comparison_track": "independent", "worker_class": "fast"},
        "identity": {"model_name": "gemma4:12b", "model_digest": "df", "task_version": "v1", "requested_think_mode": "false", "structured_output_mode": "prompt_only", "task_suite_version": "1"},
        "scores": {"correctness_score": 0.5, "completeness_score": 0.5, "safety_score": 1.0, "latency_score": 1.0, "weighted_total": 0.5},
        "score_status": "final",
        "ranking_eligible": True,
        "hard_fail": False,
    }
    heavy_high = {
        "task": {"id": "pair", "team": "general_knowledge", "role": "heavy_worker", "comparison_id": "cmp_mode", "comparison_track": "independent", "worker_class": "heavy"},
        "identity": {"model_name": "gemma4:26b", "model_digest": "dh", "task_version": "v1", "requested_think_mode": "high", "structured_output_mode": "ollama_schema", "task_suite_version": "1"},
        "scores": {"correctness_score": 0.8, "completeness_score": 0.8, "safety_score": 1.0, "latency_score": 0.8, "weighted_total": 0.8},
        "score_status": "final",
        "ranking_eligible": True,
        "hard_fail": False,
    }

    (run / "general_knowledge" / "fast_worker" / "a.result.json").write_text(json.dumps(fast_false))
    (run / "general_knowledge" / "heavy_worker" / "b.result.json").write_text(json.dumps(heavy_high))

    generate_reports(run)
    rows = list(csv.DictReader((run / "escalation_value.csv").read_text().splitlines()))
    assert len(rows) == 0


def test_unresolved_results_emit_unresolved_status(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "run_manifest.json").write_text(json.dumps({"profile": "smoke", "task_suite_version": "1"}))
    (run / "events.jsonl").write_text("")
    (run / "general_knowledge" / "fast_worker").mkdir(parents=True)
    (run / "general_knowledge" / "heavy_worker").mkdir(parents=True)

    fast = {
        "task": {"id": "pair", "team": "general_knowledge", "role": "fast_worker", "comparison_id": "cmp_unresolved", "comparison_track": "independent", "worker_class": "fast"},
        "identity": {"model_name": "gemma4:12b", "model_digest": "df", "task_version": "v1", "requested_think_mode": "false", "structured_output_mode": "prompt_only", "task_suite_version": "1"},
        "scores": {"correctness_score": 0.5, "completeness_score": 0.5, "safety_score": 1.0, "latency_score": 1.0, "weighted_total": 0.5},
        "score_status": "provisional",
        "ranking_eligible": False,
        "hard_fail": False,
    }
    heavy = {
        "task": {"id": "pair", "team": "general_knowledge", "role": "heavy_worker", "comparison_id": "cmp_unresolved", "comparison_track": "independent", "worker_class": "heavy"},
        "identity": {"model_name": "gemma4:26b", "model_digest": "dh", "task_version": "v1", "requested_think_mode": "false", "structured_output_mode": "prompt_only", "task_suite_version": "1"},
        "scores": {"correctness_score": 0.8, "completeness_score": 0.8, "safety_score": 1.0, "latency_score": 0.8, "weighted_total": 0.8},
        "score_status": "final",
        "ranking_eligible": True,
        "hard_fail": False,
    }

    (run / "general_knowledge" / "fast_worker" / "a.result.json").write_text(json.dumps(fast))
    (run / "general_knowledge" / "heavy_worker" / "b.result.json").write_text(json.dumps(heavy))

    generate_reports(run)
    rows = list(csv.DictReader((run / "escalation_value.csv").read_text().splitlines()))
    assert len(rows) == 1
    assert rows[0]["status"] == "unresolved"


def test_fixture_comparison_metadata_coverage():
    tasks = load_tasks_from_dir(Path(__file__).parent.parent / "fixtures")
    pairs: dict[tuple[str, str], set[str]] = {}
    team_groups: dict[str, set[str]] = {}
    for t in tasks:
        if not t.comparison_id:
            continue
        key = (t.comparison_id, t.comparison_track)
        pairs.setdefault(key, set()).add(t.worker_class)
        team_groups.setdefault(t.team, set()).add(t.comparison_id)

    valid_pairs = [k for k, classes in pairs.items() if {"fast", "heavy"}.issubset(classes)]
    assert len(valid_pairs) >= 12
    assert len(team_groups.get("general_knowledge", set())) >= 2
    assert len(team_groups.get("linux_infrastructure", set())) >= 2
    assert len(team_groups.get("engineering_hardware", set())) >= 2
    assert len(team_groups.get("security", set())) >= 2
    assert len(team_groups.get("architecture", set())) >= 2
    assert len(team_groups.get("development", set())) >= 2
