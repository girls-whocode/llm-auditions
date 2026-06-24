from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from llm_auditions.reporting import generate_reports


def _identity_key(identity: dict, fixture_hashes: dict | None = None) -> str:
    fixture_hashes = fixture_hashes or {}
    parts = [
        identity.get("team", ""),
        identity.get("role", ""),
        identity.get("task_id", ""),
        identity.get("task_version", "v1"),
        identity.get("model_name", ""),
        identity.get("model_digest", ""),
        identity.get("requested_think_mode", ""),
        identity.get("structured_output_mode", ""),
        str(identity.get("temperature", "")),
        str(identity.get("num_ctx", "")),
        str(identity.get("num_predict", "")),
        identity.get("system_prompt_hash", ""),
        identity.get("user_prompt_hash", ""),
        identity.get("task_suite_version", ""),
        identity.get("verifier_version", ""),
        identity.get("scoring_version", ""),
        identity.get("engine_version", ""),
        identity.get("handoff_fast_identity_key", ""),
        identity.get("handoff_fast_response_hash", ""),
        identity.get("comparison_scenario_hash", ""),
        identity.get("scenario_content_hash", ""),
        identity.get("effective_think_mode", ""),
    ]
    for key in sorted(fixture_hashes):
        parts.append(f"{key}:{fixture_hashes[key]}")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:24]


def test_handoff_report_uses_recorded_dependency_only(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "events.jsonl").write_text("", encoding="utf-8")
    (run / "run_manifest.json").write_text(json.dumps({"profile": "smoke", "task_suite_version": "2"}), encoding="utf-8")

    fast_dir = run / "security" / "worker"
    heavy_dir = run / "security" / "reviewer"
    fast_dir.mkdir(parents=True)
    heavy_dir.mkdir(parents=True)

    fast_identity = {
        "team": "security",
        "role": "worker",
        "task_id": "sec_fast",
        "task_version": "v1",
        "model_name": "fast-model",
        "model_digest": "d-fast",
        "requested_think_mode": "false",
        "effective_think_mode": "false",
        "structured_output_mode": "prompt_only",
        "temperature": 0,
        "num_ctx": 8192,
        "num_predict": 512,
        "system_prompt_hash": "a",
        "user_prompt_hash": "b",
        "comparison_scenario_hash": "scenario-h",
        "scenario_content_hash": "scenario-h",
    }
    fast_key = _identity_key(fast_identity)
    fast_response = "fast output"
    fast_hash = hashlib.sha256(fast_response.encode()).hexdigest()[:16]

    fast_result = {
        "task": {"id": "sec_fast", "team": "security", "role": "worker", "comparison_id": "sec_incident_response_002", "comparison_track": "handoff", "worker_class": "fast"},
        "identity": fast_identity,
        "response": {
            "content": fast_response,
            "metrics": {},
            "effective_prompt": {"prompt_components": {"scenario_version": "1", "comparison_information_mode": "asymmetric"}},
        },
        "scores": {"weighted_total": 0.5, "correctness_score": 0.4, "completeness_score": 0.4, "safety_score": 1.0},
        "score_status": "final",
        "ranking_eligible": True,
    }

    unrelated_fast = dict(fast_result)
    unrelated_fast["identity"] = dict(fast_identity, model_name="unused-fast", model_digest="d-unused", user_prompt_hash="c")

    heavy_identity = {
        "team": "security",
        "role": "reviewer",
        "task_id": "sec_heavy",
        "task_version": "v1",
        "model_name": "heavy-model",
        "model_digest": "d-heavy",
        "requested_think_mode": "false",
        "effective_think_mode": "false",
        "structured_output_mode": "prompt_only",
        "temperature": 0,
        "num_ctx": 8192,
        "num_predict": 512,
        "system_prompt_hash": "x",
        "user_prompt_hash": "y",
        "comparison_scenario_hash": "scenario-h",
        "scenario_content_hash": "scenario-h",
        "handoff_fast_identity_key": fast_key,
        "handoff_fast_response_hash": fast_hash,
    }
    heavy_result = {
        "task": {"id": "sec_heavy", "team": "security", "role": "reviewer", "comparison_id": "sec_incident_response_002", "comparison_track": "handoff", "worker_class": "heavy"},
        "identity": heavy_identity,
        "response": {
            "content": "heavy output",
            "metrics": {},
            "effective_prompt": {"prompt_components": {"scenario_version": "1", "comparison_information_mode": "asymmetric"}},
        },
        "scores": {"weighted_total": 0.8, "correctness_score": 0.8, "completeness_score": 0.8, "safety_score": 1.0},
        "score_status": "final",
        "ranking_eligible": True,
    }

    (fast_dir / "fast.result.json").write_text(json.dumps(fast_result), encoding="utf-8")
    (fast_dir / "unused.result.json").write_text(json.dumps(unrelated_fast), encoding="utf-8")
    (heavy_dir / "heavy.result.json").write_text(json.dumps(heavy_result), encoding="utf-8")

    generate_reports(run)
    rows = list(csv.DictReader((run / "escalation_value.csv").read_text(encoding="utf-8").splitlines()))

    handoff_rows = [r for r in rows if r["track"] == "handoff"]
    assert len(handoff_rows) == 1
    assert handoff_rows[0]["fast_result_identity"] == fast_key
    assert handoff_rows[0]["fast_model"] == "fast-model"
    assert handoff_rows[0]["heavy_model"] == "heavy-model"
