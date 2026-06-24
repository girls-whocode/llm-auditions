from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm_auditions.cli import cmd_audit_run
from llm_auditions.configuration import Configuration
from llm_auditions.models import ResultIdentity


def test_audit_run_reports_findings(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "run_manifest.json").write_text(json.dumps({"run_id": "r1", "profile": "smoke", "request_count": 1}))
    (run / "task_manifest.json").write_text(json.dumps({"run_id": "r1", "profile": "smoke", "requests": []}))
    (run / "events.jsonl").write_text(json.dumps({"identity_key": "k1", "status": "completed", "score_status": "final", "ranking_eligible": True}) + "\n")
    rc = cmd_audit_run(argparse.Namespace(run_dir=str(run)), Configuration())
    assert (run / "RUN_AUDIT.json").exists()
    assert (run / "RUN_AUDIT.md").exists()
    assert rc == 0 or rc == 1


def test_audit_run_detects_required_artifact_findings(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "run_manifest.json").write_text(json.dumps({"run_id": "r2", "profile": "smoke", "request_count": 1}))
    (run / "task_manifest.json").write_text(json.dumps({"run_id": "r2", "profile": "smoke", "requests": []}))
    (run / "events.jsonl").write_text(json.dumps({"identity_key": "k2", "status": "completed", "score_status": "final", "ranking_eligible": True}) + "\n")

    result_dir = run / "baseline" / "baseline_all"
    result_dir.mkdir(parents=True)
    bad_result = {
        "identity": {"team": "baseline", "role": "baseline_all", "task_id": "t1", "task_version": "v1", "model_name": "m", "requested_think_mode": "false", "effective_think_mode": "false"},
        "task": {"id": "t1", "team": "baseline", "role": "baseline_all", "verification_classification": "rubric_assisted"},
        "response": {"content": "", "thinking": "hidden chain of thought", "metrics": {"done_reason": "length"}},
        "scores": {"score_status": "final"},
        "score_status": "final",
        "ranking_eligible": True,
        "schema_errors": ["bad json"],
        "verifier_output": {"checks": []},
    }
    (result_dir / "bad.result.json").write_text(json.dumps(bad_result))

    rc = cmd_audit_run(argparse.Namespace(run_dir=str(run)), Configuration())
    audit = json.loads((run / "RUN_AUDIT.json").read_text())
    types = {item["type"] for item in audit.get("findings", [])}

    assert rc == 1
    assert "thinking_under_think_false" in types
    assert "empty_final_answer" in types
    assert "length_truncation" in types
    assert "schema_failure" in types
    assert "missing_request_payload" in types
    assert "missing_effective_prompt" in types
    assert "missing_rubric_output" in types


def test_audit_run_reports_timing_and_rubric_provisional_issues(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "run_manifest.json").write_text(
        json.dumps(
            {
                "run_id": "r3",
                "profile": "smoke",
                "request_count": 1,
                "engine_version": "0.9.0",
                "task_suite_version": "1",
                "scoring_version": "1",
                "verifier_version": "1",
                "report_version": "1",
                "model_digests": {},
            }
        )
    )
    (run / "task_manifest.json").write_text(json.dumps({"run_id": "r3", "profile": "smoke", "requests": []}))
    (run / "events.jsonl").write_text(json.dumps({"identity_key": "k3", "status": "completed", "score_status": "final", "ranking_eligible": True}) + "\n")

    result_dir = run / "baseline" / "baseline_all"
    result_dir.mkdir(parents=True)
    payload = {
        "identity": {
            "team": "baseline",
            "role": "baseline_all",
            "task_id": "t1",
            "task_version": "v1",
            "model_name": "m",
            "requested_think_mode": "false",
            "effective_think_mode": "false",
        },
        "task": {
            "id": "t1",
            "team": "baseline",
            "role": "baseline_all",
            "verification_classification": "rubric_assisted",
            "rubric_finalization": "deterministic",
        },
        "response": {
            "content": "ok",
            "request_payload": {},
            "effective_prompt": {},
            "metrics": {
                "load_seconds": 1.0,
                "prompt_eval_seconds": 1.0,
                "generation_seconds": 1.0,
                "ollama_total_seconds": 3.0,
                "wall_clock_seconds": 2.0,
                "overhead_seconds": -1.0,
            },
        },
        "scores": {"score_status": "provisional"},
        "score_status": "provisional",
        "ranking_eligible": False,
        "schema_errors": [],
        "verifier_output": {
            "rubric_rules": [
                {"rule_id": "r1", "status": "pass"},
                {"rule_id": "r2", "status": "pass"},
            ]
        },
    }
    (result_dir / "x.result.json").write_text(json.dumps(payload))

    rc = cmd_audit_run(argparse.Namespace(run_dir=str(run)), Configuration())
    audit = json.loads((run / "RUN_AUDIT.json").read_text())
    types = {f["type"] for f in audit.get("findings", [])}

    assert rc == 1
    assert "rubric_stuck_provisional" in types
    assert "negative_timing_overhead" in types
    assert "timing_field_inconsistency" in types


def test_audit_run_reports_required_pass8_finding_names(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()

    (run / "run_manifest.json").write_text(
        json.dumps(
            {
                "run_id": "r4",
                "profile": "smoke",
                "request_count": 2,
                "fixture_hashes": {},
            }
        )
    )
    (run / "task_manifest.json").write_text(json.dumps({"run_id": "r4", "profile": "smoke", "requests": []}))
    (run / "events.jsonl").write_text(json.dumps({"identity_key": "k4", "status": "completed", "score_status": "final", "ranking_eligible": True}) + "\n")

    out_dir = run / "linux_infrastructure" / "escalation"
    out_dir.mkdir(parents=True)

    fast_identity = {
        "team": "linux_infrastructure",
        "role": "fast_worker",
        "task_id": "proof_cmp",
        "task_version": "v1",
        "model_name": "fast-a",
        "model_digest": "d-fast",
        "requested_think_mode": "false",
        "effective_think_mode": "false",
        "structured_output_mode": "prompt_only",
        "temperature": 0.0,
        "num_ctx": 1024,
        "num_predict": 128,
        "system_prompt_hash": "a",
        "user_prompt_hash": "b",
        "scenario_content_hash": "hash-a",
    }
    fast_key = ResultIdentity.model_validate(fast_identity).key()

    fast = {
        "identity": fast_identity,
        "task": {
            "id": "proof_cmp",
            "team": "linux_infrastructure",
            "role": "fast_worker",
            "comparison_id": "linux_firewall_change_002",
            "comparison_track": "handoff",
            "worker_class": "fast",
            "comparison_scenario_ref": "fixtures/comparisons/linux_firewall_change_002.json",
            "verification_classification": "rubric_assisted",
            "rubric_finalization": "mixed",
            "comparison_shared_rubric_version": "1",
        },
        "response": {
            "content": "fast",
            "request_payload": {},
            "effective_prompt": {
                "prompt_components": {
                    "scenario_content_hash": "hash-a",
                    "scenario_version": "1",
                    "comparison_information_mode": "symmetric",
                }
            },
            "metrics": {},
        },
        "verifier_output": {"rubric_rules": [{"rule_id": "requires_backup_rules", "status": "pass"}]},
        "scores": {"score_status": "final"},
        "score_status": "final",
        "ranking_eligible": True,
        "schema_errors": [],
    }

    heavy = {
        "identity": {
            "team": "linux_infrastructure",
            "role": "escalation",
            "task_id": "proof_cmp",
            "task_version": "v1",
            "model_name": "heavy-a",
            "model_digest": "d-heavy",
            "requested_think_mode": "low",
            "effective_think_mode": "low",
            "structured_output_mode": "ollama_schema",
            "temperature": 0.0,
            "num_ctx": 1024,
            "num_predict": 128,
            "system_prompt_hash": "a",
            "user_prompt_hash": "b",
            "handoff_fast_identity_key": fast_key,
            "handoff_fast_response_hash": "bad-hash",
            "scenario_content_hash": "hash-b",
        },
        "task": {
            "id": "proof_cmp",
            "team": "linux_infrastructure",
            "role": "escalation",
            "comparison_id": "linux_firewall_change_002",
            "comparison_track": "handoff",
            "worker_class": "heavy",
            "comparison_scenario_ref": "fixtures/comparisons/linux_firewall_change_002.json",
            "verification_classification": "rubric_assisted",
            "rubric_finalization": "mixed",
            "comparison_shared_rubric_version": "999",
        },
        "response": {
            "content": "heavy",
            "request_payload": {
                "handoff_payload": {"fast_result_identity": fast_key}
            },
            "effective_prompt": {
                "prompt_components": {
                    "scenario_content_hash": "hash-b",
                    "scenario_version": "2",
                    "comparison_information_mode": "asymmetric",
                }
            },
            "metrics": {},
        },
        "verifier_output": {"rubric_rules": [{"rule_id": "only_role_rule", "status": "pass"}]},
        "scores": {"score_status": "final"},
        "score_status": "final",
        "ranking_eligible": True,
        "schema_errors": [],
    }

    (out_dir / "fast.result.json").write_text(json.dumps(fast))
    (out_dir / "heavy.result.json").write_text(json.dumps(heavy))

    rc = cmd_audit_run(argparse.Namespace(run_dir=str(run)), Configuration())
    audit = json.loads((run / "RUN_AUDIT.json").read_text())
    types = {f["type"] for f in audit.get("findings", [])}

    assert rc == 1
    assert "handoff_think_mode_mismatch" in types
    assert "handoff_output_mode_mismatch" in types
    assert "handoff_scenario_hash_mismatch" in types
    assert "handoff_scenario_version_mismatch" in types
    assert "handoff_information_mode_mismatch" in types
    assert "comparison_rubric_version_mismatch" in types
    assert "comparison_shared_rubric_missing" in types
    assert "comparison_scenario_fixture_untracked" in types
    assert "mixed_result_incorrectly_final" in types
