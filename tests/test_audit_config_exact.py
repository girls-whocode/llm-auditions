from __future__ import annotations

import argparse
from pathlib import Path

from llm_auditions.cli import cmd_audit_config
from llm_auditions.configuration import Configuration
from llm_auditions.models import TaskDefinition


def test_audit_config_exact_exit_status():
    rc = cmd_audit_config(argparse.Namespace(), Configuration())
    assert rc == 0


def test_audit_config_reports_missing_rubric_finalization_and_missing_partner(monkeypatch, capsys):
    cfg = Configuration()
    cfg.load()

    task = TaskDefinition(
        id="audit_task",
        team="general_knowledge",
        role="fast_worker",
        prompt="p",
        verification_classification="rubric_assisted",
        rubric_rules=[
            {
                "rule_id": "r1",
                "description": "required",
                "type": "required",
                "weight": 1.0,
                "matcher": {"type": "phrase_aliases", "phrases": ["userspace"]},
            }
        ],
    )

    monkeypatch.setattr("llm_auditions.cli._load_tasks", lambda: [task])
    monkeypatch.setattr("llm_auditions.cli.validate_tasks", lambda tasks: [])
    monkeypatch.setattr("llm_auditions.cli.detect_duplicate_tasks", lambda tasks: {"duplicate_ids": [], "duplicate_prompts": []})
    monkeypatch.setattr(
        "llm_auditions.cli._load_raw_task_entries",
        lambda _: [
            (
                Path("fixtures/test.yaml"),
                {
                    "id": "raw_task_missing_finalization",
                    "team": "general_knowledge",
                    "role": "fast_worker",
                    "prompt": "p",
                    "verification_classification": "rubric_assisted",
                    "rubric_rules": [
                        {
                            "rule_id": "r1",
                            "description": "required",
                            "type": "required",
                            "weight": 1.0,
                            "matcher": {"type": "phrase_aliases", "phrases": ["userspace"]},
                        }
                    ],
                    "comparison_id": "cmp_missing_partner",
                    "comparison_track": "independent",
                    "worker_class": "fast",
                },
            )
        ],
    )
    monkeypatch.setattr("llm_auditions.cli.OllamaClient.list_models", lambda self: [])
    monkeypatch.setattr(cfg, "list_teams", lambda: [])
    monkeypatch.setattr(cfg, "get_audit_policy", lambda: {
        "require_rubric_finalization_field": True,
        "require_comparison_fast_heavy_pairs": True,
        "require_optional_rubric_bonus_semantics": True,
        "forbid_development_execution_enable": True,
    })

    rc = cmd_audit_config(argparse.Namespace(), cfg)
    out = capsys.readouterr().out

    assert rc == 1
    assert "type=rubric_finalization_missing" in out
    assert "type=comparison_missing_heavy_partner" in out


def test_audit_config_reports_required_pass8_comparison_finding_names(monkeypatch, capsys):
    cfg = Configuration()
    cfg.load()

    valid_task = TaskDefinition(
        id="cmp_task",
        team="linux_infrastructure",
        role="fast_worker",
        prompt="p",
        verification_classification="rubric_assisted",
        rubric_finalization="mixed",
        rubric_rules=[
            {
                "rule_id": "r1",
                "description": "required",
                "type": "required",
                "weight": 1.0,
                "matcher": {"type": "phrase_aliases", "phrases": ["safe"]},
            }
        ],
        comparison_id="linux_ops_safety_001",
        comparison_track="independent",
        worker_class="fast",
        comparison_scenario_ref="fixtures/comparisons/linux_ops_safety_001.json",
    )
    missing_fixture_task = TaskDefinition(
        id="cmp_task_missing_fixture",
        team="linux_infrastructure",
        role="escalation",
        prompt="p",
        verification_classification="rubric_assisted",
        rubric_finalization="mixed",
        rubric_rules=[
            {
                "rule_id": "r2",
                "description": "required",
                "type": "required",
                "weight": 1.0,
                "matcher": {"type": "phrase_aliases", "phrases": ["safe"]},
            }
        ],
        comparison_id="linux_ops_safety_001",
        comparison_track="independent",
        worker_class="heavy",
        comparison_scenario_ref="fixtures/comparisons/does_not_exist.json",
    )

    monkeypatch.setattr("llm_auditions.cli._load_tasks", lambda: [valid_task, missing_fixture_task])
    monkeypatch.setattr("llm_auditions.cli.validate_tasks", lambda tasks: [])
    monkeypatch.setattr("llm_auditions.cli.detect_duplicate_tasks", lambda tasks: {"duplicate_ids": [], "duplicate_prompts": []})
    monkeypatch.setattr(
        "llm_auditions.cli._load_raw_task_entries",
        lambda _: [
            (
                Path("fixtures/test.yaml"),
                {
                    "id": "cmp_fast",
                    "team": "linux_infrastructure",
                    "role": "fast_worker",
                    "prompt": "p",
                    "verification_classification": "rubric_assisted",
                    "rubric_finalization": "deterministic",
                    "rubric_rules": [],
                    "comparison_id": "linux_ops_safety_001",
                    "comparison_track": "handoff",
                    "worker_class": "fast",
                    "comparison_scenario_ref": "fixtures/comparisons/linux_ops_safety_001.json",
                    "comparison_shared_rubric_version": "1",
                    "use_shared_scenario_rubric": False,
                    "role_rubric_rules": [
                        {
                            "rule_id": "requires_device_latency",
                            "description": "legacy conflict",
                            "type": "required",
                            "weight": 1.0,
                            "source_fact_ids": ["unknown_fact"],
                            "matcher": {"type": "phrase_aliases", "phrases": ["du -ah"]},
                        }
                    ],
                    "required_concepts": [],
                },
            ),
            (
                Path("fixtures/test.yaml"),
                {
                    "id": "cmp_heavy",
                    "team": "linux_infrastructure",
                    "role": "escalation",
                    "prompt": "p",
                    "verification_classification": "rubric_assisted",
                    "rubric_finalization": "mixed",
                    "rubric_rules": [],
                    "comparison_id": "linux_ops_safety_001",
                    "comparison_track": "handoff",
                    "worker_class": "heavy",
                    "comparison_scenario_ref": "fixtures/comparisons/linux_ops_safety_001.json",
                    "comparison_shared_rubric_version": "2",
                    "use_shared_scenario_rubric": True,
                    "role_rubric_rules": [],
                    "required_concepts": [],
                },
            ),
            (
                Path("fixtures/test.yaml"),
                {
                    "id": "complex_analysis_det",
                    "team": "architecture",
                    "role": "worker",
                    "prompt": "p",
                    "verification_classification": "rubric_assisted",
                    "rubric_finalization": "deterministic",
                    "rubric_rules": [],
                },
            ),
        ],
    )
    monkeypatch.setattr("llm_auditions.cli.OllamaClient.list_models", lambda self: [])
    monkeypatch.setattr(cfg, "list_teams", lambda: [])
    monkeypatch.setattr(
        "llm_auditions.cli._build_execution_plan",
        lambda config, profile, **kwargs: (
            [],
            [
                {
                    "plan_row_id": "f1",
                    "comparison_id": "c1",
                    "comparison_track": "handoff",
                    "worker_class": "fast",
                    "requested_think_mode": "false",
                    "structured_output_mode": "prompt_only",
                    "scenario_content_hash": "a",
                    "scenario_version": "1",
                    "comparison_information_mode": "symmetric",
                },
                {
                    "plan_row_id": "h1",
                    "comparison_id": "c1",
                    "comparison_track": "handoff",
                    "worker_class": "heavy",
                    "fast_plan_row_id": "f1",
                    "requested_think_mode": "low",
                    "structured_output_mode": "ollama_schema",
                    "scenario_content_hash": "b",
                    "scenario_version": "2",
                    "comparison_information_mode": "asymmetric",
                },
            ],
            {},
        ),
    )
    monkeypatch.setattr(
        cfg,
        "get_audit_policy",
        lambda: {
            "require_rubric_finalization_field": True,
            "require_comparison_fast_heavy_pairs": True,
            "require_optional_rubric_bonus_semantics": True,
            "forbid_development_execution_enable": True,
        },
    )

    rc = cmd_audit_config(argparse.Namespace(), cfg)
    out = capsys.readouterr().out

    assert rc == 1
    assert "type=comparison_shared_rubric_missing" in out
    assert "type=comparison_required_fact_uncovered" in out
    assert "type=comparison_role_rule_conflict" in out
    assert "type=comparison_legacy_rule_detected" in out
    assert "type=comparison_rubric_version_mismatch" in out
    assert "type=handoff_cross_think_dependency" in out
    assert "type=handoff_cross_output_dependency" in out
    assert "type=handoff_cross_scenario_dependency" in out
    assert "type=handoff_cross_information_mode_dependency" in out
    assert "type=comparison_scenario_missing_from_manifest_inventory" in out
    assert "type=complex_deterministic_without_override" in out
