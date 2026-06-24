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
