from __future__ import annotations

import csv
import json
from pathlib import Path

from llm_auditions.models import ModelResponse, ResultIdentity, TaskDefinition, TaskResult
from llm_auditions.reporting import generate_reports
from llm_auditions.scoring import score_result


def _identity() -> ResultIdentity:
    return ResultIdentity(
        team="general_knowledge",
        role="fast_worker",
        task_id="gk_1",
        model_name="gemma4:12b",
        model_digest="digest",
        requested_think_mode="false",
        effective_think_mode="false",
        temperature=0.0,
        num_ctx=8192,
        num_predict=1024,
        system_prompt_hash="a",
        user_prompt_hash="b",
    )


def _task(rubric_finalization: str = "deterministic") -> TaskDefinition:
    return TaskDefinition(
        id="rubric_task",
        team="general_knowledge",
        role="fast_worker",
        prompt="p",
        verifier="",
        verification_classification="rubric_assisted",
        rubric_finalization=rubric_finalization,
        rubric_rules=[
            {
                "rule_id": "must_mention",
                "description": "must mention userspace",
                "weight": 1.0,
                "type": "required",
                "matcher": {"type": "phrase_aliases", "phrases": ["userspace"]},
            },
            {
                "rule_id": "forbid_kernel",
                "description": "must not claim kernel module",
                "weight": 1.0,
                "type": "forbidden",
                "matcher": {"type": "forbidden_claim", "phrases": ["kernel module"]},
            },
        ],
    )


def _result(task: TaskDefinition, content: str) -> TaskResult:
    return TaskResult(
        identity=_identity(),
        task=task,
        response=ModelResponse(
            model="gemma4:12b",
            requested_think_mode="false",
            effective_think_mode="false",
            content=content,
        ),
    )


def test_deterministic_rubric_all_passes_is_final():
    task = _task("deterministic")
    result = _result(task, "systemd is a userspace init system")
    scores = score_result(result, "schemas")

    assert scores.score_status == "final"
    assert scores.provisional is False
    assert scores.ranking_eligible is True
    assert scores.human_review_required is False


def test_deterministic_rubric_failure_is_still_final_when_not_hard_gate():
    task = _task("deterministic")
    result = _result(task, "this answer omits the required term")
    scores = score_result(result, "schemas")

    assert scores.correctness_score == 0.5
    assert scores.score_status == "final"
    assert scores.ranking_eligible is True


def test_deterministic_hard_gate_failure_disqualifies():
    task = TaskDefinition(
        id="rubric_hard_gate",
        team="general_knowledge",
        role="fast_worker",
        prompt="p",
        verifier="",
        verification_classification="rubric_assisted",
        rubric_finalization="deterministic",
        rubric_rules=[
            {
                "rule_id": "hg",
                "description": "forbid fabricated claim",
                "weight": 1.0,
                "type": "hard_gate",
                "matcher": {"type": "forbidden_claim", "phrases": ["fabricated claim"]},
            }
        ],
    )
    result = _result(task, "this includes fabricated claim")
    scores = score_result(result, "schemas")

    assert scores.score_status == "disqualified"
    assert scores.ranking_eligible is False


def test_human_review_rubric_requires_human():
    task = _task("human_review")
    result = _result(task, "systemd is a userspace init system")
    scores = score_result(result, "schemas")

    assert scores.score_status == "human_required"
    assert scores.ranking_eligible is False
    assert scores.human_review_required is True


def test_mixed_rubric_with_uncertain_rule_is_provisional():
    task = TaskDefinition(
        id="rubric_mixed",
        team="general_knowledge",
        role="fast_worker",
        prompt="p",
        verifier="",
        verification_classification="rubric_assisted",
        rubric_finalization="mixed",
        rubric_rules=[
            {
                "rule_id": "must_mention",
                "description": "must mention userspace",
                "weight": 1.0,
                "type": "required",
                "matcher": {"type": "phrase_aliases", "phrases": ["userspace"]},
            },
            {
                "rule_id": "human_judgment",
                "description": "requires human review if absent",
                "weight": 1.0,
                "type": "optional",
                "requires_human_review": True,
                "matcher": {"type": "phrase_aliases", "phrases": ["manual-review-anchor"]},
            },
        ],
    )
    result = _result(task, "systemd is a userspace init system")
    scores = score_result(result, "schemas")

    assert scores.score_status == "provisional"
    assert scores.ranking_eligible is False
    assert scores.human_review_required is True


def test_mixed_rubric_deterministic_ready_is_still_human_required_provisional():
    task = TaskDefinition(
        id="rubric_mixed_ready",
        team="general_knowledge",
        role="fast_worker",
        prompt="p",
        verifier="",
        verification_classification="rubric_assisted",
        rubric_finalization="mixed",
        rubric_rules=[
            {
                "rule_id": "must_mention",
                "description": "must mention userspace",
                "weight": 1.0,
                "type": "required",
                "matcher": {"type": "phrase_aliases", "phrases": ["userspace"]},
            }
        ],
    )
    result = _result(task, "userspace mentioned")
    scores = score_result(result, "schemas")

    assert scores.score_status == "provisional"
    assert scores.provisional is True
    assert scores.ranking_eligible is False
    assert scores.human_review_required is True


def test_only_final_eligible_results_enter_definitive_leaderboard(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "run_manifest.json").write_text(json.dumps({"profile": "smoke", "task_suite_version": "1"}))
    (run / "events.jsonl").write_text("")

    role_dir = run / "general_knowledge" / "fast_worker"
    role_dir.mkdir(parents=True)

    def _write(name: str, status: str, eligible: bool, score: float) -> None:
        payload = {
            "status": "completed",
            "task": {"id": "gk", "team": "general_knowledge", "role": "fast_worker", "verification_classification": "rubric_assisted"},
            "identity": {
                "model_name": "gemma4:12b",
                "model_digest": "d1",
                "requested_think_mode": "false",
                "effective_think_mode": "false",
                "structured_output_mode": "prompt_only",
                "task_suite_version": "1",
            },
            "response": {"metrics": {"wall_clock_seconds": 1.0, "net_generation_seconds": 1.0}, "truncated_length_stop": False, "empty_final_content": False},
            "scores": {"weighted_total": score},
            "score_status": status,
            "ranking_eligible": eligible,
            "schema_errors": [],
            "safety_flags": [],
            "verifier_output": {"rubric_rules": []},
            "deterministic_results": [],
            "hard_fail": False,
        }
        (role_dir / f"{name}.result.json").write_text(json.dumps(payload))

    _write("a", "final", True, 0.9)
    _write("b", "provisional", True, 0.2)
    _write("c", "human_required", False, 0.7)
    _write("d", "final", False, 0.1)

    generate_reports(run)
    rows = list(csv.DictReader((run / "leaderboard_by_role.csv").read_text().splitlines()))

    assert len(rows) == 1
    assert rows[0]["sample_size"] == "1"
    assert rows[0]["eligible_count"] == "1"
    assert float(rows[0]["avg_score"]) == 0.9
