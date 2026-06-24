from __future__ import annotations

from llm_auditions.models import ModelResponse, ResultIdentity, TaskDefinition, TaskResult
from llm_auditions.scoring import score_result


def _identity() -> ResultIdentity:
    return ResultIdentity(
        team="general_knowledge",
        role="fast_worker",
        task_id="optional_case",
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


def test_missing_optional_rule_does_not_lower_base_score():
    task = TaskDefinition(
        id="optional_1",
        team="general_knowledge",
        role="fast_worker",
        prompt="p",
        verification_classification="rubric_assisted",
        rubric_finalization="deterministic",
        rubric_optional_bonus_cap=0.10,
        rubric_rules=[
            {
                "rule_id": "required_userspace",
                "description": "must mention userspace",
                "type": "required",
                "weight": 1.0,
                "matcher": {"type": "phrase_aliases", "phrases": ["userspace"]},
            },
            {
                "rule_id": "optional_bonus",
                "description": "bonus mention",
                "type": "optional",
                "weight": 1.0,
                "matcher": {"type": "phrase_aliases", "phrases": ["bonus-term"]},
            },
        ],
    )

    result = _result(task, "systemd is userspace")
    scores = score_result(result, "schemas")

    assert scores.correctness_score == 1.0


def test_passing_optional_rule_adds_bonus_up_to_cap():
    task = TaskDefinition(
        id="optional_2",
        team="general_knowledge",
        role="fast_worker",
        prompt="p",
        verification_classification="rubric_assisted",
        rubric_finalization="deterministic",
        rubric_optional_bonus_cap=0.10,
        rubric_rules=[
            {
                "rule_id": "required_main",
                "description": "required base hit",
                "type": "required",
                "weight": 1.0,
                "matcher": {"type": "phrase_aliases", "phrases": ["userspace"]},
            },
            {
                "rule_id": "optional_bonus",
                "description": "bonus mention",
                "type": "optional",
                "weight": 1.0,
                "matcher": {"type": "phrase_aliases", "phrases": ["bonus-term"]},
            },
        ],
    )

    result = _result(task, "userspace and bonus-term both present")
    scores = score_result(result, "schemas")

    assert scores.correctness_score == 1.0


def test_optional_failure_does_not_trigger_hard_gate():
    task = TaskDefinition(
        id="optional_3",
        team="general_knowledge",
        role="fast_worker",
        prompt="p",
        verification_classification="rubric_assisted",
        rubric_finalization="deterministic",
        rubric_rules=[
            {
                "rule_id": "required_main",
                "description": "required base hit",
                "type": "required",
                "weight": 1.0,
                "matcher": {"type": "phrase_aliases", "phrases": ["userspace"]},
            },
            {
                "rule_id": "optional_bonus",
                "description": "optional missing",
                "type": "optional",
                "weight": 1.0,
                "matcher": {"type": "phrase_aliases", "phrases": ["bonus-term"]},
            },
        ],
    )

    result = _result(task, "userspace only")
    scores = score_result(result, "schemas")

    assert scores.score_status == "final"
    assert scores.ranking_eligible is True
    assert result.hard_fail is False
