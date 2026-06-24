from __future__ import annotations

from llm_auditions.models import ModelResponse, ResultIdentity, TaskDefinition, TaskResult
from llm_auditions.scoring import score_result


def _identity() -> ResultIdentity:
    return ResultIdentity(
        team="baseline",
        role="baseline_all",
        task_id="task1",
        model_name="gemma4:12b",
        model_digest="digest",
        requested_think_mode="false",
        effective_think_mode="false",
        temperature=0.0,
        num_ctx=8192,
        num_predict=512,
        system_prompt_hash="a",
        user_prompt_hash="b",
    )


def test_removed_negation_fails():
    task = TaskDefinition(
        id="negation_task",
        team="baseline",
        role="baseline_all",
        prompt="p",
        verifier="structure",
        verification_classification="rubric_assisted",
        rubric_rules=[
            {
                "rule_id": "fake_rfc_claim",
                "description": "Must not claim the fake RFC exists",
                "type": "forbidden",
                "weight": 2.0,
                "phrase": "RFC 9999 exists",
                "aliases": ["RFC 9999 is real", "RFC 9999 is valid"],
            }
        ],
    )
    good = TaskResult(identity=_identity(), task=task, response=ModelResponse(model="m", requested_think_mode="false", effective_think_mode="false", content="RFC 9999 does not exist."))
    bad = TaskResult(identity=_identity(), task=task, response=ModelResponse(model="m", requested_think_mode="false", effective_think_mode="false", content="RFC 9999 exists."))

    good_scores = score_result(good, "schemas")
    bad_scores = score_result(bad, "schemas")
    assert good_scores.correctness_score == 1.0
    assert bad_scores.correctness_score == 0.0
    assert bad_scores.score_status != "final"
