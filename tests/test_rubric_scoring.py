from __future__ import annotations

from llm_auditions.models import ModelResponse, ResultIdentity, TaskDefinition, TaskResult
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


def test_rubric_rules_change_score():
    task = TaskDefinition(
        id="rubric_task",
        team="general_knowledge",
        role="fast_worker",
        prompt="p",
        verifier="structure",
        verification_classification="rubric_assisted",
        rubric_rules=[
            {"rule_id": "r1", "description": "must mention userspace", "phrase": "userspace", "weight": 2.0, "type": "required"},
            {"rule_id": "r2", "description": "must not claim kernel module", "phrase": "kernel module", "weight": 1.0, "type": "forbidden"},
        ],
    )
    good = TaskResult(identity=_identity(), task=task, response=ModelResponse(model="m", requested_think_mode="false", effective_think_mode="false", content="systemd is a userspace init system"))
    bad = TaskResult(identity=_identity(), task=task, response=ModelResponse(model="m", requested_think_mode="false", effective_think_mode="false", content="systemd is a kernel module"))

    sg = score_result(good, "schemas")
    sb = score_result(bad, "schemas")
    assert sg.correctness_score is not None
    assert sb.correctness_score is not None
    assert sg.correctness_score > sb.correctness_score
