from __future__ import annotations

from llm_auditions.models import ModelResponse, ResultIdentity, TaskDefinition, TaskResult
from llm_auditions.scoring import score_result


def _identity() -> ResultIdentity:
    return ResultIdentity(
        team="security",
        role="worker",
        task_id="sec_1",
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


def test_empty_answer_disqualified_hard_gate():
    task = TaskDefinition(
        id="sec_hard_gate",
        team="security",
        role="worker",
        prompt="p",
        verifier="command_safety",
        hard_gate=True,
    )
    resp = ModelResponse(model="m", requested_think_mode="false", effective_think_mode="false", content="", empty_final_content=True)
    result = TaskResult(identity=_identity(), task=task, response=resp)
    scores = score_result(result, "schemas")
    assert result.hard_fail
    assert result.score_status == "disqualified"
    assert not result.ranking_eligible
    assert "empty_required_response" in result.hard_fail_reasons
