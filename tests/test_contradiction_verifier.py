from __future__ import annotations

from llm_auditions.models import ModelResponse, TaskDefinition
from llm_auditions.verifiers.contradiction import ContradictionVerifier


def _task_with_reference_facts(reference_facts):
    return TaskDefinition(
        id="contradiction_case",
        team="integration_review",
        role="primary",
        prompt="Identify conflicts.",
        verification_classification="deterministic",
        verifier="contradiction",
        reference_facts=reference_facts,
    )


def test_contradiction_verifier_supports_normalized_reference_facts_list():
    task = _task_with_reference_facts(
        [
            {
                "fact_id": "expected_contradictions",
                "expected": [
                    "Service A requires strict consistency while Service B is eventually consistent",
                ],
                "required": True,
            }
        ]
    )
    response = ModelResponse(
        model="m",
        requested_think_mode="false",
        effective_think_mode="false",
        content="There is a contradiction: Service A requires strict consistency while Service B is eventually consistent.",
    )

    out = ContradictionVerifier().verify(task, response)
    assert out.passed is True
    assert out.score == 1.0


def test_contradiction_verifier_supports_legacy_dict_reference_facts():
    task = _task_with_reference_facts(
        {
            "expected_contradictions": [
                "Component X requires online writes while component Y demands write freeze",
            ]
        }
    )
    response = ModelResponse(
        model="m",
        requested_think_mode="false",
        effective_think_mode="false",
        content="The output notes component X requires online writes while component Y demands write freeze.",
    )

    out = ContradictionVerifier().verify(task, response)
    assert out.passed is True
    assert out.score == 1.0
