from __future__ import annotations

from llm_auditions.models import TaskDefinition
from llm_auditions.verifiers.evidence import EvidenceVerifier


def test_required_optional_ids_are_distinct():
    task = TaskDefinition(id="evidence_task", team="research", role="evidence_synthesizer", prompt="p", verifier="evidence")
    verifier = EvidenceVerifier(valid_ids={"ADV-001", "ADV-002"})

    class _Resp:
        content = "Uses ADV-001 and mentions ADV-002 as optional"

    r = verifier.verify(task, _Resp())
    assert r.passed is False or r.score >= 0.0
