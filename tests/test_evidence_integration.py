from __future__ import annotations

from pathlib import Path

from llm_auditions.models import TaskDefinition
from llm_auditions.verifiers.evidence import EvidenceVerifier, load_valid_ids_from_fixture


def test_fixture_backed_ids_require_real_files(tmp_path: Path):
    fixture = tmp_path / "DOC-001.md"
    fixture.write_text("ID: DOC-001\nFact: hello\n")
    valid = load_valid_ids_from_fixture(fixture)
    assert valid == {"DOC-001"}

    task = TaskDefinition(
        id="ev1",
        team="research",
        role="evidence_synthesizer",
        prompt="p",
        verifier="evidence",
        fixture_paths=[],
    )
    verifier = EvidenceVerifier(valid_ids=valid)
    class _Resp:
        content = "Uses DOC-999"
    r = verifier.verify(task, _Resp())
    assert not r.passed
    assert "DOC-999" in r.extra["hallucinated_ids"]
