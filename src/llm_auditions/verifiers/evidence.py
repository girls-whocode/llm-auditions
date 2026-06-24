"""Evidence citation verifier — checks that claims are backed by fixture evidence."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .base import BaseVerifier, VerifierResult

# Pattern for fixture evidence IDs: DOC-001, ADV-002, POLICY-003, etc.
_EVIDENCE_ID_RE = re.compile(r"\b(?:DOC|ADV|POLICY|LOG|CFG)-\d{3}\b", re.IGNORECASE)


class EvidenceVerifier(BaseVerifier):
    """
    Verifies that a model's evidence synthesis cites real fixture IDs
    and does not hallucinate unsupported claims.
    """

    name = "evidence"

    def __init__(self, valid_ids: set[str] | None = None) -> None:
        # Set of evidence IDs that actually exist in the fixture packet
        self.valid_ids: set[str] = valid_ids or set()

    def verify(self, task: Any, response: Any) -> VerifierResult:
        content = response.content if hasattr(response, "content") else str(response)

        # Derive valid IDs from the task fixture packet when not injected.
        valid_ids = set(self.valid_ids)
        if not valid_ids and task is not None and hasattr(task, "fixture_paths"):
            project_root = Path(__file__).parent.parent.parent.parent
            for fp in getattr(task, "fixture_paths", []):
                valid_ids |= load_valid_ids_from_fixture(project_root / fp)

        # Collect all IDs cited in the response
        cited: set[str] = {m.upper() for m in _EVIDENCE_ID_RE.findall(content)}
        evidence_cfg = getattr(task, "evidence", None)
        required_ids = {item.upper() for item in getattr(evidence_cfg, "required_ids", [])}
        optional_ids = {item.upper() for item in getattr(evidence_cfg, "optional_ids", [])}

        # If we have a known set of valid IDs, cross-check
        if valid_ids:
            hallucinated = cited - valid_ids
            valid_cited = cited & valid_ids
            uncited = valid_ids - cited
        else:
            hallucinated = set()
            valid_cited = cited
            uncited = set()

        if required_ids:
            missing_required = required_ids - cited
        else:
            missing_required = set()
        optional_used = cited & optional_ids

        # Score
        if valid_ids:
            required_coverage = (len(required_ids & cited) / len(required_ids)) if required_ids else (len(valid_cited) / len(valid_ids))
            optional_credit = 0.0 if not optional_ids else min(1.0, len(optional_used) / len(optional_ids)) * 0.2
            hallucination_penalty = len(hallucinated) * 0.25
            missing_required_penalty = len(missing_required) * 0.4
            score = max(0.0, min(1.0, required_coverage + optional_credit - hallucination_penalty - missing_required_penalty))
            passed = len(hallucinated) == 0 and not missing_required and required_coverage == 1.0 and bool(valid_cited or required_ids)
        else:
            # Strict mode: unknown fixture set cannot receive citation credit.
            score = 0.0
            passed = False

        details = (
            f"Cited IDs: {sorted(cited)}. "
            f"Hallucinated: {sorted(hallucinated)}. "
            f"Uncited valid: {sorted(uncited)}."
        )

        return VerifierResult(
            passed,
            score,
            details,
            {
                "has_reference_fixture_ids": bool(valid_ids),
                "cited_ids": sorted(cited),
                "hallucinated_ids": sorted(hallucinated),
                "valid_cited": sorted(valid_cited),
                "uncited_valid": sorted(uncited),
                "required_ids": sorted(required_ids),
                "optional_ids": sorted(optional_ids),
                "missing_required_ids": sorted(missing_required),
                "optional_ids_used": sorted(optional_used),
            },
        )


def load_valid_ids_from_fixture(fixture_path: Path) -> set[str]:
    """Extract all evidence IDs defined in a fixture file."""
    if not fixture_path.exists():
        return set()
    text = fixture_path.read_text()
    return {m.upper() for m in _EVIDENCE_ID_RE.findall(text)}
