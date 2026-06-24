"""Contradiction verifier — detects contradictions and conflicts in integration review outputs."""

from __future__ import annotations

import re
from typing import Any

from .base import BaseVerifier, VerifierResult


class ContradictionVerifier(BaseVerifier):
    """
    Verifies that a reviewer correctly identifies contradictions and conflicts
    in supplied multi-team integration review fixtures.
    """

    name = "contradiction"

    @staticmethod
    def _expected_contradictions(reference_facts: Any) -> list[str]:
        # Backward-compatible extraction for both legacy dict and normalized list forms.
        if isinstance(reference_facts, dict):
            values = reference_facts.get("expected_contradictions", [])
            return [str(v) for v in values if str(v).strip()]

        expected: list[str] = []
        if isinstance(reference_facts, list):
            for fact in reference_facts:
                if not isinstance(fact, dict):
                    continue
                fact_id = str(fact.get("fact_id", "")).strip().lower()
                if fact_id in {"expected_contradictions", "contradictions", "expected_conflicts"}:
                    expected.extend(str(v) for v in (fact.get("expected") or []) if str(v).strip())
        return expected

    def verify(self, task: Any, response: Any) -> VerifierResult:
        content = response.content if hasattr(response, "content") else str(response)

        # Expected contradictions from task reference_facts
        expected: list[str] = []
        if hasattr(task, "reference_facts"):
            expected = self._expected_contradictions(task.reference_facts)

        if not expected:
            return VerifierResult(
                True, 1.0,
                "No expected contradictions specified for this task"
            )

        content_lower = content.lower()
        found = []
        missed = []

        for contradiction in expected:
            # Check if key concepts from the contradiction description appear in output
            keywords = [w for w in contradiction.lower().split() if len(w) > 4]
            hits = sum(1 for kw in keywords if kw in content_lower)
            if hits >= max(1, len(keywords) // 2):
                found.append(contradiction)
            else:
                missed.append(contradiction)

        score = len(found) / len(expected) if expected else 1.0
        passed = score >= 0.6

        return VerifierResult(
            passed,
            score,
            f"Found {len(found)}/{len(expected)} expected contradictions",
            {
                "found": found,
                "missed": missed,
                "expected": expected,
            },
        )
