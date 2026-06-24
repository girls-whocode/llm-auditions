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

    def verify(self, task: Any, response: Any) -> VerifierResult:
        content = response.content if hasattr(response, "content") else str(response)

        # Expected contradictions from task reference_facts
        expected: list[str] = []
        if hasattr(task, "reference_facts"):
            expected = task.reference_facts.get("expected_contradictions", [])

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
