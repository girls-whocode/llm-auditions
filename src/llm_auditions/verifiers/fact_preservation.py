"""Fact preservation verifier — checks that editing does not alter approved facts."""

from __future__ import annotations

import re
from typing import Any

from .base import BaseVerifier, VerifierResult

# Categories of facts that must be preserved verbatim or semantically
_NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)*(?:\s*%|\s*GB|\s*MB|\s*TB|\s*ms|\s*s)?\b")
_DATE_RE = re.compile(
    r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}|"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4})\b",
    re.IGNORECASE,
)
_COMMAND_RE = re.compile(r"`[^`]+`|(?<=\$\s)\S+")
_PATH_RE = re.compile(r"(?:/[\w./-]+){2,}")
_CERTAINTY_RE = re.compile(
    r"\b(?:must|must not|never|always|should|should not|may|may not|"
    r"cannot|will not|is not|are not|does not|do not|would not|"
    r"no guarantee|uncertain|unclear|unknown|unconfirmed|unverified)\b",
    re.IGNORECASE,
)
_NEGATION_RE = re.compile(r"\bnot\b|\bno\b|\bnever\b|\bnon-?\w+", re.IGNORECASE)


def _extract_numbers(text: str) -> list[str]:
    return _NUMBER_RE.findall(text)


def _extract_dates(text: str) -> list[str]:
    return _DATE_RE.findall(text)


def _extract_paths(text: str) -> list[str]:
    return _PATH_RE.findall(text)


def _extract_certainty(text: str) -> list[str]:
    return _CERTAINTY_RE.findall(text)


def _extract_negations(text: str) -> list[str]:
    return _NEGATION_RE.findall(text)


class FactPreservationVerifier(BaseVerifier):
    """
    Verifies that a language edit preserves approved facts.
    Requires both original and edited text in the task reference_facts.
    """

    name = "fact_preservation"

    @staticmethod
    def _reference_fact_map(reference_facts: Any) -> dict[str, Any]:
        # Backward compatibility: support both legacy dict and normalized list forms.
        if isinstance(reference_facts, dict):
            return dict(reference_facts)
        if not isinstance(reference_facts, list):
            return {}

        mapped: dict[str, Any] = {}
        for item in reference_facts:
            if not isinstance(item, dict):
                continue
            fact_id = str(item.get("fact_id", "")).strip()
            if not fact_id:
                continue
            expected = item.get("expected", [])
            if isinstance(expected, list):
                mapped[fact_id] = [str(v) for v in expected if str(v).strip()]
            elif expected is None:
                mapped[fact_id] = []
            else:
                mapped[fact_id] = [str(expected)]
        return mapped

    @staticmethod
    def _to_expected_values(value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(v) for v in value if str(v).strip()]
        if value is None:
            return []
        return [str(value)]

    def verify(self, task: Any, response: Any) -> VerifierResult:
        content = response.content if hasattr(response, "content") else str(response)

        # Retrieve original text from task reference_facts
        original = ""
        ref_map: dict[str, Any] = {}
        if hasattr(task, "reference_facts"):
            ref_map = self._reference_fact_map(task.reference_facts)
            original_values = self._to_expected_values(ref_map.get("original_text", ""))
            original = original_values[0] if original_values else ""

        if not original:
            return VerifierResult(
                True, 1.0, "No original text provided for fact comparison"
            )

        issues: list[str] = []
        scores: list[float] = []

        # --- Numbers ---
        orig_nums = set(_extract_numbers(original))
        edit_nums = set(_extract_numbers(content))
        lost_nums = orig_nums - edit_nums
        if lost_nums:
            issues.append(f"Missing numbers: {', '.join(sorted(lost_nums)[:5])}")
            scores.append(0.0)
        else:
            scores.append(1.0)

        # --- Dates ---
        orig_dates = set(_extract_dates(original))
        edit_dates = set(_extract_dates(content))
        lost_dates = orig_dates - edit_dates
        if lost_dates:
            issues.append(f"Missing dates: {', '.join(sorted(lost_dates)[:5])}")
            scores.append(0.0)
        else:
            scores.append(1.0)

        # --- Paths ---
        orig_paths = set(_extract_paths(original))
        edit_paths = set(_extract_paths(content))
        lost_paths = orig_paths - edit_paths
        if lost_paths:
            issues.append(f"Missing paths: {', '.join(list(lost_paths)[:3])}")
            scores.append(0.5)
        else:
            scores.append(1.0)

        # --- Certainty/negation words ---
        orig_cert = set(w.lower() for w in _extract_certainty(original))
        edit_cert = set(w.lower() for w in _extract_certainty(content))
        lost_cert = orig_cert - edit_cert
        if lost_cert:
            issues.append(f"Lost certainty/qualification words: {', '.join(sorted(lost_cert)[:5])}")
            scores.append(0.3)
        else:
            scores.append(1.0)

        orig_neg = set(w.lower() for w in _extract_negations(original))
        edit_neg = set(w.lower() for w in _extract_negations(content))
        lost_neg = orig_neg - edit_neg
        if lost_neg:
            issues.append(f"Lost negation words: {', '.join(sorted(lost_neg)[:5])}")
            scores.append(0.0)
        else:
            scores.append(1.0)

        # --- Check for reference_facts specific checks ---
        if ref_map:
            for key, expected in ref_map.items():
                if key == "original_text":
                    continue
                expected_values = self._to_expected_values(expected)
                if not expected_values:
                    continue
                missing = [value for value in expected_values if value not in content]
                if missing:
                    issues.append(f"Reference fact '{key}' values not found in output: {', '.join(missing[:3])}")
                    scores.append(0.0)
                else:
                    scores.append(1.0)

        overall = sum(scores) / len(scores) if scores else 1.0
        passed = overall >= 0.8 and not any("Missing numbers" in i or "Lost certainty" in i or "Lost negation" in i for i in issues)

        return VerifierResult(
            passed,
            overall,
            "; ".join(issues) if issues else "All facts preserved",
            {"issues": issues, "dimension_scores": scores},
        )
