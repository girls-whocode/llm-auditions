"""Base class for all deterministic verifiers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class VerifierResult:
    """Result from a deterministic verifier."""

    def __init__(
        self,
        passed: bool,
        score: float,
        details: str,
        extra: dict[str, Any] | None = None,
    ) -> None:
        self.passed = passed
        self.score = score  # 0.0–1.0
        self.details = details
        self.extra = extra or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "score": self.score,
            "details": self.details,
            **self.extra,
        }


class BaseVerifier(ABC):
    """Abstract base for all verifiers."""

    name: str = "base"

    @abstractmethod
    def verify(self, task: Any, response: Any) -> VerifierResult:
        """Run verification. Return VerifierResult."""
        ...
