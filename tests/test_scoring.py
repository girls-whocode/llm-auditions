"""Tests for scoring engine."""

from __future__ import annotations

from llm_auditions.models import ScoreBreakdown
from llm_auditions.scoring import get_weights


class TestWeightedScoring:
    def test_weights_sum_to_nonzero(self):
        weights = get_weights("mathematics", "solver")
        assert sum(weights.values()) > 0

    def test_mathematics_weights_heavy_on_deterministic(self):
        weights = get_weights("mathematics", "solver")
        assert weights.get("deterministic_test_score", 0) >= 0.5

    def test_security_weights_heavy_on_safety(self):
        weights = get_weights("security", "worker")
        assert weights.get("safety_score", 0) >= 0.25

    def test_editor_weights_heavy_on_fact_preservation(self):
        weights = get_weights("language_knowledge", "editor")
        assert weights.get("fact_preservation_score", 0) >= 0.35

    def test_weighted_total_in_range(self):
        s = ScoreBreakdown(
            correctness_score=1.0,
            deterministic_test_score=1.0,
            contract_score=1.0,
            efficiency_score=1.0,
        )
        weights = get_weights("mathematics", "solver")
        total = s.compute_weighted_total(weights)
        assert 0.0 <= total <= 1.0

    def test_zero_weights_fallback(self):
        s = ScoreBreakdown()
        total = s.compute_weighted_total({"correctness_score": 1.0})
        assert total == 0.0  # all scores default 0

    def test_safety_zero_reduces_total(self):
        s = ScoreBreakdown(
            correctness_score=1.0,
            safety_score=0.0,
        )
        weights = {"correctness_score": 0.5, "safety_score": 0.5}
        total = s.compute_weighted_total(weights)
        assert total == 0.5

    def test_fallback_weights_for_unknown_role(self):
        weights = get_weights("unknown_team", "unknown_role")
        assert "correctness_score" in weights


class TestScoreBreakdown:
    def test_default_safety_is_one(self):
        s = ScoreBreakdown()
        assert s.safety_score == 1.0

    def test_default_fact_preservation_is_one(self):
        s = ScoreBreakdown()
        assert s.fact_preservation_score == 1.0

    def test_weighted_total_stored(self):
        s = ScoreBreakdown(correctness_score=0.8)
        s.compute_weighted_total({"correctness_score": 1.0})
        assert s.weighted_total == pytest.approx(0.8, abs=0.001)


import pytest
