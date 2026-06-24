"""Tests for resume identity key generation and task loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_auditions.models import ResultIdentity, TaskDefinition
from llm_auditions.task_loader import load_tasks_from_dir, filter_tasks


PROJECT_ROOT = Path(__file__).parent.parent


class TestResultIdentity:
    def _make_identity(self, **overrides) -> ResultIdentity:
        defaults = dict(
            team="general_knowledge",
            role="fast_worker",
            task_id="gk_fast_capital",
            model_name="gemma4:12b",
            model_digest="4eb23ef187e2",
            requested_think_mode="false",
            effective_think_mode="false",
            temperature=0.0,
            num_ctx=8192,
            num_predict=4096,
            system_prompt_hash="abc123def456",
            user_prompt_hash="deadbeef1234",
            fixture_hashes={"fixtures/sample.txt": "abc"},
        )
        defaults.update(overrides)
        return ResultIdentity(**defaults)

    def test_key_is_deterministic(self):
        id1 = self._make_identity()
        id2 = self._make_identity()
        assert id1.key() == id2.key()

    def test_different_think_mode_different_key(self):
        id1 = self._make_identity(requested_think_mode="false", effective_think_mode="false")
        id2 = self._make_identity(requested_think_mode="low", effective_think_mode="low")
        assert id1.key() != id2.key()

    def test_different_model_different_key(self):
        id1 = self._make_identity(model_name="gemma4:12b")
        id2 = self._make_identity(model_name="gemma4:26b")
        assert id1.key() != id2.key()

    def test_different_digest_different_key(self):
        id1 = self._make_identity(model_digest="aaaa111122223333")
        id2 = self._make_identity(model_digest="bbbb444455556666")
        assert id1.key() != id2.key()

    def test_different_prompt_hash_different_key(self):
        id1 = self._make_identity(user_prompt_hash="hash1")
        id2 = self._make_identity(user_prompt_hash="hash2")
        assert id1.key() != id2.key()

    def test_key_length(self):
        identity = self._make_identity()
        assert len(identity.key()) == 24


class TestTaskLoader:
    def test_loads_tasks_from_fixtures(self):
        fixtures_dir = PROJECT_ROOT / "fixtures"
        tasks = load_tasks_from_dir(fixtures_dir)
        assert len(tasks) > 0

    def test_tasks_have_required_fields(self):
        fixtures_dir = PROJECT_ROOT / "fixtures"
        tasks = load_tasks_from_dir(fixtures_dir)
        for task in tasks:
            assert task.id, f"Task missing id"
            assert task.prompt, f"Task {task.id} missing prompt"
            assert task.team, f"Task {task.id} missing team"
            assert task.role, f"Task {task.id} missing role"

    def test_smoke_filter(self):
        fixtures_dir = PROJECT_ROOT / "fixtures"
        all_tasks = load_tasks_from_dir(fixtures_dir)
        smoke_tasks = filter_tasks(all_tasks, smoke_only=True)
        assert len(smoke_tasks) > 0
        assert all(t.smoke for t in smoke_tasks)
        assert len(smoke_tasks) < len(all_tasks)

    def test_team_filter(self):
        fixtures_dir = PROJECT_ROOT / "fixtures"
        all_tasks = load_tasks_from_dir(fixtures_dir)
        baseline_tasks = filter_tasks(all_tasks, teams=["baseline"])
        assert len(baseline_tasks) > 0
        assert all(t.team == "baseline" for t in baseline_tasks)

    def test_edge_case_exclusion(self):
        fixtures_dir = PROJECT_ROOT / "fixtures"
        all_tasks = load_tasks_from_dir(fixtures_dir)
        no_edge = filter_tasks(all_tasks, include_edge_cases=False)
        edge_only = [t for t in all_tasks if t.edge_case]
        for t in edge_only:
            assert t not in no_edge

    def test_math_tasks_loaded(self):
        fixtures_dir = PROJECT_ROOT / "fixtures"
        all_tasks = load_tasks_from_dir(fixtures_dir)
        math_tasks = [t for t in all_tasks if t.team == "mathematics"]
        assert len(math_tasks) >= 3  # At least the 3 preserved problems

    def test_preserved_math_task_ids(self):
        fixtures_dir = PROJECT_ROOT / "fixtures"
        all_tasks = load_tasks_from_dir(fixtures_dir)
        task_ids = {t.id for t in all_tasks}
        assert "math_trailing_zeros_1000_base12" in task_ids
        assert "math_pell_x2_61y2_1" in task_ids
        assert "math_eigenvalues_min_matrix_10x10" in task_ids

    def test_all_teams_have_smoke_tasks(self):
        fixtures_dir = PROJECT_ROOT / "fixtures"
        all_tasks = load_tasks_from_dir(fixtures_dir)
        smoke_by_team: set[str] = {t.team for t in all_tasks if t.smoke}
        expected_teams = {
            "baseline", "general_knowledge", "research", "linux_infrastructure",
            "engineering_hardware", "development", "mathematics", "security",
            "architecture", "document_analysis", "integration_review", "language_knowledge",
        }
        missing = expected_teams - smoke_by_team
        assert missing == set(), f"Teams missing smoke tasks: {missing}"
