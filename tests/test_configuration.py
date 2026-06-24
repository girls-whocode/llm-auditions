"""Tests for configuration loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_auditions.configuration import Configuration

PROJECT_ROOT = Path(__file__).parent.parent


def _config() -> Configuration:
    return Configuration(project_root=PROJECT_ROOT)


class TestConfigurationLoading:
    def test_load_succeeds(self):
        cfg = _config()
        cfg.load()
        assert cfg._loaded

    def test_models_loaded(self):
        cfg = _config()
        cfg.load()
        models = cfg.get_configured_models()
        assert len(models) == 10

    def test_exact_model_names(self):
        cfg = _config()
        cfg.load()
        names = cfg.get_all_configured_model_names()
        expected = [
            "qwen3-coder:30b",
            "phi4-reasoning:14b",
            "gemma4:26b",
            "gemma4:12b",
            "qwen3.5:9b",
            "qwen2.5-coder:14b",
            "qwen2.5:14b-instruct",
            "codestral:22b",
            "qwen2.5-coder:7b",
            "phi4:14b",
        ]
        for name in expected:
            assert name in names, f"Missing model: {name}"

    def test_teams_loaded(self):
        cfg = _config()
        cfg.load()
        teams = cfg.list_teams()
        expected_teams = [
            "baseline",
            "general_knowledge",
            "research",
            "linux_infrastructure",
            "engineering_hardware",
            "development",
            "mathematics",
            "security",
            "architecture",
            "document_analysis",
            "integration_review",
            "language_knowledge",
        ]
        for t in expected_teams:
            assert t in teams, f"Missing team: {t}"

    def test_default_temperature(self):
        cfg = _config()
        cfg.load()
        assert cfg.default_temperature == 0.0

    def test_default_keep_alive(self):
        cfg = _config()
        cfg.load()
        assert cfg.default_keep_alive == 0

    def test_role_candidates_populated(self):
        cfg = _config()
        cfg.load()
        candidates = cfg.get_role_candidates("general_knowledge", "fast_worker")
        assert len(candidates) > 0
        assert "gemma4:12b" in candidates

    def test_validate_no_live_models(self):
        """Validation without live model check should only flag config errors."""
        cfg = _config()
        cfg.load()
        errors = cfg.validate(live_models=None)
        # Should be no config-only errors
        assert errors == [], f"Unexpected errors: {errors}"

    def test_validate_missing_model(self):
        """Validation with a partial live inventory should flag missing models."""
        cfg = _config()
        cfg.load()
        errors = cfg.validate(live_models=["gemma4:12b"])  # incomplete
        # Should flag all other configured models as missing
        assert len(errors) > 0

    def test_ollama_url_from_env(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_URL", "http://custom-host:11434")
        cfg = _config()
        cfg.load()
        assert cfg.ollama_base_url == "http://custom-host:11434"
