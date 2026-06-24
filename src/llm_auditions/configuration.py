"""Configuration loading and validation."""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Any

import yaml

from .models import ModelInfo, TaskDefinition

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent  # llm-auditions/


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return yaml.safe_load(f) or {}


class Configuration:
    """Loads and validates the full project configuration."""

    def __init__(self, project_root: Path = PROJECT_ROOT) -> None:
        self.root = project_root
        self.config_dir = project_root / "config"
        self._models_config: list[dict[str, Any]] = []
        self._defaults: dict[str, Any] = {}
        self._teams: dict[str, dict[str, Any]] = {}
        self._profiles: dict[str, dict[str, Any]] = {}
        self._loaded = False

    def load(self) -> None:
        """Load all configuration files."""
        self._defaults = _load_yaml(self.config_dir / "defaults.yaml")
        raw = _load_yaml(self.config_dir / "models.yaml")
        self._models_config = raw.get("models", [])

        # Load team configs
        teams_dir = self.config_dir / "teams"
        for team_file in sorted(teams_dir.glob("*.yaml")):
            data = _load_yaml(team_file)
            team_name = data.get("team", team_file.stem)
            self._teams[team_name] = data

        # Load profile configs
        profiles_dir = self.config_dir / "profiles"
        for profile_file in sorted(profiles_dir.glob("*.yaml")):
            data = _load_yaml(profile_file)
            profile_name = data.get("profile", profile_file.stem)
            self._profiles[profile_name] = data

        self._loaded = True
        logger.debug("Configuration loaded from %s", self.root)

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def ollama_base_url(self) -> str:
        """OLLAMA_URL environment variable overrides config."""
        return os.environ.get(
            "OLLAMA_URL",
            self._defaults.get("engine", {}).get("ollama_base_url", "http://localhost:11434"),
        )

    @property
    def default_temperature(self) -> float:
        self._ensure_loaded()
        return float(self._defaults.get("defaults", {}).get("temperature", 0.0))

    @property
    def default_num_ctx(self) -> int:
        self._ensure_loaded()
        return int(self._defaults.get("defaults", {}).get("num_ctx", 8192))

    @property
    def default_num_predict(self) -> int:
        self._ensure_loaded()
        return int(self._defaults.get("defaults", {}).get("num_predict", 4096))

    @property
    def default_keep_alive(self) -> int:
        self._ensure_loaded()
        return int(self._defaults.get("engine", {}).get("keep_alive", 0))

    @property
    def concurrency(self) -> int:
        self._ensure_loaded()
        return int(self._defaults.get("engine", {}).get("concurrency", 1))

    @property
    def request_timeout(self) -> float | None:
        self._ensure_loaded()
        val = self._defaults.get("engine", {}).get("request_timeout_seconds")
        if val is None:
            return None
        return float(val)

    @property
    def results_dir(self) -> Path:
        self._ensure_loaded()
        rdir = self._defaults.get("output", {}).get("results_dir", "results")
        return self.root / rdir

    def get_configured_models(self) -> list[ModelInfo]:
        self._ensure_loaded()
        result = []
        for m in self._models_config:
            result.append(
                ModelInfo(
                    name=m["name"],
                    id=m.get("id", ""),
                    full_digest=m.get("full_digest", m.get("id", "")),
                    size=str(m.get("size_gb", "?")),
                    modified="",
                    family=m.get("family", ""),
                    parameter_size=m.get("parameter_size", ""),
                    capabilities=m.get("capabilities", []),
                    supports_thinking=m.get("supports_thinking", False),
                )
            )
        return result

    def get_all_configured_model_names(self) -> list[str]:
        return [m.name for m in self.get_configured_models()]

    def get_team_config(self, team: str) -> dict[str, Any]:
        self._ensure_loaded()
        return self._teams.get(team, {})

    def get_profile_config(self, profile: str) -> dict[str, Any]:
        self._ensure_loaded()
        return self._profiles.get(profile, {})

    def list_teams(self) -> list[str]:
        self._ensure_loaded()
        return sorted(self._teams.keys())

    def list_profiles(self) -> list[str]:
        self._ensure_loaded()
        return sorted(self._profiles.keys())

    def get_team_roles(self, team: str) -> list[str]:
        self._ensure_loaded()
        tc = self._teams.get(team, {})
        return [r["id"] for r in tc.get("roles", [])]

    def get_role_candidates(self, team: str, role: str) -> list[str]:
        self._ensure_loaded()
        tc = self._teams.get(team, {})
        for r in tc.get("roles", []):
            if r["id"] == role:
                return r.get("candidates", [])
        return []

    def get_role_config(self, team: str, role: str) -> dict[str, Any]:
        self._ensure_loaded()
        tc = self._teams.get(team, {})
        for r in tc.get("roles", []):
            if r.get("id") == role:
                return r
        return {}

    def get_smoke_candidates(self, team: str, role: str) -> list[str]:
        role_cfg = self.get_role_config(team, role)
        if not role_cfg:
            return []
        if role_cfg.get("smoke_candidate"):
            return [str(role_cfg["smoke_candidate"])]
        if role_cfg.get("primary_candidate"):
            return [str(role_cfg["primary_candidate"])]
        if role_cfg.get("candidates"):
            return [str(role_cfg["candidates"][0])]
        return []

    def config_hashes(self) -> dict[str, str]:
        self._ensure_loaded()
        hashes: dict[str, str] = {}
        for p in sorted(self.config_dir.rglob("*.yaml")):
            hashes[str(p.relative_to(self.root))] = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
        return hashes

    def get_role_scoring_weights(self, team: str, role: str) -> dict[str, float]:
        self._ensure_loaded()
        tc = self._teams.get(team, {})
        for r in tc.get("roles", []):
            if r["id"] == role:
                return {k: float(v) for k, v in r.get("scoring_weights", {}).items()}
        return {}

    def get_audit_policy(self) -> dict[str, Any]:
        self._ensure_loaded()
        return dict(self._defaults.get("audit", {}))

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, live_models: list[str] | None = None) -> list[str]:
        """Return list of validation errors. Empty list means valid."""
        self._ensure_loaded()
        errors: list[str] = []

        # Check every configured model name exists
        configured_names = self.get_all_configured_model_names()
        if live_models is not None:
            for name in configured_names:
                if name not in live_models:
                    errors.append(f"Configured model '{name}' not found in Ollama inventory")

        # Check every team has at least one role
        for team, cfg in self._teams.items():
            roles = cfg.get("roles", [])
            if not roles:
                errors.append(f"Team '{team}' has no roles defined")
            for role in roles:
                if "id" not in role:
                    errors.append(f"Team '{team}' has a role without an id")
                if "candidates" not in role or not role["candidates"]:
                    errors.append(f"Team '{team}' role '{role.get('id', '?')}' has no candidates")
                for key in ("smoke_candidate", "primary_candidate"):
                    v = role.get(key)
                    if v is not None and not isinstance(v, str):
                        errors.append(f"Team '{team}' role '{role.get('id', '?')}' {key} must be a string")

        # Check team candidate model names exist in models.yaml
        for team, cfg in self._teams.items():
            for role in cfg.get("roles", []):
                for cand in role.get("candidates", []):
                    if cand not in configured_names:
                        errors.append(
                            f"Team '{team}' role '{role['id']}' candidate '{cand}' "
                            "not in models.yaml"
                        )

        return errors
