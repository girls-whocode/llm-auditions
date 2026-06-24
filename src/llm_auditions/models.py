"""Pydantic data models for the audition framework."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class ThinkMode(str, Enum):
    FALSE = "false"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    TRUE = "true"


class StructuredOutputMode(str, Enum):
    NONE = "none"
    PROMPT_ONLY = "prompt_only"
    OLLAMA_JSON = "ollama_json"
    OLLAMA_SCHEMA = "ollama_schema"


class VerificationClassification(str, Enum):
    DETERMINISTIC = "deterministic"
    RUBRIC_ASSISTED = "rubric_assisted"
    HUMAN_REQUIRED = "human_required"


class RubricRuleType(str, Enum):
    REQUIRED = "required"
    FORBIDDEN = "forbidden"
    OPTIONAL = "optional"
    HARD_GATE = "hard_gate"


class RubricMatcherType(str, Enum):
    PHRASE_ALIASES = "phrase_aliases"
    REGEX = "regex"
    NUMERIC_EXACT = "numeric_exact"
    NUMERIC_TOLERANCE = "numeric_tolerance"
    JSON_FIELD = "json_field"
    REQUIRED_SECTION = "required_section"
    DISPOSITION = "disposition"
    REFERENCE_FACT = "reference_fact"
    CITATION_ID = "citation_id"
    FORBIDDEN_CLAIM = "forbidden_claim"


class WorkerDisposition(str, Enum):
    ANSWER = "ANSWER"
    ESCALATE_HEAVY = "ESCALATE_HEAVY"
    RESEARCH_REQUIRED = "RESEARCH_REQUIRED"
    SPECIALIST_REROUTE = "SPECIALIST_REROUTE"
    NEEDS_CLARIFICATION = "NEEDS_CLARIFICATION"
    CANNOT_RESOLVE = "CANNOT_RESOLVE"


class ReviewerVerdict(str, Enum):
    APPROVE = "APPROVE"
    REVISE = "REVISE"
    ESCALATE_HEAVY = "ESCALATE_HEAVY"
    RESEARCH_REQUIRED = "RESEARCH_REQUIRED"
    SPECIALIST_REROUTE = "SPECIALIST_REROUTE"
    REJECT = "REJECT"


class ConfidenceLevel(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    NONE = "NONE"


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class Completeness(str, Enum):
    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    INCOMPLETE = "INCOMPLETE"


class EvidenceConfidence(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


# ---------------------------------------------------------------------------
# Ollama model inventory
# ---------------------------------------------------------------------------


class ModelInfo(BaseModel):
    """Single model entry from ollama list."""

    name: str
    id: str
    full_digest: str = ""
    size: str
    modified: str
    family: str = ""
    parameter_size: str = ""
    quantization_level: str = ""
    capabilities: list[str] = Field(default_factory=list)
    supports_thinking: bool = False
    supports_vision: bool = False
    raw_show: dict[str, Any] = Field(default_factory=dict)


class ReferenceFactDefinition(BaseModel):
    fact_id: str
    expected: list[str] = Field(default_factory=list)
    required: bool = True


class EvidenceConfig(BaseModel):
    required_ids: list[str] = Field(default_factory=list)
    optional_ids: list[str] = Field(default_factory=list)


class RubricMatcher(BaseModel):
    type: str
    phrases: list[str] = Field(default_factory=list)
    pattern: str = ""
    flags: list[str] = Field(default_factory=list)
    field_path: str = ""
    expected_value: Any = None
    expected_values: list[Any] = Field(default_factory=list)
    section_name: str = ""
    fact_id: str = ""
    citation_ids: list[str] = Field(default_factory=list)
    value: str = ""
    tolerance: float | None = None
    unit: str = ""
    min_items: int | None = None
    max_items: int | None = None
    allow_negated_mentions: bool = False
    negation_window: int = 4

    @field_validator("type")
    @classmethod
    def _validate_type(cls, v: str) -> str:
        allowed = {m.value for m in RubricMatcherType}
        if v not in allowed:
            raise ValueError(f"Unsupported rubric matcher '{v}'. Allowed: {sorted(allowed)}")
        return v


class RubricRule(BaseModel):
    rule_id: str
    description: str
    type: str
    weight: float = 1.0
    requires_human_review: bool = False
    matcher: RubricMatcher

    @model_validator(mode="before")
    @classmethod
    def _upgrade_legacy_rule_shape(cls, value: Any) -> Any:
        if not isinstance(value, dict) or "matcher" in value:
            return value

        legacy = dict(value)
        phrase = legacy.pop("phrase", "")
        aliases = legacy.pop("aliases", [])
        pattern = legacy.pop("pattern", "")
        allow_negated_mentions = bool(legacy.pop("allow_negated_mentions", False))
        negation_window = int(legacy.pop("negation_window", 4))
        rule_type = str(legacy.get("type", RubricRuleType.REQUIRED.value))

        if pattern:
            matcher = {
                "type": RubricMatcherType.REGEX.value,
                "pattern": pattern,
                "allow_negated_mentions": allow_negated_mentions,
                "negation_window": negation_window,
            }
        elif rule_type in (RubricRuleType.FORBIDDEN.value, RubricRuleType.HARD_GATE.value):
            matcher = {
                "type": RubricMatcherType.FORBIDDEN_CLAIM.value,
                "phrases": [str(phrase)] + [str(alias) for alias in aliases],
                "allow_negated_mentions": allow_negated_mentions,
                "negation_window": negation_window,
            }
        else:
            matcher = {
                "type": RubricMatcherType.PHRASE_ALIASES.value,
                "phrases": [str(phrase)] + [str(alias) for alias in aliases],
                "allow_negated_mentions": allow_negated_mentions,
                "negation_window": negation_window,
            }

        legacy["matcher"] = matcher
        return legacy

    @field_validator("type")
    @classmethod
    def _validate_type(cls, v: str) -> str:
        allowed = {m.value for m in RubricRuleType}
        if v not in allowed:
            raise ValueError(f"Unsupported rubric rule type '{v}'. Allowed: {sorted(allowed)}")
        return v

    @field_validator("weight")
    @classmethod
    def _validate_weight(cls, v: float) -> float:
        if v < 0:
            raise ValueError("Rubric rule weight must be non-negative")
        return v


# ---------------------------------------------------------------------------
# Task definition
# ---------------------------------------------------------------------------


class TaskDefinition(BaseModel):
    """A single audition task."""

    id: str
    team: str
    role: str
    category: str = ""
    prompt: str
    system_prompt: str = ""
    expected_disposition: str = ""
    required_sections: list[str] = Field(default_factory=list)
    required_json_schema: str = ""
    required_concepts: list[str] = Field(default_factory=list)
    forbidden_claims: list[str] = Field(default_factory=list)
    reference_facts: list[ReferenceFactDefinition] = Field(default_factory=list)
    fixture_paths: list[str] = Field(default_factory=list)
    evidence: EvidenceConfig = Field(default_factory=EvidenceConfig)
    verifier: str = ""
    timeout: Optional[int] = None
    num_ctx: int = 8192
    num_predict: int = 4096
    temperature: float = 0.0
    think_modes: list[str] = Field(default_factory=lambda: ["false"])
    weight: float = 1.0
    smoke: bool = False
    edge_case: bool = False
    description: str = ""
    task_version: str = "v1"
    structured_output_mode: str = StructuredOutputMode.PROMPT_ONLY.value
    required_caveats: list[str] = Field(default_factory=list)
    forbidden_recommendations: list[str] = Field(default_factory=list)
    hard_gate: bool = False
    criticality: str = "normal"
    verification_classification: str = VerificationClassification.RUBRIC_ASSISTED.value
    schema_name: str = ""
    rubric_rules: list[RubricRule] = Field(default_factory=list)
    rubric_finalization: str = ""
    rubric_optional_bonus_cap: float = 0.10
    comparison_id: str = ""
    comparison_track: str = ""
    worker_class: str = ""
    comparison_scenario_ref: str = ""
    development: dict[str, Any] = Field(default_factory=dict)
    smoke_candidates: list[str] = Field(default_factory=list)

    @field_validator("structured_output_mode")
    @classmethod
    def _validate_structured_output_mode(cls, v: str) -> str:
        allowed = {m.value for m in StructuredOutputMode}
        if v not in allowed:
            raise ValueError(f"Unsupported structured_output_mode '{v}'. Allowed: {sorted(allowed)}")
        return v

    @field_validator("rubric_finalization")
    @classmethod
    def _validate_rubric_finalization(cls, v: str) -> str:
        if v == "":
            return v
        allowed = {"deterministic", "human_review", "mixed"}
        if v not in allowed:
            raise ValueError(f"Unsupported rubric_finalization '{v}'. Allowed: {sorted(allowed)}")
        return v

    @field_validator("reference_facts", mode="before")
    @classmethod
    def _normalize_reference_facts(cls, value: Any) -> Any:
        if isinstance(value, dict):
            normalized: list[dict[str, Any]] = []
            for key, expected in value.items():
                if isinstance(expected, list):
                    expected_values = [str(item) for item in expected]
                else:
                    expected_values = [str(expected)]
                normalized.append({"fact_id": str(key), "expected": expected_values, "required": True})
            return normalized
        return value

    def prompt_hash(self) -> str:
        """Stable hash of the prompt content."""
        h = hashlib.sha256()
        h.update(self.system_prompt.encode())
        h.update(self.prompt.encode())
        return h.hexdigest()[:16]

    def fixture_hashes(self, project_root: Path) -> dict[str, str]:
        """SHA-256 hashes of all fixture files."""
        hashes: dict[str, str] = {}
        for fp in self.fixture_paths:
            p = project_root / fp
            if p.exists():
                h = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
                hashes[fp] = h
        return hashes


# ---------------------------------------------------------------------------
# Result identity (for resume matching)
# ---------------------------------------------------------------------------


class ResultIdentity(BaseModel):
    """Stable identity for deduplication and resume logic."""

    team: str
    role: str
    task_id: str
    task_version: str = "v1"
    model_name: str
    model_digest: str
    requested_think_mode: str
    effective_think_mode: str
    think_mode_accepted: bool = True
    structured_output_mode: str = StructuredOutputMode.PROMPT_ONLY.value
    schema_hash: str = ""
    temperature: float
    num_ctx: int
    num_predict: int
    system_prompt_hash: str
    user_prompt_hash: str
    fixture_hashes: dict[str, str] = Field(default_factory=dict)
    verifier_version: str = "2"
    scoring_version: str = "2"
    engine_version: str = "0.10.0"

    task_suite_version: str = "2"
    handoff_fast_identity_key: str = ""
    handoff_fast_response_hash: str = ""
    comparison_scenario_hash: str = ""

    def key(self, include_effective_mode: bool = True) -> str:
        """Deterministic string key for this identity."""
        parts = [
            self.team,
            self.role,
            self.task_id,
            self.task_version,
            self.model_name,
            self.model_digest,
            self.requested_think_mode,
            self.structured_output_mode,
            self.schema_hash,
            str(self.temperature),
            str(self.num_ctx),
            str(self.num_predict),
            self.system_prompt_hash,
            self.user_prompt_hash,
            self.task_suite_version,
            self.verifier_version,
            self.scoring_version,
            self.engine_version,
            self.handoff_fast_identity_key,
            self.handoff_fast_response_hash,
            self.comparison_scenario_hash,
        ]
        if include_effective_mode:
            parts.append(self.effective_think_mode)
        for k in sorted(self.fixture_hashes):
            parts.append(f"{k}:{self.fixture_hashes[k]}")
        combined = "|".join(parts)
        return hashlib.sha256(combined.encode()).hexdigest()[:24]

    def pre_request_key(self) -> str:
        """Identity key used for skip checks before inference call."""
        return self.key(include_effective_mode=False)


# ---------------------------------------------------------------------------
# Ollama raw timing metrics
# ---------------------------------------------------------------------------


class OllamaMetrics(BaseModel):
    """Timing and token metrics from the Ollama API response."""

    prompt_eval_count: int = 0
    prompt_eval_duration_ns: int = 0
    eval_count: int = 0
    eval_duration_ns: int = 0
    load_duration_ns: int = 0
    total_duration_ns: int = 0
    wall_clock_seconds_local: float = 0.0
    done: bool = False
    done_reason: str = ""

    @property
    def prompt_eval_seconds(self) -> float:
        return self.prompt_eval_duration_ns / 1e9 if self.prompt_eval_duration_ns > 0 else 0.0

    @property
    def generation_seconds(self) -> float:
        return self.eval_duration_ns / 1e9 if self.eval_duration_ns > 0 else 0.0

    @property
    def ollama_total_seconds(self) -> float:
        return self.total_duration_ns / 1e9 if self.total_duration_ns > 0 else 0.0

    @property
    def prompt_eval_rate(self) -> float:
        """Tokens per second for prompt processing."""
        if self.prompt_eval_duration_ns > 0:
            return self.prompt_eval_count / (self.prompt_eval_duration_ns / 1e9)
        return 0.0

    @property
    def generation_rate(self) -> float:
        """Tokens per second for generation."""
        if self.eval_duration_ns > 0:
            return self.eval_count / (self.eval_duration_ns / 1e9)
        return 0.0

    @property
    def wall_clock_seconds(self) -> float:
        if self.wall_clock_seconds_local > 0:
            return self.wall_clock_seconds_local
        return self.ollama_total_seconds

    @property
    def load_seconds(self) -> float:
        return self.load_duration_ns / 1e9 if self.load_duration_ns > 0 else 0.0

    @property
    def overhead_seconds(self) -> float:
        return max(0.0, self.wall_clock_seconds - self.ollama_total_seconds)

    @property
    def net_generation_seconds(self) -> float:
        """Backward-compatible alias for generation duration."""
        return self.generation_seconds


# ---------------------------------------------------------------------------
# Model response
# ---------------------------------------------------------------------------


class ModelResponse(BaseModel):
    """Full response from a single model call."""

    model: str
    requested_think_mode: str
    effective_think_mode: str
    think_mode_accepted: bool = True
    raw_json: dict[str, Any] = Field(default_factory=dict)
    request_payload: dict[str, Any] = Field(default_factory=dict)
    effective_prompt: dict[str, Any] = Field(default_factory=dict)
    content: str = ""
    thinking: str = ""
    has_thinking_content: bool = False
    truncated_length_stop: bool = False
    empty_final_content: bool = False
    metrics: OllamaMetrics = Field(default_factory=OllamaMetrics)
    error: Optional[str] = None
    json_recovered_from_fence: bool = False

    # Ollama ps snapshots
    ollama_ps_before: str = ""
    ollama_ps_after: str = ""


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


class ScoreBreakdown(BaseModel):
    """Per-dimension scores for a single task result."""

    contract_score: float = 0.0        # 0–1: JSON schema compliance
    correctness_score: Optional[float] = None     # 0–1 or null when human review required
    completeness_score: Optional[float] = None    # 0–1 or null when human review required
    safety_score: float = 1.0          # 0–1: no unsafe commands/claims (default safe)
    evidence_score: float = 0.0        # 0–1: evidence citation quality
    instruction_score: float = 0.0     # 0–1: follows instructions (format, length, schema)
    escalation_score: float = 0.0      # 0–1: correct escalation/reroute decisions
    fact_preservation_score: float = 1.0  # 0–1: preserves facts during editing
    deterministic_test_score: float = 0.0  # 0–1: passes automated verifiers
    latency_score: float = 0.0         # 0–1: relative speed (normalised)
    efficiency_score: float = 0.0      # 0–1: tokens per correct output

    weighted_total: float = 0.0
    weights_used: dict[str, float] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)
    provisional: bool = False
    human_review_required: bool = False
    verification_classification: str = VerificationClassification.RUBRIC_ASSISTED.value
    ranking_eligible: bool = True
    score_status: str = "provisional"  # final | provisional | human_required | disqualified

    def compute_weighted_total(self, weights: dict[str, float]) -> float:
        total_weight = sum(weights.values())
        if total_weight == 0:
            return 0.0
        score = 0.0
        for dim, w in weights.items():
            raw = getattr(self, dim, 0.0)
            val = 0.0 if raw is None else raw
            score += val * (w / total_weight)
        self.weighted_total = round(score, 4)
        self.weights_used = weights
        return self.weighted_total


# ---------------------------------------------------------------------------
# Task result
# ---------------------------------------------------------------------------


class TaskResult(BaseModel):
    """Complete result for one task/model/think-mode combination."""

    identity: ResultIdentity
    task: TaskDefinition
    response: ModelResponse
    parsed_output: Optional[dict[str, Any]] = None
    schema_errors: list[str] = Field(default_factory=list)
    verifier_output: dict[str, Any] = Field(default_factory=dict)
    deterministic_results: list[dict[str, Any]] = Field(default_factory=list)
    safety_flags: list[str] = Field(default_factory=list)
    scores: ScoreBreakdown = Field(default_factory=ScoreBreakdown)
    run_order: int = 0
    completed_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    status: str = "completed"  # completed | error | skipped | unsupported_mode
    warnings: list[str] = Field(default_factory=list)
    hard_fail: bool = False
    hard_fail_reasons: list[str] = Field(default_factory=list)
    ranking_eligible: bool = True
    score_status: str = "provisional"
    human_review_required: bool = False


class PlannedRequest(BaseModel):
    team: str
    role: str
    task_id: str
    task_version: str
    model: str
    full_model_digest: str = ""
    requested_think_mode: str
    structured_output_mode: str
    temperature: float
    num_ctx: int
    num_predict: int
    schema_hash: str = ""
    fixture_hashes: dict[str, str] = Field(default_factory=dict)


class TaskSnapshot(BaseModel):
    team: str
    role: str
    task_id: str
    task_version: str
    task_hash: str
    schema_name: str = ""
    schema_hash: str = ""
    schema_content: dict[str, Any] = Field(default_factory=dict)
    fixture_hashes: dict[str, str] = Field(default_factory=dict)
    task: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Run manifest
# ---------------------------------------------------------------------------


class EnvironmentInfo(BaseModel):
    """Non-secret operational environment snapshot."""

    date_iso: str = ""
    timezone: str = ""
    hostname: str = ""
    os_release: str = ""
    kernel: str = ""
    cpu_model: str = ""
    cpu_count: int = 0
    memory_total: str = ""
    gpu_model: str = ""
    gpu_vram: str = ""
    nvidia_driver: str = ""
    cuda_version: str = ""
    ollama_version: str = ""
    python_version: str = ""
    project_version: str = ""
    git_commit: str = ""
    free_output: str = ""
    uptime_output: str = ""
    nvidia_smi_output: str = ""
    ollama_ps_output: str = ""
    df_output: str = ""


class RunManifest(BaseModel):
    """Complete manifest for a single audition run."""

    run_id: str
    created_at: str
    profile: str
    engine_version: str = "0.10.0"
    task_suite_version: str = "2"
    scoring_version: str = "2"
    verifier_version: str = "2"
    report_version: str = "2"
    ollama_version: str = "unknown"
    ollama_base_url: str = "http://localhost:11434"
    project_version: str = "1.0.0"
    git_commit: str = ""
    git_dirty: bool = False
    models: list[ModelInfo] = Field(default_factory=list)
    model_digests: dict[str, str] = Field(default_factory=dict)
    environment: EnvironmentInfo = Field(default_factory=EnvironmentInfo)
    task_count: int = 0
    candidate_count: int = 0
    model_count: int = 0
    request_count: int = 0
    teams_included: list[str] = Field(default_factory=list)
    roles_included: list[str] = Field(default_factory=list)
    models_included: list[str] = Field(default_factory=list)
    requested_think_modes: list[str] = Field(default_factory=list)
    structured_output_modes: list[str] = Field(default_factory=list)
    task_hashes: dict[str, str] = Field(default_factory=dict)
    config_hashes: dict[str, str] = Field(default_factory=dict)
    schema_hashes: dict[str, str] = Field(default_factory=dict)
    fixture_hashes: dict[str, str] = Field(default_factory=dict)
    execution_source_hashes: dict[str, str] = Field(default_factory=dict)
    task_manifest_hash: str = ""
    execution_plan_hash: str = ""


class RunState(BaseModel):
    """Mutable run status (separate from immutable run_manifest.json)."""

    run_id: str
    profile: str
    started_at: str
    updated_at: str
    status: str = "running"
    completed_identity_keys: list[str] = Field(default_factory=list)
    completed_count: int = 0
    error_count: int = 0
    unsupported_mode_count: int = 0
