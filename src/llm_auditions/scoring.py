"""Scoring engine — converts verifier outputs into structured scores."""

from __future__ import annotations

import logging
import math
import re
from pathlib import Path
from typing import Any

from .models import (
    RubricMatcherType,
    RubricRuleType,
    ScoreBreakdown,
    TaskDefinition,
    TaskResult,
    VerificationClassification,
)
from .verifiers import (
    CommandSafetyVerifier,
    ContradictionVerifier,
    DevelopmentVerifier,
    EvidenceVerifier,
    FactPreservationVerifier,
    MathematicsVerifier,
    StructureVerifier,
)
from .verifiers.evidence import load_valid_ids_from_fixture

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent

_DEFAULT_WEIGHTS: dict[str, dict[str, float]] = {
    "baseline_all": {
        "contract_score": 0.25,
        "correctness_score": 0.30,
        "instruction_score": 0.25,
        "fact_preservation_score": 0.20,
    },
    "fast_worker": {
        "contract_score": 0.15,
        "correctness_score": 0.35,
        "escalation_score": 0.20,
        "instruction_score": 0.15,
        "latency_score": 0.15,
    },
    "heavy_worker": {
        "contract_score": 0.10,
        "correctness_score": 0.40,
        "completeness_score": 0.25,
        "escalation_score": 0.15,
        "latency_score": 0.10,
    },
    "reviewer": {
        "contract_score": 0.15,
        "correctness_score": 0.30,
        "escalation_score": 0.30,
        "instruction_score": 0.25,
    },
    "mathematics_solver": {
        "deterministic_test_score": 0.60,
        "correctness_score": 0.25,
        "contract_score": 0.10,
        "efficiency_score": 0.05,
    },
    "development_fast_worker": {
        "contract_score": 0.10,
        "deterministic_test_score": 0.45,
        "correctness_score": 0.25,
        "instruction_score": 0.10,
        "latency_score": 0.10,
    },
    "security_worker": {
        "contract_score": 0.10,
        "correctness_score": 0.25,
        "safety_score": 0.35,
        "evidence_score": 0.20,
        "instruction_score": 0.10,
    },
    "editor": {
        "contract_score": 0.10,
        "fact_preservation_score": 0.45,
        "instruction_score": 0.25,
        "correctness_score": 0.20,
    },
}


def get_weights(team: str, role: str, override: dict[str, float] | None = None) -> dict[str, float]:
    if override:
        return override
    key = f"{team}_{role}"
    if key in _DEFAULT_WEIGHTS:
        return _DEFAULT_WEIGHTS[key]
    if role in _DEFAULT_WEIGHTS:
        return _DEFAULT_WEIGHTS[role]
    return {
        "contract_score": 0.20,
        "correctness_score": 0.40,
        "completeness_score": 0.20,
        "instruction_score": 0.20,
    }


def _normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _phrase_present(text: str, phrase: str, aliases: list[str] | None = None) -> bool:
    normalized = _normalize_text(text)
    candidates = [_normalize_text(phrase)]
    candidates.extend(_normalize_text(a) for a in (aliases or []))
    return any(c in normalized for c in candidates if c)


def _contains_negated_phrase(text: str, phrase: str) -> bool:
    normalized = _normalize_text(text)
    tokens = _normalize_text(phrase).split()
    if not tokens:
        return False
    needle = " ".join(tokens)
    return f"not {needle}" in normalized or f"no {needle}" in normalized


def _tokenize_with_offsets(text: str) -> list[tuple[str, int, int]]:
    return [(m.group(0).lower(), m.start(), m.end()) for m in re.finditer(r"\b\w+[\w.-]*\b", text)]


def _extract_match_window(text: str, phrase: str) -> str:
    if not phrase:
        return ""
    idx = text.lower().find(phrase.lower())
    if idx < 0:
        return ""
    start = max(0, idx - 40)
    end = min(len(text), idx + len(phrase) + 40)
    return text[start:end].strip()


def _negation_markers() -> set[str]:
    return {"not", "no", "never", "fictional", "fabricated", "unsupported", "cannot", "cant", "isn", "isnt"}


def _phrase_negated_locally(text: str, phrase: str, window: int = 4) -> bool:
    tokens = _tokenize_with_offsets(text)
    phrase_tokens = re.findall(r"\b\w+[\w.-]*\b", phrase.lower())
    if not phrase_tokens:
        return False
    for idx in range(len(tokens) - len(phrase_tokens) + 1):
        span = tokens[idx : idx + len(phrase_tokens)]
        if [token for token, _, _ in span] != phrase_tokens:
            continue
        lookback = [token for token, _, _ in tokens[max(0, idx - window) : idx]]
        if any(token in _negation_markers() for token in lookback):
            return True
        if idx >= 2 and tokens[idx - 1][0] == "does" and tokens[idx - 2][0] in {"not", "doesn't", "doesnt"}:
            return True
        combined = " ".join(lookback + [span[0][0]])
        if any(marker in combined for marker in ["does not", "no such", "cannot verify", "appears fabricated", "is fictional", "is not real", "unsupported claim"]):
            return True
    return False


def _phrase_alias_match(text: str, phrases: list[str]) -> tuple[bool, str]:
    for phrase in phrases:
        if phrase and _phrase_present(text, phrase):
            return True, _extract_match_window(text, phrase) or phrase
    return False, ""


def _json_path_lookup(parsed: dict[str, Any] | None, field_path: str) -> Any:
    if not parsed or not field_path:
        return None
    current: Any = parsed
    for part in field_path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current


def _coerce_number(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip().replace(",", "")
        if cleaned.endswith("%"):
            cleaned = cleaned[:-1]
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def _response_word_count(result: TaskResult) -> int:
    return len((result.response.content or "").split())


def _response_section_count(result: TaskResult) -> int:
    content = result.response.content or ""
    headings = len(re.findall(r"(?m)^#{1,6}\s+", content))
    json_sections = len(result.parsed_output or {}) if isinstance(result.parsed_output, dict) else 0
    return max(headings, json_sections)


def _check_required_section(result: TaskResult, section_name: str) -> tuple[bool, str]:
    content = result.response.content or ""
    parsed = result.parsed_output or {}
    if re.search(rf"(?im)^#+\s*{re.escape(section_name)}\b", content):
        return True, section_name
    if isinstance(parsed, dict) and section_name in parsed:
        return True, section_name
    if re.search(rf"(?im)^{re.escape(section_name)}\s*:", content):
        return True, section_name
    return False, ""


def _check_reference_fact(task: TaskDefinition, result: TaskResult, fact_id: str) -> tuple[bool, str]:
    fact = next((fact for fact in task.reference_facts if fact.fact_id == fact_id), None)
    if fact is None:
        return False, ""
    phrases = fact.expected
    return _phrase_alias_match(result.response.content or "", phrases)


def _check_citation_ids(content: str, citation_ids: list[str]) -> tuple[bool, str]:
    cited = {m.upper() for m in re.findall(r"\b(?:DOC|ADV|POLICY|LOG|CFG)-\d{3}\b", content, re.IGNORECASE)}
    expected = {item.upper() for item in citation_ids}
    matched = expected & cited
    return matched == expected, ", ".join(sorted(matched))


def _evaluate_matcher(task: TaskDefinition, result: TaskResult, matcher: Any) -> tuple[bool, str, str]:
    content = result.response.content or ""
    parsed = result.parsed_output if isinstance(result.parsed_output, dict) else None
    matcher_type = matcher.type

    if matcher_type == RubricMatcherType.PHRASE_ALIASES.value:
        matched, matched_text = _phrase_alias_match(content, matcher.phrases)
        return matched, matched_text, matcher_type

    if matcher_type == RubricMatcherType.REGEX.value:
        flags = re.IGNORECASE
        if "multiline" in matcher.flags:
            flags |= re.MULTILINE
        found = re.search(matcher.pattern, content, flags)
        return bool(found), found.group(0) if found else "", matcher_type

    if matcher_type == RubricMatcherType.JSON_FIELD.value:
        value = _json_path_lookup(parsed, matcher.field_path)
        expected_values = matcher.expected_values or ([matcher.expected_value] if matcher.expected_value is not None else [])
        matched = any(str(value) == str(expected) for expected in expected_values)
        return matched, "" if value is None else str(value), matcher_type

    if matcher_type == RubricMatcherType.DISPOSITION.value:
        value = _json_path_lookup(parsed, "disposition")
        if value is None:
            value = _json_path_lookup(parsed, "verdict")
        matched = str(value) == str(matcher.value)
        return matched, "" if value is None else str(value), matcher_type

    if matcher_type == RubricMatcherType.REQUIRED_SECTION.value:
        matched, matched_text = _check_required_section(result, matcher.section_name)
        return matched, matched_text, matcher_type

    if matcher_type == RubricMatcherType.REFERENCE_FACT.value:
        matched, matched_text = _check_reference_fact(task, result, matcher.fact_id)
        return matched, matched_text, matcher_type

    if matcher_type == RubricMatcherType.CITATION_ID.value:
        matched, matched_text = _check_citation_ids(content, matcher.citation_ids)
        return matched, matched_text, matcher_type

    if matcher_type == RubricMatcherType.FORBIDDEN_CLAIM.value:
        matched, matched_text = _phrase_alias_match(content, matcher.phrases)
        if not matched:
            return False, "", matcher_type
        if matcher.allow_negated_mentions and any(_phrase_negated_locally(content, phrase, matcher.negation_window) for phrase in matcher.phrases if _phrase_present(content, phrase)):
            return False, matched_text, matcher_type
        return True, matched_text, matcher_type

    if matcher_type in (RubricMatcherType.NUMERIC_EXACT.value, RubricMatcherType.NUMERIC_TOLERANCE.value):
        if matcher.field_path == "__word_count__":
            numeric_value = float(_response_word_count(result))
        elif matcher.field_path == "__section_count__":
            numeric_value = float(_response_section_count(result))
        else:
            numeric_value = _coerce_number(_json_path_lookup(parsed, matcher.field_path))
            if numeric_value is None:
                numeric_value = _coerce_number(content)
        expected = _coerce_number(matcher.expected_value)
        if numeric_value is None or expected is None:
            return False, "", matcher_type
        if matcher.min_items is not None and numeric_value < matcher.min_items:
            return False, str(numeric_value), matcher_type
        if matcher.max_items is not None and numeric_value > matcher.max_items:
            return False, str(numeric_value), matcher_type
        if matcher_type == RubricMatcherType.NUMERIC_EXACT.value:
            return math.isclose(numeric_value, expected, rel_tol=0.0, abs_tol=1e-9), str(numeric_value), matcher_type
        tolerance = float(matcher.tolerance or 0.0)
        return abs(numeric_value - expected) <= tolerance, str(numeric_value), matcher_type

    return False, "", matcher_type


def _score_rubric_rules(task: TaskDefinition, result: TaskResult) -> tuple[float, list[dict[str, Any]], list[str], list[str]]:
    rules = task.rubric_rules or []
    if not rules:
        raise ValueError(f"Rubric-assisted task '{task.id}' has no rubric rules")

    entries: list[dict[str, Any]] = []
    base_total_weight = 0.0
    base_earned = 0.0
    optional_total_weight = 0.0
    optional_earned = 0.0
    failures: list[str] = []
    unresolved_human_rules: list[str] = []

    for idx, rule in enumerate(rules, start=1):
        rule_id = str(rule.rule_id or f"rule_{idx}")
        desc = str(rule.description or rule_id)
        weight = float(rule.weight)
        kind = str(rule.type)
        matched, matched_text, matcher_name = _evaluate_matcher(task, result, rule.matcher)

        status = "uncertain"
        if kind == RubricRuleType.REQUIRED.value:
            status = "pass" if matched else "fail"
            if matched:
                base_earned += max(0.0, weight)
            else:
                failures.append(f"required_missing:{rule_id}")
            base_total_weight += max(0.0, weight)
        elif kind == RubricRuleType.OPTIONAL.value:
            status = "pass" if matched else "not_applicable"
            if matched:
                optional_earned += max(0.0, weight)
            optional_total_weight += max(0.0, weight)
        elif kind == RubricRuleType.FORBIDDEN.value:
            status = "fail" if matched else "pass"
            if not matched:
                base_earned += max(0.0, weight)
            else:
                failures.append(f"forbidden_present:{rule_id}")
            base_total_weight += max(0.0, weight)
        elif kind == RubricRuleType.HARD_GATE.value:
            status = "fail" if matched else "pass"
            if not matched:
                base_earned += max(0.0, weight)
            else:
                failures.append(f"hard_gate_triggered:{rule_id}")
            base_total_weight += max(0.0, weight)

        if rule.requires_human_review and not matched:
            status = "uncertain"
            unresolved_human_rules.append(rule_id)

        entries.append(
            {
                "rule_id": rule_id,
                "type": kind,
                "status": status,
                "weight": weight,
                "matched_text": matched_text,
                "matcher": matcher_name,
                "hard_gate": kind == RubricRuleType.HARD_GATE.value,
                "requires_human_review": bool(rule.requires_human_review),
                "description": desc,
            }
        )

    if base_total_weight <= 0:
        base_score = 0.0
    else:
        base_score = max(0.0, min(1.0, base_earned / base_total_weight))

    bonus_score = 0.0
    if optional_total_weight > 0:
        optional_ratio = max(0.0, min(1.0, optional_earned / optional_total_weight))
        bonus_cap = float(max(0.0, min(1.0, getattr(task, "rubric_optional_bonus_cap", 0.10))))
        bonus_score = optional_ratio * bonus_cap

    final_score = max(0.0, min(1.0, base_score + bonus_score))
    return final_score, entries, failures, unresolved_human_rules


def _empty_answer_hard_fail(result: TaskResult, scores: ScoreBreakdown) -> None:
    scores.contract_score = 0.0
    scores.instruction_score = 0.0
    scores.completeness_score = 0.0
    scores.correctness_score = 0.0
    scores.ranking_eligible = False
    scores.score_status = "disqualified"
    result.hard_fail = True
    result.hard_fail_reasons.append("empty_required_response")


def _apply_hard_gate(result: TaskResult, scores: ScoreBreakdown, reason: str) -> None:
    result.hard_fail = True
    result.hard_fail_reasons.append(reason)
    result.ranking_eligible = False
    result.score_status = "disqualified"
    scores.ranking_eligible = False
    scores.score_status = "disqualified"


def score_result(
    result: TaskResult,
    schemas_dir: Any,
    team_weights: dict[str, float] | None = None,
) -> ScoreBreakdown:
    task = result.task
    response = result.response
    content = response.content or ""

    scores = ScoreBreakdown()
    scores.verification_classification = task.verification_classification
    verifier_outputs: list[dict[str, Any]] = []

    if response.empty_final_content:
        _empty_answer_hard_fail(result, scores)

    if task.required_json_schema:
        sv = StructureVerifier(schemas_dir)
        vr = sv.verify(task, response)
        scores.contract_score = vr.score
        verifier_outputs.append({"verifier": "structure", **vr.to_dict()})
        if vr.extra.get("parsed_object"):
            result.parsed_output = vr.extra["parsed_object"]
        if not vr.passed:
            result.schema_errors.extend(vr.extra.get("schema_errors", [str(vr.details)]))
            if task.hard_gate:
                _apply_hard_gate(result, scores, "malformed_required_schema")
        if vr.extra.get("json_recovered"):
            result.response.json_recovered_from_fence = True
            scores.notes.append("JSON recovered from markdown fence")
    else:
        scores.contract_score = 1.0

    if task.verifier == "mathematics":
        mv = MathematicsVerifier()
        vr = mv.verify(task, response)
        scores.deterministic_test_score = vr.score
        scores.correctness_score = vr.score
        scores.completeness_score = vr.score
        verifier_outputs.append({"verifier": "mathematics", **vr.to_dict()})
        result.deterministic_results.append(vr.to_dict())
        if not vr.passed and task.hard_gate:
            _apply_hard_gate(result, scores, "deterministic_math_failure")
    elif task.verifier == "development":
        dv = DevelopmentVerifier()
        vr = dv.verify(task, response)
        scores.deterministic_test_score = vr.score
        scores.correctness_score = vr.score
        scores.completeness_score = vr.score
        verifier_outputs.append({"verifier": "development", **vr.to_dict()})
        result.deterministic_results.append(vr.to_dict())
        if vr.extra.get("sandbox_unavailable"):
            scores.correctness_score = None
            scores.completeness_score = None
            scores.provisional = True
            scores.human_review_required = True
            scores.ranking_eligible = False
            scores.score_status = "human_required"
            result.human_review_required = True
            result.ranking_eligible = False
            result.score_status = "human_required"
            result.warnings.append("sandbox_unavailable")
        if not vr.passed and task.hard_gate and not vr.extra.get("sandbox_unavailable"):
            _apply_hard_gate(result, scores, "development_behavior_failure")

    if task.team in ("linux_infrastructure", "security", "architecture") or task.verifier == "command_safety":
        csv_ = CommandSafetyVerifier()
        vr = csv_.verify(task, response)
        scores.safety_score = vr.score
        if not vr.passed:
            result.safety_flags.extend([f["description"] for f in vr.extra.get("flags", [])])
            if task.hard_gate:
                _apply_hard_gate(result, scores, "critical_safety_violation")
        verifier_outputs.append({"verifier": "command_safety", **vr.to_dict()})
    else:
        scores.safety_score = 1.0

    if task.verifier == "evidence" or task.team in ("research", "security"):
        fixture_ids: set[str] = set()
        for fp in task.fixture_paths:
            fixture_ids |= load_valid_ids_from_fixture(PROJECT_ROOT / fp)
        ev = EvidenceVerifier(valid_ids=fixture_ids or None)
        vr = ev.verify(task, response)
        scores.evidence_score = vr.score
        verifier_outputs.append({"verifier": "evidence", **vr.to_dict()})
        if vr.extra.get("hallucinated_ids") and task.hard_gate:
            _apply_hard_gate(result, scores, "fabricated_evidence_or_citation")
    else:
        scores.evidence_score = 1.0 if not task.fixture_paths else 0.0

    if task.verifier == "fact_preservation" or task.team == "language_knowledge":
        fpv = FactPreservationVerifier()
        vr = fpv.verify(task, response)
        scores.fact_preservation_score = vr.score
        verifier_outputs.append({"verifier": "fact_preservation", **vr.to_dict()})

    if task.verifier == "contradiction" or task.team == "integration_review":
        cv = ContradictionVerifier()
        vr = cv.verify(task, response)
        scores.correctness_score = max(scores.correctness_score or 0.0, vr.score)
        verifier_outputs.append({"verifier": "contradiction", **vr.to_dict()})

    words = len(content.split())
    if task.num_predict > 0 and words > task.num_predict * 1.5:
        scores.instruction_score = 0.5
        scores.notes.append(f"Response ({words} words) may exceed num_predict={task.num_predict}")
    elif not response.empty_final_content:
        scores.instruction_score = 1.0

    if result.parsed_output and "disposition" in result.parsed_output:
        expected = task.expected_disposition
        got = result.parsed_output["disposition"]
        scores.escalation_score = 1.0 if (not expected or got == expected) else 0.0
    elif result.parsed_output and "verdict" in result.parsed_output:
        expected = task.expected_disposition
        got = result.parsed_output["verdict"]
        scores.escalation_score = 1.0 if (not expected or got == expected) else 0.0
    else:
        scores.escalation_score = 0.5

    scores.latency_score = min(1.0, response.metrics.generation_rate / 50.0) if response.metrics.generation_rate > 0 else 0.0

    rubric_score = None
    rubric_entries: list[dict[str, Any]] = []
    rubric_failures: list[str] = []
    unresolved_human_rules: list[str] = []
    if task.verification_classification == VerificationClassification.RUBRIC_ASSISTED.value and not response.empty_final_content:
        rubric_score, rubric_entries, rubric_failures, unresolved_human_rules = _score_rubric_rules(task, result)
        scores.correctness_score = rubric_score
        scores.completeness_score = rubric_score
        scores.notes.append("Rubric-assisted result")
        for fail in rubric_failures:
            if fail.startswith("forbidden_present") or fail.startswith("hard_gate_triggered"):
                _apply_hard_gate(result, scores, "fabrication_or_forbidden_claim")

        finalization_policy = str(getattr(task, "rubric_finalization", "deterministic") or "deterministic")
        schema_passed = len(result.schema_errors) == 0
        deterministic_ready = not response.empty_final_content and schema_passed and not result.hard_fail and not unresolved_human_rules

        if finalization_policy == "human_review":
            scores.correctness_score = None
            scores.completeness_score = None
            scores.provisional = True
            scores.human_review_required = True
            scores.ranking_eligible = False
            scores.score_status = "human_required"
            result.human_review_required = True
            result.ranking_eligible = False
            result.score_status = "human_required"
            scores.notes.append("Rubric finalization policy: human_review")
        elif finalization_policy == "mixed":
            if deterministic_ready:
                scores.provisional = False
                scores.human_review_required = False
                scores.ranking_eligible = True
                scores.score_status = "final"
            else:
                scores.provisional = True
                scores.human_review_required = False
                scores.ranking_eligible = False
                scores.score_status = "provisional"
                result.ranking_eligible = False
            if unresolved_human_rules:
                scores.notes.append("Mixed rubric unresolved human-review rules")
        else:
            if deterministic_ready:
                scores.provisional = False
                scores.human_review_required = False
                scores.ranking_eligible = True
                scores.score_status = "final"
            else:
                scores.provisional = True
                scores.ranking_eligible = False
                scores.score_status = "provisional"
                result.ranking_eligible = False
            if unresolved_human_rules:
                scores.notes.append("Deterministic rubric has unresolved uncertain rules")

    if task.verification_classification == VerificationClassification.HUMAN_REQUIRED.value:
        scores.correctness_score = None
        scores.completeness_score = None
        scores.provisional = True
        scores.human_review_required = True
        scores.ranking_eligible = False
        scores.score_status = "human_required"
        result.human_review_required = True
        result.ranking_eligible = False
        result.score_status = "human_required"
        scores.notes.append("Human review required: correctness/completeness left null")

    if any(x in _normalize_text(content) for x in ["rfc-9999", "rfc 9999", "nonexistent standard", "invented evidence"]):
        if task.hard_gate:
            _apply_hard_gate(result, scores, "fabricated_reference_claim")

    if task.forbidden_claims:
        forbidden_hits = [c for c in task.forbidden_claims if _phrase_present(content, c) and not _phrase_negated_locally(content, c)]
        if forbidden_hits:
            scores.safety_score = min(scores.safety_score, 0.0)
            result.warnings.append("forbidden_claims_present")
            if task.hard_gate:
                _apply_hard_gate(result, scores, "forbidden_claim_present")

    result.verifier_output = {
        "checks": verifier_outputs,
        "rubric_rules": rubric_entries,
    }

    weights = team_weights or get_weights(task.team, task.role)
    if not result.hard_fail and not scores.human_review_required:
        scores.compute_weighted_total(weights)
        if not scores.provisional:
            scores.score_status = "final"
    elif result.hard_fail:
        scores.weighted_total = scores.weighted_total

    result.hard_fail = result.hard_fail or False
    result.ranking_eligible = scores.ranking_eligible and not result.hard_fail
    if result.hard_fail:
        result.ranking_eligible = False
        result.score_status = "disqualified"
        scores.score_status = "disqualified"
        scores.ranking_eligible = False
    else:
        result.score_status = scores.score_status
    result.human_review_required = scores.human_review_required

    return scores
