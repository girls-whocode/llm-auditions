from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

REQUIRED_SCENARIO_FIELDS = (
    "comparison_id",
    "scenario_version",
    "title",
    "scenario",
    "constraints",
    "required_facts",
)


def _normalize_line_endings(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _canonical_payload_hash(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def load_comparison_scenario(project_root: Path, scenario_ref: str) -> dict[str, Any]:
    scenario_path = project_root / scenario_ref
    if not scenario_path.exists():
        raise FileNotFoundError(f"Comparison scenario fixture does not exist: {scenario_ref}")

    raw_text = _normalize_line_endings(scenario_path.read_text(encoding="utf-8"))
    if scenario_path.suffix.lower() in (".yaml", ".yml"):
        payload = yaml.safe_load(raw_text) or {}
    else:
        payload = json.loads(raw_text)

    if not isinstance(payload, dict):
        raise ValueError(f"Comparison scenario fixture must be an object: {scenario_ref}")

    missing = [field for field in REQUIRED_SCENARIO_FIELDS if field not in payload]
    if missing:
        raise ValueError(f"Comparison scenario fixture missing required fields {missing}: {scenario_ref}")

    scenario_content_hash = _canonical_payload_hash(payload)
    comparison_information_mode = str(payload.get("comparison_information_mode", "symmetric") or "symmetric")

    return {
        "path": str(scenario_path),
        "ref": scenario_ref,
        "payload": payload,
        "scenario_content_hash": scenario_content_hash,
        "scenario_version": str(payload.get("scenario_version", "")),
        "comparison_information_mode": comparison_information_mode,
    }


def render_shared_scenario(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"Comparison ID: {payload.get('comparison_id', '')}")
    lines.append(f"Scenario version: {payload.get('scenario_version', '')}")
    lines.append(f"Title: {str(payload.get('title', '')).strip()}")
    lines.append("")
    lines.append("Scenario:")
    lines.append(str(payload.get("scenario", "")).strip())

    constraints = payload.get("constraints") or []
    if constraints:
        lines.append("")
        lines.append("Constraints:")
        for item in constraints:
            lines.append(f"- {str(item).strip()}")

    required_facts = payload.get("required_facts") or []
    if required_facts:
        lines.append("")
        lines.append("Required facts:")
        for item in required_facts:
            lines.append(f"- {str(item).strip()}")

    reference_facts = payload.get("reference_facts") or []
    if reference_facts:
        lines.append("")
        lines.append("Reference facts:")
        for item in reference_facts:
            lines.append(f"- {str(item).strip()}")

    safety_reqs = payload.get("safety_requirements") or []
    if safety_reqs:
        lines.append("")
        lines.append("Safety requirements:")
        for item in safety_reqs:
            lines.append(f"- {str(item).strip()}")

    expected_escalation = payload.get("expected_escalation_conditions") or []
    if expected_escalation:
        lines.append("")
        lines.append("Expected escalation conditions:")
        for item in expected_escalation:
            lines.append(f"- {str(item).strip()}")

    return "\n".join(lines).strip()
