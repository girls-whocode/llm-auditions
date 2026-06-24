"""Task loader — loads task definitions from YAML fixture files."""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Any

import yaml

from .comparisons import load_comparison_scenario
from .models import RubricRuleType, TaskDefinition
from .verifiers.evidence import load_valid_ids_from_fixture

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).parent.parent.parent
_EVIDENCE_ID_RE = re.compile(r"\b(?:DOC|ADV|POLICY|LOG|CFG)-\d{3}\b", re.IGNORECASE)


def _load_yaml(path: Path) -> Any:
    with path.open("r") as f:
        return yaml.safe_load(f)


def load_tasks_from_dir(tasks_dir: Path) -> list[TaskDefinition]:
    """Recursively load all task definitions from YAML files in a directory."""
    tasks: list[TaskDefinition] = []
    if not tasks_dir.exists():
        return tasks

    for path in sorted(tasks_dir.rglob("*.yaml")):
        try:
            data = _load_yaml(path)
            if isinstance(data, list):
                for item in data:
                    tasks.append(_parse_task(item, path))
            elif isinstance(data, dict):
                if "tasks" in data:
                    for item in data["tasks"]:
                        tasks.append(_parse_task(item, path))
                elif "id" in data:
                    tasks.append(_parse_task(data, path))
        except Exception as exc:
            logger.error("Failed to load task file %s: %s", path, exc)

    # Soft validation warnings; hard failures are handled by CLI audit/validate.
    dup = detect_duplicate_tasks(tasks)
    if dup["duplicate_ids"] or dup["duplicate_prompts"]:
        logger.warning(
            "Duplicate tasks detected: ids=%d prompts=%d",
            len(dup["duplicate_ids"]),
            len(dup["duplicate_prompts"]),
        )

    return tasks


def _parse_task(data: dict[str, Any], source_path: Path) -> TaskDefinition:
    """Parse a raw dict into a TaskDefinition."""
    # Ensure required fields
    if "id" not in data:
        raise ValueError(f"Task in {source_path} missing 'id' field")
    if "prompt" not in data:
        raise ValueError(f"Task '{data['id']}' in {source_path} missing 'prompt' field")
    if "team" not in data:
        data["team"] = source_path.parent.name
    if "role" not in data:
        data["role"] = source_path.stem

    item = dict(data)
    if item.get("comparison_id") and item.get("comparison_scenario_ref"):
        use_shared = bool(item.get("use_shared_scenario_rubric", True))
        role_rules = list(item.get("role_rubric_rules") or item.get("rubric_rules") or [])
        item["use_shared_scenario_rubric"] = use_shared
        item["role_rubric_rules"] = role_rules
        if use_shared:
            scenario = load_comparison_scenario(PROJECT_ROOT, str(item.get("comparison_scenario_ref")))
            payload = scenario["payload"]
            shared_rules = list(payload.get("shared_rubric_rules") or [])
            item["comparison_shared_rubric_version"] = str(payload.get("shared_rubric_version", ""))
            item["rubric_rules"] = [*shared_rules, *role_rules]
        else:
            item.setdefault("comparison_shared_rubric_version", "")
            item["rubric_rules"] = role_rules

    return TaskDefinition(**{k: v for k, v in item.items() if k in TaskDefinition.model_fields})


def filter_tasks(
    tasks: list[TaskDefinition],
    smoke_only: bool = False,
    teams: list[str] | None = None,
    roles: list[str] | None = None,
    include_edge_cases: bool = False,
) -> list[TaskDefinition]:
    """Filter tasks by profile requirements."""
    result = tasks

    if teams:
        result = [t for t in result if t.team in teams]

    if roles:
        result = [t for t in result if t.role in roles]

    if smoke_only:
        result = [t for t in result if t.smoke]

    if not include_edge_cases:
        result = [t for t in result if not t.edge_case]

    return result


def detect_duplicate_tasks(tasks: list[TaskDefinition]) -> dict[str, list[dict[str, Any]]]:
    """Return duplicate task-id and prompt collisions for auditing."""
    by_id: dict[str, list[TaskDefinition]] = {}
    by_prompt: dict[str, list[TaskDefinition]] = {}
    for t in tasks:
        by_id.setdefault(t.id, []).append(t)
        prompt_key = " ".join(t.prompt.split()).strip().lower()
        by_prompt.setdefault(prompt_key, []).append(t)

    duplicate_ids = []
    for tid, items in by_id.items():
        if len(items) > 1:
            duplicate_ids.append(
                {
                    "id": tid,
                    "occurrences": [f"{x.team}.{x.role}" for x in items],
                }
            )

    duplicate_prompts = []
    for pkey, items in by_prompt.items():
        if len(items) > 1:
            duplicate_prompts.append(
                {
                    "prompt_hash": hashlib.sha256(pkey.encode()).hexdigest()[:16],
                    "task_refs": [f"{x.id}@{x.team}.{x.role}" for x in items],
                }
            )

    return {
        "duplicate_ids": duplicate_ids,
        "duplicate_prompts": duplicate_prompts,
    }


def validate_tasks(tasks: list[TaskDefinition]) -> list[str]:
    """Return validation errors for task integrity."""
    errors: list[str] = []
    dup = detect_duplicate_tasks(tasks)
    task_index = {f"{t.id}@{t.team}.{t.role}": t for t in tasks}
    for item in dup["duplicate_ids"]:
        errors.append(f"Duplicate task id '{item['id']}' in {item['occurrences']}")
    for item in dup["duplicate_prompts"]:
        refs = item.get("task_refs", [])
        ref_tasks = [task_index.get(ref) for ref in refs if task_index.get(ref)]
        if ref_tasks and all(t.comparison_id for t in ref_tasks):
            continue
        errors.append(f"Duplicate prompt collision {item['prompt_hash']} in {item['task_refs']}")

    for t in tasks:
        if not t.prompt.strip():
            errors.append(f"Task '{t.id}' has empty prompt")
        if t.required_json_schema and not t.required_json_schema.strip():
            errors.append(f"Task '{t.id}' has blank required_json_schema")
        if t.verification_classification == "rubric_assisted" and not t.rubric_rules:
            errors.append(f"Task '{t.id}' is rubric_assisted but has no explicit rubric_rules")
        if t.verification_classification == "rubric_assisted" and t.rubric_finalization not in ("deterministic", "human_review", "mixed"):
            errors.append(f"Task '{t.id}' has invalid rubric_finalization '{t.rubric_finalization}'")
        if t.verification_classification == "deterministic" and not t.verifier:
            errors.append(f"Task '{t.id}' is deterministic but has no verifier")
        if t.comparison_id:
            if t.worker_class not in ("fast", "heavy"):
                errors.append(f"Task '{t.id}' comparison worker_class must be 'fast' or 'heavy'")
            if t.comparison_track not in ("independent", "handoff"):
                errors.append(f"Task '{t.id}' comparison_track must be 'independent' or 'handoff'")
            if not t.comparison_scenario_ref.strip():
                errors.append(f"Task '{t.id}' comparison_scenario_ref is required when comparison_id is set")

        if t.verifier == "evidence":
            fixture_ids: set[str] = set()
            for fp in t.fixture_paths:
                fixture_ids |= load_valid_ids_from_fixture(PROJECT_ROOT / fp)
            prompt_ids = {item.upper() for item in _EVIDENCE_ID_RE.findall(t.prompt)}
            configured_ids = {item.upper() for item in t.evidence.required_ids + t.evidence.optional_ids}
            unknown_configured = configured_ids - fixture_ids
            if unknown_configured:
                errors.append(f"Task '{t.id}' evidence ids not present in fixtures: {sorted(unknown_configured)}")
            prompt_only_ids = prompt_ids - fixture_ids
            if prompt_only_ids:
                errors.append(f"Task '{t.id}' prompt evidence ids do not match fixtures: {sorted(prompt_only_ids)}")

        rule_ids: set[str] = set()
        for rule in t.rubric_rules:
            if rule.rule_id in rule_ids:
                errors.append(f"Task '{t.id}' has duplicate rubric rule id '{rule.rule_id}'")
            rule_ids.add(rule.rule_id)

            if rule.type == RubricRuleType.REQUIRED.value and rule.weight == 0:
                errors.append(f"Task '{t.id}' required rubric rule '{rule.rule_id}' has zero weight")
            if rule.type == RubricRuleType.HARD_GATE.value and t.criticality == "normal":
                errors.append(f"Task '{t.id}' hard_gate rubric rule '{rule.rule_id}' requires non-normal criticality")

    return errors
