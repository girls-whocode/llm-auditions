"""CLI entry point for the audition framework."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from pydantic import ValidationError

_HERE = Path(__file__).parent
_SRC = _HERE.parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from llm_auditions.configuration import Configuration, PROJECT_ROOT
from llm_auditions.comparisons import load_comparison_scenario
from llm_auditions.models import ResultIdentity, TaskDefinition
from llm_auditions.ollama_client import OllamaClient
from llm_auditions.task_loader import (
    detect_duplicate_tasks,
    filter_tasks,
    load_tasks_from_dir,
    validate_tasks,
)
from llm_auditions.versioning import (
    ENGINE_VERSION,
    REPORT_VERSION,
    SCORING_VERSION,
    TASK_SUITE_VERSION,
    VERIFIER_VERSION,
    execution_source_hashes,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("audition")

LEGACY_COMPARISON_RUBRIC_TERMS = [
    "du -ah",
    "sort -rh",
    "largest files",
    "vm.dirty_ratio",
    "vm.dirty_background_ratio",
    "setenforce 0",
    "selinux",
    "adv-001",
    "nginx 1.25.0",
    "ai systemic risk",
    "ups utilization",
    "gpu inference in 0.1 seconds",
    "database query bottleneck",
    "read replica",
]


def _normalize_think_modes(values: list[Any]) -> list[str]:
    out: list[str] = []
    for v in values:
        if isinstance(v, bool):
            out.append("true" if v else "false")
        else:
            out.append(str(v).strip().lower())
    return out


def _make_run_dir(config: Configuration, profile: str) -> Path:
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return config.results_dir / f"{profile}-{ts}"


def _load_tasks() -> list[TaskDefinition]:
    return load_tasks_from_dir(PROJECT_ROOT / "fixtures")


def _build_execution_plan(
    config: Configuration,
    profile: str,
    team_filter: str | None = None,
    role_filter: str | None = None,
    model_filter: str | None = None,
    task_filter: str | None = None,
    think_mode_filter: str | None = None,
) -> tuple[list[TaskDefinition], list[dict[str, Any]], dict[str, Any]]:
    profile_cfg = config.get_profile_config(profile)
    if not profile_cfg:
        raise ValueError(f"Unknown profile '{profile}'")

    tasks = _load_tasks()

    smoke_only = profile_cfg.get("task_filter", {}).get("smoke_only", False)
    include_edge = profile_cfg.get("task_filter", {}).get("include_edge_cases", False)
    teams = profile_cfg.get("include_teams")
    if team_filter:
        teams = [team_filter]

    tasks = filter_tasks(
        tasks,
        smoke_only=smoke_only,
        teams=teams,
        roles=[role_filter] if role_filter else None,
        include_edge_cases=include_edge,
    )
    if task_filter:
        tasks = [t for t in tasks if t.id == task_filter]

    think_modes = _normalize_think_modes(profile_cfg.get("think_modes", [False]))
    if profile == "smoke":
        think_modes = ["false"]
    if think_mode_filter is not None:
        think_modes = _normalize_think_modes([think_mode_filter])

    use_smoke_candidates = bool(profile_cfg.get("use_smoke_candidates", False))
    smoke_ceiling = int(profile_cfg.get("max_requests", 40 if profile == "smoke" else 5000))
    role_task_budget = int(profile_cfg.get("smoke_tasks_per_role", 2 if profile == "smoke" else 9999))
    configured_digests = {m.name: (m.full_digest or m.id) for m in config.get_configured_models()}

    rows: list[dict[str, Any]] = []
    handoff_groups: dict[tuple[str, str, str, str, str, str, str, str], dict[str, list[dict[str, Any]]]] = {}
    role_counts: dict[tuple[str, str], int] = {}
    base_independent_requests = 0
    valid_handoff_fast_rows = 0
    valid_handoff_dependent_heavy_rows = 0

    def _handoff_compatibility_key(row: dict[str, Any]) -> str:
        parts = [
            str(row.get("comparison_id", "")),
            str(row.get("comparison_track", "")),
            str(row.get("scenario_content_hash", "")),
            str(row.get("scenario_version", "")),
            str(row.get("comparison_information_mode", "")),
            str(row.get("requested_think_mode", "")),
            str(row.get("structured_output_mode", "")),
            str(row.get("task_suite_version", "")),
        ]
        return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]

    def _row_id(payload: dict[str, Any]) -> str:
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    for t in tasks:
        if profile == "smoke":
            mode = t.structured_output_mode
            if mode not in ("prompt_only", "ollama_json", "ollama_schema"):
                mode = "prompt_only"
        else:
            mode = t.structured_output_mode

        candidates = config.get_role_candidates(t.team, t.role)
        if use_smoke_candidates:
            rk = (t.team, t.role)
            if role_counts.get(rk, 0) >= role_task_budget:
                continue
            role_counts[rk] = role_counts.get(rk, 0) + 1
            task_level = t.smoke_candidates or []
            role_level = config.get_smoke_candidates(t.team, t.role)
            candidates = task_level or role_level or candidates[:1]

        if model_filter:
            candidates = [m for m in candidates if m == model_filter]

        schema_hash = ""
        if t.required_json_schema:
            schema_path = PROJECT_ROOT / "schemas" / f"{t.required_json_schema}.schema.json"
            if schema_path.exists():
                schema_hash = __import__("hashlib").sha256(schema_path.read_bytes()).hexdigest()[:16]

        fixture_hashes = t.fixture_hashes(PROJECT_ROOT)

        scenario_ref = str(t.comparison_scenario_ref or "")
        scenario_version = ""
        scenario_content_hash = ""
        comparison_information_mode = ""
        if t.comparison_id and scenario_ref:
            scenario_info = load_comparison_scenario(PROJECT_ROOT, scenario_ref)
            scenario_version = scenario_info["scenario_version"]
            scenario_content_hash = scenario_info["scenario_content_hash"]
            comparison_information_mode = scenario_info["comparison_information_mode"]

        base_rows: list[dict[str, Any]] = []
        for model_name in candidates:
            for think_mode in think_modes:
                row = {
                    "team": t.team,
                    "role": t.role,
                    "task_id": t.id,
                    "task_version": t.task_version,
                    "model": model_name,
                    "full_model_digest": configured_digests.get(model_name, "unknown"),
                    "requested_think_mode": think_mode,
                    "structured_output_mode": mode,
                    "temperature": t.temperature,
                    "num_ctx": t.num_ctx,
                    "num_predict": t.num_predict,
                    "schema_hash": schema_hash,
                    "fixture_hashes": fixture_hashes,
                    "comparison_id": t.comparison_id,
                    "comparison_track": t.comparison_track,
                    "worker_class": t.worker_class,
                    "comparison_scenario_ref": scenario_ref,
                    "scenario_version": scenario_version,
                    "scenario_content_hash": scenario_content_hash,
                    "comparison_information_mode": comparison_information_mode,
                    "comparison_shared_rubric_version": t.comparison_shared_rubric_version,
                    "task_suite_version": TASK_SUITE_VERSION,
                    "handoff_compatibility_key": "",
                    "fast_plan_row_id": "",
                }
                row["handoff_compatibility_key"] = _handoff_compatibility_key(row)
                row["plan_row_id"] = _row_id(row)
                base_rows.append(row)

        if t.comparison_track == "handoff" and t.comparison_id and t.worker_class in ("fast", "heavy"):
            group_key = (
                t.comparison_id,
                t.comparison_track,
                scenario_content_hash,
                scenario_version,
                comparison_information_mode,
                "",
                "",
                TASK_SUITE_VERSION,
            )
            # Think mode and output mode are row-level dimensions, so we key at row granularity below.
            handoff_groups.setdefault(group_key, {"fast": [], "heavy": []})
            handoff_groups[group_key][t.worker_class].extend(base_rows)
            continue

        rows.extend(base_rows)
        base_independent_requests += len(base_rows)

    by_full_compat: dict[tuple[str, str, str, str, str, str, str, str], dict[str, list[dict[str, Any]]]] = {}
    for (_, track, _, _, _, _, _, _), grouped in sorted(handoff_groups.items()):
        if track != "handoff":
            rows.extend(grouped.get("fast", []))
            rows.extend(grouped.get("heavy", []))
            continue

        for worker_class in ("fast", "heavy"):
            for row in grouped.get(worker_class, []):
                full_key = (
                    str(row.get("comparison_id", "")),
                    str(row.get("comparison_track", "")),
                    str(row.get("scenario_content_hash", "")),
                    str(row.get("scenario_version", "")),
                    str(row.get("comparison_information_mode", "")),
                    str(row.get("requested_think_mode", "")),
                    str(row.get("structured_output_mode", "")),
                    str(row.get("task_suite_version", TASK_SUITE_VERSION)),
                )
                by_full_compat.setdefault(full_key, {"fast": [], "heavy": []})
                by_full_compat[full_key][worker_class].append(row)

    for _, grouped in sorted(by_full_compat.items()):
        fast_rows = grouped.get("fast", [])
        heavy_rows = grouped.get("heavy", [])
        rows.extend(fast_rows)
        valid_handoff_fast_rows += len(fast_rows)
        for heavy in heavy_rows:
            for fast in fast_rows:
                dependent = dict(heavy)
                dependent["fast_plan_row_id"] = fast["plan_row_id"]
                dependent["handoff_compatibility_key"] = fast.get("handoff_compatibility_key", "")
                dependent["plan_row_id"] = _row_id(dependent)
                rows.append(dependent)
                valid_handoff_dependent_heavy_rows += 1

    metadata = {
        "profile": profile,
        "teams": sorted({r["team"] for r in rows}),
        "roles": sorted({f"{r['team']}.{r['role']}" for r in rows}),
        "tasks": sorted({(r["team"], r["role"], r["task_id"], r["task_version"]) for r in rows}),
        "models": sorted({r["model"] for r in rows}),
        "think_modes": sorted({r["requested_think_mode"] for r in rows}),
        "structured_output_modes": sorted({r["structured_output_mode"] for r in rows}),
        "request_count": len(rows),
        "base_independent_requests": base_independent_requests,
        "valid_handoff_fast_rows": valid_handoff_fast_rows,
        "valid_handoff_dependent_heavy_rows": valid_handoff_dependent_heavy_rows,
        "cross_think_handoff_dependencies": 0,
        "cross_output_handoff_dependencies": 0,
        "cross_scenario_handoff_dependencies": 0,
        "cross_information_mode_dependencies": 0,
        "task_count": len({(r["team"], r["role"], r["task_id"], r["task_version"]) for r in rows}),
        "model_count": len({r["model"] for r in rows}),
        "candidate_count": len({(r["team"], r["role"], r["model"]) for r in rows}),
        "smoke_ceiling": smoke_ceiling,
        "use_smoke_candidates": use_smoke_candidates,
    }

    if profile == "smoke" and len(rows) > smoke_ceiling:
        raise ValueError(
            f"Smoke plan too large: {len(rows)} requests exceeds ceiling {smoke_ceiling}. "
            "Tighten smoke candidates or filters."
        )

    return tasks, rows, metadata


def _task_key(t: TaskDefinition) -> tuple[str, str, str, str, str]:
    return (t.team, t.role, t.id, t.task_version, t.structured_output_mode)


# ---------------------------------------------------------------------------
# Subcommand implementations
# ---------------------------------------------------------------------------


def cmd_list_models(args: argparse.Namespace, config: Configuration) -> int:
    config.load()
    client = OllamaClient(base_url=config.ollama_base_url)
    live = client.list_models()
    configured = config.get_all_configured_model_names()

    print(f"\n{'Model':<35} {'Configured':<12} {'Live in Ollama'}")
    print("-" * 65)
    all_names = sorted(set([m.name for m in live] + configured))
    for name in all_names:
        in_config = "yes" if name in configured else "NO"
        in_live = next((m for m in live if m.name == name), None)
        live_str = f"yes (id={in_live.id}, size={in_live.size}GB)" if in_live else "NOT FOUND"
        print(f"{name:<35} {in_config:<12} {live_str}")
    return 0


def cmd_list_teams(args: argparse.Namespace, config: Configuration) -> int:
    config.load()
    for team in config.list_teams():
        cfg = config.get_team_config(team)
        desc = cfg.get("description", "")
        print(f"  {team:<30} {desc}")
    return 0


def cmd_list_roles(args: argparse.Namespace, config: Configuration) -> int:
    config.load()
    for team in config.list_teams():
        for role in config.get_team_roles(team):
            candidates = config.get_role_candidates(team, role)
            print(f"  {team:<30} {role:<25} candidates={len(candidates)}")
    return 0


def cmd_list_tasks(args: argparse.Namespace, config: Configuration) -> int:
    config.load()
    tasks = _load_tasks()
    print(f"\nTotal tasks found: {len(tasks)}")
    for t in tasks:
        if args.team and t.team != args.team:
            continue
        if args.role and t.role != args.role:
            continue
        smoke = "[smoke]" if t.smoke else ""
        edge = "[edge]" if t.edge_case else ""
        print(f"  {t.team:<28} {t.role:<25} {t.id:<40} {smoke}{edge}")
    return 0


def cmd_validate(args: argparse.Namespace, config: Configuration) -> int:
    config.load()
    client = OllamaClient(base_url=config.ollama_base_url)
    live_models = [m.name for m in client.list_models()]

    errors = config.validate(live_models=live_models)
    tasks = _load_tasks()
    errors.extend(validate_tasks(tasks))

    if errors:
        print(f"\nValidation FAILED ({len(errors)} error(s)):")
        for err in errors:
            print(f"  - {err}")
        return 1

    print(f"\nValidation passed. {len(tasks)} tasks found.")
    return 0


def cmd_plan(args: argparse.Namespace, config: Configuration) -> int:
    config.load()
    _, rows, md = _build_execution_plan(
        config=config,
        profile=args.profile,
        team_filter=getattr(args, "team", None),
        role_filter=getattr(args, "role", None),
        model_filter=getattr(args, "model", None),
        task_filter=getattr(args, "task", None),
        think_mode_filter=getattr(args, "think_mode", None),
    )

    print("Execution plan")
    print("-------------")
    print(f"profile: {md['profile']}")
    print(f"teams: {md['teams']}")
    print(f"roles: {md['roles']}")
    print(f"tasks: {len(md['tasks'])}")
    print(f"models: {md['models']}")
    print(f"think modes: {md['think_modes']}")
    print(f"structured-output modes: {md['structured_output_modes']}")
    print(f"exact request count: {md['request_count']}")
    print(f"base independent requests: {md['base_independent_requests']}")
    print(f"valid handoff fast rows: {md['valid_handoff_fast_rows']}")
    print(f"valid handoff dependent heavy rows: {md['valid_handoff_dependent_heavy_rows']}")
    print(f"cross_think_handoff_dependencies: {md['cross_think_handoff_dependencies']}")
    print(f"cross_output_handoff_dependencies: {md['cross_output_handoff_dependencies']}")
    print(f"cross_scenario_handoff_dependencies: {md['cross_scenario_handoff_dependencies']}")
    print(f"cross_information_mode_dependencies: {md['cross_information_mode_dependencies']}")
    if args.profile == "smoke":
        print(f"smoke ceiling: {md['smoke_ceiling']}")
    return 0


def _load_raw_task_entries(tasks_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    import yaml

    entries: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(tasks_dir.rglob("*.yaml")):
        data = yaml.safe_load(path.read_text()) or {}
        if isinstance(data, dict) and "tasks" in data and isinstance(data["tasks"], list):
            for item in data["tasks"]:
                if isinstance(item, dict):
                    entries.append((path, item))
        elif isinstance(data, dict) and "id" in data:
            entries.append((path, data))
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    entries.append((path, item))
    return entries


def _rubric_finalization_inventory(raw_entries: list[tuple[Path, dict[str, Any]]]) -> dict[str, dict[str, int]]:
    inventory: dict[str, dict[str, int]] = {}
    for _, item in raw_entries:
        if item.get("verification_classification") != "rubric_assisted":
            continue
        key = f"{item.get('team', '')}.{item.get('role', '')}"
        bucket = str(item.get("rubric_finalization", "missing") or "missing")
        inventory.setdefault(key, {"deterministic": 0, "mixed": 0, "human_review": 0, "missing": 0})
        if bucket not in inventory[key]:
            bucket = "missing"
        inventory[key][bucket] += 1
    return inventory


def cmd_audit_config(args: argparse.Namespace, config: Configuration) -> int:
    config.load()
    audit_policy = config.get_audit_policy()
    tasks = _load_tasks()
    task_errors = validate_tasks(tasks)
    dup = detect_duplicate_tasks(tasks)
    task_index = {f"{t.id}@{t.team}.{t.role}": t for t in tasks}

    findings: dict[str, list[str]] = {
        "blocking": [],
        "warnings": [],
    }

    if dup["duplicate_ids"]:
        findings["blocking"].append(f"duplicate task ids: {len(dup['duplicate_ids'])}")
    if dup["duplicate_prompts"]:
        non_exempt_collisions = 0
        for collision in dup["duplicate_prompts"]:
            refs = collision.get("task_refs", [])
            referenced_tasks = [task_index.get(ref) for ref in refs if task_index.get(ref)]
            if referenced_tasks and all(t.comparison_id for t in referenced_tasks):
                continue
            non_exempt_collisions += 1
        if non_exempt_collisions:
            findings["blocking"].append(f"duplicate prompts: {non_exempt_collisions}")

    raw_entries = _load_raw_task_entries(PROJECT_ROOT / "fixtures")
    known_fields = set(TaskDefinition.model_fields.keys())
    comparison_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    comparison_versions: dict[tuple[str, str], set[str]] = {}
    for path, item in raw_entries:
        unknown = sorted(set(item.keys()) - known_fields)
        if unknown:
            findings["blocking"].append(f"{path}: task '{item.get('id', '?')}' has unknown fields {unknown}")

        if audit_policy.get("require_rubric_finalization_field", False) and item.get("verification_classification") == "rubric_assisted" and "rubric_finalization" not in item:
            findings["blocking"].append(f"type=rubric_finalization_missing task={item.get('id', '?')} file={path}")

        if item.get("verification_classification") == "rubric_assisted" and item.get("rubric_finalization") == "deterministic":
            for rule in item.get("rubric_rules", []) or []:
                matcher = (rule or {}).get("matcher", {})
                matcher_type = matcher.get("type")
                if matcher_type and matcher_type not in {"phrase_aliases", "regex", "numeric_exact", "numeric_tolerance", "json_field", "required_section", "disposition", "reference_fact", "citation_id", "forbidden_claim"}:
                    findings["blocking"].append(
                        f"type=deterministic_unsupported_matcher task={item.get('id', '?')} matcher={matcher_type}"
                    )

        for fp in item.get("fixture_paths", []) or []:
            fixture_path = PROJECT_ROOT / fp
            if not fixture_path.exists():
                findings["blocking"].append(f"{path}: task '{item.get('id', '?')}' missing fixture {fp}")

        comparison_id = item.get("comparison_id")
        comparison_track = item.get("comparison_track", "")
        if comparison_id:
            scenario_ref = str(item.get("comparison_scenario_ref", "") or "")
            if not scenario_ref:
                findings["blocking"].append(
                    f"type=comparison_scenario_missing_ref comparison_id={comparison_id} track={comparison_track} task={item.get('id', '?')}"
                )
            else:
                scenario_path = PROJECT_ROOT / scenario_ref
                if not scenario_path.exists():
                    findings["blocking"].append(
                        f"type=comparison_scenario_missing_fixture comparison_id={comparison_id} track={comparison_track} task={item.get('id', '?')} ref={scenario_ref}"
                    )
                else:
                    try:
                        scenario_info = load_comparison_scenario(PROJECT_ROOT, scenario_ref)
                        scenario_payload = scenario_info["payload"]
                        shared_rules = scenario_payload.get("shared_rubric_rules") or []
                        required_facts = scenario_payload.get("required_facts") or []
                        fact_ids = {
                            str(fact.get("fact_id", ""))
                            for fact in required_facts
                            if isinstance(fact, dict) and str(fact.get("fact_id", ""))
                        }
                        shared_rule_ids = {
                            str(rule.get("rule_id", ""))
                            for rule in shared_rules
                            if isinstance(rule, dict) and str(rule.get("rule_id", ""))
                        }

                        if not shared_rules:
                            findings["blocking"].append(
                                f"type=comparison_shared_rubric_missing comparison_id={comparison_id} track={comparison_track} ref={scenario_ref}"
                            )

                        use_shared = bool(item.get("use_shared_scenario_rubric", False))
                        if not use_shared and not str(item.get("shared_rubric_disable_reason", "")).strip():
                            findings["blocking"].append(
                                f"type=comparison_shared_rubric_missing comparison_id={comparison_id} track={comparison_track} task={item.get('id', '?')}"
                            )

                        role_rules = item.get("role_rubric_rules") or []
                        role_rule_ids = {str(rule.get("rule_id", "")) for rule in role_rules if isinstance(rule, dict)}
                        overlap = sorted(x for x in role_rule_ids if x in shared_rule_ids)
                        if overlap:
                            findings["blocking"].append(
                                f"type=comparison_role_rule_conflict comparison_id={comparison_id} track={comparison_track} task={item.get('id', '?')} rule_ids={overlap}"
                            )

                        effective_rules = (shared_rules if use_shared else []) + role_rules
                        covered_fact_ids: set[str] = set()
                        for rule in effective_rules:
                            if not isinstance(rule, dict):
                                continue
                            for source_fact_id in (rule.get("source_fact_ids") or []):
                                covered_fact_ids.add(str(source_fact_id))
                        if fact_ids and not fact_ids.issubset(covered_fact_ids):
                            findings["blocking"].append(
                                f"type=comparison_required_fact_uncovered comparison_id={comparison_id} track={comparison_track} task={item.get('id', '?')}"
                            )

                        role_blob = json.dumps(role_rules, ensure_ascii=False).lower()
                        if any(term in role_blob for term in LEGACY_COMPARISON_RUBRIC_TERMS):
                            findings["blocking"].append(
                                f"type=comparison_legacy_rule_detected comparison_id={comparison_id} track={comparison_track} task={item.get('id', '?')}"
                            )

                        required_concepts_blob = " ".join(str(x) for x in (item.get("required_concepts") or [])).lower()
                        missing_alias_coverage = 0
                        for fact in required_facts:
                            if not isinstance(fact, dict):
                                continue
                            aliases = [str(x).lower() for x in (fact.get("aliases") or [])]
                            if aliases and not any(alias in required_concepts_blob for alias in aliases):
                                missing_alias_coverage += 1
                        if required_facts and missing_alias_coverage:
                            findings["blocking"].append(
                                f"type=comparison_required_fact_uncovered comparison_id={comparison_id} track={comparison_track} task={item.get('id', '?')}"
                            )

                        comparison_versions.setdefault((comparison_id, comparison_track), set()).add(
                            str(item.get("comparison_shared_rubric_version", ""))
                        )

                        if str(scenario_payload.get("comparison_id", "")) != str(comparison_id):
                            findings["blocking"].append(
                                f"type=comparison_scenario_fixture_mismatch comparison_id={comparison_id} track={comparison_track} task={item.get('id', '?')} ref={scenario_ref}"
                            )
                        if not str(scenario_payload.get("scenario", "") or "").strip() or len(str(scenario_payload.get("scenario", "") or "").split()) < 8:
                            findings["blocking"].append(
                                f"type=comparison_scenario_generic_or_empty comparison_id={comparison_id} track={comparison_track} task={item.get('id', '?')} ref={scenario_ref}"
                            )
                        if comparison_track and str(item.get("comparison_track", "")) != str(comparison_track):
                            findings["blocking"].append(
                                f"type=comparison_scenario_version_mismatch comparison_id={comparison_id} track={comparison_track} task={item.get('id', '?')} ref={scenario_ref}"
                            )
                    except Exception:
                        findings["blocking"].append(
                            f"type=comparison_scenario_fixture_invalid_json comparison_id={comparison_id} track={comparison_track} task={item.get('id', '?')} ref={scenario_ref}"
                        )
            comparison_groups.setdefault((comparison_id, comparison_track), []).append(item)

    for (comparison_id, comparison_track), items in comparison_groups.items():
        classes = {str(x.get("worker_class", "")) for x in items}
        if audit_policy.get("require_comparison_fast_heavy_pairs", True) and "fast" not in classes:
            findings["blocking"].append(f"type=comparison_missing_fast_partner comparison_id={comparison_id} track={comparison_track}")
        if audit_policy.get("require_comparison_fast_heavy_pairs", True) and "heavy" not in classes:
            findings["blocking"].append(f"type=comparison_missing_heavy_partner comparison_id={comparison_id} track={comparison_track}")
        refs = {str(x.get("comparison_scenario_ref", "") or "") for x in items}
        if len(refs) > 1:
            findings["blocking"].append(f"type=comparison_scenario_mismatch comparison_id={comparison_id} track={comparison_track}")
        if comparison_track == "handoff" and "fast" not in classes:
            findings["blocking"].append(f"type=handoff_pair_with_no_fast_candidate comparison_id={comparison_id} track={comparison_track}")
        if comparison_track == "handoff" and "heavy" not in classes:
            findings["blocking"].append(f"type=handoff_pair_with_no_heavy_candidate comparison_id={comparison_id} track={comparison_track}")
        versions = comparison_versions.get((comparison_id, comparison_track), set())
        non_empty_versions = {v for v in versions if v}
        if len(non_empty_versions) > 1:
            findings["blocking"].append(
                f"type=comparison_rubric_version_mismatch comparison_id={comparison_id} track={comparison_track}"
            )

    complex_keywords = ("architecture", "synthesis", "tradeoff", "analysis", "review", "integration", "security")
    for _, item in raw_entries:
        if item.get("verification_classification") != "rubric_assisted":
            continue
        if str(item.get("rubric_finalization", "")) != "deterministic":
            continue
        text = f"{item.get('id', '')} {item.get('description', '')} {item.get('role', '')}".lower()
        is_complex = any(keyword in text for keyword in complex_keywords)
        if is_complex and (not str(item.get("rubric_finalization_rationale", "")).strip() or not bool(item.get("deterministic_override", False))):
            findings["blocking"].append(
                f"type=complex_deterministic_without_override task={item.get('id', '?')} finalization=deterministic"
            )

    # Audit planned handoff dependencies for cross-mode / cross-output / cross-scenario drift.
    for profile in ("standard", "exhaustive"):
        try:
            _, plan_rows, _ = _build_execution_plan(config=config, profile=profile)
        except Exception:
            continue
        by_id = {row.get("plan_row_id", ""): row for row in plan_rows if row.get("plan_row_id")}
        for row in plan_rows:
            if row.get("comparison_track") != "handoff" or row.get("worker_class") != "heavy":
                continue
            fast_id = str(row.get("fast_plan_row_id", ""))
            if not fast_id:
                continue
            fast = by_id.get(fast_id)
            if not fast:
                continue
            if str(row.get("requested_think_mode", "")) != str(fast.get("requested_think_mode", "")):
                findings["blocking"].append("type=handoff_cross_think_dependency")
            if str(row.get("structured_output_mode", "")) != str(fast.get("structured_output_mode", "")):
                findings["blocking"].append("type=handoff_cross_output_dependency")
            if (
                str(row.get("scenario_content_hash", "")) != str(fast.get("scenario_content_hash", ""))
                or str(row.get("scenario_version", "")) != str(fast.get("scenario_version", ""))
            ):
                findings["blocking"].append("type=handoff_cross_scenario_dependency")
            if str(row.get("comparison_information_mode", "")) != str(fast.get("comparison_information_mode", "")):
                findings["blocking"].append("type=handoff_cross_information_mode_dependency")

    # Comparison scenarios must be present in fixture inventory derivation for task snapshots/manifests.
    for task in tasks:
        if task.comparison_id and task.comparison_scenario_ref:
            fixture_hashes = task.fixture_hashes(PROJECT_ROOT)
            if task.comparison_scenario_ref not in fixture_hashes:
                findings["blocking"].append(
                    f"type=comparison_scenario_missing_from_manifest_inventory task={task.id} ref={task.comparison_scenario_ref}"
                )

    src = PROJECT_ROOT / "src" / "llm_auditions"
    required_sources = {
        src / "runner.py",
        src / "ollama_client.py",
        src / "models.py",
        src / "scoring.py",
        src / "reporting.py",
        src / "task_loader.py",
        src / "cli.py",
        src / "configuration.py",
        src / "versioning.py",
        src / "packaging.py",
    }
    missing_source_files = [str(p.relative_to(PROJECT_ROOT)) for p in sorted(required_sources) if not p.exists()]
    if missing_source_files:
        findings["blocking"].append(
            "type=execution_source_hashing_configuration_incomplete missing=" + ",".join(missing_source_files)
        )

    # Audit invariant: optional rubric rules must not reduce base score when absent.
    try:
        from llm_auditions.models import ModelResponse, ResultIdentity, TaskResult
        from llm_auditions.scoring import score_result

        sanity_task = TaskDefinition(
            id="audit_optional_rule_sanity",
            team="general_knowledge",
            role="fast_worker",
            prompt="p",
            verification_classification="rubric_assisted",
            rubric_finalization="deterministic",
            rubric_rules=[
                {
                    "rule_id": "required",
                    "description": "required",
                    "type": "required",
                    "weight": 1.0,
                    "matcher": {"type": "phrase_aliases", "phrases": ["userspace"]},
                },
                {
                    "rule_id": "optional",
                    "description": "optional",
                    "type": "optional",
                    "weight": 1.0,
                    "matcher": {"type": "phrase_aliases", "phrases": ["bonus-term"]},
                },
            ],
        )
        sanity_result = TaskResult(
            identity=ResultIdentity(
                team="general_knowledge",
                role="fast_worker",
                task_id="audit_optional_rule_sanity",
                model_name="m",
                model_digest="d",
                requested_think_mode="false",
                effective_think_mode="false",
                temperature=0.0,
                num_ctx=8192,
                num_predict=256,
                system_prompt_hash="a",
                user_prompt_hash="b",
            ),
            task=sanity_task,
            response=ModelResponse(model="m", requested_think_mode="false", effective_think_mode="false", content="userspace"),
        )
        sanity_scores = score_result(sanity_result, PROJECT_ROOT / "schemas")
        if audit_policy.get("require_optional_rubric_bonus_semantics", True) and float(sanity_scores.correctness_score or 0.0) < 1.0:
            findings["blocking"].append("type=optional_rules_in_base_denominator")
    except Exception as exc:
        findings["warnings"].append(f"optional-denominator audit skipped: {exc}")

    if audit_policy.get("forbid_development_execution_enable", True) and os.environ.get("AUDITION_ENABLE_DEVELOPMENT_SANDBOX", "").lower() in ("1", "true", "yes"):
        findings["blocking"].append("type=development_execution_enabled_without_real_backend")

    rubric_missing = [t.id for t in tasks if t.verification_classification == "rubric_assisted" and not t.rubric_rules]
    if audit_policy.get("require_explicit_rubrics", True) and rubric_missing:
        findings["blocking"].append(f"rubric-assisted tasks missing effective rubric rules: {sorted(rubric_missing)}")

    client = OllamaClient(base_url=config.ollama_base_url)
    live_models = {m.name for m in client.list_models()}
    for team in config.list_teams():
        roles = config.get_team_roles(team)
        if not roles:
            findings["blocking"].append(f"team {team} has no roles")
        for role in roles:
            candidates = config.get_role_candidates(team, role)
            if not candidates:
                findings["blocking"].append(f"team {team} role {role} has no candidates")
            missing = [m for m in candidates if m not in live_models]
            if missing:
                findings["warnings"].append(f"team {team} role {role} candidates not installed: {missing}")

    for e in task_errors:
        if e.startswith("Duplicate prompt collision"):
            continue
        findings["blocking"].append(e)

    print("Config/fixture audit")
    print("--------------------")
    print(f"tasks_loaded: {len(tasks)}")
    print(f"duplicate_ids: {len(dup['duplicate_ids'])}")
    print(f"duplicate_prompts: {len(dup['duplicate_prompts'])}")
    print(f"blocking_findings: {len(findings['blocking'])}")
    print(f"warnings: {len(findings['warnings'])}")

    inventory = _rubric_finalization_inventory(raw_entries)
    for team_role in sorted(inventory.keys()):
        inv = inventory[team_role]
        print(
            "  rubric_finalization_inventory "
            f"{team_role} deterministic={inv.get('deterministic', 0)} "
            f"mixed={inv.get('mixed', 0)} "
            f"human_review={inv.get('human_review', 0)} "
            f"missing={inv.get('missing', 0)}"
        )

    for item in findings["blocking"][:40]:
        print(f"  BLOCK: {item}")
    for item in findings["warnings"][:20]:
        print(f"  WARN: {item}")

    return 1 if findings["blocking"] else 0


def cmd_audit_run(args: argparse.Namespace, config: Configuration) -> int:
    run_dir = Path(args.run_dir)
    manifest_path = run_dir / "run_manifest.json"
    state_path = run_dir / "run_state.json"
    events_path = run_dir / "events.jsonl"

    if not manifest_path.exists() or not events_path.exists():
        print("ERROR: run_manifest.json and events.jsonl are required")
        return 1

    manifest = json.loads(manifest_path.read_text())
    events = [json.loads(x) for x in events_path.read_text().splitlines() if x.strip()]
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    findings: list[dict[str, Any]] = []
    result_files = sorted(run_dir.rglob("*.result.json"))
    results = []
    for path in result_files:
        try:
            results.append((path, json.loads(path.read_text())))
        except Exception:
            findings.append({"severity": "error", "type": "invalid_result_artifact", "detail": str(path)})

    result_by_identity: dict[str, dict[str, Any]] = {}
    for _, result in results:
        try:
            key = ResultIdentity.model_validate(result.get("identity", {})).key()
            result_by_identity[key] = result
        except Exception:
            continue

    if not (run_dir / "task_manifest.json").exists():
        findings.append({"severity": "error", "type": "missing_artifact", "detail": "task_manifest.json missing"})
    else:
        task_manifest = json.loads((run_dir / "task_manifest.json").read_text())
        if manifest.get("execution_plan_hash"):
            current_hash = hashlib.sha256(json.dumps(task_manifest.get("requests", []), sort_keys=True).encode()).hexdigest()[:16]
            if current_hash != manifest.get("execution_plan_hash"):
                findings.append({"severity": "error", "type": "resume_version_mismatch", "detail": "execution_plan_hash mismatch"})

    expected_versions = {
        "engine_version": ENGINE_VERSION,
        "task_suite_version": TASK_SUITE_VERSION,
        "scoring_version": SCORING_VERSION,
        "verifier_version": VERIFIER_VERSION,
        "report_version": REPORT_VERSION,
    }
    for key, expected in expected_versions.items():
        current = str(manifest.get(key, ""))
        if current != expected:
            findings.append({"severity": "error", "type": "resume_version_mismatch", "detail": f"{key} stored={current} current={expected}"})

    stored_source_hashes = manifest.get("execution_source_hashes") or {}
    if stored_source_hashes:
        current_source_hashes = execution_source_hashes(PROJECT_ROOT)
        for name, stored in stored_source_hashes.items():
            current = current_source_hashes.get(name, "missing")
            if current != stored:
                findings.append(
                    {
                        "severity": "error",
                        "type": "resume_source_hash_mismatch",
                        "detail": f"{name} stored={stored} current={current}",
                    }
                )

    try:
        live = OllamaClient(base_url=config.ollama_base_url).list_models()
        live_digests = {m.name: (m.full_digest or m.id) for m in live}
        for model_name, stored_digest in (manifest.get("model_digests") or {}).items():
            current_digest = live_digests.get(model_name)
            if current_digest != stored_digest:
                findings.append(
                    {
                        "severity": "error",
                        "type": "live_model_digest_mismatch",
                        "detail": f"{model_name}: stored={stored_digest} current={current_digest or 'missing'}",
                    }
                )
    except Exception as exc:
        findings.append({"severity": "warning", "type": "live_model_digest_audit_skipped", "detail": str(exc)})

    seen = set()
    for e in events:
        key = (e.get("identity_key"), e.get("status"))
        if key in seen:
            findings.append({"severity": "error", "type": "duplicate_request", "detail": str(key)})
        seen.add(key)

        if e.get("requested_think_mode") == "false" and e.get("status") == "completed":
            if e.get("effective_think_mode") not in ("false", "") and not e.get("think_mode_accepted", True):
                findings.append({"severity": "warning", "type": "think_mode_fallback", "detail": e.get("task_id")})

        if e.get("empty_final_content"):
            findings.append({"severity": "error", "type": "empty_final_answer", "detail": e.get("task_id")})

        if "requested_think_mode" not in e and "think_mode" in e:
            findings.append({"severity": "error", "type": "missing_correctness_dimension", "detail": f"legacy think_mode field on {e.get('task_id')}"})

        if e.get("score_status") == "final" and e.get("ranking_eligible") is False:
            findings.append({"severity": "error", "type": "final_but_ineligible", "detail": e.get("task_id")})

    if manifest.get("request_count", 0):
        non_skipped_keys = {
            e.get("identity_key")
            for e in events
            if e.get("status") in ("completed", "unsupported_mode", "error") and e.get("identity_key")
        }
        if len(non_skipped_keys) > manifest.get("request_count", 0):
            findings.append({"severity": "error", "type": "manifest_inconsistency", "detail": "unique executed identities exceed manifest request_count"})

    if not result_files:
        findings.append({"severity": "error", "type": "missing_result_artifact", "detail": "no .result.json files found"})

    for path, result in results:
        identity = result.get("identity", {})
        response = result.get("response", {})
        task = result.get("task", {})
        scores = result.get("scores", {})
        verifier_output = result.get("verifier_output", {})
        prompt_components = (response.get("effective_prompt", {}) or {}).get("prompt_components", {})
        rubric_rules = [item for item in (verifier_output.get("rubric_rules") or []) if isinstance(item, dict)]

        check_entries = [item for item in (verifier_output.get("checks") or []) if isinstance(item, dict)]
        development_extras = [
            (check.get("extra") or {})
            for check in check_entries
            if check.get("verifier") == "development" and isinstance(check.get("extra"), dict)
        ]
        development_sandbox_policy_refusal = any(
            bool(extra.get("policy_refusal"))
            and (bool(extra.get("sandbox_unavailable")) or str(extra.get("reason", "")) == "sandbox_unavailable")
            for extra in development_extras
        )
        verifier_error_present = any(
            bool(check.get("error"))
            or bool((check.get("extra") or {}).get("error"))
            for check in check_entries
        )

        if "requested_think_mode" not in identity and "think_mode" in identity:
            findings.append({"severity": "error", "type": "missing_correctness_dimension", "detail": f"legacy identity think_mode in {path.name}"})

        think_false = str(identity.get("requested_think_mode", identity.get("think_mode", ""))).lower() == "false"
        if think_false and response.get("thinking"):
            findings.append({"severity": "error", "type": "thinking_under_think_false", "detail": path.name})

        if not response.get("content", ""):
            findings.append({"severity": "error", "type": "empty_final_answer", "detail": path.name})

        done_reason = ((response.get("metrics") or {}).get("done_reason") or "").lower()
        if done_reason == "length" or response.get("truncated_length_stop"):
            findings.append({"severity": "warning", "type": "length_truncation", "detail": path.name})

        if result.get("schema_errors"):
            findings.append({"severity": "error", "type": "schema_failure", "detail": path.name})

        if "request_payload" not in response:
            findings.append({"severity": "error", "type": "missing_request_payload", "detail": path.name})

        if "effective_prompt" not in response:
            findings.append({"severity": "error", "type": "missing_effective_prompt", "detail": path.name})

        if task.get("comparison_id") and task.get("comparison_scenario_ref") and prompt_components.get("scenario_content_hash"):
            try:
                scenario_info = load_comparison_scenario(PROJECT_ROOT, str(task.get("comparison_scenario_ref")))
                if scenario_info["scenario_content_hash"] != prompt_components.get("scenario_content_hash"):
                    findings.append({"severity": "error", "type": "comparison_scenario_hash_mismatch", "detail": path.name})
                shared_rule_ids = {
                    str(rule.get("rule_id", ""))
                    for rule in (scenario_info["payload"].get("shared_rubric_rules") or [])
                    if isinstance(rule, dict)
                }
                output_rule_ids = {
                    str(rule.get("rule_id", ""))
                    for rule in rubric_rules
                }
                if shared_rule_ids and not shared_rule_ids.issubset(output_rule_ids) and not development_sandbox_policy_refusal:
                    findings.append({"severity": "error", "type": "comparison_shared_rubric_missing", "detail": path.name})

                manifest_fixtures = manifest.get("fixture_hashes") or {}
                if str(task.get("comparison_scenario_ref")) not in manifest_fixtures:
                    findings.append({"severity": "error", "type": "comparison_scenario_fixture_untracked", "detail": path.name})

                stored_rubric_version = str(task.get("comparison_shared_rubric_version", ""))
                scenario_rubric_version = str(scenario_info.get("shared_rubric_version", ""))
                if stored_rubric_version and scenario_rubric_version and stored_rubric_version != scenario_rubric_version:
                    findings.append({"severity": "error", "type": "comparison_rubric_version_mismatch", "detail": path.name})
            except Exception:
                findings.append({"severity": "error", "type": "comparison_scenario_hash_mismatch", "detail": path.name})

        if task.get("verification_classification", "rubric_assisted") == "rubric_assisted" and not verifier_output.get("rubric_rules"):
            findings.append({"severity": "error", "type": "missing_rubric_output", "detail": path.name})

        if task.get("verification_classification", "rubric_assisted") == "rubric_assisted":
            rubric_finalization = task.get("rubric_finalization", "")
            statuses = {item.get("status") for item in rubric_rules}
            deterministic_complete = bool(statuses) and statuses.isdisjoint({"uncertain"})
            unresolved_required_rule = any(
                str(item.get("status", "")) == "uncertain" and str(item.get("type", "")).lower() == "required"
                for item in rubric_rules
            )
            if (
                rubric_finalization == "deterministic"
                and deterministic_complete
                and result.get("score_status") == "provisional"
                and not result.get("schema_errors")
                and not result.get("hard_fail")
                and not unresolved_required_rule
                and not development_sandbox_policy_refusal
                and not verifier_error_present
            ):
                findings.append({"severity": "error", "type": "rubric_stuck_provisional", "detail": path.name})
            if rubric_finalization == "mixed" and result.get("score_status") == "final":
                findings.append({"severity": "error", "type": "mixed_result_incorrectly_final", "detail": path.name})

        if task.get("comparison_track") == "handoff" and str(task.get("worker_class", "")).lower() == "heavy":
            handoff_payload = response.get("request_payload", {}).get("handoff_payload")
            if not handoff_payload:
                findings.append({"severity": "error", "type": "handoff_payload_missing", "detail": path.name})
            dependency_identity = identity.get("handoff_fast_identity_key", "")
            dependency_hash = identity.get("handoff_fast_response_hash", "")
            if not dependency_identity:
                findings.append({"severity": "error", "type": "handoff_dependency_missing", "detail": path.name})
            if not dependency_hash:
                findings.append({"severity": "error", "type": "handoff_fast_hash_mismatch", "detail": path.name})
            fast_result = result_by_identity.get(dependency_identity)
            if fast_result:
                fast_response = (fast_result.get("response") or {}).get("content", "")
                expected_hash = hashlib.sha256(str(fast_response).encode()).hexdigest()[:16]
                if dependency_hash and dependency_hash != expected_hash:
                    findings.append({"severity": "error", "type": "handoff_fast_hash_mismatch", "detail": path.name})
                fast_identity = fast_result.get("identity", {})
                fast_prompt_components = ((fast_result.get("response", {}) or {}).get("effective_prompt", {}) or {}).get("prompt_components", {})
                if str(identity.get("requested_think_mode", "")) != str(fast_identity.get("requested_think_mode", "")):
                    findings.append({"severity": "error", "type": "handoff_think_mode_mismatch", "detail": path.name})
                if str(identity.get("structured_output_mode", "")) != str(fast_identity.get("structured_output_mode", "")):
                    findings.append({"severity": "error", "type": "handoff_output_mode_mismatch", "detail": path.name})
                if str(identity.get("scenario_content_hash", "")) != str(fast_identity.get("scenario_content_hash", "")):
                    findings.append({"severity": "error", "type": "handoff_scenario_hash_mismatch", "detail": path.name})
                if str(prompt_components.get("scenario_version", "")) != str(fast_prompt_components.get("scenario_version", "")):
                    findings.append({"severity": "error", "type": "handoff_scenario_version_mismatch", "detail": path.name})
                if str(prompt_components.get("comparison_information_mode", "")) != str(fast_prompt_components.get("comparison_information_mode", "")):
                    findings.append({"severity": "error", "type": "handoff_information_mode_mismatch", "detail": path.name})

        if task.get("verification_classification") == "deterministic" and not result.get("deterministic_results"):
            findings.append({"severity": "error", "type": "missing_deterministic_output", "detail": path.name})

        if scores.get("score_status") == "human_required" and result.get("ranking_eligible", True):
            findings.append({"severity": "error", "type": "human_review_unresolved", "detail": path.name})

        dev_checks = verifier_output.get("checks") or []
        for check in dev_checks:
            if check.get("verifier") == "development":
                extra = check.get("extra", {})
                if extra.get("sandbox_backend") and not extra.get("policy_refusal") and not extra.get("sandbox_unavailable"):
                    findings.append({"severity": "error", "type": "development_host_execution", "detail": path.name})

        if str(task.get("team", "")) == "development" and development_sandbox_policy_refusal and result.get("score_status") == "final":
            findings.append({"severity": "error", "type": "development_final_without_sandbox", "detail": path.name})

        metrics = response.get("metrics", {})
        load_seconds = float(metrics.get("load_seconds", 0.0) or 0.0)
        prompt_eval_seconds = float(metrics.get("prompt_eval_seconds", 0.0) or 0.0)
        generation_seconds = float(metrics.get("generation_seconds", metrics.get("net_generation_seconds", 0.0)) or 0.0)
        ollama_total_seconds = float(metrics.get("ollama_total_seconds", 0.0) or 0.0)
        wall_clock_seconds = float(metrics.get("wall_clock_seconds", 0.0) or 0.0)
        overhead_seconds = float(metrics.get("overhead_seconds", 0.0) or 0.0)
        if overhead_seconds < 0:
            findings.append({"severity": "error", "type": "negative_timing_overhead", "detail": path.name})
        calc_overhead = max(0.0, wall_clock_seconds - ollama_total_seconds)
        if abs(calc_overhead - overhead_seconds) > 0.05:
            findings.append({"severity": "error", "type": "timing_field_inconsistency", "detail": path.name})
        if any(x < 0 for x in (load_seconds, prompt_eval_seconds, generation_seconds, ollama_total_seconds, wall_clock_seconds)):
            findings.append({"severity": "error", "type": "timing_field_inconsistency", "detail": path.name})

    # Definitive leaderboard must contain only final + ranking_eligible rows.
    leaderboard_path = run_dir / "leaderboard_by_role.csv"
    if leaderboard_path.exists():
        import csv

        with leaderboard_path.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            if int(row.get("eligible_result_count", "0") or 0) > int(row.get("sample_size", "0") or 0):
                findings.append({"severity": "error", "type": "provisional_in_definitive_leaderboard", "detail": row.get("model", "")})
            weighted = float(row.get("weighted_task_score", row.get("avg_score", 0.0)) or 0.0)
            unweighted = float(row.get("unweighted_task_score", weighted) or weighted)
            if float(row.get("eligible_weight_sum", 0.0) or 0.0) > 0 and weighted > 1.0:
                findings.append({"severity": "error", "type": "incorrect_task_weight_aggregation", "detail": row.get("model", "")})
            if weighted < 0 or unweighted < 0:
                findings.append({"severity": "error", "type": "incorrect_task_weight_aggregation", "detail": row.get("model", "")})

    esc_path = run_dir / "escalation_value.csv"
    if esc_path.exists():
        import csv

        with esc_path.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if rows and any(not row.get("fast_model") or not row.get("heavy_model") for row in rows):
            findings.append({"severity": "error", "type": "escalation_candidate_combinations_missing", "detail": "missing fast/heavy candidate in escalation row"})
        for row in rows:
            if row.get("track") == "handoff" and row.get("status") == "invalid_dependency":
                findings.append({"severity": "error", "type": "handoff_dependency_missing", "detail": "invalid dependency in escalation row"})

    audit_json = {
        "run_id": manifest.get("run_id"),
        "profile": manifest.get("profile"),
        "events": len(events),
        "state": {
            "completed_keys": len(state.get("completed_identity_keys", [])),
            "error_count": state.get("error_count", 0),
            "unsupported_mode_count": state.get("unsupported_mode_count", 0),
        },
        "findings": findings,
    }

    (run_dir / "RUN_AUDIT.json").write_text(json.dumps(audit_json, indent=2))

    md_lines = [
        "# Run Audit",
        "",
        f"- run_id: {manifest.get('run_id')}",
        f"- profile: {manifest.get('profile')}",
        f"- events: {len(events)}",
        f"- findings: {len(findings)}",
        "",
        "## Findings",
    ]
    if findings:
        for f in findings:
            md_lines.append(f"- [{f['severity']}] {f['type']}: {f['detail']}")
    else:
        md_lines.append("- None")
    (run_dir / "RUN_AUDIT.md").write_text("\n".join(md_lines) + "\n")

    print("Run audit")
    print("---------")
    print(f"run_id: {manifest.get('run_id')}")
    print(f"profile: {manifest.get('profile')}")
    print(f"events: {len(events)}")
    print(f"findings: {len(findings)}")
    print(f"wrote: {run_dir / 'RUN_AUDIT.json'}")
    print(f"wrote: {run_dir / 'RUN_AUDIT.md'}")

    return 1 if any(f["severity"] == "error" for f in findings) else 0


def cmd_run(args: argparse.Namespace, config: Configuration) -> int:
    from llm_auditions.runner import AuditionRunner

    config.load()

    profile = getattr(args, "profile", None) or "smoke"

    if profile == "exhaustive":
        confirmed = (
            os.environ.get("AUDITION_YES", "").lower() in ("1", "true", "yes")
            or getattr(args, "yes", False)
        )
        if not confirmed:
            print(
                "\nWARNING: exhaustive profile requires --yes or AUDITION_YES=1"
            )
            return 1

    run_dir = Path(args.run_dir) if getattr(args, "run_dir", None) else _make_run_dir(config, profile)

    tasks, plan_rows, md = _build_execution_plan(
        config=config,
        profile=profile,
        team_filter=getattr(args, "team", None),
        role_filter=getattr(args, "role", None),
        model_filter=getattr(args, "model", None),
        task_filter=getattr(args, "task", None),
        think_mode_filter=getattr(args, "think_mode", None),
    )

    runner = AuditionRunner(config=config, run_dir=run_dir, profile=profile)

    if runner.has_existing_manifest():
        manifest = runner.load_existing_run()
        existing_profile = manifest.profile
        if existing_profile != profile:
            print(
                f"ERROR: Existing run directory profile '{existing_profile}' does not match requested profile '{profile}'."
            )
            return 1
        task_manifest_path = run_dir / "task_manifest.json"
        if task_manifest_path.exists():
            task_manifest = json.loads(task_manifest_path.read_text())
            stored_rows = task_manifest.get("requests", [])
            if stored_rows:
                plan_rows = stored_rows
                all_tasks = _load_tasks()
                keys = {(r["team"], r["role"], r["task_id"]) for r in plan_rows}
                tasks = [t for t in all_tasks if (t.team, t.role, t.id) in keys]
        print(f"Resuming existing run: {run_dir}")
    else:
        runner.setup_new_run(tasks=tasks, plan_rows=plan_rows)

    if not plan_rows:
        print("No tasks matched the filter criteria.")
        return 0

    print(f"\nRunning {len(plan_rows)} planned requests in {run_dir.name} ...")

    total = 0
    errors = 0
    for result in runner.run_plan_rows(tasks, plan_rows):
        total += 1
        if result.status == "error":
            errors += 1
            print(f"  ERROR: {result.task.id} / {result.identity.model_name}: {result.response.error}")
        else:
            print(
                f"  [{total:4d}] {result.task.id:<40} {result.identity.model_name:<25} "
                f"think={result.identity.requested_think_mode}->{result.identity.effective_think_mode:<8} score={result.scores.weighted_total:.3f}"
            )

    print(f"\nRun complete: {total} results, {errors} errors")
    print(f"Run directory: {run_dir}")
    return 0


def cmd_resume(args: argparse.Namespace, config: Configuration) -> int:
    run_dir = Path(args.run_dir)
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        print("ERROR: cannot resume without run_manifest.json")
        return 1

    manifest = json.loads(manifest_path.read_text())
    args.profile = manifest.get("profile", "smoke")
    args.yes = True
    args.team = None
    args.role = None
    args.model = None
    args.task = None
    return cmd_run(args, config)


def cmd_report(args: argparse.Namespace, config: Configuration) -> int:
    from llm_auditions.reporting import generate_reports

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"ERROR: Run directory does not exist: {run_dir}")
        return 1

    created = generate_reports(run_dir)
    print(f"\nReports generated in {run_dir}:")
    for p in created:
        print(f"  {p.name}")
    return 0


def cmd_package(args: argparse.Namespace, config: Configuration) -> int:
    from llm_auditions.packaging import create_package
    from llm_auditions.sanitization import SanitizationError

    run_dir = Path(args.run_dir)
    safe_override = getattr(args, "safe_override", False)

    try:
        archive, sha = create_package(run_dir, safe_override=safe_override)
        print("\nPackage created:")
        print(f"  Archive:  {archive}")
        print(f"  SHA-256:  {sha}")
        print(f"\n  SHA-256 hash: {sha.read_text().split()[0]}")
    except SanitizationError as exc:
        print(f"\nSANITIZATION FAILED:\n{exc}")
        return 1
    except Exception as exc:
        print(f"\nPackaging failed: {exc}")
        return 1
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="audition",
        description="LLM Audition Framework — test Ollama-served models for team roles.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list-models", help="List configured and installed Ollama models")
    sub.add_parser("list-teams", help="List configured audition teams")
    sub.add_parser("list-roles", help="List all team roles and candidates")

    p_lt = sub.add_parser("list-tasks", help="List available audition tasks")
    p_lt.add_argument("--team", help="Filter by team name")
    p_lt.add_argument("--role", help="Filter by role name")

    sub.add_parser("validate", help="Validate configuration against installed models")

    p_plan = sub.add_parser("plan", help="Show dry-run request plan for a profile")
    p_plan.add_argument("--profile", default="smoke", choices=["smoke", "standard", "exhaustive"])
    p_plan.add_argument("--team")
    p_plan.add_argument("--role")
    p_plan.add_argument("--model")
    p_plan.add_argument("--task")
    p_plan.add_argument("--think-mode")

    sub.add_parser("audit-config", help="Audit fixture/config integrity")

    p_audit_run = sub.add_parser("audit-run", help="Audit run artifacts")
    p_audit_run.add_argument("--run-dir", required=True, help="Run directory")

    p_run = sub.add_parser("run", help="Run audition tasks")
    p_run.add_argument("--profile", choices=["smoke", "standard", "exhaustive"], default="smoke")
    p_run.add_argument("--team", help="Run only this team")
    p_run.add_argument("--role", help="Run only this role (within team)")
    p_run.add_argument("--model", help="Run only this model")
    p_run.add_argument("--task", help="Run only this task id")
    p_run.add_argument("--think-mode", help="Override think mode")
    p_run.add_argument("--run-dir", help="Output directory (default: auto-generated)")
    p_run.add_argument("--yes", action="store_true", help="Confirm exhaustive run")

    p_resume = sub.add_parser("resume", help="Resume an interrupted run")
    p_resume.add_argument("--run-dir", required=True, help="Existing run directory to resume")

    p_report = sub.add_parser("report", help="Generate reports from a completed run")
    p_report.add_argument("--run-dir", required=True, help="Run directory")

    p_pkg = sub.add_parser("package", help="Package run results into a sanitized archive")
    p_pkg.add_argument("--run-dir", required=True, help="Run directory to package")
    p_pkg.add_argument("--safe-override", action="store_true", help="Bypass sanitization blocker")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    config = Configuration()

    handlers: dict[str, Any] = {
        "list-models": cmd_list_models,
        "list-teams": cmd_list_teams,
        "list-roles": cmd_list_roles,
        "list-tasks": cmd_list_tasks,
        "validate": cmd_validate,
        "plan": cmd_plan,
        "audit-config": cmd_audit_config,
        "audit-run": cmd_audit_run,
        "run": cmd_run,
        "resume": cmd_resume,
        "report": cmd_report,
        "package": cmd_package,
    }

    handler = handlers.get(args.command)
    if handler is None:
        print(f"Unknown command: {args.command}")
        return 1

    try:
        return handler(args, config) or 0
    except (ValidationError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    except Exception as exc:
        logger.exception("Unhandled error: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
