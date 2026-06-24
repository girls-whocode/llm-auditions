"""Reporting — generates summary files and REPORT.md from a completed run directory."""

from __future__ import annotations

import csv
import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

logger = logging.getLogger(__name__)

try:
    from .models import ResultIdentity
except Exception:  # pragma: no cover
    ResultIdentity = None  # type: ignore[assignment]


def _float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _enrich_result_timing(result: dict[str, Any]) -> dict[str, Any]:
    response = result.get("response")
    if not isinstance(response, dict):
        return result
    metrics = response.get("metrics")
    if not isinstance(metrics, dict):
        return result

    load_duration_ns = _float(metrics.get("load_duration_ns", metrics.get("load_duration", 0.0)))
    prompt_eval_duration_ns = _float(metrics.get("prompt_eval_duration_ns", metrics.get("prompt_eval_duration", 0.0)))
    eval_duration_ns = _float(metrics.get("eval_duration_ns", metrics.get("eval_duration", 0.0)))
    total_duration_ns = _float(metrics.get("total_duration_ns", metrics.get("total_duration", 0.0)))
    wall_clock_seconds_local = _float(metrics.get("wall_clock_seconds_local", 0.0))
    eval_count = _float(metrics.get("eval_count", 0.0))

    load_seconds = load_duration_ns / 1e9 if load_duration_ns > 0 else _float(metrics.get("load_seconds", 0.0))
    prompt_eval_seconds = prompt_eval_duration_ns / 1e9 if prompt_eval_duration_ns > 0 else _float(metrics.get("prompt_eval_seconds", 0.0))
    generation_seconds = eval_duration_ns / 1e9 if eval_duration_ns > 0 else _float(metrics.get("generation_seconds", metrics.get("net_generation_seconds", 0.0)))
    ollama_total_seconds = total_duration_ns / 1e9 if total_duration_ns > 0 else _float(metrics.get("ollama_total_seconds", 0.0))
    wall_clock_seconds = wall_clock_seconds_local if wall_clock_seconds_local > 0 else _float(metrics.get("wall_clock_seconds", ollama_total_seconds))
    overhead_seconds = max(0.0, wall_clock_seconds - ollama_total_seconds)
    if eval_count > 0 and generation_seconds > 0:
        generation_rate = eval_count / generation_seconds
    else:
        generation_rate = _float(metrics.get("generation_rate", 0.0))

    metrics["load_seconds"] = round(load_seconds, 4)
    metrics["prompt_eval_seconds"] = round(prompt_eval_seconds, 4)
    metrics["generation_seconds"] = round(generation_seconds, 4)
    metrics["ollama_total_seconds"] = round(ollama_total_seconds, 4)
    metrics["wall_clock_seconds"] = round(wall_clock_seconds, 4)
    metrics["overhead_seconds"] = round(overhead_seconds, 4)
    metrics["generation_rate"] = round(generation_rate, 4)
    return result


def _load_events(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "events.jsonl"
    if not path.exists():
        return []
    events = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return events


def _load_results(run_dir: Path) -> list[dict[str, Any]]:
    results = []
    for path in sorted(run_dir.rglob("*.result.json")):
        try:
            results.append(_enrich_result_timing(json.loads(path.read_text(encoding="utf-8"))))
        except Exception as exc:
            logger.warning("Failed to load result %s: %s", path, exc)
    return results


def generate_reports(run_dir: Path) -> list[Path]:
    events = _load_events(run_dir)
    results = _load_results(run_dir)
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8")) if (run_dir / "run_manifest.json").exists() else {}

    created: list[Path] = []

    summary = _build_summary(events, results, manifest)
    p = run_dir / "summary.json"
    p.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    created.append(p)

    p = run_dir / "summary.csv"
    _write_summary_csv(p, events)
    created.append(p)

    p = run_dir / "leaderboard_by_role.csv"
    _write_leaderboard_csv(p, events, manifest)
    created.append(p)

    p = run_dir / "leaderboard_by_role_mode.csv"
    _write_leaderboard_mode_csv(p, events, manifest)
    created.append(p)

    p = run_dir / "failures.csv"
    _write_failures_csv(p, events)
    created.append(p)

    p = run_dir / "unsupported_modes.csv"
    _write_unsupported_csv(p, events)
    created.append(p)

    p = run_dir / "deterministic_results.json"
    _write_deterministic_json(p, results)
    created.append(p)

    p = run_dir / "escalation_value.csv"
    _write_escalation_csv(p, results)
    created.append(p)

    p = run_dir / "human_review_queue.csv"
    _write_human_review_queue_csv(p, run_dir, results)
    created.append(p)

    p = run_dir / "human_review_queue.md"
    _write_human_review_queue_md(p, run_dir, results)
    created.append(p)

    p = run_dir / "REPORT.md"
    p.write_text(_build_report_md(run_dir, summary, events, results, manifest), encoding="utf-8")
    created.append(p)

    p = run_dir / "REPORT_FOR_EXTERNAL_REVIEW.md"
    p.write_text(_build_external_review_report(run_dir, summary, events, results, manifest), encoding="utf-8")
    created.append(p)

    return created


def _status_bucket(event: dict[str, Any]) -> str:
    status = event.get("score_status", "provisional")
    if status == "final" and event.get("ranking_eligible", True):
        return "Qualified"
    if status == "provisional":
        return "Provisionally qualified"
    if status == "disqualified" or event.get("ranking_eligible") is False:
        return "Disqualified"
    if status == "human_required":
        return "Human review required"
    return "Insufficient evidence"


def _build_summary(events: list[dict[str, Any]], results: list[dict[str, Any]], manifest: dict[str, Any]) -> dict[str, Any]:
    total = len(results) or len(events)
    completed_results = [r for r in results if r.get("status") in ("completed", "unsupported_mode")]
    if not completed_results and events:
        completed_results = [
            {
                "status": e.get("status"),
                "score_status": e.get("score_status"),
                "ranking_eligible": e.get("ranking_eligible", True),
                "response": {
                    "metrics": {
                        "wall_clock_seconds": e.get("wall_clock_s", 0.0),
                        "load_seconds": e.get("load_seconds", 0.0),
                        "prompt_eval_seconds": e.get("prompt_eval_seconds", 0.0),
                        "generation_seconds": e.get("generation_seconds", e.get("wall_clock_s", 0.0)),
                        "ollama_total_seconds": e.get("ollama_total_seconds", e.get("wall_clock_s", 0.0)),
                        "overhead_seconds": e.get("overhead_seconds", 0.0),
                        "generation_rate": e.get("generation_rate", 0.0),
                    },
                    "truncated_length_stop": e.get("truncated_length_stop", False),
                    "empty_final_content": e.get("empty_final_content", False),
                },
                "scores": {"weighted_total": e.get("weighted_total")},
                "task": {"verification_classification": "rubric_assisted", "team": e.get("team", ""), "role": e.get("role", "")},
                "identity": {
                    "requested_think_mode": e.get("requested_think_mode", ""),
                    "effective_think_mode": e.get("effective_think_mode", ""),
                    "structured_output_mode": e.get("structured_output_mode", ""),
                    "model_name": e.get("model", ""),
                    "model_digest": "",
                    "task_suite_version": manifest.get("task_suite_version", "1"),
                },
                "schema_errors": [],
                "safety_flags": e.get("safety_flags", []),
                "verifier_output": {"rubric_rules": []},
                "deterministic_results": [],
            }
            for e in events
            if e.get("status") in ("completed", "unsupported_mode")
        ]
    definitive_results = [r for r in completed_results if r.get("score_status") == "final" and r.get("ranking_eligible", True)]

    def _rate(rows: list[dict[str, Any]], predicate) -> float:
        if not rows:
            return 0.0
        return round(sum(1 for row in rows if predicate(row)) / len(rows), 4)

    def _metric(result: dict[str, Any], path: str, default: Any = None) -> Any:
        current: Any = result
        for part in path.split("."):
            if isinstance(current, dict):
                current = current.get(part, default)
            else:
                return default
        return current

    load = [float(_metric(r, "response.metrics.load_seconds", 0.0) or 0.0) for r in completed_results]
    prompt_eval = [float(_metric(r, "response.metrics.prompt_eval_seconds", 0.0) or 0.0) for r in completed_results]
    generation = [float(_metric(r, "response.metrics.generation_seconds", _metric(r, "response.metrics.net_generation_seconds", 0.0)) or 0.0) for r in completed_results]
    ollama_total = [float(_metric(r, "response.metrics.ollama_total_seconds", 0.0) or 0.0) for r in completed_results]
    wall = [float(_metric(r, "response.metrics.wall_clock_seconds", 0.0) or 0.0) for r in completed_results]
    overhead = [float(_metric(r, "response.metrics.overhead_seconds", 0.0) or 0.0) for r in completed_results]
    generation_rate = [float(_metric(r, "response.metrics.generation_rate", 0.0) or 0.0) for r in completed_results]

    deterministic_rows = [r for r in completed_results if r.get("task", {}).get("verification_classification") == "deterministic"]
    rubric_rows = [r for r in completed_results if r.get("task", {}).get("verification_classification") == "rubric_assisted"]
    development_excluded = [
        r
        for r in completed_results
        if r.get("task", {}).get("team") == "development"
        and (
            r.get("score_status") in ("human_required", "provisional")
            or bool(r.get("human_review_required", False))
            or bool(r.get("verifier_output", {}).get("policy_refusal", False))
        )
    ]
    development_reasons: dict[str, int] = {}
    for row in development_excluded:
        reason = str(row.get("verifier_output", {}).get("reason") or "unknown")
        development_reasons[reason] = development_reasons.get(reason, 0) + 1

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": manifest.get("run_id"),
        "profile": manifest.get("profile"),
        "task_suite_version": manifest.get("task_suite_version", "1"),
        "total_requests": total,
        "completed": len(completed_results),
        "errors": sum(1 for r in results if r.get("status") == "error") or sum(1 for e in events if e.get("status") == "error"),
        "skipped": sum(1 for e in events if e.get("status") == "skipped"),
        "unsupported_modes": sum(1 for r in results if r.get("status") == "unsupported_mode") or sum(1 for e in events if e.get("status") == "unsupported_mode"),
        "final_count": len(definitive_results),
        "provisional_count": sum(1 for r in completed_results if r.get("score_status") == "provisional"),
        "human_review_count": sum(1 for r in completed_results if r.get("score_status") == "human_required" or r.get("human_review_required", False)),
        "disqualified_count": sum(1 for r in completed_results if r.get("score_status") == "disqualified"),
        "deterministic_pass_rate": _rate(deterministic_rows, lambda r: bool(r.get("deterministic_results")) and all(item.get("passed", False) for item in r.get("deterministic_results", []))),
        "schema_pass_rate": _rate(completed_results, lambda r: not bool(r.get("schema_errors"))),
        "safety_pass_rate": _rate(completed_results, lambda r: not bool(r.get("safety_flags")) and not bool(r.get("hard_fail"))),
        "rubric_pass_rate": _rate(rubric_rows, lambda r: all(item.get("status") != "fail" for item in r.get("verifier_output", {}).get("rubric_rules", []))),
        "truncation_rate": _rate(completed_results, lambda r: bool(_metric(r, "response.truncated_length_stop", False))),
        "empty_answer_rate": _rate(completed_results, lambda r: bool(_metric(r, "response.empty_final_content", False))),
        "median_load_seconds": round(median(load), 4) if load else 0.0,
        "median_prompt_eval_seconds": round(median(prompt_eval), 4) if prompt_eval else 0.0,
        "median_generation_seconds": round(median(generation), 4) if generation else 0.0,
        "median_ollama_total_seconds": round(median(ollama_total), 4) if ollama_total else 0.0,
        "median_wall_clock_seconds": round(median(wall), 4) if wall else 0.0,
        "median_overhead_seconds": round(median(overhead), 4) if overhead else 0.0,
        "median_generation_rate": round(median(generation_rate), 4) if generation_rate else 0.0,
        "development_excluded_count": len(development_excluded),
        "development_excluded_reasons": development_reasons,
        "sample_size": len(definitive_results),
    }


def _write_summary_csv(path: Path, events: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "team",
                "role",
                "task_id",
                "model",
                "requested_think_mode",
                "effective_think_mode",
                "structured_output_mode",
                "status",
                "score_status",
                "ranking_eligible",
                "weighted_total",
                "wall_clock_s",
            ],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(events)


def _write_leaderboard_csv(path: Path, events: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    results = _load_results(path.parent)
    key_scores: dict[tuple[str, str, str, str, str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    if not results:
        for event in events:
            if event.get("status") != "completed":
                continue
            key = (
                event.get("team", ""),
                event.get("role", ""),
                event.get("requested_think_mode", ""),
                event.get("effective_think_mode", ""),
                event.get("structured_output_mode", ""),
                manifest.get("profile", ""),
                manifest.get("task_suite_version", "1"),
                "",
                event.get("model", ""),
            )
            key_scores[key].append(
                {
                    "task": {"weight": event.get("task_weight", 1.0)},
                    "score_status": event.get("score_status"),
                    "ranking_eligible": event.get("ranking_eligible", True),
                    "scores": {"weighted_total": event.get("weighted_total", 0.0)},
                }
            )
    for result in results:
        if result.get("status") != "completed":
            continue
        identity = result.get("identity", {})
        task = result.get("task", {})
        key = (
            task.get("team", ""),
            task.get("role", ""),
            identity.get("requested_think_mode", ""),
            identity.get("effective_think_mode", ""),
            identity.get("structured_output_mode", ""),
            manifest.get("profile", ""),
            identity.get("task_suite_version", manifest.get("task_suite_version", "1")),
            identity.get("model_digest", ""),
            identity.get("model_name", ""),
        )
        key_scores[key].append(result)

    rows = []
    for k, v in key_scores.items():
        definitive = [item for item in v if item.get("score_status") == "final" and item.get("ranking_eligible", True)]
        definitive_scores = [float(item.get("scores", {}).get("weighted_total", 0.0) or 0.0) for item in definitive]
        definitive_weights = [float(item.get("task", {}).get("weight", 1.0) or 1.0) for item in definitive]
        eligible_weight_sum = round(sum(definitive_weights), 4)
        weighted_numerator = sum(score * weight for score, weight in zip(definitive_scores, definitive_weights))
        weighted_task_score = round(weighted_numerator / eligible_weight_sum, 4) if eligible_weight_sum > 0 else 0.0
        unweighted_task_score = round(sum(definitive_scores) / len(definitive_scores), 4) if definitive_scores else 0.0
        provisional_count = sum(1 for item in v if item.get("score_status") == "provisional")
        human_review_count = sum(1 for item in v if item.get("score_status") == "human_required" or item.get("human_review_required", False))
        disqualified_count = sum(1 for item in v if item.get("score_status") == "disqualified")
        ineligible_count = sum(1 for item in v if not bool(item.get("ranking_eligible", True)))
        def _median_metric(metric_path: str) -> float:
            values = []
            for item in definitive:
                cur: Any = item
                for part in metric_path.split("."):
                    if isinstance(cur, dict):
                        cur = cur.get(part)
                    else:
                        cur = None
                        break
                values.append(float(cur or 0.0))
            return round(median(values), 4) if values else 0.0

        rows.append(
            {
                "team": k[0],
                "role": k[1],
                "requested_think_mode": k[2],
                "effective_think_mode": k[3],
                "structured_output_mode": k[4],
                "profile": k[5],
                "task_suite_version": k[6],
                "model_digest": k[7],
                "model": k[8],
                "avg_score": weighted_task_score,
                "sample_size": len(definitive_scores),
                "eligible_count": len(definitive_scores),
                "total_count": len(v),
                "eligible_weight_sum": eligible_weight_sum,
                "eligible_result_count": len(definitive_scores),
                "total_result_count": len(v),
                "weighted_task_score": weighted_task_score,
                "unweighted_task_score": unweighted_task_score,
                "provisional_count": provisional_count,
                "human_review_count": human_review_count,
                "disqualified_count": disqualified_count,
                "ineligible_count": ineligible_count,
                "median_load_seconds": _median_metric("response.metrics.load_seconds"),
                "median_prompt_eval_seconds": _median_metric("response.metrics.prompt_eval_seconds"),
                "median_generation_seconds": _median_metric("response.metrics.generation_seconds"),
                "median_ollama_total_seconds": _median_metric("response.metrics.ollama_total_seconds"),
                "median_wall_clock_seconds": _median_metric("response.metrics.wall_clock_seconds"),
                "median_overhead_seconds": _median_metric("response.metrics.overhead_seconds"),
                "median_generation_rate": _median_metric("response.metrics.generation_rate"),
            }
        )

    rows.sort(key=lambda r: (-r["avg_score"], r["team"], r["role"], r["model"]))

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "team",
                "role",
                "requested_think_mode",
                "effective_think_mode",
                "structured_output_mode",
                "profile",
                "task_suite_version",
                "model_digest",
                "model",
                "avg_score",
                "sample_size",
                "eligible_count",
                "total_count",
                "eligible_weight_sum",
                "eligible_result_count",
                "total_result_count",
                "weighted_task_score",
                "unweighted_task_score",
                "provisional_count",
                "human_review_count",
                "disqualified_count",
                "ineligible_count",
                "median_load_seconds",
                "median_prompt_eval_seconds",
                "median_generation_seconds",
                "median_ollama_total_seconds",
                "median_wall_clock_seconds",
                "median_overhead_seconds",
                "median_generation_rate",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_leaderboard_mode_csv(path: Path, events: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    rows = []
    for e in events:
        if e.get("status") != "completed":
            continue
        row = {
            "team": e.get("team", ""),
            "role": e.get("role", ""),
            "model": e.get("model", ""),
            "requested_think_mode": e.get("requested_think_mode", ""),
            "effective_think_mode": e.get("effective_think_mode", ""),
            "structured_output_mode": e.get("structured_output_mode", ""),
            "profile": manifest.get("profile", ""),
            "task_suite_version": manifest.get("task_suite_version", "1"),
            "status_bucket": _status_bucket(e),
            "score_status": e.get("score_status", "provisional"),
            "ranking_eligible": e.get("ranking_eligible", True),
            "weighted_total": e.get("weighted_total"),
        }
        rows.append(row)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "team",
                "role",
                "model",
                "requested_think_mode",
                "effective_think_mode",
                "structured_output_mode",
                "profile",
                "task_suite_version",
                "status_bucket",
                "score_status",
                "ranking_eligible",
                "weighted_total",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_failures_csv(path: Path, events: list[dict[str, Any]]) -> None:
    failures = [e for e in events if e.get("status") == "error"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["team", "role", "task_id", "model", "requested_think_mode", "effective_think_mode", "error"],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(failures)


def _write_unsupported_csv(path: Path, events: list[dict[str, Any]]) -> None:
    unsupported = [e for e in events if e.get("status") == "unsupported_mode"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["team", "role", "task_id", "model", "requested_think_mode", "effective_think_mode", "structured_output_mode"],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(unsupported)


def _write_deterministic_json(path: Path, results: list[dict[str, Any]]) -> None:
    det_results = []
    for r in results:
        det = r.get("deterministic_results", [])
        if det:
            det_results.append(
                {
                    "task_id": r.get("task", {}).get("id", ""),
                    "model": r.get("identity", {}).get("model_name", ""),
                    "requested_think_mode": r.get("identity", {}).get("requested_think_mode", ""),
                    "effective_think_mode": r.get("identity", {}).get("effective_think_mode", ""),
                    "deterministic_results": det,
                }
            )
    path.write_text(json.dumps(det_results, indent=2), encoding="utf-8")


def _write_escalation_csv(path: Path, results: list[dict[str, Any]]) -> None:
    def _delta(a: Any, b: Any) -> float | None:
        if a is None or b is None:
            return None
        return round(float(b) - float(a), 4)

    def _result_identity_key(result: dict[str, Any]) -> str:
        identity = result.get("identity", {})
        if ResultIdentity is not None:
            try:
                return ResultIdentity.model_validate(identity).key()
            except Exception:
                pass
        parts = [
            identity.get("team", ""),
            identity.get("role", ""),
            identity.get("task_id", ""),
            identity.get("task_version", "v1"),
            identity.get("model_name", ""),
            identity.get("model_digest", ""),
            identity.get("requested_think_mode", ""),
            identity.get("structured_output_mode", ""),
            str(identity.get("temperature", "")),
            str(identity.get("num_ctx", "")),
            str(identity.get("num_predict", "")),
            identity.get("system_prompt_hash", ""),
            identity.get("user_prompt_hash", ""),
            identity.get("task_suite_version", ""),
            identity.get("verifier_version", ""),
            identity.get("scoring_version", ""),
            identity.get("engine_version", ""),
            identity.get("handoff_fast_identity_key", ""),
            identity.get("handoff_fast_response_hash", ""),
            identity.get("comparison_scenario_hash", ""),
            identity.get("scenario_content_hash", ""),
            identity.get("effective_think_mode", ""),
        ]
        fixture_hashes = identity.get("fixture_hashes", {}) or {}
        for key in sorted(fixture_hashes):
            parts.append(f"{key}:{fixture_hashes[key]}")
        return __import__("hashlib").sha256("|".join(parts).encode()).hexdigest()[:24]

    def _summary_row(result: dict[str, Any]) -> dict[str, Any]:
        identity = result.get("identity", {})
        task = result.get("task", {})
        response = result.get("response", {})
        prompt_components = (response.get("effective_prompt", {}) or {}).get("prompt_components", {})
        metrics = response.get("metrics", {})
        scores = result.get("scores", {})
        response_text = str(response.get("content", "") or "")
        fast_hash = __import__("hashlib").sha256(response_text.encode()).hexdigest()[:16] if response_text else ""
        return {
            "comparison_id": task.get("comparison_id", ""),
            "track": task.get("comparison_track", ""),
            "scenario_version": str(prompt_components.get("scenario_version", "")),
            "scenario_content_hash": identity.get("scenario_content_hash") or identity.get("comparison_scenario_hash", ""),
            "comparison_information_mode": prompt_components.get("comparison_information_mode", ""),
            "worker_class": task.get("worker_class", ""),
            "task_id": task.get("id", ""),
            "model": identity.get("model_name", ""),
            "digest": identity.get("model_digest", ""),
            "identity": _result_identity_key(result),
            "response_hash": fast_hash,
            "weighted_total": scores.get("weighted_total"),
            "correctness": scores.get("correctness_score"),
            "completeness": scores.get("completeness_score"),
            "safety": scores.get("safety_score"),
            "score_status": result.get("score_status", scores.get("score_status", "provisional")),
            "ranking_eligible": bool(result.get("ranking_eligible", True)),
            "hard_fail": bool(result.get("hard_fail", False)),
            "generation_seconds": float(metrics.get("generation_seconds", 0.0) or 0.0),
            "wall_clock_seconds": float(metrics.get("wall_clock_seconds", 0.0) or 0.0),
            "token_count": int(metrics.get("eval_count", 0) or 0),
            "requested_think_mode": str(identity.get("requested_think_mode", "")),
            "structured_output_mode": str(identity.get("structured_output_mode", "")),
            "task_suite_version": str(identity.get("task_suite_version", "")),
            "dependency_fast_identity": identity.get("handoff_fast_identity_key", ""),
            "dependency_fast_hash": identity.get("handoff_fast_response_hash", ""),
        }

    summaries = [_summary_row(r) for r in results if r.get("task", {}).get("comparison_id")]
    identity_index = {r["identity"]: r for r in summaries}

    rows = []
    for heavy in [r for r in summaries if r.get("track") and r.get("comparison_id") and r.get("dependency_fast_identity")]:
        if heavy.get("track") != "handoff":
            continue
        fast = identity_index.get(heavy.get("dependency_fast_identity", ""))
        status = "resolved"
        if not fast:
            status = "invalid_dependency"
        elif heavy.get("dependency_fast_hash", "") != fast.get("response_hash", ""):
            status = "invalid_dependency"
        unresolved = (
            status == "invalid_dependency"
            or heavy.get("score_status") in ("human_required", "provisional", "disqualified")
            or not heavy.get("ranking_eligible", True)
            or (fast and (fast.get("score_status") in ("human_required", "provisional", "disqualified") or not fast.get("ranking_eligible", True)))
        )
        correctness_delta = _delta(fast.get("correctness") if fast else None, heavy.get("correctness"))
        completeness_delta = _delta(fast.get("completeness") if fast else None, heavy.get("completeness"))
        safety_delta = _delta(fast.get("safety") if fast else None, heavy.get("safety"))
        generation_time_delta = _delta(fast.get("generation_seconds") if fast else None, heavy.get("generation_seconds"))
        wall_time_delta = _delta(fast.get("wall_clock_seconds") if fast else None, heavy.get("wall_clock_seconds"))
        token_delta = _delta(fast.get("token_count") if fast else None, heavy.get("token_count"))
        material = bool((correctness_delta or 0) > 0.05 or (completeness_delta or 0) > 0.05 or (safety_delta or 0) > 0.05)
        justified = bool(material and not unresolved and not heavy.get("hard_fail", False))

        rows.append(
            {
                "comparison_id": heavy.get("comparison_id", ""),
                "track": "handoff",
                "scenario_version": heavy.get("scenario_version", ""),
                "scenario_content_hash": heavy.get("scenario_content_hash", ""),
                "comparison_information_mode": heavy.get("comparison_information_mode", ""),
                "fast_task_id": fast.get("task_id", "") if fast else "",
                "heavy_task_id": heavy.get("task_id", ""),
                "fast_model": fast.get("model", "") if fast else "",
                "fast_digest": fast.get("digest", "") if fast else "",
                "heavy_model": heavy.get("model", ""),
                "heavy_digest": heavy.get("digest", ""),
                "fast_result_identity": heavy.get("dependency_fast_identity", ""),
                "heavy_result_identity": heavy.get("identity", ""),
                "fast_response_hash": fast.get("response_hash", "") if fast else "",
                "heavy_dependency_fast_response_hash": heavy.get("dependency_fast_hash", ""),
                "fast_score": fast.get("weighted_total") if fast else None,
                "heavy_score": heavy.get("weighted_total"),
                "correctness_delta": correctness_delta,
                "completeness_delta": completeness_delta,
                "safety_delta": safety_delta,
                "hard_gate_change": f"fast={fast.get('hard_fail', False) if fast else ''} heavy={heavy.get('hard_fail', False)}",
                "generation_time_delta": generation_time_delta,
                "wall_time_delta": wall_time_delta,
                "token_delta": token_delta,
                "material_improvement": material,
                "escalation_justified": justified,
                "status": status if status == "invalid_dependency" else ("unresolved" if unresolved else "resolved"),
            }
        )

    independent_groups: dict[tuple[str, str, str, str, str, str, str], dict[str, list[dict[str, Any]]]] = defaultdict(lambda: {"fast": [], "heavy": []})
    for row in summaries:
        if row.get("track") != "independent":
            continue
        key = (
            row.get("comparison_id", ""),
            row.get("scenario_content_hash", ""),
            row.get("scenario_version", ""),
            row.get("comparison_information_mode", ""),
            row.get("requested_think_mode", ""),
            row.get("structured_output_mode", ""),
            row.get("task_suite_version", ""),
        )
        worker = row.get("worker_class", "")
        if worker not in ("fast", "heavy"):
            continue
        independent_groups[key][worker].append(row)

    for key, grouped in sorted(independent_groups.items()):
        for fast in grouped.get("fast", []):
            for heavy in grouped.get("heavy", []):
                unresolved = any(
                    x.get("score_status") in ("human_required", "provisional", "disqualified") or not x.get("ranking_eligible", True)
                    for x in (fast, heavy)
                )
                correctness_delta = _delta(fast.get("correctness"), heavy.get("correctness"))
                completeness_delta = _delta(fast.get("completeness"), heavy.get("completeness"))
                safety_delta = _delta(fast.get("safety"), heavy.get("safety"))
                generation_time_delta = _delta(fast.get("generation_seconds"), heavy.get("generation_seconds"))
                wall_time_delta = _delta(fast.get("wall_clock_seconds"), heavy.get("wall_clock_seconds"))
                token_delta = _delta(fast.get("token_count"), heavy.get("token_count"))
                material = bool((correctness_delta or 0) > 0.05 or (completeness_delta or 0) > 0.05 or (safety_delta or 0) > 0.05)
                justified = bool(material and not unresolved and not heavy.get("hard_fail", False))
                rows.append(
                    {
                        "comparison_id": key[0],
                        "track": "independent",
                        "scenario_version": key[2],
                        "scenario_content_hash": key[1],
                        "comparison_information_mode": key[3],
                        "fast_task_id": fast.get("task_id", ""),
                        "heavy_task_id": heavy.get("task_id", ""),
                        "fast_model": fast.get("model", ""),
                        "fast_digest": fast.get("digest", ""),
                        "heavy_model": heavy.get("model", ""),
                        "heavy_digest": heavy.get("digest", ""),
                        "fast_result_identity": fast.get("identity", ""),
                        "heavy_result_identity": heavy.get("identity", ""),
                        "fast_response_hash": fast.get("response_hash", ""),
                        "heavy_dependency_fast_response_hash": "",
                        "fast_score": fast.get("weighted_total"),
                        "heavy_score": heavy.get("weighted_total"),
                        "correctness_delta": correctness_delta,
                        "completeness_delta": completeness_delta,
                        "safety_delta": safety_delta,
                        "hard_gate_change": f"fast={fast.get('hard_fail', False)} heavy={heavy.get('hard_fail', False)}",
                        "generation_time_delta": generation_time_delta,
                        "wall_time_delta": wall_time_delta,
                        "token_delta": token_delta,
                        "material_improvement": material,
                        "escalation_justified": justified,
                        "status": "unresolved" if unresolved else "resolved",
                    }
                )

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "comparison_id",
                "track",
                "scenario_version",
                "scenario_content_hash",
                "comparison_information_mode",
                "fast_task_id",
                "heavy_task_id",
                "fast_model",
                "fast_digest",
                "heavy_model",
                "heavy_digest",
                "fast_result_identity",
                "heavy_result_identity",
                "fast_response_hash",
                "heavy_dependency_fast_response_hash",
                "fast_score",
                "heavy_score",
                "correctness_delta",
                "completeness_delta",
                "safety_delta",
                "hard_gate_change",
                "generation_time_delta",
                "wall_time_delta",
                "token_delta",
                "material_improvement",
                "escalation_justified",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_human_review_queue_csv(path: Path, run_dir: Path, results: list[dict[str, Any]]) -> None:
    rows = []
    for r in results:
        if r.get("score_status") not in ("human_required", "provisional") and not r.get("human_review_required", False):
            continue
        task = r.get("task", {})
        identity = r.get("identity", {})
        safe_model = identity.get("model_name", "").replace(":", "_").replace("/", "_")
        base = f"{task.get('id')}__{safe_model}__think_{identity.get('requested_think_mode')}-effective_{identity.get('effective_think_mode')}"
        response_path = run_dir / task.get("team", "") / task.get("role", "") / f"{base}.result.json"
        rows.append(
            {
                "team": task.get("team", ""),
                "role": task.get("role", ""),
                "task": task.get("id", ""),
                "model": identity.get("model_name", ""),
                "response_path": str(response_path),
                "rubric": task.get("verification_classification", ""),
                "reason_for_review": r.get("score_status", "provisional"),
                "unresolved_dimensions": "correctness,completeness",
            }
        )

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "team",
                "role",
                "task",
                "model",
                "response_path",
                "rubric",
                "reason_for_review",
                "unresolved_dimensions",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_human_review_queue_md(path: Path, run_dir: Path, results: list[dict[str, Any]]) -> None:
    csv_path = run_dir / "human_review_queue.csv"
    lines = [
        "# Human Review Queue",
        "",
        f"Source: {csv_path.name}",
        "",
    ]
    if csv_path.exists():
        rows = list(csv.DictReader(csv_path.read_text(encoding="utf-8").splitlines()))
    else:
        rows = []

    if not rows:
        lines.append("No entries.")
    else:
        lines += [
            "| team | role | task | model | reason |",
            "|---|---|---|---|---|",
        ]
        for r in rows:
            lines.append(f"| {r['team']} | {r['role']} | {r['task']} | {r['model']} | {r['reason_for_review']} |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_report_md(
    run_dir: Path,
    summary: dict[str, Any],
    events: list[dict[str, Any]],
    results: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> str:
    buckets = defaultdict(int)
    for e in events:
        buckets[_status_bucket(e)] += 1

    lines = [
        "# LLM Auditions — Run Report",
        "",
        f"Generated: {summary['generated_at']}",
        f"Run directory: {run_dir.name}",
        "",
        "## Summary",
        "",
        f"- sample size: {summary.get('sample_size', 0)}",
        f"- deterministic pass rate: {summary.get('deterministic_pass_rate', 0.0)}",
        f"- schema pass rate: {summary.get('schema_pass_rate', 0.0)}",
        f"- safety pass rate: {summary.get('safety_pass_rate', 0.0)}",
        f"- truncation rate: {summary.get('truncation_rate', 0.0)}",
        f"- empty-answer rate: {summary.get('empty_answer_rate', 0.0)}",
        f"- median generation time: {summary.get('median_generation_time', 0.0)}",
        f"- hard-gate failures: {summary.get('disqualified_count', 0)}",
        f"- human-review count: {summary.get('human_review_count', 0)}",
        "",
        "## Buckets",
        "",
    ]
    for k in [
        "Qualified",
        "Provisionally qualified",
        "Disqualified",
        "Human review required",
        "Insufficient evidence",
    ]:
        lines.append(f"- {k}: {buckets.get(k, 0)}")

    lines += [
        "",
        "## Outputs",
        "",
        "- leaderboard_by_role.csv",
        "- leaderboard_by_role_mode.csv",
        "- human_review_queue.csv",
        "- human_review_queue.md",
        "- escalation_value.csv",
    ]

    return "\n".join(lines) + "\n"


def _build_external_review_report(
    run_dir: Path,
    summary: dict[str, Any],
    events: list[dict[str, Any]],
    results: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> str:
    lines = [
        "# LLM Auditions — External Review Report",
        "",
        "No final role assignments are made in this report.",
        "",
        f"Generated: {summary['generated_at']}",
        f"Run directory: {run_dir.name}",
        "",
        "## Required Artifacts",
        "",
        "- run_manifest.json",
        "- environment.json",
        "- model_inventory.json",
        "- events.jsonl",
        "- summary.json",
        "- leaderboard_by_role.csv",
        "- leaderboard_by_role_mode.csv",
        "- human_review_queue.csv",
        "- escalation_value.csv",
    ]
    return "\n".join(lines) + "\n"
