from __future__ import annotations

from pathlib import Path

from llm_auditions.configuration import Configuration
from llm_auditions.models import ModelResponse, OllamaMetrics, TaskDefinition
from llm_auditions.runner import AuditionRunner


PROJECT_ROOT = Path(__file__).parent.parent


class _FakeClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        text = "\n".join(m.get("content", "") for m in kwargs.get("messages", []))
        if "Escalation handoff" in text:
            content = f"heavy-for-{kwargs['model']}"
        else:
            content = f"fast-for-{kwargs['model']}"
        return ModelResponse(
            model=kwargs["model"],
            requested_think_mode=str(kwargs.get("think_mode", "false")),
            effective_think_mode=str(kwargs.get("think_mode", "false")),
            content=content,
            metrics=OllamaMetrics(eval_count=5, eval_duration_ns=1_000_000_000, total_duration_ns=1_100_000_000),
        )


def _cfg() -> Configuration:
    cfg = Configuration(project_root=PROJECT_ROOT)
    cfg.load()
    return cfg


def _task(worker_class: str) -> TaskDefinition:
    return TaskDefinition(
        id="handoff_matrix_case",
        team="security",
        role="fast_worker" if worker_class == "fast" else "worker",
        prompt="role instruction",
        verification_classification="rubric_assisted",
        rubric_finalization="deterministic",
        rubric_rules=[
            {
                "rule_id": "r1",
                "description": "required",
                "type": "required",
                "weight": 1.0,
                "matcher": {"type": "phrase_aliases", "phrases": ["safe"]},
            }
        ],
        comparison_id="sec_incident_response_002",
        comparison_track="handoff",
        worker_class=worker_class,
    )


def test_two_fast_two_heavy_produces_four_dependent_heavy_requests(tmp_path: Path):
    runner = AuditionRunner(_cfg(), tmp_path / "run", profile="smoke")
    fake = _FakeClient()
    runner.client = fake

    fast_task = _task("fast")
    heavy_task = _task("heavy")
    tasks = [fast_task, heavy_task]

    plan_rows = [
        {
            "plan_row_id": "f1",
            "team": fast_task.team,
            "role": fast_task.role,
            "task_id": fast_task.id,
            "task_version": fast_task.task_version,
            "model": "fast-a",
            "full_model_digest": "d-fast-a",
            "requested_think_mode": "false",
            "structured_output_mode": "prompt_only",
            "temperature": 0.0,
            "num_ctx": 4096,
            "num_predict": 256,
            "schema_hash": "",
            "fixture_hashes": {},
            "comparison_id": "sec_incident_response_002",
            "comparison_track": "handoff",
            "scenario_content_hash": "scenario-1",
            "fast_plan_row_id": "",
        },
        {
            "plan_row_id": "f2",
            "team": fast_task.team,
            "role": fast_task.role,
            "task_id": fast_task.id,
            "task_version": fast_task.task_version,
            "model": "fast-b",
            "full_model_digest": "d-fast-b",
            "requested_think_mode": "false",
            "structured_output_mode": "prompt_only",
            "temperature": 0.0,
            "num_ctx": 4096,
            "num_predict": 256,
            "schema_hash": "",
            "fixture_hashes": {},
            "comparison_id": "sec_incident_response_002",
            "comparison_track": "handoff",
            "scenario_content_hash": "scenario-1",
            "fast_plan_row_id": "",
        },
    ]
    for hid, hmodel, fid in (
        ("h1", "heavy-a", "f1"),
        ("h2", "heavy-a", "f2"),
        ("h3", "heavy-b", "f1"),
        ("h4", "heavy-b", "f2"),
    ):
        plan_rows.append(
            {
                "plan_row_id": hid,
                "team": heavy_task.team,
                "role": heavy_task.role,
                "task_id": heavy_task.id,
                "task_version": heavy_task.task_version,
                "model": hmodel,
                "full_model_digest": f"d-{hmodel}",
                "requested_think_mode": "false",
                "structured_output_mode": "prompt_only",
                "temperature": 0.0,
                "num_ctx": 4096,
                "num_predict": 256,
                "schema_hash": "",
                "fixture_hashes": {},
                "comparison_id": "sec_incident_response_002",
                "comparison_track": "handoff",
                "scenario_content_hash": "scenario-1",
                "fast_plan_row_id": fid,
            }
        )

    results = list(runner.run_plan_rows(tasks=tasks, plan_rows=plan_rows))

    heavy_results = [r for r in results if r.task.worker_class == "heavy"]
    assert len(heavy_results) == 4
    dependency_pairs = {(r.identity.handoff_fast_identity_key, r.identity.model_name) for r in heavy_results}
    assert len(dependency_pairs) == 4
