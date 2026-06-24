from __future__ import annotations

import json
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
        content = "fast-completed" if "Escalation handoff" not in text else "heavy-using-fast"
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
        id="handoff_resume_case",
        team="linux_infrastructure",
        role="fast_worker" if worker_class == "fast" else "escalation",
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
        comparison_id="linux_firewall_change_002",
        comparison_track="handoff",
        worker_class=worker_class,
    )


def _fast_row(task: TaskDefinition) -> dict:
    return {
        "plan_row_id": "fast-row",
        "team": task.team,
        "role": task.role,
        "task_id": task.id,
        "task_version": task.task_version,
        "model": "fast-model",
        "full_model_digest": "",
        "requested_think_mode": "false",
        "structured_output_mode": "prompt_only",
        "temperature": 0.0,
        "num_ctx": 4096,
        "num_predict": 256,
        "schema_hash": "",
        "fixture_hashes": {},
        "comparison_id": "linux_firewall_change_002",
        "comparison_track": "handoff",
        "scenario_content_hash": "scenario-1",
        "fast_plan_row_id": "",
    }


def _heavy_row(task: TaskDefinition) -> dict:
    return {
        "plan_row_id": "heavy-row",
        "team": task.team,
        "role": task.role,
        "task_id": task.id,
        "task_version": task.task_version,
        "model": "heavy-model",
        "full_model_digest": "d-heavy",
        "requested_think_mode": "false",
        "structured_output_mode": "prompt_only",
        "temperature": 0.0,
        "num_ctx": 4096,
        "num_predict": 256,
        "schema_hash": "",
        "fixture_hashes": {},
        "comparison_id": "linux_firewall_change_002",
        "comparison_track": "handoff",
        "scenario_content_hash": "scenario-1",
        "fast_plan_row_id": "fast-row",
    }


def test_resume_loads_completed_fast_artifact_without_new_fast_call(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()

    fast_task = _task("fast")
    heavy_task = _task("heavy")

    first = AuditionRunner(_cfg(), run, profile="smoke")
    first_client = _FakeClient()
    first.client = first_client
    list(first.run_plan_rows([fast_task, heavy_task], [_fast_row(fast_task)]))

    second = AuditionRunner(_cfg(), run, profile="smoke")
    second_client = _FakeClient()
    second.client = second_client
    second.load_completed()
    fast_row = _fast_row(fast_task)
    loaded = second._load_handoff_context_from_saved_fast_result(  # pylint: disable=protected-access
        fast_task=fast_task,
        fast_row=type("Row", (), fast_row)(),
        schemas_dir=PROJECT_ROOT / "schemas",
    )

    assert loaded is not None
    assert loaded["fast_response"] == "fast-completed"

    messages, _ = second._build_messages(heavy_task, handoff_context=loaded)  # pylint: disable=protected-access
    prompt_text = "\n".join(m.get("content", "") for m in messages)
    assert "fast-completed" in prompt_text


def test_resume_missing_fast_artifact_blocks_heavy(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()

    fast_task = _task("fast")
    heavy_task = _task("heavy")

    runner = AuditionRunner(_cfg(), run, profile="smoke")
    client = _FakeClient()
    runner.client = client
    list(runner.run_plan_rows([fast_task, heavy_task], [_fast_row(fast_task)]))

    for path in run.rglob("*.result.json"):
        path.unlink()

    runner2 = AuditionRunner(_cfg(), run, profile="smoke")
    client2 = _FakeClient()
    runner2.client = client2
    out = list(runner2.run_plan_rows([fast_task, heavy_task], [_heavy_row(heavy_task)]))

    assert out == []
    assert client2.calls == []
    events = (run / "events.jsonl").read_text(encoding="utf-8").splitlines()
    assert any("invalid_dependency" in line for line in events)
