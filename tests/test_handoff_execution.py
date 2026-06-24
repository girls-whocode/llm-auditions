from __future__ import annotations

from pathlib import Path

from llm_auditions.configuration import Configuration
from llm_auditions.models import ModelResponse, OllamaMetrics, TaskDefinition
from llm_auditions.runner import AuditionRunner

PROJECT_ROOT = Path(__file__).parent.parent


class _FakeClient:
    def __init__(self):
        self.calls: list[dict] = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        is_heavy = any("heavy_worker" in m.get("content", "") for m in kwargs.get("messages", []))
        content = "fast-result-content" if not is_heavy else "heavy-result-content"
        return ModelResponse(
            model=kwargs["model"],
            requested_think_mode=str(kwargs.get("think_mode", "false")),
            effective_think_mode=str(kwargs.get("think_mode", "false")),
            content=content,
            metrics=OllamaMetrics(eval_count=10, eval_duration_ns=1_000_000_000, total_duration_ns=1_200_000_000),
        )


def _cfg() -> Configuration:
    cfg = Configuration(project_root=PROJECT_ROOT)
    cfg.load()
    return cfg


def _task(worker_class: str) -> TaskDefinition:
    return TaskDefinition(
        id="handoff_scenario",
        team="linux_infrastructure",
        role="fast_worker" if worker_class == "fast" else "escalation_reviewer",
        prompt=f"{worker_class}_worker prompt",
        think_modes=["false"],
        verification_classification="rubric_assisted",
        rubric_finalization="deterministic",
        rubric_rules=[
            {
                "rule_id": "r1",
                "description": "required",
                "type": "required",
                "weight": 1.0,
                "matcher": {"type": "phrase_aliases", "phrases": ["result"]},
            }
        ],
        comparison_id="linux_handoff_001",
        comparison_track="handoff",
        worker_class=worker_class,
    )


def test_handoff_executes_fast_before_heavy_and_injects_payload(tmp_path: Path):
    cfg = _cfg()
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    runner = AuditionRunner(cfg, run_dir, profile="smoke")
    fake = _FakeClient()
    runner.client = fake

    fast = _task("fast")
    heavy = _task("heavy")
    tasks = [fast, heavy]
    plan_rows = [
        {
            "plan_row_id": "heavy-row",
            "team": heavy.team,
            "role": heavy.role,
            "task_id": heavy.id,
            "task_version": heavy.task_version,
            "model": "gemma4:26b",
            "full_model_digest": "d-heavy",
            "requested_think_mode": "false",
            "structured_output_mode": "prompt_only",
            "temperature": 0.0,
            "num_ctx": 8192,
            "num_predict": 512,
            "schema_hash": "",
            "fixture_hashes": {},
            "comparison_id": "linux_handoff_001",
            "comparison_track": "handoff",
            "scenario_content_hash": "scenario-hash",
            "fast_plan_row_id": "fast-row",
        },
        {
            "plan_row_id": "fast-row",
            "team": fast.team,
            "role": fast.role,
            "task_id": fast.id,
            "task_version": fast.task_version,
            "model": "gemma4:12b",
            "full_model_digest": "d-fast",
            "requested_think_mode": "false",
            "structured_output_mode": "prompt_only",
            "temperature": 0.0,
            "num_ctx": 8192,
            "num_predict": 512,
            "schema_hash": "",
            "fixture_hashes": {},
            "comparison_id": "linux_handoff_001",
            "comparison_track": "handoff",
            "scenario_content_hash": "scenario-hash",
            "fast_plan_row_id": "",
        },
    ]

    out = list(runner.run_plan_rows(tasks=tasks, plan_rows=plan_rows))

    assert len(out) == 2
    assert len(fake.calls) == 2
    assert fake.calls[0]["model"] == "gemma4:12b"
    assert fake.calls[1]["model"] == "gemma4:26b"

    heavy_msgs = fake.calls[1]["messages"]
    heavy_text = "\n".join(m.get("content", "") for m in heavy_msgs)
    assert "fast-result-content" in heavy_text
    assert "fast_result_identity" in heavy_text

    heavy_result = [r for r in out if r.task.worker_class == "heavy"][0]
    assert heavy_result.response.request_payload.get("handoff_context", {}).get("fast_response") == "fast-result-content"
    assert heavy_result.identity.handoff_fast_identity_key
    assert heavy_result.identity.handoff_fast_response_hash
    assert heavy_result.identity.comparison_scenario_hash


def test_handoff_uses_actual_fast_output_not_placeholder(tmp_path: Path):
    cfg = _cfg()
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    runner = AuditionRunner(cfg, run_dir, profile="smoke")
    fake = _FakeClient()
    runner.client = fake

    fast = _task("fast")
    heavy = _task("heavy")
    tasks = [fast, heavy]
    plan_rows = [
        {
            "plan_row_id": "fast-row",
            "team": fast.team,
            "role": fast.role,
            "task_id": fast.id,
            "task_version": fast.task_version,
            "model": "gemma4:12b",
            "full_model_digest": "d-fast",
            "requested_think_mode": "false",
            "structured_output_mode": "prompt_only",
            "temperature": 0.0,
            "num_ctx": 8192,
            "num_predict": 512,
            "schema_hash": "",
            "fixture_hashes": {},
            "comparison_id": "linux_handoff_001",
            "comparison_track": "handoff",
            "scenario_content_hash": "scenario-hash",
            "fast_plan_row_id": "",
        },
        {
            "plan_row_id": "heavy-row",
            "team": heavy.team,
            "role": heavy.role,
            "task_id": heavy.id,
            "task_version": heavy.task_version,
            "model": "gemma4:26b",
            "full_model_digest": "d-heavy",
            "requested_think_mode": "false",
            "structured_output_mode": "prompt_only",
            "temperature": 0.0,
            "num_ctx": 8192,
            "num_predict": 512,
            "schema_hash": "",
            "fixture_hashes": {},
            "comparison_id": "linux_handoff_001",
            "comparison_track": "handoff",
            "scenario_content_hash": "scenario-hash",
            "fast_plan_row_id": "fast-row",
        },
    ]

    out = list(runner.run_plan_rows(tasks=tasks, plan_rows=plan_rows))
    heavy_result = [r for r in out if r.task.worker_class == "heavy"][0]

    payload = heavy_result.response.request_payload.get("handoff_context", {})
    assert payload.get("fast_response") == "fast-result-content"
    assert "TODO" not in payload.get("fast_response", "")
