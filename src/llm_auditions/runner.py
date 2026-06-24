"""Audition runner — orchestrates task execution, resume logic, and result recording."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import socket
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from jsonschema import validate as jsonschema_validate

from .comparisons import load_comparison_scenario, render_shared_scenario
from .configuration import Configuration
from .models import (
    EnvironmentInfo,
    ModelInfo,
    ModelResponse,
    PlannedRequest,
    ResultIdentity,
    RunManifest,
    RunState,
    TaskDefinition,
    TaskSnapshot,
    TaskResult,
)
from .ollama_client import OllamaClient
from .scoring import score_result
from .versioning import (
    ENGINE_VERSION,
    REPORT_VERSION,
    SCORING_VERSION,
    TASK_SUITE_VERSION,
    VERIFIER_VERSION,
    execution_source_hashes,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
IMMUTABLE_RUN_FILES = {
    "run_manifest.json",
    "environment.json",
    "model_inventory.json",
    "task_manifest.json",
}


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _json_sha256(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:16]


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _atomic_write(path: Path, content: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        fh.write(content)
        fh.flush()
        os.fsync(fh.fileno())
    tmp.rename(path)


def _capture_command(cmd: list[str], timeout: int = 10) -> str:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return result.stdout.strip()
    except Exception as exc:
        return f"capture failed: {exc}"


def _collect_environment(ollama_version: str, project_root: Path) -> EnvironmentInfo:
    env = EnvironmentInfo()
    env.date_iso = datetime.now(timezone.utc).isoformat()
    env.timezone = str(datetime.now(timezone.utc).astimezone().tzname())
    try:
        env.hostname = socket.gethostname()
    except Exception:
        env.hostname = "unknown"
    env.os_release = _capture_command(["cat", "/etc/os-release"])
    env.kernel = _capture_command(["uname", "-r"])
    env.cpu_model = _capture_command(
        ["bash", "-c", "grep 'model name' /proc/cpuinfo | head -1 | cut -d: -f2"]
    )
    env.cpu_count = os.cpu_count() or 0
    env.memory_total = _capture_command(
        ["bash", "-c", "grep MemTotal /proc/meminfo | awk '{print $2, $3}'"]
    )
    env.gpu_model = _capture_command(
        ["bash", "-c", "nvidia-smi --query-gpu=gpu_name --format=csv,noheader 2>/dev/null | head -1"]
    )
    env.gpu_vram = _capture_command(
        ["bash", "-c", "nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1"]
    )
    env.nvidia_driver = _capture_command(
        ["bash", "-c", "nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1"]
    )
    env.cuda_version = _capture_command(
        ["bash", "-c", "nvidia-smi | grep 'CUDA Version' | awk '{print $NF}' 2>/dev/null"]
    )
    env.ollama_version = ollama_version
    env.python_version = platform.python_version()
    env.project_version = "1.0.0"
    env.git_commit = _capture_command(["git", "-C", str(project_root), "rev-parse", "--short", "HEAD"])
    env.free_output = _capture_command(["free", "-h"])
    env.uptime_output = _capture_command(["uptime"])
    env.nvidia_smi_output = _capture_command(["nvidia-smi"])
    env.df_output = _capture_command(["df", "-h", str(project_root)])
    return env


def _config_hashes(project_root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    config_dir = project_root / "config"
    if not config_dir.exists():
        return hashes
    for p in sorted(config_dir.rglob("*.yaml")):
        hashes[str(p.relative_to(project_root))] = _sha256_file(p)
    return hashes


def _git_status(project_root: Path) -> tuple[str, bool]:
    commit = _capture_command(["git", "-C", str(project_root), "rev-parse", "HEAD"])
    dirty = bool(_capture_command(["git", "-C", str(project_root), "status", "--porcelain"]).strip())
    return commit, dirty


class AuditionRunner:
    """Runs audition tasks, records results, and supports immutable resume."""

    def __init__(
        self,
        config: Configuration,
        run_dir: Path,
        profile: str = "smoke",
    ) -> None:
        self.config = config
        self.run_dir = run_dir
        self.profile = profile
        self.client = OllamaClient(
            base_url=config.ollama_base_url,
            timeout=config.request_timeout,
            keep_alive=config.default_keep_alive,
        )
        self._inventory_by_name: dict[str, ModelInfo] = {}
        self._manifest: RunManifest | None = None
        self._completed_keys: set[str] = set()
        self._run_order = 0

    def _task_snapshot_dir(self) -> Path:
        return self.run_dir / "task_snapshots"

    def _task_ref(self, task: TaskDefinition) -> str:
        return f"{task.team}::{task.role}::{task.id}::{task.task_version}"

    def _task_hash(self, task: TaskDefinition) -> str:
        return hashlib.sha256(json.dumps(task.model_dump(mode="json"), sort_keys=True).encode()).hexdigest()[:16]

    def _comparison_pair_key(self, task: TaskDefinition, row: PlannedRequest) -> tuple[str, str, str, str, str]:
        return (
            task.comparison_id,
            task.comparison_track,
            str(row.requested_think_mode),
            str(row.structured_output_mode),
            row.scenario_content_hash or self._comparison_scenario_hash(task),
        )

    def _comparison_scenario_hash(self, task: TaskDefinition) -> str:
        if not task.comparison_scenario_ref.strip():
            return ""
        try:
            scenario = load_comparison_scenario(PROJECT_ROOT, task.comparison_scenario_ref)
            return scenario["scenario_content_hash"]
        except Exception:
            return ""

    def _handoff_context_from_result(
        self,
        result: TaskResult,
        scenario_hash: str,
        fast_plan_row_id: str = "",
    ) -> dict[str, str]:
        fast_response = (result.response.content or "").strip()
        fast_response_hash = hashlib.sha256(fast_response.encode()).hexdigest()[:16] if fast_response else ""
        return {
            "fast_plan_row_id": fast_plan_row_id,
            "fast_identity_key": result.identity.key(),
            "fast_model": result.identity.model_name,
            "fast_model_digest": result.identity.model_digest,
            "fast_response": fast_response,
            "fast_response_hash": fast_response_hash,
            "comparison_scenario_hash": scenario_hash,
        }

    def _write_task_snapshots(self, tasks: list[TaskDefinition], plan_rows: list[dict[str, Any]]) -> dict[str, str]:
        snapshot_dir = self._task_snapshot_dir()
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        selected = {(row["team"], row["role"], row["task_id"], row["task_version"]) for row in plan_rows}
        hashes: dict[str, str] = {}
        for task in tasks:
            key = (task.team, task.role, task.id, task.task_version)
            if key not in selected:
                continue
            scenario_version = ""
            scenario_content_hash = ""
            comparison_information_mode = ""
            if task.comparison_id and task.comparison_scenario_ref:
                scenario_info = load_comparison_scenario(PROJECT_ROOT, task.comparison_scenario_ref)
                scenario_version = scenario_info["scenario_version"]
                scenario_content_hash = scenario_info["scenario_content_hash"]
                comparison_information_mode = scenario_info["comparison_information_mode"]
            schema_content: dict[str, Any] = {}
            schema_hash = ""
            if task.required_json_schema:
                schema_path = PROJECT_ROOT / "schemas" / f"{task.required_json_schema}.schema.json"
                if schema_path.exists():
                    schema_content = json.loads(schema_path.read_text(encoding="utf-8"))
                    schema_hash = _sha256_file(schema_path)
            snapshot = TaskSnapshot(
                team=task.team,
                role=task.role,
                task_id=task.id,
                task_version=task.task_version,
                task_hash=self._task_hash(task),
                schema_name=task.required_json_schema,
                schema_hash=schema_hash,
                schema_content=schema_content,
                fixture_hashes=task.fixture_hashes(PROJECT_ROOT),
                comparison_scenario_ref=task.comparison_scenario_ref,
                scenario_version=scenario_version,
                scenario_content_hash=scenario_content_hash,
                comparison_information_mode=comparison_information_mode,
                task=task.model_dump(mode="json"),
            )
            path = snapshot_dir / f"{task.team}__{task.role}__{task.id}__{task.task_version}.json"
            _atomic_write(path, snapshot.model_dump_json(indent=2))
            hashes[self._task_ref(task)] = snapshot.task_hash
        return hashes

    def has_existing_manifest(self) -> bool:
        return (self.run_dir / "run_manifest.json").exists()

    def setup_new_run(
        self,
        tasks: list[TaskDefinition],
        plan_rows: list[dict[str, Any]],
    ) -> RunManifest:
        """Create a new run and write immutable manifests exactly once."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        for name in IMMUTABLE_RUN_FILES:
            if (self.run_dir / name).exists():
                raise RuntimeError(f"Refusing new-run setup: immutable file already exists: {name}")

        ollama_version = self.client.get_version()
        live_models = self.client.discover_model_capabilities()
        self._inventory_by_name = {m.name: m for m in live_models}
        env = _collect_environment(ollama_version, PROJECT_ROOT)

        teams_included = sorted({r["team"] for r in plan_rows})
        roles_included = sorted({f"{r['team']}.{r['role']}" for r in plan_rows})
        models_included = sorted({r["model"] for r in plan_rows})
        requested_think_modes = sorted({str(r["requested_think_mode"]).lower() for r in plan_rows})
        structured_output_modes = sorted({str(r["structured_output_mode"]) for r in plan_rows})

        task_hashes = self._write_task_snapshots(tasks, plan_rows)
        fixture_hashes: dict[str, str] = {}
        schema_hashes: dict[str, str] = {}
        for t in tasks:
            fixture_hashes.update(t.fixture_hashes(PROJECT_ROOT))
            if t.required_json_schema:
                schema_path = PROJECT_ROOT / "schemas" / f"{t.required_json_schema}.schema.json"
                if schema_path.exists():
                    schema_hashes[t.required_json_schema] = _sha256_file(schema_path)

        model_digests: dict[str, str] = {}
        configured_digests = {m.name: m.full_digest or m.id for m in self.config.get_configured_models()}
        for model_name in models_included:
            inv = self._inventory_by_name.get(model_name)
            model_digests[model_name] = (inv.full_digest if inv else "") or configured_digests.get(model_name, "unknown")

        task_manifest = {
            "run_id": str(uuid.uuid4()),
            "profile": self.profile,
            "requests": [
                {
                    "plan_row_id": row.get("plan_row_id", ""),
                    "team": row["team"],
                    "role": row["role"],
                    "task_id": row["task_id"],
                    "task_version": row.get("task_version", "v1"),
                    "model": row["model"],
                    "full_model_digest": row.get("full_model_digest", ""),
                    "requested_think_mode": row.get("requested_think_mode", "false"),
                    "structured_output_mode": row.get("structured_output_mode", "prompt_only"),
                    "temperature": row.get("temperature", 0.0),
                    "num_ctx": row.get("num_ctx", 8192),
                    "num_predict": row.get("num_predict", 4096),
                    "schema_hash": row.get("schema_hash", ""),
                    "fixture_hashes": row.get("fixture_hashes", {}),
                    "comparison_id": row.get("comparison_id", ""),
                    "comparison_track": row.get("comparison_track", ""),
                    "comparison_scenario_ref": row.get("comparison_scenario_ref", ""),
                    "scenario_version": row.get("scenario_version", ""),
                    "scenario_content_hash": row.get("scenario_content_hash", ""),
                    "comparison_information_mode": row.get("comparison_information_mode", ""),
                    "fast_plan_row_id": row.get("fast_plan_row_id", ""),
                }
                for row in plan_rows
            ],
        }
        task_manifest_hash = _json_sha256(task_manifest)
        execution_plan_hash = _json_sha256(plan_rows)
        git_commit, git_dirty = _git_status(PROJECT_ROOT)

        manifest = RunManifest(
            run_id=task_manifest["run_id"],
            created_at=datetime.now(timezone.utc).isoformat(),
            profile=self.profile,
            engine_version=ENGINE_VERSION,
            scoring_version=SCORING_VERSION,
            verifier_version=VERIFIER_VERSION,
            report_version=REPORT_VERSION,
            task_suite_version=TASK_SUITE_VERSION,
            ollama_version=ollama_version,
            ollama_base_url=self.config.ollama_base_url,
            git_commit=git_commit,
            git_dirty=git_dirty,
            models=live_models,
            model_digests=model_digests,
            environment=env,
            task_count=len({(r["team"], r["role"], r["task_id"], r["task_version"]) for r in plan_rows}),
            candidate_count=len({(r["team"], r["role"], r["model"]) for r in plan_rows}),
            model_count=len(models_included),
            request_count=len(plan_rows),
            teams_included=teams_included,
            roles_included=roles_included,
            models_included=models_included,
            requested_think_modes=requested_think_modes,
            structured_output_modes=structured_output_modes,
            task_hashes=task_hashes,
            config_hashes=_config_hashes(PROJECT_ROOT),
            schema_hashes=schema_hashes,
            fixture_hashes=fixture_hashes,
            execution_source_hashes=execution_source_hashes(PROJECT_ROOT),
            task_manifest_hash=task_manifest_hash,
            execution_plan_hash=execution_plan_hash,
        )

        self._validate_manifest(manifest)
        self._manifest = manifest

        state = RunState(
            run_id=manifest.run_id,
            profile=self.profile,
            started_at=manifest.created_at,
            updated_at=manifest.created_at,
        )

        _atomic_write(self.run_dir / "run_manifest.json", manifest.model_dump_json(indent=2))
        _atomic_write(self.run_dir / "environment.json", env.model_dump_json(indent=2))
        _atomic_write(self.run_dir / "model_inventory.json", json.dumps([m.model_dump() for m in live_models], indent=2))
        _atomic_write(self.run_dir / "task_manifest.json", json.dumps(task_manifest, indent=2))
        _atomic_write(self.run_dir / "run_state.json", state.model_dump_json(indent=2))

        logger.info("New run initialized in %s", self.run_dir)
        return manifest

    def load_existing_run(self) -> RunManifest:
        """Load immutable run context without rewriting immutable files."""
        manifest_path = self.run_dir / "run_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError("run_manifest.json not found for resume")

        self._manifest = RunManifest.model_validate_json(manifest_path.read_text())

        version_mismatches: list[str] = []
        expected_versions = {
            "engine_version": ENGINE_VERSION,
            "task_suite_version": TASK_SUITE_VERSION,
            "scoring_version": SCORING_VERSION,
            "verifier_version": VERIFIER_VERSION,
            "report_version": REPORT_VERSION,
        }
        for name, expected in expected_versions.items():
            current = str(getattr(self._manifest, name, ""))
            if current != expected:
                version_mismatches.append(f"{name}: stored={current} current={expected}")

        live_models = self.client.list_models()
        self._inventory_by_name = {m.name: m for m in live_models}
        live_digests = {m.name: (m.full_digest or m.id) for m in live_models}

        digest_mismatches: list[str] = []
        for model_name, stored_digest in self._manifest.model_digests.items():
            current_digest = live_digests.get(model_name)
            if not current_digest:
                digest_mismatches.append(
                    "\n".join(
                        [
                            f"model: {model_name}",
                            f"stored digest: {stored_digest}",
                            "current digest: missing",
                            "Create a new run directory.",
                        ]
                    )
                )
                continue
            if stored_digest != current_digest:
                digest_mismatches.append(
                    "\n".join(
                        [
                            f"model: {model_name}",
                            f"stored digest: {stored_digest}",
                            f"current digest: {current_digest}",
                            "Create a new run directory.",
                        ]
                    )
                )

        task_manifest_path = self.run_dir / "task_manifest.json"
        if not task_manifest_path.exists():
            raise RuntimeError("Resume refused: missing immutable task_manifest.json")
        task_manifest = json.loads(task_manifest_path.read_text(encoding="utf-8"))
        current_task_manifest_hash = _json_sha256(task_manifest)
        current_plan_hash = _json_sha256(task_manifest.get("requests", []))
        hash_mismatches: list[str] = []
        if self._manifest.task_manifest_hash and current_task_manifest_hash != self._manifest.task_manifest_hash:
            hash_mismatches.append(
                f"task_manifest_hash: stored={self._manifest.task_manifest_hash} current={current_task_manifest_hash}"
            )
        if self._manifest.execution_plan_hash and current_plan_hash != self._manifest.execution_plan_hash:
            hash_mismatches.append(
                f"execution_plan_hash: stored={self._manifest.execution_plan_hash} current={current_plan_hash}"
            )

        current_config_hashes = _config_hashes(PROJECT_ROOT)
        config_mismatches = [
            f"{path}: stored={stored} current={current_config_hashes.get(path, 'missing')}"
            for path, stored in self._manifest.config_hashes.items()
            if current_config_hashes.get(path) != stored
        ]
        current_source_hashes = execution_source_hashes(PROJECT_ROOT)
        source_hash_mismatches = [
            f"{name}: stored={stored} current={current_source_hashes.get(name, 'missing')}"
            for name, stored in (self._manifest.execution_source_hashes or {}).items()
            if current_source_hashes.get(name) != stored
        ]

        if digest_mismatches or config_mismatches or version_mismatches or hash_mismatches or source_hash_mismatches:
            details = []
            if digest_mismatches:
                details.append("model digests changed:\n" + "\n---\n".join(digest_mismatches))
            if config_mismatches:
                details.append("config hashes changed: " + "; ".join(config_mismatches))
            if version_mismatches:
                details.append("version mismatch: " + "; ".join(version_mismatches))
            if hash_mismatches:
                details.append("manifest/hash mismatch: " + "; ".join(hash_mismatches))
            if source_hash_mismatches:
                details.append("execution source hash mismatch: " + "; ".join(source_hash_mismatches))
            raise RuntimeError("Resume refused: " + " | ".join(details))

        current_tasks = {
            self._task_ref(task): task
            for task in __import__("llm_auditions.task_loader", fromlist=["load_tasks_from_dir"]).load_tasks_from_dir(PROJECT_ROOT / "fixtures")
        }
        snapshot_mismatches: list[str] = []
        for path in sorted(self._task_snapshot_dir().glob("*.json")):
            snapshot = TaskSnapshot.model_validate_json(path.read_text(encoding="utf-8"))
            task_ref = f"{snapshot.team}::{snapshot.role}::{snapshot.task_id}::{snapshot.task_version}"
            current_task = current_tasks.get(task_ref)
            if current_task is None:
                snapshot_mismatches.append(f"missing task source for {task_ref}")
                continue
            if snapshot.task_hash != self._task_hash(current_task):
                snapshot_mismatches.append(f"task changed for {task_ref}")
            if snapshot.schema_name:
                schema_path = PROJECT_ROOT / "schemas" / f"{snapshot.schema_name}.schema.json"
                current_schema_hash = _sha256_file(schema_path) if schema_path.exists() else "missing"
                if current_schema_hash != snapshot.schema_hash:
                    snapshot_mismatches.append(f"schema changed for {task_ref}")
            current_fixture_hashes = current_task.fixture_hashes(PROJECT_ROOT)
            if current_fixture_hashes != snapshot.fixture_hashes:
                snapshot_mismatches.append(f"fixtures changed for {task_ref}")
            if snapshot.comparison_scenario_ref:
                try:
                    current_scenario = load_comparison_scenario(PROJECT_ROOT, snapshot.comparison_scenario_ref)
                    if current_scenario["scenario_content_hash"] != snapshot.scenario_content_hash:
                        snapshot_mismatches.append(f"scenario changed for {task_ref}")
                except Exception:
                    snapshot_mismatches.append(f"scenario missing or invalid for {task_ref}")
        if snapshot_mismatches:
            raise RuntimeError("Resume refused: " + "; ".join(snapshot_mismatches))

        self.load_completed()
        return self._manifest

    def run_plan_rows(
        self,
        tasks: list[TaskDefinition],
        plan_rows: list[dict[str, Any]],
        team_weights: dict[str, dict[str, float]] | None = None,
    ) -> Iterator[TaskResult]:
        tasks_by_key = {(t.team, t.role, t.id, t.task_version): t for t in tasks}
        schemas_dir = PROJECT_ROOT / "schemas"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        events_path = self.run_dir / "events.jsonl"
        events_fh = events_path.open("a", encoding="utf-8")

        planned_rows = [PlannedRequest.model_validate(row) for row in plan_rows]
        row_index_by_id = {row.plan_row_id: idx for idx, row in enumerate(planned_rows) if row.plan_row_id}
        handoff_context_by_fast_row_id: dict[str, dict[str, str]] = {}

        executed_indices: set[int] = set()
        try:
            for idx, row in enumerate(planned_rows):
                if idx in executed_indices:
                    continue
                task = tasks_by_key[(row.team, row.role, row.task_id, row.task_version)]

                handoff_context: dict[str, str] | None = None
                if task.comparison_track == "handoff" and task.worker_class == "heavy":
                    if not row.fast_plan_row_id:
                        self._write_event(
                            events_fh,
                            {
                                "identity_key": "",
                                "status": "invalid_dependency",
                                "team": row.team,
                                "role": row.role,
                                "task_id": row.task_id,
                                "model": row.model,
                                "requested_think_mode": row.requested_think_mode,
                                "structured_output_mode": row.structured_output_mode,
                                "run_order": self._run_order,
                                "reason": "missing_fast_plan_row_id",
                            },
                        )
                        executed_indices.add(idx)
                        continue

                    if row.fast_plan_row_id not in handoff_context_by_fast_row_id:
                        fast_idx = row_index_by_id.get(row.fast_plan_row_id)
                        if fast_idx is None:
                            self._write_event(
                                events_fh,
                                {
                                    "identity_key": "",
                                    "status": "invalid_dependency",
                                    "team": row.team,
                                    "role": row.role,
                                    "task_id": row.task_id,
                                    "model": row.model,
                                    "requested_think_mode": row.requested_think_mode,
                                    "structured_output_mode": row.structured_output_mode,
                                    "run_order": self._run_order,
                                    "reason": "fast_plan_row_not_found",
                                },
                            )
                            executed_indices.add(idx)
                            continue

                        if fast_idx not in executed_indices:
                            fast_row = planned_rows[fast_idx]
                            fast_task = tasks_by_key[(fast_row.team, fast_row.role, fast_row.task_id, fast_row.task_version)]
                            self._run_order += 1
                            fast_results = list(
                                self._run_single_planned_request(
                                    task=fast_task,
                                    row=fast_row,
                                    schemas_dir=schemas_dir,
                                    team_weights=(team_weights or {}).get(f"{fast_task.team}.{fast_task.role}"),
                                    events_fh=events_fh,
                                    handoff_context=None,
                                    comparison_scenario_hash=fast_row.scenario_content_hash or self._comparison_scenario_hash(fast_task),
                                )
                            )
                            executed_indices.add(fast_idx)
                            for produced in fast_results:
                                ctx = self._handoff_context_from_result(
                                    produced,
                                    fast_row.scenario_content_hash or self._comparison_scenario_hash(fast_task),
                                    fast_plan_row_id=fast_row.plan_row_id,
                                )
                                handoff_context_by_fast_row_id[fast_row.plan_row_id] = ctx
                                yield produced

                        if row.fast_plan_row_id not in handoff_context_by_fast_row_id:
                            fast_row = planned_rows[fast_idx]
                            fast_task = tasks_by_key[(fast_row.team, fast_row.role, fast_row.task_id, fast_row.task_version)]
                            loaded_ctx = self._load_handoff_context_from_saved_fast_result(
                                fast_task=fast_task,
                                fast_row=fast_row,
                                schemas_dir=schemas_dir,
                            )
                            if loaded_ctx:
                                handoff_context_by_fast_row_id[fast_row.plan_row_id] = loaded_ctx

                    handoff_context = handoff_context_by_fast_row_id.get(row.fast_plan_row_id)
                    if not handoff_context:
                        self._write_event(
                            events_fh,
                            {
                                "identity_key": "",
                                "status": "invalid_dependency",
                                "team": row.team,
                                "role": row.role,
                                "task_id": row.task_id,
                                "model": row.model,
                                "requested_think_mode": row.requested_think_mode,
                                "structured_output_mode": row.structured_output_mode,
                                "run_order": self._run_order,
                                "reason": "fast_artifact_missing_or_inconsistent",
                            },
                        )
                        executed_indices.add(idx)
                        continue

                    handoff_context = dict(handoff_context)
                    handoff_context["fast_plan_row_id"] = row.fast_plan_row_id
                    handoff_context["heavy_model"] = row.model
                    handoff_context["heavy_model_digest"] = row.full_model_digest

                self._run_order += 1
                produced_rows = list(
                    self._run_single_planned_request(
                        task=task,
                        row=row,
                        schemas_dir=schemas_dir,
                        team_weights=(team_weights or {}).get(f"{task.team}.{task.role}"),
                        events_fh=events_fh,
                        handoff_context=handoff_context,
                        comparison_scenario_hash=row.scenario_content_hash or self._comparison_scenario_hash(task),
                    )
                )
                executed_indices.add(idx)
                if task.comparison_track == "handoff" and task.worker_class == "fast":
                    for produced in produced_rows:
                        if row.plan_row_id:
                            handoff_context_by_fast_row_id[row.plan_row_id] = self._handoff_context_from_result(
                                produced,
                                row.scenario_content_hash or self._comparison_scenario_hash(task),
                                fast_plan_row_id=row.plan_row_id,
                            )
                for produced in produced_rows:
                    yield produced
        finally:
            events_fh.close()

    def _load_handoff_context_from_saved_fast_result(
        self,
        fast_task: TaskDefinition,
        fast_row: PlannedRequest,
        schemas_dir: Path,
    ) -> dict[str, str] | None:
        expected_scenario_hash = fast_row.scenario_content_hash or self._comparison_scenario_hash(fast_task)
        schema = self._load_schema_for_task(fast_task, schemas_dir)
        pre_identity = self._make_identity(
            task=fast_task,
            model_name=fast_row.model,
            requested_think_mode=fast_row.requested_think_mode,
            effective_think_mode=fast_row.requested_think_mode,
            schema=schema,
            model_response=None,
            handoff_context=None,
            comparison_scenario_hash=expected_scenario_hash,
        )
        pre_key = pre_identity.pre_request_key()
        fallback_candidates: list[TaskResult] = []
        for result_path in sorted(self.run_dir.rglob("*.result.json")):
            try:
                result = TaskResult.model_validate_json(result_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if result.status not in ("completed", "unsupported_mode"):
                continue
            if result.identity.pre_request_key() != pre_key:
                if not (
                    result.identity.team == fast_task.team
                    and result.identity.role == fast_task.role
                    and result.identity.task_id == fast_task.id
                    and result.identity.model_name == fast_row.model
                    and result.identity.requested_think_mode == fast_row.requested_think_mode
                    and (
                        result.identity.scenario_content_hash == expected_scenario_hash
                        or result.identity.comparison_scenario_hash == expected_scenario_hash
                    )
                ):
                    continue
                fallback_candidates.append(result)
                continue
            if fast_row.full_model_digest and result.identity.model_digest not in (fast_row.full_model_digest, "unknown"):
                return None
            ctx = self._handoff_context_from_result(
                result,
                expected_scenario_hash,
                fast_plan_row_id=fast_row.plan_row_id,
            )
            if not ctx.get("fast_response"):
                return None
            return ctx

        if fallback_candidates:
            result = fallback_candidates[-1]
            if fast_row.full_model_digest and result.identity.model_digest not in (fast_row.full_model_digest, "unknown"):
                return None
            ctx = self._handoff_context_from_result(
                result,
                expected_scenario_hash,
                fast_plan_row_id=fast_row.plan_row_id,
            )
            if not ctx.get("fast_response"):
                return None
            return ctx
        return None

    def _run_single_planned_request(
        self,
        task: TaskDefinition,
        row: PlannedRequest,
        schemas_dir: Path,
        team_weights: dict[str, float] | None,
        events_fh: Any,
        handoff_context: dict[str, str] | None = None,
        comparison_scenario_hash: str = "",
    ) -> Iterator[TaskResult]:
        schema = self._load_schema_for_task(task, schemas_dir)
        messages, prompt_components = self._build_messages(task, handoff_context=handoff_context)
        pre_identity = self._make_identity(
            task=task,
            model_name=row.model,
            requested_think_mode=row.requested_think_mode,
            effective_think_mode=row.requested_think_mode,
            schema=schema,
            model_response=None,
            handoff_context=handoff_context,
            comparison_scenario_hash=comparison_scenario_hash,
        )
        pre_key = pre_identity.pre_request_key()
        if pre_key in self._completed_keys:
            self._write_event(
                events_fh,
                {
                    "identity_key": pre_key,
                    "status": "skipped",
                    "team": row.team,
                    "role": row.role,
                    "task_id": row.task_id,
                    "model": row.model,
                    "requested_think_mode": row.requested_think_mode,
                    "structured_output_mode": row.structured_output_mode,
                    "comparison_id": row.comparison_id,
                    "comparison_track": row.comparison_track,
                    "scenario_content_hash": row.scenario_content_hash,
                    "scenario_version": row.scenario_version,
                    "run_order": self._run_order,
                    "reason": "completed_identity",
                },
            )
            return

        response = self.client.chat(
            model=row.model,
            messages=messages,
            think_mode=row.requested_think_mode,
            num_ctx=row.num_ctx,
            num_predict=row.num_predict,
            temperature=row.temperature,
            structured_output_mode=row.structured_output_mode,
            schema=schema,
        )
        response.effective_prompt = {
            "messages": messages,
            "structured_output_mode": row.structured_output_mode,
            "schema_name": task.required_json_schema,
            "prompt_components": prompt_components,
        }
        if handoff_context:
            response.effective_prompt["fast_worker_response"] = handoff_context.get("fast_response", "")
            response.effective_prompt["handoff_context"] = handoff_context
            response.request_payload["handoff_context"] = handoff_context
        if prompt_components.get("handoff_payload"):
            response.request_payload["handoff_payload"] = prompt_components["handoff_payload"]
            response.effective_prompt["handoff_payload"] = prompt_components["handoff_payload"]
        identity = self._make_identity(
            task=task,
            model_name=row.model,
            requested_think_mode=row.requested_think_mode,
            effective_think_mode=response.effective_think_mode,
            schema=schema,
            model_response=response,
            handoff_context=handoff_context,
            comparison_scenario_hash=comparison_scenario_hash,
        )
        status = "completed"
        if response.raw_json.get("_think_mode_unsupported"):
            status = "unsupported_mode"
        if response.error:
            status = "error"
        result = TaskResult(identity=identity, task=task, response=response, run_order=self._run_order, status=status)
        if status in ("completed", "unsupported_mode"):
            weights = team_weights or self.config.get_role_scoring_weights(task.team, task.role) or None
            result.scores = score_result(result, schemas_dir, weights)
            result.score_status = result.scores.score_status
            result.ranking_eligible = result.scores.ranking_eligible
            result.human_review_required = result.scores.human_review_required
        self._save_result(result)
        self._write_event(
            events_fh,
            {
                "identity_key": pre_key,
                "response_identity_key": identity.key(),
                "status": status,
                "team": row.team,
                "role": row.role,
                "task_id": row.task_id,
                "model": row.model,
                "requested_think_mode": row.requested_think_mode,
                "effective_think_mode": response.effective_think_mode,
                "think_mode_accepted": response.think_mode_accepted,
                "structured_output_mode": row.structured_output_mode,
                "comparison_id": row.comparison_id,
                "comparison_track": row.comparison_track,
                "scenario_content_hash": row.scenario_content_hash,
                "scenario_version": row.scenario_version,
                "comparison_information_mode": row.comparison_information_mode,
                "fast_plan_row_id": row.fast_plan_row_id,
                "run_order": self._run_order,
                "weighted_total": result.scores.weighted_total,
                "score_status": result.score_status,
                "ranking_eligible": result.ranking_eligible,
                "wall_clock_s": response.metrics.wall_clock_seconds,
                "error": response.error,
                "empty_final_content": response.empty_final_content,
                "truncated_length_stop": response.truncated_length_stop,
            },
        )
        self._update_run_state(pre_key, status)
        yield result

    def _validate_manifest(self, manifest: RunManifest) -> None:
        schema_path = PROJECT_ROOT / "schemas" / "run_manifest.schema.json"
        schema = json.loads(schema_path.read_text())
        jsonschema_validate(instance=manifest.model_dump(), schema=schema)

    def load_completed(self) -> None:
        state_path = self.run_dir / "run_state.json"
        if state_path.exists():
            try:
                state = RunState.model_validate_json(state_path.read_text())
                self._completed_keys = set(state.completed_identity_keys)
                logger.info("Resume state loaded: %d completed keys", len(self._completed_keys))
                return
            except Exception as exc:
                logger.warning("Failed to load run_state.json (%s), falling back to events", exc)

        events_path = self.run_dir / "events.jsonl"
        if not events_path.exists():
            return
        with events_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                    if event.get("status") in ("completed", "unsupported_mode"):
                        key = event.get("identity_key")
                        if key:
                            self._completed_keys.add(key)
                except json.JSONDecodeError:
                    pass

    def run_tasks(
        self,
        tasks: list[TaskDefinition],
        team_weights: dict[str, dict[str, float]] | None = None,
    ) -> Iterator[TaskResult]:
        schemas_dir = PROJECT_ROOT / "schemas"
        events_path = self.run_dir / "events.jsonl"
        events_fh = events_path.open("a", encoding="utf-8")

        try:
            for task in tasks:
                for think_mode in task.think_modes:
                    self._run_order += 1
                    yield from self._run_single(
                        task=task,
                        think_mode=str(think_mode).lower(),
                        schemas_dir=schemas_dir,
                        team_weights=(team_weights or {}).get(f"{task.team}.{task.role}"),
                        events_fh=events_fh,
                    )
        finally:
            events_fh.close()

    def _run_single(
        self,
        task: TaskDefinition,
        think_mode: str,
        schemas_dir: Path,
        team_weights: dict[str, float] | None,
        events_fh: Any,
    ) -> Iterator[TaskResult]:
        candidates = self._get_candidates(task)
        if not candidates:
            logger.warning("No candidates for task %s / %s / %s", task.team, task.role, task.id)
            return

        for model_name in candidates:
            schema = self._load_schema_for_task(task, schemas_dir)
            messages, prompt_components = self._build_messages(task)

            pre_identity = self._make_identity(
                task=task,
                model_name=model_name,
                requested_think_mode=think_mode,
                effective_think_mode=think_mode,
                schema=schema,
                model_response=None,
            )
            pre_key = pre_identity.pre_request_key()

            if pre_key in self._completed_keys:
                self._write_event(
                    events_fh,
                    {
                        "identity_key": pre_key,
                        "status": "skipped",
                        "team": task.team,
                        "role": task.role,
                        "task_id": task.id,
                        "model": model_name,
                        "requested_think_mode": str(think_mode),
                        "structured_output_mode": task.structured_output_mode,
                        "run_order": self._run_order,
                        "reason": "completed_identity",
                    },
                )
                continue

            response = self.client.chat(
                model=model_name,
                messages=messages,
                think_mode=think_mode,
                num_ctx=task.num_ctx,
                num_predict=task.num_predict,
                temperature=task.temperature,
                structured_output_mode=task.structured_output_mode,
                schema=schema,
            )
            response.effective_prompt = {
                "messages": messages,
                "structured_output_mode": task.structured_output_mode,
                "schema_name": task.required_json_schema,
                "prompt_components": prompt_components,
            }

            identity = self._make_identity(
                task=task,
                model_name=model_name,
                requested_think_mode=think_mode,
                effective_think_mode=response.effective_think_mode,
                schema=schema,
                model_response=response,
            )

            status = "completed"
            if response.raw_json.get("_think_mode_unsupported"):
                status = "unsupported_mode"
            if response.error:
                status = "error"

            result = TaskResult(
                identity=identity,
                task=task,
                response=response,
                run_order=self._run_order,
                status=status,
            )

            if status in ("completed", "unsupported_mode"):
                weights = team_weights or self.config.get_role_scoring_weights(task.team, task.role) or None
                result.scores = score_result(result, schemas_dir, weights)
                result.score_status = result.scores.score_status
                result.ranking_eligible = result.scores.ranking_eligible
                result.human_review_required = result.scores.human_review_required

            self._save_result(result)

            self._write_event(
                events_fh,
                {
                    "identity_key": pre_key,
                    "response_identity_key": identity.key(),
                    "status": status,
                    "team": task.team,
                    "role": task.role,
                    "task_id": task.id,
                    "model": model_name,
                    "requested_think_mode": str(think_mode),
                    "effective_think_mode": response.effective_think_mode,
                    "think_mode_accepted": response.think_mode_accepted,
                    "structured_output_mode": task.structured_output_mode,
                    "run_order": self._run_order,
                    "weighted_total": result.scores.weighted_total,
                    "score_status": result.score_status,
                    "ranking_eligible": result.ranking_eligible,
                    "wall_clock_s": response.metrics.wall_clock_seconds,
                    "error": response.error,
                    "empty_final_content": response.empty_final_content,
                    "truncated_length_stop": response.truncated_length_stop,
                },
            )

            self._update_run_state(pre_key, status)
            yield result

    def _write_event(self, events_fh: Any, payload: dict[str, Any]) -> None:
        events_fh.write(json.dumps(payload) + "\n")
        events_fh.flush()

    def _get_candidates(self, task: TaskDefinition) -> list[str]:
        return self.config.get_role_candidates(task.team, task.role) or []

    def _make_identity(
        self,
        task: TaskDefinition,
        model_name: str,
        requested_think_mode: str,
        effective_think_mode: str,
        schema: dict[str, Any] | None,
        model_response: ModelResponse | None,
        handoff_context: dict[str, str] | None = None,
        comparison_scenario_hash: str = "",
    ) -> ResultIdentity:
        inv = self._inventory_by_name.get(model_name)
        configured = {m.name: m.full_digest or m.id for m in self.config.get_configured_models()}
        digest = (inv.full_digest if inv else "") or configured.get(model_name, "unknown")

        schema_hash = ""
        if schema is not None:
            schema_hash = hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest()[:16]

        user_prompt = task.prompt
        if model_response and model_response.effective_prompt:
            try:
                user_prompt = "\n".join(
                    m.get("content", "")
                    for m in model_response.effective_prompt.get("messages", [])
                    if m.get("role") == "user"
                )
            except Exception:
                pass

        return ResultIdentity(
            team=task.team,
            role=task.role,
            task_id=task.id,
            task_version=task.task_version,
            model_name=model_name,
            model_digest=digest,
            requested_think_mode=str(requested_think_mode),
            effective_think_mode=str(effective_think_mode),
            think_mode_accepted=True if model_response is None else model_response.think_mode_accepted,
            structured_output_mode=task.structured_output_mode,
            schema_hash=schema_hash,
            temperature=task.temperature,
            num_ctx=task.num_ctx,
            num_predict=task.num_predict,
            system_prompt_hash=_sha256_text(task.system_prompt),
            user_prompt_hash=_sha256_text(user_prompt),
            fixture_hashes=task.fixture_hashes(PROJECT_ROOT),
            handoff_fast_identity_key=(handoff_context or {}).get("fast_identity_key", ""),
            handoff_fast_response_hash=(handoff_context or {}).get("fast_response_hash", ""),
            comparison_scenario_hash=comparison_scenario_hash,
            scenario_content_hash=comparison_scenario_hash,
        )

    def _build_messages(
        self,
        task: TaskDefinition,
        handoff_context: dict[str, str] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        messages: list[dict[str, Any]] = []

        if task.system_prompt:
            messages.append({"role": "system", "content": task.system_prompt})

        shared_scenario_content = ""
        scenario_content_hash = ""
        scenario_version = ""
        comparison_information_mode = ""
        if task.comparison_id and task.comparison_scenario_ref:
            scenario = load_comparison_scenario(PROJECT_ROOT, task.comparison_scenario_ref)
            shared_scenario_content = render_shared_scenario(scenario["payload"])
            scenario_content_hash = scenario["scenario_content_hash"]
            scenario_version = scenario["scenario_version"]
            comparison_information_mode = scenario["comparison_information_mode"]

        role_instruction = task.prompt.strip()
        output_contract = ""
        if task.structured_output_mode == "prompt_only" and task.required_json_schema:
            output_contract = "Return ONLY valid JSON (no markdown fences, no prose)."

        injected_chunks: list[str] = []
        for fp in task.fixture_paths:
            p = PROJECT_ROOT / fp
            if not p.exists():
                continue
            injected_chunks.append(f"[Fixture: {fp}]\n{p.read_text(encoding='utf-8')}")

        user_sections: list[str] = []
        if shared_scenario_content:
            user_sections.append("Shared scenario:\n" + shared_scenario_content)
        user_sections.append("Role instruction:\n" + role_instruction)
        if injected_chunks:
            user_sections.append("Reference fixtures:\n\n" + "\n\n".join(injected_chunks))

        handoff_payload: dict[str, Any] | None = None
        if handoff_context and task.comparison_track == "handoff" and task.worker_class == "heavy":
            handoff_payload = {
                "comparison_id": task.comparison_id,
                "comparison_track": task.comparison_track,
                "scenario_content_hash": scenario_content_hash,
                "fast_plan_row_id": handoff_context.get("fast_plan_row_id", ""),
                "fast_result_identity": handoff_context.get("fast_identity_key", ""),
                "fast_model": handoff_context.get("fast_model", ""),
                "fast_model_digest": handoff_context.get("fast_model_digest", ""),
                "fast_response_hash": handoff_context.get("fast_response_hash", ""),
                "fast_response": handoff_context.get("fast_response", ""),
                "heavy_model": handoff_context.get("heavy_model", ""),
                "heavy_model_digest": handoff_context.get("heavy_model_digest", ""),
                "heavy_instruction": role_instruction,
            }
            user_sections.append(
                "Escalation handoff from fast worker (use this as upstream context):\n"
                + json.dumps(handoff_payload, sort_keys=True)
            )

        if output_contract:
            user_sections.append("Output contract:\n" + output_contract)

        user_prompt = "\n\n".join(section.strip() for section in user_sections if section.strip())
        messages.append({"role": "user", "content": user_prompt})

        prompt_components = {
            "shared_scenario_content": shared_scenario_content,
            "role_instruction": role_instruction,
            "output_contract": output_contract,
            "scenario_content_hash": scenario_content_hash,
            "scenario_version": scenario_version,
            "comparison_information_mode": comparison_information_mode,
            "handoff_payload": handoff_payload,
        }
        return messages, prompt_components

    def _load_schema_for_task(self, task: TaskDefinition, schemas_dir: Path) -> dict[str, Any] | None:
        schema_name = task.required_json_schema
        if not schema_name:
            return None
        path = schemas_dir / f"{schema_name}.schema.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _update_run_state(self, identity_key: str, status: str) -> None:
        state_path = self.run_dir / "run_state.json"
        now = datetime.now(timezone.utc).isoformat()
        if state_path.exists():
            try:
                state = RunState.model_validate_json(state_path.read_text())
            except Exception:
                state = RunState(
                    run_id=(self._manifest.run_id if self._manifest else "unknown"),
                    profile=self.profile,
                    started_at=now,
                    updated_at=now,
                )
        else:
            state = RunState(
                run_id=(self._manifest.run_id if self._manifest else "unknown"),
                profile=self.profile,
                started_at=now,
                updated_at=now,
            )

        if status in ("completed", "unsupported_mode") and identity_key not in state.completed_identity_keys:
            state.completed_identity_keys.append(identity_key)
            self._completed_keys.add(identity_key)

        state.completed_count = len(state.completed_identity_keys)
        if status == "error":
            state.error_count += 1
        if status == "unsupported_mode":
            state.unsupported_mode_count += 1
        state.updated_at = now
        _atomic_write(state_path, state.model_dump_json(indent=2))

    def _save_result(self, result: TaskResult) -> None:
        team_dir = self.run_dir / result.task.team / result.task.role
        team_dir.mkdir(parents=True, exist_ok=True)

        safe_model = result.identity.model_name.replace(":", "_").replace("/", "_")
        base = (
            f"{result.task.id}__{safe_model}__"
            f"think_{result.identity.requested_think_mode}-effective_{result.identity.effective_think_mode}"
        )
        if result.task.comparison_track == "handoff" and result.task.worker_class == "heavy":
            dep = (result.identity.handoff_fast_identity_key or result.identity.handoff_fast_response_hash or "")[:12]
            if dep:
                base = f"{base}__dep_{dep}"

        _atomic_write(team_dir / f"{base}.request_payload.json", json.dumps(result.response.request_payload, indent=2))
        _atomic_write(team_dir / f"{base}.effective_prompt.json", json.dumps(result.response.effective_prompt, indent=2))
        _atomic_write(team_dir / f"{base}.raw.json", json.dumps(result.response.raw_json, indent=2))
        _atomic_write(team_dir / f"{base}.result.json", result.model_dump_json(indent=2))
        _atomic_write(team_dir / f"{base}.report.md", self._render_markdown(result))

    def _render_markdown(self, result: TaskResult) -> str:
        r = result
        t = result.task
        resp = result.response
        m = resp.metrics

        lines = [
            f"# {t.team} / {t.role} / {t.id}",
            f"**Model:** `{r.identity.model_name}`  ",
            f"**Think mode:** requested=`{r.identity.requested_think_mode}`, effective=`{r.identity.effective_think_mode}`  ",
            f"**Status:** {r.status}  ",
            f"**Run order:** {r.run_order}  ",
            "",
            "## Task",
            f"```\n{t.prompt}\n```",
            "",
            "## Response",
        ]

        if resp.thinking:
            lines += [
                "<details><summary>Thinking content</summary>",
                "",
                f"```\n{resp.thinking[:2000]}\n```",
                "",
                "</details>",
                "",
            ]

        lines.append(resp.content or "_[empty]_")
        lines.append("")
        lines += [
            "## Metrics",
            "| Metric | Value |",
            "|---|---|",
            f"| Wall clock | {m.wall_clock_seconds:.1f}s |",
            f"| Load time | {m.load_seconds:.1f}s |",
            f"| Prompt tokens | {m.prompt_eval_count} |",
            f"| Generation tokens | {m.eval_count} |",
            f"| Prompt rate | {m.prompt_eval_rate:.1f} tok/s |",
            f"| Generation rate | {m.generation_rate:.1f} tok/s |",
            f"| done_reason | {m.done_reason} |",
            "",
            "## Scores",
            "| Dimension | Score |",
            "|---|---|",
        ]
        s = result.scores
        for dim in [
            "contract_score", "correctness_score", "completeness_score",
            "safety_score", "evidence_score", "instruction_score",
            "escalation_score", "fact_preservation_score",
            "deterministic_test_score", "latency_score", "weighted_total",
        ]:
            val = getattr(s, dim, None)
            if val is not None:
                lines.append(f"| {dim} | {val:.3f} |")

        if result.safety_flags:
            lines += ["", "## Safety Flags"]
            for flag in result.safety_flags:
                lines.append(f"- {flag}")

        if result.schema_errors:
            lines += ["", "## Schema Errors"]
            for err in result.schema_errors:
                lines.append(f"- {err}")

        if resp.error:
            lines += ["", "## Error", f"```\n{resp.error}\n```"]

        return "\n".join(lines) + "\n"
