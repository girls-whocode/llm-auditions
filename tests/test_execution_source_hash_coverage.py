from __future__ import annotations

from pathlib import Path

from llm_auditions.versioning import execution_source_hashes


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _minimal_tree(root: Path) -> None:
    src = root / "src" / "llm_auditions"
    _write(src / "runner.py", "RUNNER = 1\n")
    _write(src / "ollama_client.py", "CLIENT = 1\n")
    _write(src / "models.py", "MODELS = 1\n")
    _write(src / "scoring.py", "SCORING = 1\n")
    _write(src / "reporting.py", "REPORT = 1\n")
    _write(src / "task_loader.py", "LOADER = 1\n")
    _write(src / "cli.py", "CLI = 1\n")
    _write(src / "configuration.py", "CFG = 1\n")
    _write(src / "versioning.py", "VER = 1\n")
    _write(src / "packaging.py", "PKG = 1\n")
    _write(src / "verifiers" / "sample.py", "VERIFY = 1\n")
    _write(root / "config" / "defaults.yaml", "engine: {}\n")
    _write(root / "schemas" / "worker_result.schema.json", "{}\n")


def test_execution_hash_changes_when_task_loader_changes(tmp_path: Path):
    _minimal_tree(tmp_path)
    before = execution_source_hashes(tmp_path)
    _write(tmp_path / "src" / "llm_auditions" / "task_loader.py", "LOADER = 2\n")
    after = execution_source_hashes(tmp_path)
    assert after["execution"] != before["execution"]


def test_execution_hash_changes_when_cli_changes(tmp_path: Path):
    _minimal_tree(tmp_path)
    before = execution_source_hashes(tmp_path)
    _write(tmp_path / "src" / "llm_auditions" / "cli.py", "CLI = 2\n")
    after = execution_source_hashes(tmp_path)
    assert after["execution"] != before["execution"]


def test_execution_hash_changes_when_versioning_changes(tmp_path: Path):
    _minimal_tree(tmp_path)
    before = execution_source_hashes(tmp_path)
    _write(tmp_path / "src" / "llm_auditions" / "versioning.py", "VER = 2\n")
    after = execution_source_hashes(tmp_path)
    assert after["execution"] != before["execution"]


def test_execution_hash_changes_when_verifier_changes(tmp_path: Path):
    _minimal_tree(tmp_path)
    before = execution_source_hashes(tmp_path)
    _write(tmp_path / "src" / "llm_auditions" / "verifiers" / "sample.py", "VERIFY = 2\n")
    after = execution_source_hashes(tmp_path)
    assert after["execution"] != before["execution"]
