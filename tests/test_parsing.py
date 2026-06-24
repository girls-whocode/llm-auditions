"""Tests for JSON parsing and fence recovery."""

from __future__ import annotations

import json

import pytest

from llm_auditions.verifiers.structure import recover_json_from_fence, StructureVerifier


class TestJsonRecovery:
    def test_plain_json_passthrough(self):
        text = '{"disposition": "ANSWER", "answer": "test"}'
        result, recovered = recover_json_from_fence(text)
        assert not recovered
        assert json.loads(result)["disposition"] == "ANSWER"

    def test_json_in_markdown_fence(self):
        text = '```json\n{"disposition": "ANSWER", "answer": "test"}\n```'
        result, recovered = recover_json_from_fence(text)
        assert recovered
        obj = json.loads(result)
        assert obj["disposition"] == "ANSWER"

    def test_json_in_plain_fence(self):
        text = '```\n{"disposition": "ANSWER"}\n```'
        result, recovered = recover_json_from_fence(text)
        assert recovered

    def test_no_json_at_all(self):
        text = "Just some text with no JSON here"
        result, recovered = recover_json_from_fence(text)
        assert not recovered

    def test_json_with_surrounding_text(self):
        text = 'Here is my answer:\n```json\n{"verdict": "APPROVE"}\n```\nHope this helps!'
        result, recovered = recover_json_from_fence(text)
        assert recovered
        obj = json.loads(result)
        assert obj["verdict"] == "APPROVE"


class TestStructureVerifier:
    def test_valid_worker_json(self, tmp_path):
        # Write schema
        import shutil
        src = __file__
        schemas_dir = tmp_path / "schemas"
        schemas_dir.mkdir()
        # Copy schema from project
        from pathlib import Path
        project_schemas = Path(__file__).parent.parent / "schemas"
        shutil.copytree(str(project_schemas), str(schemas_dir), dirs_exist_ok=True)

        sv = StructureVerifier(schemas_dir)

        class FakeTask:
            required_json_schema = "worker_result"

        class FakeResponse:
            content = json.dumps({
                "disposition": "ANSWER",
                "target": "NONE",
                "confidence": "HIGH",
                "answer": "Paris",
                "assumptions": [],
                "limitations": [],
                "research_required": False,
            })

        result = sv.verify(FakeTask(), FakeResponse())
        assert result.passed
        assert result.score >= 0.8

    def test_invalid_disposition_fails(self, tmp_path):
        import shutil
        from pathlib import Path
        schemas_dir = tmp_path / "schemas"
        schemas_dir.mkdir()
        project_schemas = Path(__file__).parent.parent / "schemas"
        shutil.copytree(str(project_schemas), str(schemas_dir), dirs_exist_ok=True)

        sv = StructureVerifier(schemas_dir)

        class FakeTask:
            required_json_schema = "worker_result"

        class FakeResponse:
            content = json.dumps({
                "disposition": "INVALID_DISPOSITION",
                "target": "NONE",
                "confidence": "HIGH",
                "answer": "test",
                "assumptions": [],
                "limitations": [],
                "research_required": False,
            })

        result = sv.verify(FakeTask(), FakeResponse())
        assert not result.passed

    def test_missing_required_field_fails(self, tmp_path):
        import shutil
        from pathlib import Path
        schemas_dir = tmp_path / "schemas"
        schemas_dir.mkdir()
        project_schemas = Path(__file__).parent.parent / "schemas"
        shutil.copytree(str(project_schemas), str(schemas_dir), dirs_exist_ok=True)

        sv = StructureVerifier(schemas_dir)

        class FakeTask:
            required_json_schema = "worker_result"

        class FakeResponse:
            content = json.dumps({"disposition": "ANSWER"})  # missing required fields

        result = sv.verify(FakeTask(), FakeResponse())
        assert not result.passed

    def test_no_schema_required_passes(self, tmp_path):
        sv = StructureVerifier(tmp_path)

        class FakeTask:
            required_json_schema = ""

        class FakeResponse:
            content = "Some text answer."

        result = sv.verify(FakeTask(), FakeResponse())
        assert result.passed

    def test_truncated_json_fails_cleanly(self, tmp_path):
        import shutil
        from pathlib import Path

        schemas_dir = tmp_path / "schemas"
        schemas_dir.mkdir()
        project_schemas = Path(__file__).parent.parent / "schemas"
        shutil.copytree(str(project_schemas), str(schemas_dir), dirs_exist_ok=True)

        sv = StructureVerifier(schemas_dir)

        class FakeTask:
            required_json_schema = "worker_result"

        class FakeResponse:
            content = '{"disposition":"ANSWER","target":"NONE"'

        result = sv.verify(FakeTask(), FakeResponse())
        assert not result.passed
        assert "JSON parse error" in result.details

    def test_markdown_fence_recovery_penalizes_score(self, tmp_path):
        import shutil
        from pathlib import Path

        schemas_dir = tmp_path / "schemas"
        schemas_dir.mkdir()
        project_schemas = Path(__file__).parent.parent / "schemas"
        shutil.copytree(str(project_schemas), str(schemas_dir), dirs_exist_ok=True)

        sv = StructureVerifier(schemas_dir)

        class FakeTask:
            required_json_schema = "worker_result"

        class FakeResponse:
            content = "```json\n" + json.dumps({
                "disposition": "ANSWER",
                "target": "NONE",
                "confidence": "HIGH",
                "answer": "Paris",
                "assumptions": [],
                "limitations": [],
                "research_required": False,
            }) + "\n```"

        result = sv.verify(FakeTask(), FakeResponse())
        assert result.passed
        assert result.score < 1.0
