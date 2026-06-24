"""JSON schema structure verifier."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import jsonschema

from ..models import TaskDefinition
from .base import BaseVerifier, VerifierResult

logger = logging.getLogger(__name__)

# Regex to extract JSON from a markdown code fence
_FENCE_RE = re.compile(r"```(?:json)?\s*([\[{].*?)```", re.DOTALL | re.IGNORECASE)


def recover_json_from_fence(text: str) -> tuple[str, bool]:
    """
    Attempt to extract JSON from a markdown code fence.
    Returns (json_text, was_recovered).
    """
    m = _FENCE_RE.search(text)
    if m:
        return m.group(1).strip(), True
    # Try parsing as-is
    stripped = text.strip()
    if stripped.startswith(("{", "[")):
        return stripped, False
    return text, False


class StructureVerifier(BaseVerifier):
    """Validates model output against a JSON schema."""

    name = "structure"

    def __init__(self, schemas_dir: Any) -> None:
        from pathlib import Path

        self.schemas_dir = Path(schemas_dir)
        self._cache: dict[str, dict[str, Any]] = {}

    def _load_schema(self, schema_name: str) -> dict[str, Any]:
        if schema_name in self._cache:
            return self._cache[schema_name]
        path = self.schemas_dir / f"{schema_name}.schema.json"
        if not path.exists():
            raise FileNotFoundError(f"Schema not found: {path}")
        with path.open() as f:
            schema = json.load(f)
        self._cache[schema_name] = schema
        return schema

    def verify(self, task: TaskDefinition, response: Any) -> VerifierResult:
        """
        Validate response content against the task's required_json_schema.
        Also detects and records markdown fence wrapping.
        """
        schema_name = task.required_json_schema
        if not schema_name:
            return VerifierResult(True, 1.0, "No JSON schema required for this task")

        content = response.content if hasattr(response, "content") else str(response)

        # Attempt JSON extraction
        json_text, recovered = recover_json_from_fence(content)

        try:
            obj = json.loads(json_text)
        except json.JSONDecodeError as exc:
            return VerifierResult(
                False,
                0.0,
                f"JSON parse error: {exc}. Content snippet: {content[:200]}",
                {"json_recovered": recovered},
            )

        # Load schema
        try:
            schema = self._load_schema(schema_name)
        except FileNotFoundError as exc:
            return VerifierResult(False, 0.0, str(exc))

        # Validate
        validator = jsonschema.Draft7Validator(schema)
        errors = list(validator.iter_errors(obj))

        if errors:
            error_strs = [e.message for e in errors[:5]]
            return VerifierResult(
                False,
                max(0.0, 1.0 - len(errors) * 0.2),
                f"Schema validation failed ({len(errors)} errors): "
                + "; ".join(error_strs),
                {
                    "json_recovered": recovered,
                    "schema_errors": [e.message for e in errors],
                    "parsed_object": obj,
                },
            )

        return VerifierResult(
            True,
            1.0 if not recovered else 0.8,  # Penalise fence wrapping
            "Schema valid" + (" (JSON recovered from markdown fence)" if recovered else ""),
            {
                "json_recovered": recovered,
                "parsed_object": obj,
            },
        )
