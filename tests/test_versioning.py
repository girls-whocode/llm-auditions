from __future__ import annotations

from pathlib import Path

from llm_auditions.versioning import (
    ENGINE_VERSION,
    REPORT_VERSION,
    SCORING_VERSION,
    TASK_SUITE_VERSION,
    VERIFIER_VERSION,
    execution_source_hashes,
    version_payload,
)


PROJECT_ROOT = Path(__file__).parent.parent


def test_version_constants_match_pass6_targets():
    assert ENGINE_VERSION == "0.10.0"
    assert TASK_SUITE_VERSION == "2"
    assert SCORING_VERSION == "2"
    assert VERIFIER_VERSION == "2"
    assert REPORT_VERSION == "2"


def test_version_payload_contains_expected_keys():
    payload = version_payload()
    assert payload["engine_version"] == ENGINE_VERSION
    assert payload["task_suite_version"] == TASK_SUITE_VERSION
    assert payload["scoring_version"] == SCORING_VERSION
    assert payload["verifier_version"] == VERIFIER_VERSION
    assert payload["report_version"] == REPORT_VERSION


def test_execution_source_hashes_populated():
    hashes = execution_source_hashes(PROJECT_ROOT)
    assert set(hashes.keys()) == {"engine", "scoring", "verifier", "report", "execution", "config", "schemas"}
    assert all(len(v) == 16 for v in hashes.values())
