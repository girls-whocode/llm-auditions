from __future__ import annotations

from llm_auditions.cli import _build_execution_plan
from llm_auditions.configuration import Configuration


def _check_profile(profile: str) -> None:
    cfg = Configuration()
    cfg.load()
    _, rows, md = _build_execution_plan(config=cfg, profile=profile)

    assert md["request_count"] == len(rows)
    assert md["base_independent_requests"] + md["valid_handoff_fast_rows"] + md["valid_handoff_dependent_heavy_rows"] == md["request_count"]
    assert md["cross_think_handoff_dependencies"] == 0
    assert md["cross_output_handoff_dependencies"] == 0
    assert md["cross_scenario_handoff_dependencies"] == 0
    assert md["cross_information_mode_dependencies"] == 0


def test_standard_profile_count_decomposition_exact():
    _check_profile("standard")


def test_exhaustive_profile_count_decomposition_exact():
    _check_profile("exhaustive")
