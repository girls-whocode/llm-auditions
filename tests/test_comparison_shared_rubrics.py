from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent


def test_all_comparison_scenarios_define_shared_rubrics_with_fact_ids():
    scenario_files = sorted((PROJECT_ROOT / "fixtures" / "comparisons").glob("*.json"))
    assert len(scenario_files) == 12

    for path in scenario_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["comparison_id"]
        assert payload["scenario_version"]
        assert payload["comparison_information_mode"] in {"symmetric", "asymmetric"}

        required_facts = payload.get("required_facts")
        assert isinstance(required_facts, list)
        assert required_facts
        for fact in required_facts:
            assert isinstance(fact, dict)
            assert fact.get("fact_id")
            assert fact.get("description")
            aliases = fact.get("aliases")
            assert isinstance(aliases, list)
            assert aliases

        fact_ids = {fact["fact_id"] for fact in required_facts}
        shared_rules = payload.get("shared_rubric_rules")
        assert isinstance(shared_rules, list)
        assert shared_rules
        for rule in shared_rules:
            assert rule.get("rule_id")
            source_fact_ids = rule.get("source_fact_ids")
            assert isinstance(source_fact_ids, list)
            assert source_fact_ids
            for source_fact_id in source_fact_ids:
                assert source_fact_id in fact_ids
