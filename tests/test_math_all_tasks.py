from __future__ import annotations

from pathlib import Path

from llm_auditions.models import TaskDefinition
from llm_auditions.verifiers.mathematics import MathematicsVerifier

PROJECT_ROOT = Path(__file__).parent.parent


def test_all_math_tasks_have_deterministic_verifiers():
    tasks = []
    for path in (PROJECT_ROOT / "fixtures" / "mathematics").rglob("*.yaml"):
        text = path.read_text()
        assert "verifier: mathematics" in text
        tasks.append(path)
    assert tasks


def test_ground_truth_values_exist():
    g = MathematicsVerifier()
    assert g.TRAILING_ZEROS_1000_BASE12 == 497
    assert g.PELL_X > 0 and g.PELL_Y > 0
    assert len(g.EIGENVALUES_10X10) == 10
