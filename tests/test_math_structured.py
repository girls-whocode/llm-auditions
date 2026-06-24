from __future__ import annotations

from llm_auditions.models import TaskDefinition
from llm_auditions.verifiers.mathematics import MathematicsVerifier


class _Resp:
    def __init__(self, content: str):
        self.content = content


def test_trailing_zeros_structured_json_passes():
    task = TaskDefinition(id="math_trailing_zeros_1000_base12", team="mathematics", role="solver", prompt="p")
    resp = _Resp('{"final_answer": 497, "base_factorization": {"2": 2, "3": 1}, "v2": 994, "v3": 498}')
    r = MathematicsVerifier().verify(task, resp)
    assert r.passed


def test_eigenvalues_incomplete_fails():
    task = TaskDefinition(id="math_eigenvalues_min_matrix_10x10", team="mathematics", role="solver", prompt="p")
    resp = _Resp('{"eigenvalues_exact": ["1", "2"], "count": 2}')
    r = MathematicsVerifier().verify(task, resp)
    assert not r.passed
