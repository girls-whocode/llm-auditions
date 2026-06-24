from __future__ import annotations

from llm_auditions.models import TaskDefinition
from llm_auditions.verifiers.development import DevelopmentVerifier


def test_development_behavioral_failure_for_wrong_code():
    task = TaskDefinition(
        id="dev_case",
        team="development",
        role="fast_worker",
        prompt="p",
        verifier="development",
        development={
            "entry_file": "solution.py",
            "fixture_directory": "fixtures/development/task_001",
            "test_command": ["python", "-m", "pytest", "-q"],
            "timeout_seconds": 10,
        },
    )

    class _Resp:
        content = "```python\ndef count_words(text):\n    return {}\n```"

    r = DevelopmentVerifier().verify(task, _Resp())
    assert not r.passed
    assert r.score == 0.0
