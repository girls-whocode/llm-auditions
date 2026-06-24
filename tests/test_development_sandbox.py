from __future__ import annotations

from llm_auditions.models import TaskDefinition
from llm_auditions.verifiers.development import DevelopmentVerifier


def test_development_verifier_disables_when_no_backend(monkeypatch):
    monkeypatch.setattr("llm_auditions.verifiers.development._tool_available", lambda name: False)
    calls = {"run_safe": 0}

    def _fail_if_called(*args, **kwargs):
        calls["run_safe"] += 1
        raise AssertionError("host execution must never run")

    monkeypatch.setattr("llm_auditions.verifiers.development._run_safe", _fail_if_called)
    task = TaskDefinition(id="dev_sandbox", team="development", role="fast_worker", prompt="p", verifier="development", development={"entry_file": "solution.py", "fixture_directory": "fixtures/development/task_001", "test_command": ["python", "-m", "pytest", "-q"]})

    class _Resp:
        content = "```python\ndef count_words(text):\n    return {}\n```"

    r = DevelopmentVerifier().verify(task, _Resp())
    assert r.passed is False
    assert r.extra.get("sandbox_backend") in (None, "disabled")
    assert calls["run_safe"] == 0


def test_development_verifier_refuses_execution_by_policy(monkeypatch):
    monkeypatch.setenv("AUDITION_ENABLE_DEVELOPMENT_SANDBOX", "1")
    monkeypatch.setattr("llm_auditions.verifiers.development._detect_sandbox_backend", lambda: "bubblewrap")
    calls = {"run_safe": 0}

    def _fail_if_called(*args, **kwargs):
        calls["run_safe"] += 1
        raise AssertionError("host execution must never run")

    monkeypatch.setattr("llm_auditions.verifiers.development._run_safe", _fail_if_called)
    task = TaskDefinition(id="dev_policy", team="development", role="fast_worker", prompt="p", verifier="development", development={"entry_file": "solution.py", "fixture_directory": "fixtures/development/task_001", "test_command": ["python", "-m", "pytest", "-q"]})

    class _Resp:
        content = "```python\ndef count_words(text):\n    return {}\n```"

    r = DevelopmentVerifier().verify(task, _Resp())
    assert r.passed is False
    assert r.extra.get("sandbox_unavailable") is True
    assert r.extra.get("policy_refusal") is True
    assert r.extra.get("reason") == "sandbox_unavailable"
    assert calls["run_safe"] == 0
