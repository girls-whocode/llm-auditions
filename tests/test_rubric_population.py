from __future__ import annotations

from pathlib import Path

from llm_auditions.configuration import Configuration
from llm_auditions.task_loader import load_tasks_from_dir


PROJECT_ROOT = Path(__file__).parent.parent


def test_every_rubric_assisted_task_has_rules():
    cfg = Configuration(project_root=PROJECT_ROOT)
    cfg.load()
    tasks = load_tasks_from_dir(PROJECT_ROOT / "fixtures")
    rubric_tasks = [t for t in tasks if t.verification_classification == "rubric_assisted"]
    assert rubric_tasks, "expected rubric-assisted tasks in fixtures"
    missing = [t.id for t in rubric_tasks if not t.rubric_rules]
    assert missing == [], f"rubric-assisted tasks missing rules: {missing}"
