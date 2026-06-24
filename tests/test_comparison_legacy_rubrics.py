from __future__ import annotations

from pathlib import Path

import yaml

from llm_auditions.task_loader import load_tasks_from_dir


PROJECT_ROOT = Path(__file__).parent.parent

LEGACY_TERMS = [
    "du -ah",
    "sort -rh",
    "largest files",
    "vm.dirty_ratio",
    "vm.dirty_background_ratio",
    "setenforce 0",
    "selinux",
    "adv-001",
    "nginx 1.25.0",
    "ai systemic risk",
    "ups utilization",
    "gpu inference in 0.1 seconds",
    "database query bottleneck",
    "read replica",
]


def _raw_comparison_entries() -> list[dict]:
    entries: list[dict] = []
    for path in sorted((PROJECT_ROOT / "fixtures").rglob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        tasks = []
        if isinstance(data, dict) and isinstance(data.get("tasks"), list):
            tasks = data["tasks"]
        elif isinstance(data, dict) and "id" in data:
            tasks = [data]
        elif isinstance(data, list):
            tasks = data
        for task in tasks:
            if isinstance(task, dict) and task.get("comparison_id"):
                entries.append(task)
    return entries


def test_no_legacy_rubric_terms_remain_in_comparison_tasks():
    entries = _raw_comparison_entries()
    assert len(entries) == 24
    combined = "\n".join(str(item).lower() for item in entries)
    for term in LEGACY_TERMS:
        assert term not in combined


def test_domain_specific_alignment_examples():
    tasks = load_tasks_from_dir(PROJECT_ROOT / "fixtures")
    indexed = {task.id: task for task in tasks}

    linux_iowait = indexed["linux_fast_disk_usage"]
    linux_text = str(linux_iowait.model_dump(mode="json")).lower()
    assert "per-device latency" in linux_text or "queue depth" in linux_text or "await" in linux_text
    assert "du -ah" not in linux_text

    sec_incident = indexed["sec_worker_diagnosis_not_remediation"]
    sec_text = str(sec_incident.model_dump(mode="json")).lower()
    assert "contain" in sec_text and "evidence" in sec_text
    assert "adv-001" not in sec_text and "nginx 1.25.0" not in sec_text

    philosophy = indexed["gk_fast_escalation_recognition"]
    philosophy_text = str(philosophy.model_dump(mode="json")).lower()
    assert "kant" in philosophy_text or "utilitarian" in philosophy_text or "uncertainty" in philosophy_text
    assert "ai systemic risk" not in philosophy_text

    engineering = indexed["eng_hw_power_calculation"]
    engineering_text = str(engineering.model_dump(mode="json")).lower()
    assert "headroom" in engineering_text or "requests per minute" in engineering_text
    assert "ups utilization" not in engineering_text and "gpu inference in 0.1 seconds" not in engineering_text

    architecture = indexed["arch_worker_failure_domain"]
    architecture_text = str(architecture.model_dump(mode="json")).lower()
    assert "failure domain" in architecture_text or "failover" in architecture_text or "topology" in architecture_text
    assert "database query bottleneck" not in architecture_text
