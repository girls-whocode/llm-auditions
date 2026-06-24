from __future__ import annotations

import tarfile
from pathlib import Path

from llm_auditions.packaging import create_package


def test_package_contains_comparison_scenario_fixtures_and_checksums(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "events.jsonl").write_text("", encoding="utf-8")
    (run / "run_manifest.json").write_text("{}", encoding="utf-8")

    archive, _ = create_package(run, output_dir=tmp_path, safe_override=True)
    with tarfile.open(archive, "r:gz") as tf:
        names = set(tf.getnames())
        checksum_data = tf.extractfile("SHA256SUMS").read().decode("utf-8")

    scenario_path = "project/fixtures/comparisons/linux_ops_safety_001.json"
    assert scenario_path in names
    assert scenario_path in checksum_data
