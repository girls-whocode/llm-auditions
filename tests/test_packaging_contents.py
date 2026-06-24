from __future__ import annotations

import tarfile
from pathlib import Path

from llm_auditions.packaging import create_package


PROJECT_ROOT = Path(__file__).parent.parent


def test_package_includes_fixtures_and_sha_sums(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "events.jsonl").write_text("")
    (run / "run_manifest.json").write_text("{}")

    archive, _ = create_package(run, output_dir=tmp_path, safe_override=True)
    with tarfile.open(archive, "r:gz") as tf:
        names = tf.getnames()
    assert any("project/fixtures/" in n for n in names)
    assert "SHA256SUMS" in names
