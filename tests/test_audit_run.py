from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm_auditions.cli import cmd_audit_run
from llm_auditions.configuration import Configuration


def test_audit_run_writes_artifacts(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run_manifest.json").write_text(json.dumps({"run_id": "r1", "profile": "smoke", "request_count": 1}))
    (run_dir / "events.jsonl").write_text(json.dumps({"identity_key": "k1", "status": "completed"}) + "\n")
    (run_dir / "run_state.json").write_text(json.dumps({"completed_identity_keys": ["k1"], "error_count": 0, "unsupported_mode_count": 0}))

    rc = cmd_audit_run(argparse.Namespace(run_dir=str(run_dir)), Configuration())
    assert (run_dir / "RUN_AUDIT.json").exists()
    assert (run_dir / "RUN_AUDIT.md").exists()
    assert rc in (0, 1)
