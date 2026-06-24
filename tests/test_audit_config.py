from __future__ import annotations

import argparse

from llm_auditions.cli import cmd_audit_config
from llm_auditions.configuration import Configuration


def test_audit_config_runs_and_returns_code():
    cfg = Configuration()
    rc = cmd_audit_config(argparse.Namespace(), cfg)
    assert rc in (0, 1)
