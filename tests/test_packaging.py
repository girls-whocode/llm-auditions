"""Tests for packaging and sanitization."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_auditions.sanitization import scan_directory, check_and_raise, SanitizationError


class TestSanitization:
    def test_clean_directory_passes(self, tmp_path: Path):
        (tmp_path / "results.json").write_text('{"score": 0.9, "model": "gemma4:12b"}')
        findings = scan_directory(tmp_path)
        assert findings == []

    def test_private_key_detected(self, tmp_path: Path):
        (tmp_path / "secret_key.txt").write_text("-----BEGIN PRIVATE KEY-----\nABCDEFGH\n-----END PRIVATE KEY-----")
        findings = scan_directory(tmp_path)
        assert len(findings) > 0
        assert any("key" in f["pattern"].lower() or "KEY" in f["snippet"] for f in findings)

    def test_aws_key_detected(self, tmp_path: Path):
        (tmp_path / "config.txt").write_text("aws_access_key_id = AKIAIOSFODNN7EXAMPLE")
        findings = scan_directory(tmp_path)
        assert len(findings) > 0

    def test_jwt_token_detected(self, tmp_path: Path):
        jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ0ZXN0In0.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        (tmp_path / "token.txt").write_text(f"Authorization: Bearer {jwt}")
        findings = scan_directory(tmp_path)
        assert len(findings) > 0

    def test_model_digest_not_flagged(self, tmp_path: Path):
        """Model digest IDs in manifests must not be flagged as secrets."""
        (tmp_path / "manifest.json").write_text(
            '{"model_digest": "06c1097efce0", "identity_key": "abc123def456"}'
        )
        findings = scan_directory(tmp_path)
        # Should not flag these as secrets
        secret_findings = [f for f in findings if f["pattern"] in ("Hex secret (32+ chars)", "AWS access key")]
        assert len(secret_findings) == 0, f"Unexpected findings: {secret_findings}"

    def test_check_and_raise_blocks(self, tmp_path: Path):
        (tmp_path / "secret_key.txt").write_text("-----BEGIN PRIVATE KEY-----\nABC\n-----END PRIVATE KEY-----")
        with pytest.raises(SanitizationError):
            check_and_raise(tmp_path, safe_override=False)

    def test_check_and_raise_with_override(self, tmp_path: Path):
        (tmp_path / "secret_key.txt").write_text("-----BEGIN PRIVATE KEY-----\nABC\n-----END PRIVATE KEY-----")
        # Should not raise
        findings = check_and_raise(tmp_path, safe_override=True)
        assert len(findings) > 0

    def test_binary_files_skipped(self, tmp_path: Path):
        (tmp_path / "model.bin").write_bytes(b"\x00\x01\x02\x03" * 100)
        findings = scan_directory(tmp_path)
        assert findings == []

    def test_gitignore_pattern_skipped(self, tmp_path: Path):
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        (git_dir / "config").write_text("password=supersecret123456")
        findings = scan_directory(tmp_path)
        # .git/ should be skipped
        assert not any(".git" in f["file"] for f in findings)


class TestPackagingHelpers:
    def test_human_size(self):
        from llm_auditions.packaging import _human_size
        assert "KB" in _human_size(1024)
        assert "MB" in _human_size(1024 * 1024)
        assert "GB" in _human_size(1024 * 1024 * 1024)

    def test_should_not_exclude_json(self):
        from llm_auditions.packaging import _should_exclude
        assert not _should_exclude(Path("results/run-001/summary.json"))

    def test_should_exclude_pyc(self):
        from llm_auditions.packaging import _should_exclude
        assert _should_exclude(Path("src/__pycache__/foo.pyc"))
