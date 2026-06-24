"""Sanitization — scans for likely secrets before packaging."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Patterns that suggest secrets (key names or value patterns)
_SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("AWS access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("AWS secret key", re.compile(r"(?i)aws_secret_access_key\s*[=:]\s*\S{20,}")),
    ("Private key header", re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----")),
    ("Generic password field", re.compile(r"(?i)(?:password|passwd|secret|token|api_key|apikey|auth_token)\s*[=:]\s*['\"]?\S{8,}['\"]?")),
    ("Bearer token", re.compile(r"(?i)bearer\s+[A-Za-z0-9\-_=]{20,}")),
    ("Basic auth", re.compile(r"(?i)basic\s+[A-Za-z0-9+/=]{20,}")),
    ("Private SSH key content", re.compile(r"[A-Za-z0-9+/]{60,}={0,2}\n[A-Za-z0-9+/]{60,}")),
    ("Hex secret (32+ chars)", re.compile(r"\b[0-9a-fA-F]{32,}\b")),
    ("JWT token", re.compile(r"eyJ[A-Za-z0-9\-_]+\.eyJ[A-Za-z0-9\-_]+\.[A-Za-z0-9\-_]+")),
]

# Extensions to skip (binary / large files)
_SKIP_EXTENSIONS = {
    ".gz", ".tar", ".zip", ".bz2", ".xz",
    ".png", ".jpg", ".jpeg", ".gif", ".svg",
    ".pdf", ".doc", ".docx",
    ".pyc", ".pyo", ".so", ".a",
    ".bin", ".db", ".sqlite",
}

# Files to always skip by name
_SKIP_FILES = {
    ".git",
    "node_modules",
    "__pycache__",
}

# Hex strings of 32+ characters are common in hashes — we whitelist known
# safe hex patterns (model digests, SHA256 hashes in our own manifests)
_HEX_WHITELIST = re.compile(r"(?:digest|sha256|id|hash)\s*[=:]\s*['\"]?[0-9a-fA-F]{12,64}['\"]?", re.IGNORECASE)


class SanitizationError(Exception):
    """Raised when a likely secret is detected and packaging is blocked."""
    pass


def scan_directory(directory: Path) -> list[dict[str, Any]]:
    """
    Scan a directory tree for likely secrets.
    Returns list of findings: [{file, line_number, pattern, snippet}].
    """
    findings: list[dict[str, Any]] = []

    for path in sorted(directory.rglob("*")):
        # Skip directories
        if path.is_dir():
            continue

        # Skip binary / large / unrelated extensions
        if path.suffix.lower() in _SKIP_EXTENSIONS:
            continue

        # Skip paths containing hidden git or cache dirs
        if any(part in _SKIP_FILES for part in path.parts):
            continue

        # Skip large files (>5MB)
        try:
            if path.stat().st_size > 5 * 1024 * 1024:
                continue
        except OSError:
            continue

        try:
            text = path.read_text(errors="replace")
        except Exception:
            continue

        for lineno, line in enumerate(text.splitlines(), start=1):
            for name, pattern in _SECRET_PATTERNS:
                m = pattern.search(line)
                if not m:
                    continue

                # Whitelist check for hex patterns
                if "Hex secret" in name:
                    if _HEX_WHITELIST.search(line):
                        continue
                    # Also skip lines that look like our own scoring/manifest values
                    if any(kw in line.lower() for kw in ("sha256", "digest", "identity_key", "model_digest", "prompt_hash")):
                        continue

                findings.append({
                    "file": str(path),
                    "line_number": lineno,
                    "pattern": name,
                    "snippet": line.strip()[:120],
                })

    return findings


def check_and_raise(directory: Path, safe_override: bool = False) -> list[dict[str, Any]]:
    """
    Scan for secrets. If findings exist and safe_override is False, raise SanitizationError.
    Returns findings list.
    """
    findings = scan_directory(directory)
    if findings and not safe_override:
        summary = "\n".join(
            f"  {f['file']}:{f['line_number']} [{f['pattern']}]: {f['snippet'][:80]}"
            for f in findings[:20]
        )
        raise SanitizationError(
            f"Likely secrets detected ({len(findings)} finding(s)). "
            f"Refusing to package.\n{summary}\n"
            f"Use --safe-override to package anyway (not recommended)."
        )
    return findings
