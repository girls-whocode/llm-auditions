"""Packaging — creates sanitized tar.gz archive and SHA-256 manifest."""

from __future__ import annotations

import hashlib
import json
import logging
import tarfile
from datetime import datetime, timezone
from pathlib import Path

from .sanitization import SanitizationError, check_and_raise, scan_directory

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent

_EXCLUDE_PATTERNS = [
    "*.pyc",
    "*/__pycache__/*",
    "*/.git/*",
    "*.tar.gz",
    "*.sha256",
    "*/models/blobs/*",
    "*/.ollama/*",
    "*.key",
    "*.pem",
    ".env",
    ".env.*",
    "secrets.yaml",
    "secrets.json",
    "id_rsa",
    "id_ed25519",
    "*/node_modules/*",
    "*/.venv/*",
    "*/venv/*",
]


def _should_exclude(path: Path) -> bool:
    from fnmatch import fnmatch

    name = path.name
    full = str(path)
    for pat in _EXCLUDE_PATTERNS:
        if fnmatch(name, pat.lstrip("*/")):
            return True
        if fnmatch(full, pat):
            return True
    return False


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _human_size(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size //= 1024
    return f"{size} TB"


def _add_dir(
    tar: tarfile.TarFile,
    directory: Path,
    arcname: str,
    sha256_sums: list[str],
    package_manifest: list[dict[str, str]],
) -> None:
    for path in sorted(directory.rglob("*")):
        if path.is_dir():
            continue
        if _should_exclude(path):
            continue
        rel = path.relative_to(directory)
        if ".." in rel.parts:
            continue
        arc = f"{arcname}/{rel.as_posix()}"
        tar.add(str(path), arcname=arc)
        digest = _sha256_file(path)
        sha256_sums.append(f"{digest}  {arc}")
        package_manifest.append({"path": arc, "sha256": digest})


def create_package(
    run_dir: Path,
    output_dir: Path | None = None,
    safe_override: bool = False,
) -> tuple[Path, Path]:
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    logger.info("Running sanitization scan on package scope ...")
    findings = check_and_raise(run_dir, safe_override=safe_override)

    scope_paths = [
        PROJECT_ROOT / "config",
        PROJECT_ROOT / "schemas",
        PROJECT_ROOT / "fixtures",
        PROJECT_ROOT / "src",
        PROJECT_ROOT / "bin",
        PROJECT_ROOT / "tests",
        PROJECT_ROOT / "docs",
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "requirements.txt",
        PROJECT_ROOT / "pyproject.toml",
    ]

    extra_findings = []
    for p in scope_paths:
        if p.exists() and p.is_dir():
            extra_findings.extend(scan_directory(p))
        elif p.exists() and p.is_file():
            file_findings = scan_directory(p.parent)
            extra_findings.extend([f for f in file_findings if Path(f.get("file", "")) == p])

    findings.extend(extra_findings)

    if findings and not safe_override:
        raise SanitizationError(
            f"Likely secrets detected in package scope ({len(findings)} finding(s)). "
            "Refusing to package. Use --safe-override to bypass."
        )

    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    archive_name = f"llm-auditions-{ts}.tar.gz"
    out_dir = output_dir or run_dir.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    archive_path = out_dir / archive_name
    sha256_path = out_dir / f"{archive_name}.sha256"

    sha256_sums: list[str] = []
    package_manifest: list[dict[str, str]] = []

    with tarfile.open(archive_path, "w:gz") as tar:
        _add_dir(tar, run_dir, arcname=f"run/{run_dir.name}", sha256_sums=sha256_sums, package_manifest=package_manifest)

        include_dirs = ["config", "schemas", "fixtures", "src", "bin", "tests", "docs"]
        for subdir in include_dirs:
            p = PROJECT_ROOT / subdir
            if p.exists():
                _add_dir(tar, p, arcname=f"project/{subdir}", sha256_sums=sha256_sums, package_manifest=package_manifest)

        for fname in ("pyproject.toml", "requirements.txt", "README.md"):
            p = PROJECT_ROOT / fname
            if p.exists() and not _should_exclude(p):
                arc = f"project/{fname}"
                tar.add(str(p), arcname=arc)
                digest = _sha256_file(p)
                sha256_sums.append(f"{digest}  {arc}")
                package_manifest.append({"path": arc, "sha256": digest})

        git_meta = {
            "git_available": (PROJECT_ROOT / ".git").exists(),
            "packaged_at": datetime.now(timezone.utc).isoformat(),
            "project_root": str(PROJECT_ROOT),
        }
        git_meta_path = out_dir / "GIT_METADATA_SUMMARY.json"
        git_meta_path.write_text(json.dumps(git_meta, indent=2), encoding="utf-8")
        arc = "project/GIT_METADATA_SUMMARY.json"
        tar.add(str(git_meta_path), arcname=arc)
        digest = _sha256_file(git_meta_path)
        sha256_sums.append(f"{digest}  {arc}")
        package_manifest.append({"path": arc, "sha256": digest})

        pm_path = out_dir / "package_manifest.json"
        pm_path.write_text(json.dumps({"files": package_manifest}, indent=2), encoding="utf-8")
        arc = "package_manifest.json"
        tar.add(str(pm_path), arcname=arc)
        digest = _sha256_file(pm_path)
        sha256_sums.append(f"{digest}  {arc}")

        sums_text = "\n".join(sha256_sums) + "\n"
        import io

        sums_bytes = sums_text.encode()
        info = tarfile.TarInfo(name="SHA256SUMS")
        info.size = len(sums_bytes)
        tar.addfile(info, io.BytesIO(sums_bytes))

    archive_sha256 = _sha256_file(archive_path)
    sha256_path.write_text(f"{archive_sha256}  {archive_name}\n", encoding="utf-8")

    findings_path = out_dir / "sanitization_findings.json"
    findings_path.write_text(json.dumps(findings, indent=2), encoding="utf-8")

    package_manifest_path = out_dir / "package_manifest.json"
    if not package_manifest_path.exists():
        package_manifest_path.write_text(json.dumps({"files": package_manifest}, indent=2), encoding="utf-8")

    logger.info("Archive created: %s (%s)", archive_path, _human_size(archive_path.stat().st_size))
    logger.info("SHA-256: %s", sha256_path)
    logger.info("Sanitization findings: %s", findings_path)
    logger.info("Package manifest: %s", package_manifest_path)

    return archive_path, sha256_path
