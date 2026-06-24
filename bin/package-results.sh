#!/usr/bin/env bash
# package-results.sh — Package a run directory into a sanitized tar.gz
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUDITION="$SCRIPT_DIR/audition"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <run-directory> [--safe-override]"
    echo ""
    echo "Creates:"
    echo "  llm-auditions-YYYYMMDD-HHMMSS.tar.gz"
    echo "  llm-auditions-YYYYMMDD-HHMMSS.tar.gz.sha256"
    echo ""
    echo "The sanitizer scans for secrets before packaging."
    echo "Use --safe-override to package despite warnings (not recommended)."
    exit 1
fi

RUN_DIR="$1"
shift

python3 "$AUDITION" package --run-dir "$RUN_DIR" "$@"
