#!/usr/bin/env bash
# run-exhaustive.sh — All tasks, all think modes, edge cases
# REQUIRES: --yes flag or AUDITION_YES=1 environment variable
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUDITION="$SCRIPT_DIR/audition"

echo "=============================================="
echo " LLM Auditions — EXHAUSTIVE Run"
echo " ⚠  WARNING: All think modes, edge cases"
echo " ⚠  Expected runtime: many hours"
echo " ⚠  Requires --yes or AUDITION_YES=1"
echo "=============================================="
echo ""

python3 "$AUDITION" run --profile exhaustive "$@"
