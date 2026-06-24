#!/usr/bin/env bash
# run-smoke.sh — Runs baseline suite + representative tasks per role, think=false only
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
AUDITION="$SCRIPT_DIR/audition"

echo "=============================================="
echo " LLM Auditions — Smoke Run"
echo " think=false | serial | all teams"
echo "=============================================="
echo ""

python3 "$AUDITION" run --profile smoke "$@"
