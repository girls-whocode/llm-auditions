#!/usr/bin/env bash
# run-standard.sh — Full role auditions, think=false + think=low, resume enabled
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AUDITION="$SCRIPT_DIR/audition"

echo "=============================================="
echo " LLM Auditions — Standard Run"
echo " think=false + think=low | serial | resume on"
echo "=============================================="
echo ""

python3 "$AUDITION" run --profile standard "$@"
