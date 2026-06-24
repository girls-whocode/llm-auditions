#!/usr/bin/env bash
# validate-project.sh — Runs all project validation checks
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
AUDITION="$SCRIPT_DIR/audition"

echo "=============================================="
echo " LLM Auditions — Project Validation"
echo "=============================================="
echo ""

PASS=0
FAIL=0
SKIP=0

_check() {
    local name="$1"
    local cmd="$2"
    echo -n "  [CHECK] $name ... "
    if eval "$cmd" > /dev/null 2>&1; then
        echo "PASS"
        PASS=$((PASS + 1))
    else
        echo "FAIL"
        FAIL=$((FAIL + 1))
        # Show error output
        eval "$cmd" 2>&1 | head -5 | sed 's/^/    /'
    fi
}

_check_optional() {
    local name="$1"
    local tool="$2"
    local cmd="$3"
    if command -v "$tool" > /dev/null 2>&1; then
        _check "$name" "$cmd"
    else
        echo "  [SKIP ] $name (${tool} not installed)"
        SKIP=$((SKIP + 1))
    fi
}

echo "1. Python compilation checks"
_check "src/ compileall" "python3 -m compileall -q '$PROJECT_ROOT/src'"
_check "tests/ compileall" "python3 -m compileall -q '$PROJECT_ROOT/tests'"

echo ""
echo "2. Unit tests"
_check "pytest" "python3 -m pytest -q '$PROJECT_ROOT/tests'"

echo ""
echo "3. Shell script syntax"
for sh in "$PROJECT_ROOT"/bin/*.sh; do
    _check "bash -n $(basename "$sh")" "bash -n '$sh'"
done

echo ""
echo "4. Configuration validation"
_check "audition validate" "python3 '$AUDITION' validate"

echo ""
echo "5. Optional checks"
_check_optional "ruff lint src/" "ruff" "ruff check --quiet '$PROJECT_ROOT/src'"
_check_optional "shellcheck bin/" "shellcheck" "shellcheck '$PROJECT_ROOT'/bin/*.sh"

echo ""
echo "=============================================="
echo " Results: PASS=$PASS  FAIL=$FAIL  SKIP=$SKIP"
echo "=============================================="

if [[ $FAIL -gt 0 ]]; then
    echo " ✗ Validation FAILED"
    exit 1
else
    echo " ✓ All required checks passed"
    exit 0
fi
