"""Deterministic mathematics verifiers using Python integer arithmetic and SymPy."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from .base import BaseVerifier, VerifierResult
from .structure import recover_json_from_fence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ground-truth computations
# ---------------------------------------------------------------------------


def _trailing_zeros_1000_factorial_base12() -> int:
    """
    Compute the exact number of trailing zeros of 1000! in base 12.
    base 12 = 2^2 * 3, so we need floor(v2/2) and v3 where v2, v3 are
    the exact powers of 2 and 3 in 1000!.
    Trailing zeros = min(floor(v2/2), v3).
    """
    def legendre(n: int, p: int) -> int:
        count = 0
        pk = p
        while pk <= n:
            count += n // pk
            pk *= p
        return count

    v2 = legendre(1000, 2)
    v3 = legendre(1000, 3)
    return min(v2 // 2, v3)


def _pell_fundamental_solution_61() -> tuple[int, int]:
    """
    Fundamental (minimal positive) solution to x^2 - 61*y^2 = 1.
    Uses continued fraction expansion of sqrt(61).
    """
    D = 61
    # Continued fraction of sqrt(D)
    m, d, a0 = 0, 1, int(D**0.5)
    a = a0

    # Using convergents
    h_prev, h_curr = 1, a0
    k_prev, k_curr = 0, 1

    while True:
        m = d * a - m
        d = (D - m * m) // d
        a = (a0 + m) // d
        h_prev, h_curr = h_curr, a * h_curr + h_prev
        k_prev, k_curr = k_curr, a * k_curr + k_prev
        if h_curr * h_curr - D * k_curr * k_curr == 1:
            return h_curr, k_curr


def _eigenvalues_min_matrix_10x10() -> list[float]:
    """
    Eigenvalues of the 10x10 matrix A where A[i][j] = min(i+1, j+1).
    Formula: eigenvalues are 1/(4*cos^2(k*pi/(2n+1))) for k=1..n.
    Returns sorted list of eigenvalues.
    Reference: this matrix is related to the tridiagonal form.
    """
    import math

    n = 10
    # The eigenvalues of the n×n min matrix can be computed analytically.
    # λ_k = 1 / (4 * cos²(kπ/(2n+1)))  for k = 1, ..., n
    eigvals = []
    for k in range(1, n + 1):
        lam = 1.0 / (4.0 * math.cos(k * math.pi / (2 * n + 1)) ** 2)
        eigvals.append(lam)
    return sorted(eigvals, reverse=True)


def _modular_arithmetic_answer() -> int:
    return pow(2, 100, 1_000_000_007)


def _sum_of_divisors_360() -> int:
    n = 360
    total = 0
    for divisor in range(1, n + 1):
        if n % divisor == 0:
            total += divisor
    return total


def _birthday_probability_23() -> float:
    probability_unique = 1.0
    for count in range(23):
        probability_unique *= (365 - count) / 365
    return 1.0 - probability_unique


# Compute ground truth once
_GT_TRAILING_ZEROS = _trailing_zeros_1000_factorial_base12()
_GT_PELL_X, _GT_PELL_Y = _pell_fundamental_solution_61()
_GT_EIGENVALUES = _eigenvalues_min_matrix_10x10()
_GT_MOD_2_100 = _modular_arithmetic_answer()
_GT_SIGMA_360 = _sum_of_divisors_360()
_GT_BIRTHDAY_23 = _birthday_probability_23()


# ---------------------------------------------------------------------------
# Verifier helpers
# ---------------------------------------------------------------------------


def _extract_integer(text: str) -> int | None:
    """Extract integer from structured JSON or explicit ANSWER marker first."""
    json_text, _ = recover_json_from_fence(text)
    try:
        parsed = json.loads(json_text)
        if isinstance(parsed, dict):
            for key in ("answer", "result", "value"):
                val = parsed.get(key)
                if isinstance(val, int):
                    return val
                if isinstance(val, str) and re.fullmatch(r"\s*\d+\s*", val):
                    return int(val.strip())
    except Exception:
        pass

    marker = re.search(r"(?i)\banswer\s*[:=]\s*(\d+)\b", text)
    if marker:
        return int(marker.group(1))

    nums = re.findall(r"\b(\d+)\b", text)
    if len(nums) == 1:
        return int(nums[0])
    return None


def _extract_pair(text: str) -> tuple[int, int] | None:
    """Extract an (x,y) pair from JSON or explicit x=..., y=... format."""
    json_text, _ = recover_json_from_fence(text)
    try:
        parsed = json.loads(json_text)
        if isinstance(parsed, dict):
            x = parsed.get("x")
            y = parsed.get("y")
            if isinstance(x, int) and isinstance(y, int):
                return x, y
            if isinstance(x, str) and isinstance(y, str) and x.strip().isdigit() and y.strip().isdigit():
                return int(x.strip()), int(y.strip())
    except Exception:
        pass

    m = re.search(r"(?is)\bx\s*[:=]\s*(\d+)\D+\by\s*[:=]\s*(\d+)", text)
    if m:
        return int(m.group(1)), int(m.group(2))

    nums = re.findall(r"\b(\d{8,})\b", text)
    if len(nums) == 2:
        return int(nums[0]), int(nums[1])
    return None


def _extract_float_list(text: str) -> list[float]:
    json_text, _ = recover_json_from_fence(text)
    try:
        parsed = json.loads(json_text)
        if isinstance(parsed, dict):
            for key in ("eigenvalues", "answer", "values"):
                val = parsed.get(key)
                if isinstance(val, list):
                    out: list[float] = []
                    for item in val:
                        if isinstance(item, (int, float)):
                            out.append(float(item))
                        elif isinstance(item, str):
                            out.append(float(item))
                    if out:
                        return out
        if isinstance(parsed, list):
            return [float(x) for x in parsed]
    except Exception:
        pass
    return [float(m) for m in re.findall(r"\b\d+\.?\d*(?:e[+-]?\d+)?\b", text)]


def _extract_symbolic_list(text: str) -> list[str]:
    json_text, _ = recover_json_from_fence(text)
    try:
        parsed = json.loads(json_text)
        if isinstance(parsed, dict):
            vals = parsed.get("eigenvalues_exact")
            if isinstance(vals, list):
                return [str(v).strip() for v in vals]
    except Exception:
        pass
    return []


class MathematicsVerifier(BaseVerifier):
    """Verifies math model answers against deterministic Python results."""

    name = "mathematics"

    # Ground truth (accessible for tests)
    TRAILING_ZEROS_1000_BASE12 = _GT_TRAILING_ZEROS
    PELL_X = _GT_PELL_X
    PELL_Y = _GT_PELL_Y
    EIGENVALUES_10X10 = _GT_EIGENVALUES
    MODULAR_2_100 = _GT_MOD_2_100
    SIGMA_360 = _GT_SIGMA_360
    BIRTHDAY_23 = _GT_BIRTHDAY_23

    def verify(self, task: Any, response: Any) -> VerifierResult:
        task_id = task.id if hasattr(task, "id") else str(task)
        content = response.content if hasattr(response, "content") else str(response)

        if "trailing_zeros" in task_id:
            return self._verify_trailing_zeros(content)
        if "pell" in task_id:
            return self._verify_pell(content)
        if "eigenvalue" in task_id or "min_matrix" in task_id:
            return self._verify_eigenvalues(content)
        if "modular" in task_id:
            return self._verify_modular_arithmetic(content)
        if "sum_of_divisors" in task_id:
            return self._verify_sum_of_divisors(content)
        if "birthday" in task_id:
            return self._verify_birthday_probability(content)

        return VerifierResult(
            False, 0.0, f"No math verifier defined for task_id '{task_id}'"
        )

    def _verify_trailing_zeros(self, content: str) -> VerifierResult:
        gt = _GT_TRAILING_ZEROS
        json_text, _ = recover_json_from_fence(content)
        found = None
        v2 = None
        v3 = None
        base_factorization = None
        try:
            parsed = json.loads(json_text)
            if isinstance(parsed, dict):
                if isinstance(parsed.get("final_answer"), int):
                    found = int(parsed.get("final_answer"))
                if isinstance(parsed.get("v2"), int):
                    v2 = int(parsed.get("v2"))
                if isinstance(parsed.get("v3"), int):
                    v3 = int(parsed.get("v3"))
                if isinstance(parsed.get("base_factorization"), dict):
                    base_factorization = parsed.get("base_factorization")
        except Exception:
            pass
        if found is None:
            found = _extract_integer(content)
        if found is None:
            return VerifierResult(
                False, 0.0, f"No integer found in response. Expected {gt}",
                {"expected": gt, "found": None},
            )
        expected_v2 = 994
        expected_v3 = 498
        factorization_ok = isinstance(base_factorization, dict) and str(base_factorization.get("2")) == "2" and str(base_factorization.get("3")) == "1"
        valuation_ok = v2 == expected_v2 and v3 == expected_v3
        correct = found == gt and factorization_ok and valuation_ok
        score = 0.0
        if found == gt:
            score += 0.4
        if factorization_ok:
            score += 0.2
        if v2 == expected_v2:
            score += 0.2
        if v3 == expected_v3:
            score += 0.2
        return VerifierResult(
            correct,
            round(score, 4),
            f"Expected trailing zeros={gt}, v2={expected_v2}, v3={expected_v3}; found answer={found}, v2={v2}, v3={v3}" if not correct else f"Correct: {gt}",
            {"expected": gt, "found": found, "expected_v2": expected_v2, "expected_v3": expected_v3, "found_v2": v2, "found_v3": v3, "base_factorization_ok": factorization_ok},
        )

    def _verify_pell(self, content: str) -> VerifierResult:
        gt_x, gt_y = _GT_PELL_X, _GT_PELL_Y
        pair = _extract_pair(content)
        extra = {
            "expected_x": gt_x,
            "expected_y": gt_y,
        }
        if pair is None:
            return VerifierResult(
                False, 0.0,
                f"Could not extract (x,y) pair. Expected x={gt_x}, y={gt_y}",
                extra,
            )
        x, y = pair
        extra["found_x"] = x
        extra["found_y"] = y

        # Verify substitution
        substitution_ok = x * x - 61 * y * y == 1
        if substitution_ok and x == gt_x and y == gt_y:
            return VerifierResult(True, 1.0, f"Correct fundamental solution: x={x}, y={y}", extra)
        if substitution_ok:
            # Valid solution but not minimal
            return VerifierResult(
                False, 0.5,
                f"Valid Pell solution but not fundamental. Got x={x}, y={y}; expected x={gt_x}, y={gt_y}",
                extra,
            )
        return VerifierResult(
            False, 0.0,
            f"x^2 - 61*y^2 ≠ 1 for x={x}, y={y} (got {x*x - 61*y*y})",
            extra,
        )

    def _verify_eigenvalues(self, content: str) -> VerifierResult:
        from sympy import N
        from sympy.parsing.sympy_parser import parse_expr

        gt = _GT_EIGENVALUES
        symbolic = _extract_symbolic_list(content)
        parsed_symbolic: list[float] = []
        for s in symbolic:
            try:
                parsed_symbolic.append(float(N(parse_expr(s))))
            except Exception:
                pass

        if parsed_symbolic:
            floats_found = parsed_symbolic
        else:
            # Extract floats from response text only when exact symbolic list is absent.
            # This is secondary and cannot pass unless exactly 10 values match.
            floats_found = _extract_float_list(content)

        if not floats_found:
            return VerifierResult(
                False, 0.0,
                "No numeric values found in response",
                {"expected": [round(v, 6) for v in gt]},
            )

        # Check how many eigenvalues are within 0.1% of ground truth
        tol = 0.001
        matched = 0
        for gv in gt:
            for fv in floats_found:
                if gv > 0 and abs(fv - gv) / gv < tol:
                    matched += 1
                    break

        score = matched / len(gt)
        passed = matched == 10 and len(floats_found) == 10

        return VerifierResult(
            passed,
            score,
            f"{matched}/10 eigenvalues matched within 0.1% tolerance",
            {
                "expected": [round(v, 6) for v in gt],
                "matched_count": matched,
                "found_count": len(floats_found),
            },
        )

    def _verify_modular_arithmetic(self, content: str) -> VerifierResult:
        gt = _GT_MOD_2_100
        found = _extract_integer(content)
        if found is None:
            return VerifierResult(False, 0.0, f"No modular arithmetic answer found. Expected {gt}", {"expected": gt})
        passed = found == gt
        return VerifierResult(passed, 1.0 if passed else 0.0, f"Expected {gt}, found {found}" if not passed else f"Correct: {gt}", {"expected": gt, "found": found})

    def _verify_sum_of_divisors(self, content: str) -> VerifierResult:
        gt = _GT_SIGMA_360
        found = _extract_integer(content)
        factorization_ok = "2^3" in content or "2^3 * 3^2 * 5" in content or "360 = 2^3" in content
        sigma_ok = "sigma" in content.lower() or "(1+2+4+8)" in content or "15 * 13 * 6" in content
        if found is None:
            return VerifierResult(False, 0.0, f"No divisor-sum answer found. Expected {gt}", {"expected": gt})
        score = 0.4 if found == gt else 0.0
        if factorization_ok:
            score += 0.3
        if sigma_ok:
            score += 0.3
        passed = found == gt and factorization_ok and sigma_ok
        return VerifierResult(passed, round(score, 4), f"Expected {gt}, found {found}" if not passed else f"Correct: {gt}", {"expected": gt, "found": found, "factorization_ok": factorization_ok, "sigma_ok": sigma_ok})

    def _verify_birthday_probability(self, content: str) -> VerifierResult:
        gt = _GT_BIRTHDAY_23
        json_text, _ = recover_json_from_fence(content)
        found = None
        try:
            parsed = json.loads(json_text)
            if isinstance(parsed, dict):
                found = _coerce_float(parsed.get("final_answer"))
        except Exception:
            pass
        if found is None:
            found = _coerce_float_from_text(content)
        if found is None:
            return VerifierResult(False, 0.0, f"No birthday probability found. Expected {gt:.6f}", {"expected": gt})
        event_ok = "at least two" in content.lower() or "shared birthday" in content.lower() or "1 -" in content.lower()
        tolerance = 0.001
        value_ok = abs(found - gt) <= tolerance
        score = 0.7 if value_ok else 0.0
        if event_ok:
            score += 0.3
        passed = value_ok and event_ok
        return VerifierResult(passed, round(score, 4), f"Expected {gt:.6f}, found {found}" if not passed else f"Correct: {found:.6f}", {"expected": gt, "found": found, "event_ok": event_ok, "tolerance": tolerance})


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _coerce_float_from_text(text: str) -> float | None:
    values = re.findall(r"\b\d+\.\d+\b", text)
    if not values:
        return None
    return float(values[-1])


def verify_math_ground_truth() -> dict[str, Any]:
    """Return all ground-truth values for inspection/testing."""
    return {
        "trailing_zeros_1000_base12": _GT_TRAILING_ZEROS,
        "pell_x": _GT_PELL_X,
        "pell_y": _GT_PELL_Y,
        "pell_check": _GT_PELL_X**2 - 61 * _GT_PELL_Y**2,
        "eigenvalues_10x10": [round(v, 8) for v in _GT_EIGENVALUES],
        "modular_2_100": _GT_MOD_2_100,
        "sum_of_divisors_360": _GT_SIGMA_360,
        "birthday_probability_23": round(_GT_BIRTHDAY_23, 8),
    }
