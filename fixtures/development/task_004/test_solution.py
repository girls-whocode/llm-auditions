import pytest


BUGGY = """
def find_max_subarray_sum(nums):
    max_sum = 0
    current_sum = 0
    for n in nums:
        current_sum += n
        if current_sum > max_sum:
            max_sum = current_sum
        if current_sum < 0:
            current_sum = 0
    return max_sum
"""


def test_regression_case_against_buggy_reference(tmp_path):
    ns = {}
    exec(BUGGY, ns)
    assert ns["find_max_subarray_sum"]([-2, -3, -1]) == 0


def test_fixed_solution_handles_all_negative():
    from solution import find_max_subarray_sum
    assert find_max_subarray_sum([-2, -3, -1]) == -1
