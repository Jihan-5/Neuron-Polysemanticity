import numpy as np
import pytest

from polyid.bounds.schur_horn import (
    brute_force_rotation_counts,
    group_ranges,
    is_majorized,
    nonneg_diagonal_count_range,
    rotation_group_ranges,
    span_group_ranges,
)


def test_majorization_basic():
    assert is_majorized([1, 1, 1], [3, 0, 0])
    assert is_majorized([2, 1, 0], [3, 0, 0])
    assert not is_majorized([3, 0, 0], [1, 1, 1])
    assert not is_majorized([1, 1], [3, 0])  # different sums


def test_count_range_trace_rules():
    assert nonneg_diagonal_count_range([1.0, 0.5, 0.2]) == (3, 3)
    assert nonneg_diagonal_count_range([-1.0, -0.5]) == (0, 0)
    assert nonneg_diagonal_count_range([1.0, -0.4]) == (1, 2)  # trace > 0: max g, min 1
    assert nonneg_diagonal_count_range([1.0, -2.0]) == (0, 1)  # trace < 0: min 0


def _angle_sweep(M, tau, j, n=20000):
    ths = np.linspace(0, np.pi, n, endpoint=False)
    counts = []
    for t in ths:
        R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        C = M @ R
        C = C / np.linalg.norm(C, axis=0)
        counts.append(int(np.sum(np.abs(C[j]) >= tau)))
    return min(counts), max(counts)


@pytest.mark.parametrize("seed", range(25))
def test_g2_exact_against_angle_sweep(seed):
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((5, 2)) * rng.uniform(0.2, 1.0, size=2)
    tau = float(rng.choice([0.1, 0.3, 0.5, 0.7]))
    lo, hi = rotation_group_ranges(M, tau)
    for j in range(5):
        assert (lo[j], hi[j]) == _angle_sweep(M, tau, j)


@pytest.mark.parametrize("seed", range(10))
def test_g3_range_contains_random_rotations(seed):
    rng = np.random.default_rng(100 + seed)
    M = rng.standard_normal((6, 3))
    lo, hi = rotation_group_ranges(M, 0.35)
    for j in range(6):
        blo, bhi = brute_force_rotation_counts(M, 0.35, j, n_samples=1500, rng=rng)
        assert lo[j] <= blo and bhi <= hi[j]


def test_singleton_group_reduces_to_threshold():
    a = np.array([[0.9], [0.05], [0.4]])
    lo, hi = rotation_group_ranges(a, 0.3)
    assert lo.tolist() == hi.tolist() == [1, 0, 1]


def test_span_outer_bound_contains_rotation_range():
    rng = np.random.default_rng(3)
    M = rng.standard_normal((6, 3))
    rlo, rhi = rotation_group_ranges(M, 0.3)
    slo, shi = span_group_ranges(M, 0.3)
    assert np.all(slo <= rlo) and np.all(shi >= rhi)


def test_mixed_sign_dispatches_to_span():
    rng = np.random.default_rng(4)
    A = rng.standard_normal((5, 2))
    _, _, mode = group_ranges(A, np.array([1.0, -0.5]), 0.3)
    assert mode == "span"
    _, _, mode = group_ranges(A, np.array([1.0, 0.5]), 0.3)
    assert mode == "rotation"
