"""Plan p.5 unit tests: verify known P_j from A; sign/permutation invariance."""

import numpy as np

from polyid.estimand import (
    canonical_signs,
    check_nonduplication,
    environmental_diversity,
    multiplicity,
    normalize_columns,
    smooth_multiplicity,
    visibility_stat,
)


def test_known_counts_from_handmade_A():
    A = np.array([[0.9, 0.0, 0.3],
                  [0.1, 0.8, 0.0],
                  [0.0, 0.6, 0.95]])
    An = normalize_columns(A)[0]
    expected = (np.abs(An) >= 0.2).sum(axis=1)
    assert np.array_equal(multiplicity(A, 0.2), expected)
    # row 1: 0.1 / ||(0.9, 0.1, 0)|| = 0.11 < 0.2, so only column 1 counts
    assert np.array_equal(multiplicity(A, 0.2), [2, 1, 2])


def test_closed_threshold_boundary():
    A = np.array([[0.6], [0.8]])  # unit norm already
    assert multiplicity(A, 0.6)[0] == 1  # |A| == tau counts (closed threshold)
    assert multiplicity(A, 0.6000001)[0] == 0


def test_sign_and_permutation_invariance():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((8, 6))
    P = multiplicity(A, 0.25)
    for _ in range(20):
        perm = rng.permutation(6)
        signs = rng.choice([-1.0, 1.0], size=6)
        assert np.array_equal(multiplicity(A[:, perm] * signs, 0.25), P)


def test_scale_invariance_via_normalisation():
    rng = np.random.default_rng(1)
    A = rng.standard_normal((5, 5))
    assert np.array_equal(multiplicity(A * rng.uniform(0.1, 10, size=5), 0.3), multiplicity(A, 0.3))


def test_visibility_mask_excludes_columns():
    A = np.eye(3) + 0.5
    assert multiplicity(A, 0.1, visible=np.array([True, False, True])).max() <= 2


def test_canonical_signs_positive_max():
    A = np.array([[-0.9, 0.2], [0.1, -0.8]])
    C = canonical_signs(A)
    assert np.all(C[np.argmax(np.abs(C), axis=0), [0, 1]] > 0)


def test_nonduplication_detects_near_copies():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.99, 0.141, 0.0])
    ok, pair, worst = check_nonduplication(np.stack([a, b, [0, 0, 1.0]], axis=1), 0.95)
    assert not ok and pair == (0, 1) and worst > 0.95


def test_visibility_and_diversity():
    prof = np.array([[1.0, -1.0, 0.0], [2.0, -2.0, 0.0], [0.0, 1.0, -1.0]])
    assert np.allclose(visibility_stat(prof)[:2], [np.sqrt(2 / 3), np.sqrt(8 / 3)])
    assert environmental_diversity(prof) < 1e-7  # rows 0 and 1 collinear


def test_smooth_surrogate_brackets_count():
    rng = np.random.default_rng(2)
    A = rng.standard_normal((6, 6))
    s = smooth_multiplicity(A, 0.3, bandwidth=1e-4)
    assert np.allclose(s, multiplicity(A, 0.3), atol=1e-2)
