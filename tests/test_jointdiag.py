"""Plan p.5: zero-noise recovery in an identifiable setting; failure on ambiguous settings."""

import numpy as np

from polyid.generators.planted import PlantedConfig, generate
from polyid.jointdiag import fit_joint_diag, gevd_diag
from polyid.matching import amari_distance, match_columns
from polyid.moments import compute_moment_stats, population_stats
from polyid.utils.seeding import SeedBank


def test_population_recovery_is_exact():
    for seed in range(6):
        ds = generate(PlantedConfig(noise_rel_sd=0.2), SeedBank(seed).child("jd"))
        ps = population_stats(ds.oracle.pop_cov)
        best, _ = fit_joint_diag(ps.D, ps.pooled, n_starts=4, rng=np.random.default_rng(seed))
        assert amari_distance(best.V, ds.oracle.A) < 1e-8


def test_every_start_reaches_correct_basin_on_sample_data():
    ds = generate(PlantedConfig(noise_rel_sd=0.2), SeedBank(5).child("jd-sample"))
    ms = compute_moment_stats(ds.observed, with_covariance=False)
    best, runs = fit_joint_diag(ms.D, ms.pooled, n_starts=6, rng=np.random.default_rng(0))
    assert amari_distance(best.V, ds.oracle.A) < 0.15
    assert all(amari_distance(r.V, ds.oracle.A) < 0.2 for r in runs)


def test_gevd_exact_for_two_matrices():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((4, 4))
    M1 = A @ np.diag([1.0, 2.0, 3.0, 4.0]) @ A.T
    M2 = A @ np.diag([1.0, -1.0, 0.5, 2.0]) @ A.T
    V = gevd_diag(M1, M2)
    assert amari_distance(V, A) < 1e-8


def test_collinear_pair_is_not_recovered_but_others_are():
    ds = generate(PlantedConfig(d=6, K=6, collinear_groups=[[0, 1]], noise_rel_sd=0.05), SeedBank(3).child("amb"))
    ps = population_stats(ds.oracle.pop_cov)
    best, _ = fit_joint_diag(ps.D, ps.pooled, n_starts=4, rng=np.random.default_rng(0))
    r, c, cos = match_columns(best.A, ds.oracle.A)
    cos_by_true = dict(zip(c, cos))
    assert min(cos_by_true[k] for k in range(2, 6)) > 1 - 1e-6  # identified factors exact
    # the collinear pair is only identified as a 2-D subspace
    Q = np.linalg.qr(ds.oracle.A[:, :2])[0]
    est = best.A[:, [i for i, cc in zip(r, c) if cc in (0, 1)]]
    est = est / np.linalg.norm(est, axis=0)
    assert np.allclose(np.linalg.norm(Q.T @ est, axis=0), 1.0, atol=1e-6)


def test_pair_sin2_flags_collinearity():
    ds = generate(PlantedConfig(d=6, K=6, collinear_groups=[[0, 1]]), SeedBank(4).child("sin"))
    ps = population_stats(ds.oracle.pop_cov)
    best, _ = fit_joint_diag(ps.D, ps.pooled, n_starts=4, rng=np.random.default_rng(0))
    iu = np.triu_indices(6, 1)
    assert best.pair_sin2[iu].min() < 1e-6
