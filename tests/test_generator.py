import numpy as np
import pytest

from polyid.estimand import normalize_columns
from polyid.generators.planted import (
    PlantedConfig,
    check_two_explanations,
    generate,
    latent_covariance,
    sample_loading_matrix,
)
from polyid.utils.seeding import SeedBank


def test_loading_matrix_constraints():
    cfg = PlantedConfig()
    A, exempt = sample_loading_matrix(cfg, np.random.default_rng(0))
    assert np.allclose(np.linalg.norm(A, axis=0), 1.0)
    assert np.linalg.cond(A) <= cfg.max_cond
    G = np.abs(A.T @ A)
    np.fill_diagonal(G, 0)
    assert G.max() <= cfg.max_abs_cos
    absA = np.abs(A)
    assert np.all((absA >= cfg.tau + cfg.tau_margin) | (absA <= cfg.tau - cfg.tau_margin))
    assert exempt == []


def test_sample_covariance_matches_population():
    cfg = PlantedConfig(d=5, K=5, n_envs=3, n_per_env=200_000, noise_rel_sd=0.2)
    ds = generate(cfg, SeedBank(3))
    X, env = ds.observed.X, ds.observed.env
    for e in range(3):
        C = np.cov(X[env == e].T)
        assert np.max(np.abs(C - ds.oracle.pop_cov[e])) < 0.01


def test_latent_covariance_with_copulas_matches_sampling():
    cfg = PlantedConfig(d=4, K=4, n_envs=1, n_per_env=300_000, switch_corr_blocks=[[0, 1]], switch_corr=0.6,
                        magnitude_corr_blocks=[[2, 3]], magnitude_corr=0.7, store_latents=True, min_visibility=0.0)
    ds = generate(cfg, SeedBank(5))
    emp = np.cov(ds.oracle.latents.T)
    assert np.max(np.abs(emp - ds.oracle.latent_cov[0])) < 0.004
    assert not ds.oracle.independent


def test_collinear_groups_have_identical_profiles():
    ds = generate(PlantedConfig(collinear_groups=[[0, 1, 2]]), SeedBank(7))
    P = ds.oracle.profiles
    assert np.allclose(P[0], P[1]) and np.allclose(P[0], P[2])
    assert ds.oracle.diversity < 1e-9


def test_invisible_factors_have_zero_profile():
    ds = generate(PlantedConfig(invisible_factors=[3]), SeedBank(8))
    assert np.allclose(ds.oracle.profiles[3], 0.0)
    assert not ds.oracle.visible[3]


def test_perfect_cooccurrence_shares_switches():
    cfg = PlantedConfig(d=4, K=4, n_envs=2, n_per_env=5000, switch_corr_blocks=[[0, 1]], switch_corr=1.0,
                        store_latents=True, min_visibility=0.0)
    ds = generate(cfg, SeedBank(9))
    Z = ds.oracle.latents
    assert np.array_equal(Z[:, 0] > 0, Z[:, 1] > 0)


def test_fixed_A_frequency_shift_keeps_truth():
    base = generate(PlantedConfig(), SeedBank(10).child("a"))
    other = generate(PlantedConfig(rate_center=0.3), SeedBank(10).child("b"), A=base.oracle.A)
    assert np.array_equal(base.oracle.truth, other.oracle.truth)
    assert not np.allclose(base.oracle.rates, other.oracle.rates)


def test_noise_modes_and_stationarity_flags():
    for mode, stationary in [("iid", True), ("correlated_stationary", True), ("student_t", True),
                             ("env_scaled", False), ("env_heteroscedastic", False)]:
        ds = generate(PlantedConfig(d=4, K=4, n_per_env=500, noise_mode=mode, noise_rel_sd=0.2), SeedBank(12))
        assert ds.oracle.noise_stationary is stationary


def test_two_explanations_equivalence_helper():
    rng = np.random.default_rng(0)
    A = normalize_columns(rng.standard_normal((4, 4)))[0]
    v = rng.uniform(0.05, 0.2, size=(3, 4))
    assert check_two_explanations(A, v, A, v) == 0.0


def test_overcomplete_generation():
    ds = generate(PlantedConfig(K=24, max_cond=1e9), SeedBank(13))
    assert ds.oracle.A.shape == (16, 24)


def test_truth_at_uses_the_requested_threshold():
    """Regression: scoring must use spec.tau, not the generator's own tau field."""
    from polyid.estimand import multiplicity

    ds = generate(PlantedConfig(d=6, K=6), SeedBank(14))
    for t in (0.05, 0.1, 0.3, 0.6):
        assert np.array_equal(ds.oracle.truth_at(t), multiplicity(ds.oracle.A, t, visible=ds.oracle.visible))
    assert not np.array_equal(ds.oracle.truth_at(0.1), ds.oracle.truth_at(0.6))
