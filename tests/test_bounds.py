import numpy as np
import pytest

from polyid.bounds.ar_grid import ARGridBounds, admissible_triples, fibonacci_hemisphere
from polyid.bounds.ismb import ISMBConfig, fit_ismb, ismb_population
from polyid.bounds.rank_tests import cragg_donald_rank1, test_pair as pair_test
from polyid.estimand import EstimandSpec
from polyid.generators.planted import PlantedConfig, generate
from polyid.identification import identified_set
from polyid.jointdiag import fit_joint_diag
from polyid.matching import align_to_reference
from polyid.moments import compute_moment_stats, population_stats
from polyid.utils.seeding import SeedBank


def test_cragg_donald_zero_for_exact_rank_one():
    a, b = np.array([1.0, 2.0, -1.0]), np.array([0.5, -1.0, 2.0, 1.0])
    stat, dof = cragg_donald_rank1(np.outer(a, b), np.eye(12) * 1e-4)
    assert stat < 1e-8 and dof == 6


def test_collinearity_test_size_and_power():
    rej0, rej1 = [], []
    for rep in range(60):
        cfg = PlantedConfig(d=4, K=4, n_envs=5, n_per_env=3000, collinear_groups=[[0, 1]], noise_rel_sd=0.1)
        ds = generate(cfg, SeedBank(7).child("size", rep))
        ms = compute_moment_stats(ds.observed, with_covariance=False)
        jd, _ = fit_joint_diag(ms.D, ms.pooled, n_starts=3, rng=np.random.default_rng(rep))
        A, _, _ = align_to_reference(jd.A, ds.oracle.A)
        U = ds.observed.X @ np.linalg.inv(A).T
        rej0.append(pair_test(U[:, 0], U[:, 1], ds.observed.env, 5, 0, 1).collinear_p < 0.05)
        rej1.append(pair_test(U[:, 2], U[:, 3], ds.observed.env, 5, 2, 3).collinear_p < 0.05)
    assert np.mean(rej0) < 0.15  # nominal 0.05, Monte Carlo slack for 60 reps
    assert np.mean(rej1) > 0.7


def test_population_procedure_identified_vs_not(spec, small_clean, small_collinear):
    for ds, expect_partial in ((small_clean, False), (small_collinear, True)):
        ps = population_stats(ds.oracle.pop_cov)
        pop = ismb_population(ps.D, ps.pooled, spec)
        truth = ds.oracle.truth
        assert np.all((pop["lower"] <= truth) & (truth <= pop["upper"]))
        assert (pop["upper"] - pop["lower"]).max() > 0 if expect_partial else np.all(pop["upper"] == pop["lower"])
        ident = identified_set(ds.oracle, spec)
        assert np.array_equal(ident.lower, pop["lower"]) and np.array_equal(ident.upper, pop["upper"])


def test_ismb_finite_sample_covers_clean_and_collinear(spec, small_clean, small_collinear):
    for ds in (small_clean, small_collinear):
        f = fit_ismb(ds.observed, spec, ISMBConfig(n_boot=60), np.random.default_rng(0))
        truth = ds.oracle.truth
        assert np.all((f.lower <= truth) & (truth <= f.upper))
    f = fit_ismb(small_collinear.observed, spec, ISMBConfig(n_boot=60), np.random.default_rng(0))
    assert len(f.groups) >= 1


def test_gms_selection_is_not_wider(spec, small_clean):
    a = fit_ismb(small_clean.observed, spec, ISMBConfig(n_boot=60, selection="gms"), np.random.default_rng(1))
    b = fit_ismb(small_clean.observed, spec, ISMBConfig(n_boot=60, selection="none"), np.random.default_rng(1))
    assert (a.upper - a.lower).sum() <= (b.upper - b.lower).sum()


def test_invisible_components_detected(spec):
    ds = generate(PlantedConfig(d=8, K=5, n_envs=5, n_per_env=20000), SeedBank(21), v_min=spec.v_min)
    f = fit_ismb(ds.observed, spec, ISMBConfig(n_boot=40), np.random.default_rng(0))
    assert f.diagnostics["n_invisible"] + f.diagnostics["n_ambiguous_visible"] >= 3
    assert np.all((f.lower <= ds.oracle.truth) & (ds.oracle.truth <= f.upper))


def test_fibonacci_grid_and_triples():
    D = fibonacci_hemisphere(60)
    assert D.shape == (60, 3) and np.allclose(np.linalg.norm(D, axis=1), 1) and np.all(D[:, 2] > 0)
    T = admissible_triples(D, 0.95)
    assert T.shape[1] == 3 and len(T) > 1000


def test_ar_grid_contains_truth_on_grid():
    spec = EstimandSpec(tau=0.3, v_min=0.006, max_abs_cos=0.95)
    ar = ARGridBounds({"n_dirs": 60})
    T = ar.triples(spec.max_abs_cos)
    rng = np.random.default_rng(0)
    A = np.stack([ar.dirs[i] for i in T[rng.integers(len(T))]], axis=1)
    ds = generate(PlantedConfig(d=3, K=3, n_envs=4, n_per_env=8000), SeedBank(1), A=A)
    J, dof, Tt = ar.criterion(ds.observed, spec)
    idx = [i for i, t in enumerate(Tt) if set(t.tolist()) == set(np.argmax(np.abs(ar.dirs @ A), axis=0).tolist())]
    assert idx, "truth must be an admissible grid triple"
    from scipy import stats

    assert J[idx[0]] < stats.chi2.ppf(0.9999, dof)
