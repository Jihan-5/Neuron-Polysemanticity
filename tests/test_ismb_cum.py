"""Cumulant-aware ISMB (``ismb_cum``): calibration of its two new tests, and behaviour on the
cases it exists for (collinear profiles, dependent latents) and against (near-Gaussian latents)."""

import numpy as np
import pytest

from polyid.bounds.ismb import ISMBConfig, _cross_cumulant_test, _fourth_cumulant_test, fit_ismb
from polyid.estimand import EstimandSpec
from polyid.estimators.registry import REGISTRY
from polyid.generators.planted import PlantedConfig, generate
from polyid.utils.seeding import SeedBank

SPEC = EstimandSpec(tau=0.10, v_min=0.006, max_abs_cos=0.98)


def test_fourth_cumulant_test_size_and_power():
    rng = np.random.default_rng(0)
    E, n = 5, 3000
    env = np.repeat(np.arange(E), n)
    p_gauss = [_fourth_cumulant_test(rng.standard_normal(E * n) * np.repeat(rng.uniform(0.5, 2, E), n), env, E)
               for _ in range(400)]
    assert 0.02 < np.mean(np.array(p_gauss) < 0.05) < 0.09  # size near 5 % under H0 (Gaussian)
    sparse = (rng.random(E * n) < 0.15) * rng.uniform(0.5, 1.5, E * n)
    assert _fourth_cumulant_test(sparse, env, E) < 1e-6  # sparse latents are clearly non-Gaussian


def test_cross_cumulant_test_size_and_power():
    rng = np.random.default_rng(1)
    E, n = 5, 3000
    env = np.repeat(np.arange(E), n)
    ps = []
    for _ in range(300):
        a = (rng.random(E * n) < 0.2) * rng.uniform(0.5, 1.5, E * n)
        b = (rng.random(E * n) < 0.2) * rng.uniform(0.5, 1.5, E * n)
        ps.append(_cross_cumulant_test(a - a.mean(), b - b.mean(), env, E))
    assert 0.02 < np.mean(np.array(ps) < 0.05) < 0.09  # size near 5 % under independence
    switch = rng.random(E * n) < 0.2  # shared switch: perfect co-occurrence
    a = switch * rng.uniform(0.5, 1.5, E * n)
    b = switch * rng.uniform(0.5, 1.5, E * n)
    assert _cross_cumulant_test(a - a.mean(), b - b.mean(), env, E) < 1e-6


def _fit(kw, cum, rep=0, n_boot=40):
    ds = generate(PlantedConfig(d=8, K=8, n_envs=5, n_per_env=4000, **kw), SeedBank(21).child(str(kw), rep),
                  v_min=SPEC.v_min)
    f = fit_ismb(ds.observed, SPEC, ISMBConfig(n_boot=n_boot, cumulants=cum), np.random.default_rng(rep))
    return f, ds.oracle.truth_at(SPEC.tau)


@pytest.mark.slow
def test_cumulants_resolve_a_collinear_pair_and_still_cover():
    f2, truth = _fit({"collinear_groups": [[0, 1]]}, cum=False)
    f4, _ = _fit({"collinear_groups": [[0, 1]]}, cum=True)
    for f in (f2, f4):
        assert np.all((f.lower <= truth) & (truth <= f.upper))
    assert np.mean(f4.upper - f4.lower) < np.mean(f2.upper - f2.lower)
    assert f4.diagnostics["n_groups"] == 0 and not f4.diagnostics["misfit_rejected"]


@pytest.mark.slow
def test_gaussian_latents_fall_back_to_second_moment_merging():
    f, truth = _fit({"collinear_groups": [[0, 1]], "latent_gaussianity": 1.0}, cum=True)
    assert f.diagnostics["n_nongaussian"] <= 1  # Gaussianity rarely rejected, so the pair is merged
    assert np.all((f.lower <= truth) & (truth <= f.upper))


@pytest.mark.slow
def test_cross_cumulants_flag_perfect_cooccurrence():
    f, truth = _fit({"switch_corr_blocks": [[0, 1]], "switch_corr": 1.0}, cum=True)
    assert f.diagnostics["n_cum_dependence_flags"] >= 1  # invisible to second moments (Prop. 2)
    assert np.all((f.lower <= truth) & (truth <= f.upper))


def test_registry_and_access_label():
    est = REGISTRY["ismb_cum"]({"n_boot": 5})
    assert est.name == "ismb_cum" and est.cfg.cumulants


def test_latent_gaussianity_keeps_truth_and_second_moments():
    a = generate(PlantedConfig(d=6, K=6, n_per_env=500), SeedBank(2).child("g"))
    b = generate(PlantedConfig(d=6, K=6, n_per_env=500, latent_gaussianity=0.7), SeedBank(2).child("g"))
    assert np.array_equal(a.oracle.truth_at(0.1), b.oracle.truth_at(0.1))
    assert np.allclose(a.oracle.A, b.oracle.A)


def test_ismb_cum_v2_flags_and_stability_widening():
    """v2 bundles the cumulant stack, the overcompleteness gate, stability widening and dependence
    tests on ambiguous pairs; forcing every column 'unstable' must raise U_j to K, never lower L_j."""
    from polyid.bounds.ismb import ISMB

    v2 = REGISTRY["ismb_cum_v2"]({"n_boot": 20})
    assert v2.name == "ismb_cum_v2"
    assert v2.cfg.cumulants and v2.cfg.oc_gate and v2.cfg.stab_widen and v2.cfg.dep_ambiguous
    ds = generate(PlantedConfig.from_dict({"d": 8, "K": 8, "n_per_env": 3000}), SeedBank(3).child("v2"), v_min=0.006)
    spec = EstimandSpec(tau=0.10, v_min=0.006, max_abs_cos=0.98)
    base = fit_ismb(ds.observed, spec, ISMBConfig(n_boot=20, cumulants=True), np.random.default_rng(0))
    wide = fit_ismb(ds.observed, spec, ISMBConfig(n_boot=20, cumulants=True, stab_widen=True, stab_cos=1.01),
                    np.random.default_rng(0))
    assert wide.diagnostics["n_unstable_cols"] == 8
    assert np.array_equal(wide.lower, base.lower) and np.all(wide.upper == 8)
    assert ISMB({"cumulants": True, "stab_widen": True, "dep_ambiguous": True}).name == "ismb_cum_stab_amb"
