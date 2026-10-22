"""Fourth-order cumulant rank test for overcompleteness and the ISMB gate that uses it."""

import numpy as np

from polyid.bounds.overcomplete import overcomplete_test, quadricovariance
from polyid.data import ObservedData
from polyid.estimand import EstimandSpec
from polyid.estimators import build_estimator


def _sparse_mixture(d, K, rng, E=4, n=4000, p=0.05, noise_sd=0.0):
    A = rng.normal(size=(d, K))
    A /= np.linalg.norm(A, axis=0)
    rates = np.clip(p * rng.uniform(0.4, 1.6, size=(E, K)), 1e-4, 1.0)
    X, env = [], []
    for e in range(E):
        z = (rng.random((n, K)) < rates[e]) * rng.random((n, K))
        X.append(z @ A.T + noise_sd * rng.normal(size=(n, d)))
        env.append(np.full(n, e))
    return np.concatenate(X), np.concatenate(env)


def test_quadricovariance_rank_counts_latents_past_d():
    rng = np.random.default_rng(0)
    for K in (5, 9):
        X, env = _sparse_mixture(5, K, rng, n=20000)
        s = np.linalg.svd(quadricovariance(X, env), compute_uv=False)
        gap = s[K - 1] / s[K] if K < len(s) else np.inf
        assert gap > 3, (K, s[: K + 2])


def test_size_when_latents_fit_in_d():
    rng = np.random.default_rng(1)
    rejections = sum(overcomplete_test(*_sparse_mixture(6, 6, rng, noise_sd=0.05), n_boot=30,
                                       rng=np.random.default_rng(r)).rejected for r in range(8))
    assert rejections <= 1


def test_power_and_rank_lower_bound_when_overcomplete():
    rng = np.random.default_rng(2)
    X, env = _sparse_mixture(6, 12, rng, n=8000)
    t = overcomplete_test(X, env, n_boot=30, rng=rng)
    assert t.rejected and t.ratio > 1
    assert 6 < t.rank_lower_bound <= 12


def test_ismb_oc_abstains_on_overcomplete_data_and_is_labelled():
    rng = np.random.default_rng(3)
    X, env = _sparse_mixture(6, 12, rng, n=8000)
    data = ObservedData(X=X, env=env, neuron_ids=np.arange(6), meta={})
    spec = EstimandSpec(tau=0.3, v_min=0.002, max_abs_cos=0.98, k_max=12)
    res = build_estimator("ismb_oc", {"n_boot": 30, "oc_n_boot": 30}).fit(data, spec, np.random.default_rng(0))
    assert res.method == "ismb_oc" and res.access.higher_moments
    assert res.diagnostics["oc_rejected"]
    assert np.all(res.lower == 0) and np.all(res.upper == 12)
    assert build_estimator("ismb", {}).name == "ismb" and build_estimator("ismb_cum_oc", {}).name == "ismb_cum_oc"
