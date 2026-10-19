r"""Baselines (plan p.7: "Fit univariate moments, a GMM with model selection, a
DPMI implementation, PCA/factor analysis, FastICA, multi-environment joint
diagonalization, and a sparse dictionary on the same training observations").

Fairness conventions shared by every method here
------------------------------------------------
* All methods see the same rows. Single-environment methods receive the
  pooled data (all environments concatenated) and are marked
  ``env_labels=False`` in their :class:`Access` record.
* Methods that need a component count get the *declared* bound
  ``spec.k_max`` (default ``d``) — never the true ``K``.
* Loading-based counts use the same normalisation and threshold rule as the
  estimand (:func:`polyid.estimand.multiplicity`).
* Score-only methods (moments, DPMI-style) are evaluated by ranking metrics,
  and by count MAE only after a development-split calibration map
  (:class:`DevCalibratedCountMap`), never on the test split.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy import stats
from sklearn.decomposition import PCA, FactorAnalysis, FastICA, MiniBatchDictionaryLearning
from sklearn.exceptions import ConvergenceWarning
from sklearn.isotonic import IsotonicRegression
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture

from polyid.data import ObservedData
from polyid.estimand import EstimandSpec, multiplicity, normalize_columns
from polyid.estimators.base import Access, EstimateResult
from polyid.jointdiag import fit_joint_diag
from polyid.moments import bootstrap_weights, centered_stack, compute_moment_stats, env_covariances


def _n_factors(data: ObservedData, spec: EstimandSpec) -> int:
    return int(spec.k_max or data.d)


def _subsample(X: np.ndarray, max_rows: int, rng: np.random.Generator) -> np.ndarray:
    if X.shape[0] <= max_rows:
        return X
    return X[rng.choice(X.shape[0], size=max_rows, replace=False)]


def _env_centered(data: ObservedData) -> np.ndarray:
    """Remove per-environment means (the same preprocessing the moment methods use)."""
    X = np.array(data.X, dtype=float, copy=True)
    for e in range(data.n_envs):
        idx = data.env == e
        X[idx] -= X[idx].mean(axis=0)
    return X


# --------------------------------------------------------------------------
# per-neuron distributional scores
# --------------------------------------------------------------------------

class UnivariateMoments:
    """Excess kurtosis (score) and Sarle's bimodality coefficient per neuron."""

    name = "moments"

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or {}

    def fit(self, data, spec, rng):
        X = data.X
        kurt = stats.kurtosis(X, axis=0, fisher=True, bias=False)
        skew = stats.skew(X, axis=0, bias=False)
        n = X.shape[0]
        bc = (skew**2 + 1) / (kurt + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))
        return EstimateResult(
            method=self.name, kind="score", score=kurt,
            access=Access(env_labels=False, higher_moments=True, n_observations=n),
            diagnostics={"bimodality_coefficient": bc.tolist(), "skew": skew.tolist()},
        )


class GMMBIC:
    """1-D Gaussian mixture per neuron, ``k in 1..k_max`` chosen by BIC.

    Count proxy = ``k* - 1`` (the 'all-features-off' mode is one component).
    This is a proxy, *not* the structural estimand; reported as such.
    """

    name = "gmm_bic"

    def __init__(self, cfg: dict | None = None):
        cfg = cfg or {}
        self.k_max = int(cfg.get("k_max", 6))
        self.max_rows = int(cfg.get("max_rows", 5000))

    def fit(self, data, spec, rng):
        X = _subsample(data.X, self.max_rows, rng)
        ks = np.zeros(data.d, dtype=int)
        seed = int(rng.integers(2**31 - 1))
        for j in range(data.d):
            x = X[:, [j]]
            bics = []
            for k in range(1, self.k_max + 1):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    gm = GaussianMixture(k, n_init=2, random_state=seed, reg_covar=1e-6).fit(x)
                bics.append(gm.bic(x))
            ks[j] = int(np.argmin(bics)) + 1
        return EstimateResult(
            method=self.name, kind="count", point=np.maximum(ks - 1, 0).astype(float), score=ks.astype(float),
            access=Access(env_labels=False, higher_moments=True, n_observations=X.shape[0],
                          model_selection_budget=self.k_max, restarts=2),
            label="proxy: mixture components - 1",
        )


class DPMIStyle:
    r"""Dirichlet-process mixture index in the spirit of DPMI (syllabus S17).

    Our approximation, *not* a verified reimplementation: per neuron fit a
    truncated DP Gaussian mixture; keep components with weight
    ``>= w_min``; index ``= (N_eff - 1) * mean pairwise separation`` with
    separation ``1 - exp(-D/2)`` from Ashman's ``D``. Consult the DPMI paper's
    exact definition before labelling any result "DPMI".
    """

    name = "dpmi_style"

    def __init__(self, cfg: dict | None = None):
        cfg = cfg or {}
        self.k_max = int(cfg.get("k_max", 8))
        self.w_min = float(cfg.get("w_min", 0.02))
        self.max_rows = int(cfg.get("max_rows", 4000))
        self.concentration = float(cfg.get("concentration", 0.1))

    def fit(self, data, spec, rng):
        X = _subsample(data.X, self.max_rows, rng)
        seed = int(rng.integers(2**31 - 1))
        idx = np.zeros(data.d)
        neff = np.zeros(data.d, dtype=int)
        for j in range(data.d):
            x = X[:, [j]]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                bgm = BayesianGaussianMixture(
                    n_components=self.k_max, weight_concentration_prior_type="dirichlet_process",
                    weight_concentration_prior=self.concentration, max_iter=300, random_state=seed,
                    reg_covar=1e-6,
                ).fit(x)
            keep = bgm.weights_ >= self.w_min
            mu = bgm.means_[keep, 0]
            sd = np.sqrt(bgm.covariances_[keep].reshape(-1))
            m = int(keep.sum())
            neff[j] = m
            if m < 2:
                continue
            seps = []
            for a in range(m):
                for b in range(a + 1, m):
                    D = np.sqrt(2.0) * abs(mu[a] - mu[b]) / np.sqrt(sd[a] ** 2 + sd[b] ** 2)
                    seps.append(1.0 - np.exp(-D / 2.0))
            idx[j] = (m - 1) * float(np.mean(seps))
        return EstimateResult(
            method=self.name, kind="score", score=idx,
            access=Access(env_labels=False, higher_moments=True, n_observations=X.shape[0],
                          model_selection_budget=self.k_max),
            diagnostics={"n_effective_components": neff.tolist()},
            label="DPMI-style approximation (see docstring)",
        )


# --------------------------------------------------------------------------
# loading-matrix baselines
# --------------------------------------------------------------------------

def _count_result(name, A, data, spec, access, label="", visible=None, diagnostics=None):
    return EstimateResult(
        method=name, kind="count", point=multiplicity(A, spec.tau, visible=visible).astype(float),
        loadings=normalize_columns(A)[0], access=access, label=label, diagnostics=diagnostics or {},
    )


class PCACount:
    name = "pca"

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or {}

    def fit(self, data, spec, rng):
        X = _env_centered(data)
        k = _n_factors(data, spec)
        pca = PCA(n_components=k).fit(X)
        return _count_result(self.name, pca.components_.T, data, spec,
                             Access(env_labels=False, higher_moments=False, n_observations=data.n))


class FactorAnalysisCount:
    name = "factor_analysis"

    def __init__(self, cfg: dict | None = None):
        cfg = cfg or {}
        self.rotation = cfg.get("rotation", "varimax")

    def fit(self, data, spec, rng):
        X = _env_centered(data)
        k = min(_n_factors(data, spec), data.d)
        fa = FactorAnalysis(n_components=k, rotation=self.rotation, random_state=int(rng.integers(2**31 - 1)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            fa.fit(X)
        return _count_result(self.name, fa.components_.T, data, spec,
                             Access(env_labels=False, higher_moments=False, n_observations=data.n),
                             label=f"rotation={self.rotation}")


class FastICACount:
    """FastICA on pooled data: identification from non-Gaussianity, not environments."""

    name = "fastica"

    def __init__(self, cfg: dict | None = None):
        cfg = cfg or {}
        self.max_iter = int(cfg.get("max_iter", 1000))
        self.fun = cfg.get("fun", "logcosh")

    def fit(self, data, spec, rng):
        X = _env_centered(data)
        k = min(_n_factors(data, spec), data.d)
        ica = FastICA(n_components=k, whiten="unit-variance", max_iter=self.max_iter, fun=self.fun,
                      random_state=int(rng.integers(2**31 - 1)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            ica.fit(X)
        return _count_result(self.name, ica.mixing_, data, spec,
                             Access(env_labels=False, higher_moments=True, n_observations=data.n))


class JointDiagCount:
    """Multi-environment joint diagonalisation point estimate (coroICA-style).

    With ``visibility_filter`` (default) components whose estimated
    visibility is below ``v_min`` are not counted — the same rule the
    estimand uses, so the baseline is not a straw man when ``K < d``.
    """

    name = "jointdiag"

    def __init__(self, cfg: dict | None = None):
        cfg = cfg or {}
        self.n_starts = int(cfg.get("n_starts", 6))
        self.visibility_filter = bool(cfg.get("visibility_filter", True))

    def fit(self, data, spec, rng):
        ms = compute_moment_stats(data, with_covariance=False)
        best, runs = fit_joint_diag(ms.D, ms.pooled, n_starts=self.n_starts, rng=rng)
        A, norms = normalize_columns(best.A)
        B = norms[:, None] * best.V
        prof = np.einsum("ki,eij,kj->ke", B, ms.D, B)
        vis = np.sqrt(np.mean(prof**2, axis=1))
        visible = vis >= spec.v_min if self.visibility_filter else None
        return _count_result(
            self.name if self.visibility_filter else self.name + "_nofilter", A, data, spec,
            Access(env_labels=True, higher_moments=False, n_observations=data.n, restarts=self.n_starts),
            visible=visible, diagnostics={"objective": best.objective, "visibility": vis.tolist()},
        )


class DictLearningCount:
    """Sparse dictionary (L1 sparse coding) on pooled, environment-centred data."""

    name = "dict_learning"

    def __init__(self, cfg: dict | None = None):
        cfg = cfg or {}
        self.alpha = float(cfg.get("alpha", 0.1))
        self.max_rows = int(cfg.get("max_rows", 10000))
        self.max_iter = int(cfg.get("max_iter", 50))
        self.atoms_multiplier = float(cfg.get("atoms_multiplier", 1.0))

    def fit(self, data, spec, rng):
        X = _subsample(_env_centered(data), self.max_rows, rng)
        k = int(round(_n_factors(data, spec) * self.atoms_multiplier))
        dl = MiniBatchDictionaryLearning(n_components=k, alpha=self.alpha, batch_size=256, max_iter=self.max_iter,
                                         random_state=int(rng.integers(2**31 - 1)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            codes = dl.fit_transform(X)
        alive = np.mean(np.abs(codes) > 1e-8, axis=0) > 0.001
        return _count_result(self.name, dl.components_.T, data, spec,
                             Access(env_labels=False, higher_moments=True, n_observations=X.shape[0]),
                             visible=alive, label=f"alpha={self.alpha}, atoms={k}",
                             diagnostics={"dead_atoms": int((~alive).sum())})


class BootstrapPercentile:
    """Naive uncertainty: percentile interval of the joint-diagonalisation count.

    Included because it is what one would do by default — and because it is
    expected to fail under non-identification (the count of an arbitrary
    rotation is resampled, not the identified set).
    """

    name = "boot_percentile"

    def __init__(self, cfg: dict | None = None):
        cfg = cfg or {}
        self.n_boot = int(cfg.get("n_boot", 200))
        self.alpha = float(cfg.get("alpha", 0.05))
        self.n_starts = int(cfg.get("n_starts", 6))

    def fit(self, data, spec, rng):
        ms = compute_moment_stats(data, with_covariance=False)
        best, _ = fit_joint_diag(ms.D, ms.pooled, n_starts=self.n_starts, rng=rng)
        counts = np.empty((self.n_boot, data.d))
        for b in range(self.n_boot):
            w = bootstrap_weights(data, rng)
            C, _, _ = env_covariances(data.X, data.env, data.n_envs, weights=w)
            D = centered_stack(C)
            jd, _ = fit_joint_diag(D, C.mean(axis=0), init=best.V, max_iter=300)
            A, nrm = normalize_columns(jd.A)
            B = nrm[:, None] * jd.V
            vis = np.sqrt(np.mean(np.einsum("ki,eij,kj->ke", B, D, B) ** 2, axis=1))
            counts[b] = multiplicity(A, spec.tau, visible=vis >= spec.v_min, normalize=False)
        lo = np.quantile(counts, self.alpha / 2, axis=0, method="lower")
        hi = np.quantile(counts, 1 - self.alpha / 2, axis=0, method="higher")
        A0, n0 = normalize_columns(best.A)
        B0 = n0[:, None] * best.V
        vis0 = np.sqrt(np.mean(np.einsum("ki,eij,kj->ke", B0, ms.D, B0) ** 2, axis=1))
        point = multiplicity(A0, spec.tau, visible=vis0 >= spec.v_min, normalize=False)
        return EstimateResult(
            method=self.name, kind="interval", point=point.astype(float), lower=lo, upper=hi, loadings=A0,
            label="naive percentile bootstrap (no identification analysis)",
            access=Access(env_labels=True, higher_moments=False, n_observations=data.n, restarts=self.n_starts,
                          notes=f"bootstrap={self.n_boot}"),
        )


class DevCalibratedCountMap:
    """Monotone map from a score to a count, fitted on *development* datasets only."""

    def __init__(self, k_max: int):
        self.k_max = k_max
        self.iso = IsotonicRegression(out_of_bounds="clip", increasing=True)

    def fit(self, scores: np.ndarray, truth: np.ndarray) -> "DevCalibratedCountMap":
        self.iso.fit(np.asarray(scores, dtype=float), np.asarray(truth, dtype=float))
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        return np.clip(np.rint(self.iso.predict(np.asarray(scores, dtype=float))), 0, self.k_max)
