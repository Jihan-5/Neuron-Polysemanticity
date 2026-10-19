r"""AR-grid: a weak-identification–robust confidence set for the small certified problem.

Plan p.8, "Small certified starting problem": use ``d = 3`` with bounded
``K`` and enumerate where feasible; "a finite dictionary grid yields a
grid-restricted statement, not a continuous-model guarantee". Syllabus S22
(Chernozhukov–Hong–Tamer): write ``C_n`` as a level set of a criterion.

Construction
------------
Parameters: ``A`` (``K = d`` unit columns, from a finite grid ``𝒢`` of
directions, admissible under non-duplication) and the centred profiles
``c`` (nuisance, enters linearly). Moments: ``m = (vech D̂_0, …, vech D̂_{E-2})``
with model ``m(A, c) = (I_q ⊗ G(A)) vec(c)``, ``G(A) = [vech(a_k a_k^T)]_k``.
The profiled, efficiently weighted criterion

.. math:: J(A) = \min_c \; (\hat m - X(A)c)^\top \hat V^{-1} (\hat m - X(A)c)

is, at the true ``A``, asymptotically :math:`\chi^2_{\dim m - qK}` *whether
or not ``A`` is identified* — the distribution at the truth does not involve
identification strength (Anderson–Rubin logic). Hence

.. math:: C_n = \{A \in 𝒢 : J(A) \le \chi^2_{\dim m - qK, 1-\alpha}\}

contains the true ``A`` with asymptotic probability ``>= 1 - alpha`` provided
the truth lies on the grid, and the projection
``[min_{A∈C_n} P_j(A), max_{A∈C_n} P_j(A)]`` inherits that coverage (it is
conservative: projection of a joint set). An **empty** ``C_n`` is evidence
against the declared model (e.g. co-occurrence), not a narrow answer.

Declared restrictions of this certified problem: ``K = d`` factors, all
counted (visible), latents conditionally independent, stationary noise,
grid-restricted parameter space.

The smooth surrogate range (sum of sigmoids) is reported alongside, as
syllabus S23 suggests, to show where smooth-functional inference would and
would not apply.
"""

from __future__ import annotations

import itertools

import numpy as np
from scipy import stats

from polyid.data import ObservedData
from polyid.estimand import EstimandSpec
from polyid.estimators.base import Access, EstimateResult
from polyid.moments import compute_moment_stats


def fibonacci_hemisphere(n: int) -> np.ndarray:
    """``n`` near-uniform unit vectors in R^3 with a canonical sign (z > 0)."""
    m = 2 * n
    i = np.arange(m) + 0.5
    phi = np.arccos(1 - 2 * i / m)
    theta = np.pi * (1 + 5**0.5) * i
    P = np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1)
    P = P[P[:, 2] > 0]
    return P[:n]


def admissible_triples(dirs: np.ndarray, max_abs_cos: float, min_abs_det: float = 0.05) -> np.ndarray:
    G = np.abs(dirs @ dirs.T)
    n = dirs.shape[0]
    out = []
    for i, j in itertools.combinations(range(n), 2):
        if G[i, j] > max_abs_cos:
            continue
        ks = np.arange(j + 1, n)
        ok = (G[i, ks] <= max_abs_cos) & (G[j, ks] <= max_abs_cos)
        for k in ks[ok]:
            out.append((i, j, k))
    T = np.array(out, dtype=np.int32).reshape(-1, 3)
    if T.size:
        dets = np.abs(np.linalg.det(np.stack([dirs[T[:, 0]], dirs[T[:, 1]], dirs[T[:, 2]]], axis=2)))
        T = T[dets >= min_abs_det]
    return T


def _vech_outer(a: np.ndarray) -> np.ndarray:
    """``a`` is (..., 3) -> (..., 6) vech(a a^T) in triu row-major order."""
    i, j = np.triu_indices(3)
    return a[..., i] * a[..., j]


class ARGridBounds:
    name = "ar_grid"

    def __init__(self, cfg: dict | None = None):
        cfg = cfg or {}
        self.n_dirs = int(cfg.get("n_dirs", 120))
        self.alpha = float(cfg.get("alpha", 0.05))
        self.chunk = int(cfg.get("chunk", 40000))
        self.min_abs_det = float(cfg.get("min_abs_det", 0.05))
        self.bandwidth = float(cfg.get("bandwidth", 0.02))
        self.dirs = fibonacci_hemisphere(self.n_dirs)
        self._triples_cache: dict = {}

    def triples(self, max_abs_cos: float) -> np.ndarray:
        key = round(max_abs_cos, 6)
        if key not in self._triples_cache:
            self._triples_cache[key] = admissible_triples(self.dirs, max_abs_cos, self.min_abs_det)
        return self._triples_cache[key]

    def criterion(self, data: ObservedData, spec: EstimandSpec) -> tuple[np.ndarray, int, np.ndarray]:
        """Return ``(J for every admissible triple, dof, triples)``."""
        if data.d != 3:
            raise ValueError("AR-grid is implemented for the d = 3 certified problem only")
        ms = compute_moment_stats(data, with_covariance=True)
        m = ms.moment_vector()
        V = ms.moment_covariance()
        W = np.linalg.pinv(V, hermitian=True)
        E = ms.n_envs
        q, K, p = E - 1, 3, 6
        W4 = W.reshape(q, p, q, p).transpose(0, 2, 1, 3)  # (a, b, i, j)
        y = (W @ m).reshape(q, p)
        mWm = float(m @ W @ m)
        T = self.triples(spec.max_abs_cos)
        J = np.empty(T.shape[0])
        for s in range(0, T.shape[0], self.chunk):
            tt = T[s:s + self.chunk]
            Gm = np.stack([_vech_outer(self.dirs[tt[:, k]]) for k in range(K)], axis=2)  # (t, 6, K)
            WG = np.einsum("abij,tjl->tabil", W4, Gm)
            lhs = np.einsum("tik,tabil->takbl", Gm, WG).reshape(-1, q * K, q * K)
            rhs = np.einsum("tik,ai->tak", Gm, y).reshape(-1, q * K)
            lhs = lhs + 1e-12 * np.eye(q * K)[None]
            sol = np.linalg.solve(lhs, rhs[..., None])[..., 0]
            J[s:s + len(tt)] = mWm - np.einsum("tk,tk->t", rhs, sol)
        dof = q * p - q * K
        return J, dof, T

    def fit(self, data: ObservedData, spec: EstimandSpec, rng: np.random.Generator) -> EstimateResult:
        J, dof, T = self.criterion(data, spec)
        crit = float(stats.chi2.ppf(1 - self.alpha, dof))
        acc = T[J <= crit]
        d = 3
        if acc.shape[0] == 0:
            lo = np.zeros(d)
            hi = np.full(d, 3.0)
            point = np.full(d, np.nan)
            slo, shi = lo.copy(), hi.copy()
        else:
            A_all = np.stack([self.dirs[acc[:, k]] for k in range(3)], axis=2)  # (t, d, K)
            counts = (np.abs(A_all) >= spec.tau - 1e-12).sum(axis=2)  # (t, d)
            lo, hi = counts.min(axis=0).astype(float), counts.max(axis=0).astype(float)
            sm = (1 / (1 + np.exp(-np.clip((np.abs(A_all) - spec.tau) / self.bandwidth, -60, 60)))).sum(axis=2)
            slo, shi = sm.min(axis=0), sm.max(axis=0)
            point = counts[np.argmin(J[J <= crit])].astype(float)
        return EstimateResult(
            method=self.name, kind="interval", point=point, lower=lo, upper=hi,
            label=f"confidence(AR projection, grid-restricted, n_dirs={self.n_dirs})",
            access=Access(env_labels=True, higher_moments=False, n_observations=data.n,
                          notes="fourth moments used only for the weight matrix"),
            diagnostics={
                "n_grid_triples": int(T.shape[0]), "n_accepted": int(acc.shape[0]),
                "empty": bool(acc.shape[0] == 0), "J_min": float(J.min()), "crit": crit, "dof": dof,
                "smooth_lower": np.asarray(slo).tolist(), "smooth_upper": np.asarray(shi).tolist(),
            },
        )

    def nearest_grid_matrix(self, A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Snap columns of ``A`` to the grid (canonical sign). Returns ``(A_grid, indices)``."""
        idx = np.argmax(np.abs(self.dirs @ A), axis=0)
        return self.dirs[idx].T, idx
