r"""Exploratory repair: joint diagonalisation of covariance differences **and** fourth-order
cumulant slices (JADE-style; Cardoso & Souloumiac 1993).

Motivation (lab notebook §8): in the E1 pilot, second moments alone were too weak at
``d = 16, E = 6`` while FastICA (non-Gaussianity) did well. Under the declared model with
independent latents, every slice of the fourth-order cumulant tensor of ``h`` is

.. math:: \mathcal C_4(e_p e_p^\top) = A\,\operatorname{diag}(\kappa_k a_{pk}^2)\,A^\top ,

so it is jointly diagonalised by the *same* unmixing matrix as the centred covariances.
Gaussian noise has zero fourth cumulant, so it drops out even when non-stationary.
Per-environment slices add ``d x E`` matrices whose "profiles" (``kappa_{e,k} a_{pk}^2``)
generically break the collinearity that defeats the second-moment stack.

Status: **point estimator only**, exploratory. It changes the information set (higher moments),
so it is compared with FastICA, not with the second-moment procedures, and no interval or
coverage claim is attached. Turning it into an ISMB-style bound (cumulant-aware pair tests) is
future work. It does not help with dependent latents (Proposition 2).
"""

from __future__ import annotations

import numpy as np

from polyid.data import ObservedData
from polyid.estimand import EstimandSpec, multiplicity, normalize_columns
from polyid.estimators.base import Access, EstimateResult
from polyid.jointdiag import fit_joint_diag
from polyid.moments import compute_moment_stats


def cumulant_slices(X: np.ndarray, max_rows: int | None = None, rng: np.random.Generator | None = None) -> np.ndarray:
    """``d`` slices ``cum(x_p, x_p, x_i, x_j)`` of the fourth-order cumulant of centred ``X``."""
    if max_rows is not None and X.shape[0] > max_rows:
        rng = rng or np.random.default_rng(0)
        X = X[rng.choice(X.shape[0], size=max_rows, replace=False)]
    X = X - X.mean(axis=0)
    n, d = X.shape
    C = X.T @ X / n
    out = np.empty((d, d, d))
    for p in range(d):
        w = X[:, p] ** 2
        M4 = (X * w[:, None]).T @ X / n
        out[p] = M4 - C[p, p] * C - 2.0 * np.outer(C[:, p], C[p, :])
    return out


class JointDiagCumulant:
    name = "jointdiag_cum"

    def __init__(self, cfg: dict | None = None):
        cfg = cfg or {}
        self.n_starts = int(cfg.get("n_starts", 6))
        self.per_env = bool(cfg.get("per_env", True))
        self.cum_weight = float(cfg.get("cum_weight", 1.0))
        self.max_rows = cfg.get("max_rows", 20000)

    def fit(self, data: ObservedData, spec: EstimandSpec, rng: np.random.Generator) -> EstimateResult:
        ms = compute_moment_stats(data, with_covariance=False)
        mats = [M / max(np.linalg.norm(M), 1e-300) for M in ms.D]
        envs = range(data.n_envs) if self.per_env else [None]
        for e in envs:
            Xe = data.X if e is None else data.X[data.env == e]
            for S in cumulant_slices(Xe, self.max_rows, rng):
                S = 0.5 * (S + S.T)
                nrm = np.linalg.norm(S)
                if nrm > 0:
                    mats.append(self.cum_weight * S / nrm)
        stack = np.stack(mats)
        best, _ = fit_joint_diag(stack, ms.pooled, n_starts=self.n_starts, rng=rng)
        A, norms = normalize_columns(best.A)
        B = norms[:, None] * best.V
        vis = np.sqrt(np.mean(np.einsum("ki,eij,kj->ke", B, ms.D, B) ** 2, axis=1))
        return EstimateResult(
            method=self.name, kind="count", point=multiplicity(A, spec.tau, visible=vis >= spec.v_min, normalize=False).astype(float),
            loadings=A, label="exploratory: covariance differences + 4th-order cumulant slices",
            access=Access(env_labels=True, higher_moments=True, n_observations=data.n, restarts=self.n_starts),
            diagnostics={"n_matrices": int(stack.shape[0]), "objective": best.objective},
        )
