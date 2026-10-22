r"""Overcompleteness test: are there more (non-Gaussian) latents than neurons?

Second moments cannot count past ``d``: every covariance of ``d`` neurons has rank at most
``d``, so a square model always fits as many components as there are neurons. In trained toy
models (E5) every model represents more features than dimensions, the second-moment misfit
gate misses many of them, and ISMB's upper bounds, which cannot exceed the number of fitted
components, then undercover. This module supplies the missing diagnostic.

**Statistic.** Under ``h = A z + eps`` with independent latents and Gaussian (or no) noise, the
fourth-order cumulant of ``h`` within environment ``e``, flattened over unordered index pairs,

.. math:: Q_e[(ij),(kl)] = \operatorname{cum}(h_i, h_j, h_k, h_l)
          = \sum_k \kappa_{ek}\, u_k u_k^\top, \qquad u_k = \operatorname{vech}(a_k a_k^\top),

has rank equal to the number of latents with non-zero excess kurtosis, for up to
``d(d+1)/2`` generic columns. Environments are centred separately and then averaged
(``Q = mean_e Q_e``): pooling environments whose covariances differ would add fourth-order
terms of its own.

**Test of H0: rank(Q) <= d.** By Weyl's inequality
``|sigma_i(Q_hat) - sigma_i(Q)| <= ||Q_hat - Q||_op``. With ``c`` the bootstrap ``1 - alpha``
quantile of ``||Q_hat* - Q_hat||_op`` (rows resampled within environment),
``sigma_{d+1}(Q_hat) > c`` implies ``sigma_{d+1}(Q) > 0`` up to the bootstrap approximation.
The test is conservative by construction. ``rank_lower_bound = #{i : sigma_i(Q_hat) > c}`` is a
lower bound on the number of non-Gaussian latents with the same guarantee.

**What it cannot see.** Latents with near-zero excess kurtosis (near-Gaussian, or dense
features with little data) contribute almost nothing to ``Q``. Non-Gaussian noise with
independent coordinates adds up to ``d`` rank-one terms and can trigger the test. That
causes a needless abstention, never undercoverage.

Prior art: the rank of the matricised quadricovariance underlies FOOBI (De Lathauwer,
Castaing & Cardoso, IEEE Trans. Signal Processing 2007) for underdetermined mixtures. Here it
is used only as a diagnostic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class OvercompleteTest:
    sigma: np.ndarray  # singular values of the averaged quadricovariance, descending
    floor: float  # bootstrap (1 - alpha) quantile of ||Q_hat* - Q_hat||_op
    d: int
    rejected: bool  # rank > d: more non-Gaussian latents than neurons
    rank_lower_bound: int
    ratio: float  # sigma_{d+1} / floor; > 1 means rejected


def _env_rows(env: np.ndarray, max_rows: int | None, rng: np.random.Generator | None) -> list[np.ndarray]:
    out = []
    for e in np.unique(env):
        idx = np.flatnonzero(env == e)
        if max_rows is not None and idx.size > max_rows:
            idx = np.sort((rng or np.random.default_rng(0)).choice(idx, size=max_rows, replace=False))
        out.append(idx)
    return out


def _quadricov_block(H: np.ndarray, iu: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    H = H - H.mean(axis=0)
    n = len(H)
    C = H.T @ H / n
    i, j = iu
    Z = H[:, i] * H[:, j]
    M4 = Z.T @ Z / n
    gauss = np.outer(C[i, j], C[i, j]) + C[np.ix_(i, i)] * C[np.ix_(j, j)] + C[np.ix_(i, j)] * C[np.ix_(j, i)]
    return M4 - gauss


def quadricovariance(X: np.ndarray, env: np.ndarray, rows: list[np.ndarray] | None = None) -> np.ndarray:
    """Within-environment fourth-order cumulant flattening over unordered pairs, averaged over
    environments. Shape ``(d(d+1)/2, d(d+1)/2)``."""
    iu = np.triu_indices(X.shape[1])
    rows = rows if rows is not None else _env_rows(env, None, None)
    return np.mean([_quadricov_block(X[ix], iu) for ix in rows], axis=0)


def overcomplete_test(X: np.ndarray, env: np.ndarray, alpha: float = 0.05, n_boot: int = 50,
                      max_rows: int | None = 20000, rng: np.random.Generator | None = None) -> OvercompleteTest:
    """Test H0: at most ``d`` non-Gaussian latents (module docstring)."""
    rng = rng or np.random.default_rng(0)
    d = X.shape[1]
    rows = _env_rows(env, max_rows, rng)
    Q = quadricovariance(X, env, rows)
    sigma = np.linalg.svd(Q, compute_uv=False)
    if len(sigma) <= d:  # d = 1: the flattening cannot have rank above d
        return OvercompleteTest(sigma, float("inf"), d, False, 0, 0.0)
    iu = np.triu_indices(d)
    dev = np.empty(n_boot)
    for b in range(n_boot):
        Qb = np.mean([_quadricov_block(X[rng.choice(ix, size=ix.size)], iu) for ix in rows], axis=0)
        dev[b] = np.linalg.norm(Qb - Q, 2)
    floor = float(np.quantile(dev, 1 - alpha, method="higher"))
    return OvercompleteTest(sigma=sigma, floor=floor, d=d, rejected=bool(sigma[d] > floor),
                            rank_lower_bound=int(np.sum(sigma > floor)),
                            ratio=float(sigma[d] / floor) if floor > 0 else float("inf"))
