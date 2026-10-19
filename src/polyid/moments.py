r"""Second-moment statistics and their sampling uncertainty.

Pipeline step **Fit** (plan p.7): "Center each environment separately;
estimate covariance and its uncertainty."

Objects
-------
* ``C[e]``  per-environment covariance (each environment centred by its own mean);
* ``D[e] = C[e] - mean_e C[e]``  centred covariance stack. Under the declared
  model ``D[e] = A diag(c_e) A^T`` where ``c`` is the centred variance profile;
  stationary noise cancels exactly;
* the **moment vector** ``m = (vech D[0], ..., vech D[E-2])`` — the last
  centred matrix is dropped because the stack sums to zero;
* ``Var(m)`` from the empirical fourth moments (independent environments,
  i.i.d. rows or cluster-robust sandwich when cluster labels are given).

``vech`` stacks the upper triangle row-major *including* the diagonal; all
quadratic forms in this package use that same convention.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from polyid.data import ObservedData


def vech_indices(d: int) -> tuple[np.ndarray, np.ndarray]:
    return np.triu_indices(d)


def vech(M: np.ndarray) -> np.ndarray:
    """Upper-triangle (incl. diagonal) of the last two axes."""
    d = M.shape[-1]
    i, j = vech_indices(d)
    return M[..., i, j]


def unvech(v: np.ndarray, d: int) -> np.ndarray:
    i, j = vech_indices(d)
    M = np.zeros(v.shape[:-1] + (d, d))
    M[..., i, j] = v
    M[..., j, i] = v
    return M


def env_covariances(
    X: np.ndarray, env: np.ndarray, n_envs: int, weights: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-environment covariance with per-environment centring.

    ``weights`` (non-negative, e.g. bootstrap multiplicities) are supported so
    that resampling does not need to copy the data.

    Returns ``(C, means, effective_counts)``.
    """
    d = X.shape[1]
    C = np.zeros((n_envs, d, d))
    means = np.zeros((n_envs, d))
    counts = np.zeros(n_envs)
    for e in range(n_envs):
        idx = env == e
        Xe = X[idx]
        if weights is None:
            n_e = Xe.shape[0]
            mu = Xe.mean(axis=0)
            Y = Xe - mu
            C[e] = (Y.T @ Y) / max(n_e - 1, 1)
        else:
            w = weights[idx].astype(float)
            n_e = w.sum()
            mu = (w @ Xe) / n_e
            Y = Xe - mu
            C[e] = (Y.T * w) @ Y / max(n_e - 1.0, 1.0)
        means[e] = mu
        counts[e] = n_e
    return C, means, counts


def centered_stack(C: np.ndarray) -> np.ndarray:
    return C - C.mean(axis=0, keepdims=True)


def moment_vector(D: np.ndarray) -> np.ndarray:
    """Stack ``vech(D[e])`` for ``e = 0..E-2``."""
    return vech(D[:-1]).reshape(-1)


def centering_map(n_envs: int) -> np.ndarray:
    """Matrix ``L`` with ``(D[0..E-2]) = L (C[0..E-1])`` acting on environment blocks."""
    L = np.eye(n_envs) - 1.0 / n_envs
    return L[:-1]


@dataclass
class MomentStats:
    """Sufficient second-moment summary of an :class:`ObservedData`."""

    C: np.ndarray  # E x d x d
    D: np.ndarray  # E x d x d centred stack
    pooled: np.ndarray  # d x d mean of C over environments (positive definite reference)
    counts: np.ndarray  # E
    omega: list | None = None  # per-environment Var(vech C_e) (already divided by n_e)

    @property
    def n_envs(self) -> int:
        return int(self.C.shape[0])

    @property
    def d(self) -> int:
        return int(self.C.shape[1])

    def moment_vector(self) -> np.ndarray:
        return moment_vector(self.D)

    def moment_covariance(self) -> np.ndarray:
        """``Var(m)`` = ``(L ⊗ I) blkdiag(Omega_e) (L ⊗ I)^T``."""
        if self.omega is None:
            raise ValueError("compute_moment_stats(..., with_covariance=True) is required")
        E = self.n_envs
        p = self.omega[0].shape[0]
        L = centering_map(E)
        V = np.zeros(((E - 1) * p, (E - 1) * p))
        for a in range(E - 1):
            for b in range(E - 1):
                block = np.zeros((p, p))
                for e in range(E):
                    coef = L[a, e] * L[b, e]
                    if coef != 0.0:
                        block += coef * self.omega[e]
                V[a * p:(a + 1) * p, b * p:(b + 1) * p] = block
        return V


def _vech_outer_rows(Y: np.ndarray) -> np.ndarray:
    """Row i -> vech(y_i y_i^T)."""
    d = Y.shape[1]
    i, j = vech_indices(d)
    return Y[:, i] * Y[:, j]


def vech_covariance_omega(Xe: np.ndarray, cluster: np.ndarray | None = None, max_rows: int | None = None,
                          rng: np.random.Generator | None = None) -> np.ndarray:
    r"""Estimate ``Var(vech \hat C_e)`` from fourth moments.

    i.i.d. rows: ``Var(vech(y y^T)) / n``. With cluster labels the sandwich
    ``sum_c (s_c)(s_c)^T / n^2`` of within-cluster sums of centred outer
    products is used (contexts from one document are not independent, plan
    p.9). ``max_rows`` subsamples rows for speed in high dimension (the
    variance is then rescaled to the full ``n``).
    """
    n = Xe.shape[0]
    Y = Xe - Xe.mean(axis=0)
    if cluster is None:
        if max_rows is not None and n > max_rows:
            rng = rng or np.random.default_rng(0)
            Y = Y[rng.choice(n, size=max_rows, replace=False)]
        U = _vech_outer_rows(Y)
        U = U - U.mean(axis=0)
        return (U.T @ U) / (U.shape[0] - 1) / n
    U = _vech_outer_rows(Y)
    U = U - U.mean(axis=0)
    _, inv = np.unique(cluster, return_inverse=True)
    S = np.zeros((inv.max() + 1, U.shape[1]))
    np.add.at(S, inv, U)
    G = S.shape[0]
    return (S.T @ S) / n**2 * (G / max(G - 1, 1))


def compute_moment_stats(data: ObservedData, with_covariance: bool = True, weights: np.ndarray | None = None,
                         max_rows_for_omega: int | None = 20000) -> MomentStats:
    C, _, counts = env_covariances(data.X, data.env, data.n_envs, weights=weights)
    omega = None
    if with_covariance:
        if data.d > 64:
            raise ValueError(
                "analytic moment covariance is O(d^4); for d > 64 use bootstrap-based procedures"
            )
        omega = []
        for e in range(data.n_envs):
            idx = data.env == e
            cl = None if data.cluster is None else data.cluster[idx]
            omega.append(vech_covariance_omega(data.X[idx], cl, max_rows=max_rows_for_omega))
    D = centered_stack(C)
    return MomentStats(C=C, D=D, pooled=C.mean(axis=0), counts=counts, omega=omega)


def population_stats(pop_cov: np.ndarray) -> MomentStats:
    """Exact (infinite-sample) moment summary from population covariances."""
    C = np.asarray(pop_cov, dtype=float)
    return MomentStats(C=C, D=centered_stack(C), pooled=C.mean(axis=0), counts=np.full(C.shape[0], np.inf))


def bootstrap_weights(data: ObservedData, rng: np.random.Generator) -> np.ndarray:
    """Multiplicity weights for one nonparametric bootstrap replicate.

    Rows are resampled within each environment; with cluster labels whole
    clusters are resampled within each environment (cluster bootstrap).
    """
    w = np.zeros(data.n)
    for e in range(data.n_envs):
        idx = np.flatnonzero(data.env == e)
        if data.cluster is None:
            draws = rng.integers(0, idx.size, size=idx.size)
            np.add.at(w, idx[draws], 1.0)
        else:
            cl = data.cluster[idx]
            uniq, inv = np.unique(cl, return_inverse=True)
            picks = rng.integers(0, uniq.size, size=uniq.size)
            counts = np.bincount(picks, minlength=uniq.size).astype(float)
            w[idx] = counts[inv]
    return w
