r"""Pairwise tests that decide which estimated factors are *separately* identified.

For two estimated components ``k, l`` with unmixed signals
``u_k = b_k^T x`` and ``u_l = b_l^T x`` define, for every environment ``e``,
the centred 2x2 covariance ``M_e = Cov_e(u_k, u_l) - mean_e Cov_e(u_k, u_l)``
and the ``3 x (E-1)`` matrix

.. math:: \Pi = [\operatorname{vech} M_0, \ldots, \operatorname{vech} M_{E-2}].

**Collinearity (H0: the pair is not separately identified).** If the two
factors' centred variance profiles are collinear, ``M_e = u_e Gamma`` for a
fixed ``Gamma``, i.e. ``rank(Pi) = 1``. This is invariant to any invertible
re-mixing within the pair, so it does not require the pair to be identified
— exactly what a test of non-identification needs. We use the Cragg–Donald
(1997) minimum-distance rank statistic

.. math:: CD = \min_{a,b} \operatorname{vec}(\hat\Pi - ab^\top)^\top
          \hat V^{-1} \operatorname{vec}(\hat\Pi - ab^\top)
          \;\to_d\; \chi^2_{(3-1)(E-1-1)} .

*First-order robustness to estimating the pair's subspace* (derivation in
``docs/THEORY.md``, Remark 3): under H0 the first-order effect of an error in
``(b_k, b_l)`` on ``M_e`` is itself proportional to ``u_e``, i.e. lies in the
tangent space of the rank-1 manifold, so treating the estimated basis as
fixed does not change the limiting distribution. Requires ``E >= 3``; with
``E = 2`` no pair can be separated (the procedure then merges everything,
which is the correct, honest answer).

**Residual dependence (allowed-correlation trigger).** Under the declared
model the off-diagonal row ``Pi[1, :]`` is zero at the true unmixing. A
Wald statistic on that row, with ``E - 3`` degrees of freedom to account
heuristically for the two pair parameters fitted by joint diagonalisation,
flags pairs whose latents look dependent (e.g. co-occurrence). This test is
*approximate*; it is used only to merge (widen), never to narrow.

Variances are computed from per-environment fourth moments of ``(u_k, u_l)``
(cluster-robust when cluster labels exist) — no ``O(d^4)`` objects.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from polyid.moments import centering_map


def _vech_rows(Z: np.ndarray) -> np.ndarray:
    i, j = np.triu_indices(Z.shape[1])
    return Z[:, i] * Z[:, j]


def env_second_moments(Ug: np.ndarray, env: np.ndarray, n_envs: int, cluster: np.ndarray | None = None
                       ) -> tuple[np.ndarray, list]:
    """Per-environment ``vech`` covariance of ``Ug`` (``N x g``) and its sampling covariance."""
    p = Ug.shape[1] * (Ug.shape[1] + 1) // 2
    means = np.zeros((n_envs, p))
    omegas = []
    for e in range(n_envs):
        idx = env == e
        Z = Ug[idx] - Ug[idx].mean(axis=0)
        W = _vech_rows(Z)
        n = W.shape[0]
        means[e] = W.mean(axis=0)
        Wc = W - means[e]
        if cluster is None:
            omegas.append((Wc.T @ Wc) / max(n - 1, 1) / n)
        else:
            _, inv = np.unique(cluster[idx], return_inverse=True)
            S = np.zeros((inv.max() + 1, p))
            np.add.at(S, inv, Wc)
            G = S.shape[0]
            omegas.append((S.T @ S) / n**2 * (G / max(G - 1, 1)))
    return means, omegas


def moment_matrix(Ug: np.ndarray, env: np.ndarray, n_envs: int, cluster: np.ndarray | None = None
                  ) -> tuple[np.ndarray, np.ndarray]:
    """``(Pi, V)``: the ``p x (E-1)`` centred moment matrix (``p = g(g+1)/2``) and ``Var(vec Pi)``."""
    means, omegas = env_second_moments(Ug, env, n_envs, cluster)
    p = means.shape[1]
    L = centering_map(n_envs)  # (E-1) x E
    Pi = (L @ means).T  # p x (E-1)
    q = n_envs - 1
    V = np.zeros((p * q, p * q))
    for a in range(q):
        for b in range(q):
            blk = np.zeros((p, p))
            for e in range(n_envs):
                c = L[a, e] * L[b, e]
                if c:
                    blk += c * omegas[e]
            V[p * a:p * a + p, p * b:p * b + p] = blk
    return Pi, V


def pair_moment_matrix(u_k: np.ndarray, u_l: np.ndarray, env: np.ndarray, n_envs: int,
                       cluster: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """``(Pi, V)`` for a pair: ``3 x (E-1)`` and ``Var(vec Pi)`` (column-major)."""
    return moment_matrix(np.stack([u_k, u_l], axis=1), env, n_envs, cluster)


def group_rank1_pvalue(Ug: np.ndarray, env: np.ndarray, n_envs: int, cluster: np.ndarray | None = None) -> float:
    """p-value of ``H0: all profiles in the group are collinear`` (joint rank-1 test)."""
    Pi, V = moment_matrix(Ug, env, n_envs, cluster)
    stat, dof = cragg_donald_rank1(Pi, V)
    return 1.0 if dof == 0 else float(stats.chi2.sf(stat, dof))


def profile_band(u: np.ndarray, env: np.ndarray, n_envs: int, z: float, cluster: np.ndarray | None = None
                 ) -> tuple[float, float, float]:
    r"""Visibility statistic of one component and a conservative ``[lo, hi]`` band.

    ``c_e = Var_e(u) - mean_e Var_e(u)``; its covariance follows from
    per-environment fourth moments. A per-coordinate box of half-width
    ``z * se_e`` (Bonferroni over coordinates by the caller's choice of ``z``)
    contains the joint confidence region; the RMS over the box gives the band.
    For an invisible direction the first-order effect of estimating the
    unmixing row vanishes (``D_e b = 0``), so treating ``b`` as fixed is
    first-order exact there.
    """
    means, omegas = env_second_moments(u[:, None], env, n_envs, cluster)
    s = means[:, 0]
    Lfull = np.eye(n_envs) - 1.0 / n_envs
    c = Lfull @ s
    Vs = np.diag([o[0, 0] for o in omegas])
    se = np.sqrt(np.maximum(np.diag(Lfull @ Vs @ Lfull.T), 0.0))
    lo_abs = np.maximum(np.abs(c) - z * se, 0.0)
    hi_abs = np.abs(c) + z * se
    rms = lambda x: float(np.sqrt(np.mean(x**2)))
    return rms(c), rms(lo_abs), rms(hi_abs)


def cragg_donald_rank1(Pi: np.ndarray, V: np.ndarray, max_iter: int = 200, tol: float = 1e-12) -> tuple[float, int]:
    """Minimum-distance statistic for ``H0: rank(Pi) <= 1`` and its degrees of freedom."""
    p, q = Pi.shape
    dof = (p - 1) * (q - 1)
    if dof <= 0:
        return 0.0, 0
    W = np.linalg.pinv(V, rcond=1e-12, hermitian=True)
    y = Pi.reshape(-1, order="F")
    U, s, Vt = np.linalg.svd(Pi, full_matrices=False)
    best = np.inf
    for init in (Vt[0] * s[0], np.ones(q)):
        b = np.asarray(init, dtype=float)
        if np.linalg.norm(b) == 0:
            b = np.ones(q)
        prev = np.inf
        for _ in range(max_iter):
            X = np.kron(b[:, None], np.eye(p))  # vec(a b^T) = (b ⊗ I) a
            a = np.linalg.lstsq(X.T @ W @ X, X.T @ W @ y, rcond=None)[0]
            Y = np.kron(np.eye(q), a[:, None])  # vec(a b^T) = (I ⊗ a) b
            b = np.linalg.lstsq(Y.T @ W @ Y, Y.T @ W @ y, rcond=None)[0]
            r = y - Y @ b
            val = float(r @ W @ r)
            if prev - val < tol * max(1.0, val):
                break
            prev = val
        best = min(best, val)
    return best, dof


@dataclass
class PairTest:
    k: int
    l: int
    collinear_stat: float
    collinear_dof: int
    collinear_p: float
    dependence_stat: float
    dependence_dof: int
    dependence_p: float


def test_pair(u_k: np.ndarray, u_l: np.ndarray, env: np.ndarray, n_envs: int, k: int, l: int,
              cluster: np.ndarray | None = None) -> PairTest:
    Pi, V = pair_moment_matrix(u_k, u_l, env, n_envs, cluster)
    stat, dof = cragg_donald_rank1(Pi, V)
    p_col = 1.0 if dof == 0 else float(stats.chi2.sf(stat, dof))
    # residual dependence: off-diagonal row of Pi (entries 1, 4, 7, ... in vec order)
    q = n_envs - 1
    idx = np.arange(q) * 3 + 1
    o = Pi[1]
    Vo = V[np.ix_(idx, idx)]
    w = float(o @ np.linalg.pinv(Vo, rcond=1e-12, hermitian=True) @ o)
    ddof = max(q - 2, 1)
    p_dep = float(stats.chi2.sf(w, ddof)) if q >= 3 else 1.0
    return PairTest(k, l, stat, dof, p_col, w, ddof, p_dep)


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra

    def groups(self, members: list[int]) -> list[list[int]]:
        out: dict[int, list[int]] = {}
        for m in members:
            out.setdefault(self.find(m), []).append(m)
        return sorted((sorted(v) for v in out.values()), key=lambda g: g[0])
