r"""Exact multiplicity ranges for a second-moment–unidentified factor group.

Setting
-------
Suppose the factors in a group ``G`` (size ``g``) have *collinear* centred
variance profiles, ``c_k = lambda_k u`` for ``k in G`` with all
``lambda_k > 0``. Then every centred covariance restricted to the group is
``u_e * Gamma`` with ``Gamma = A_G diag(lambda) A_G^T = M M^T``,
``M = A_G diag(lambda)^{1/2}``. Any ``M R`` with ``R`` orthogonal produces
the same ``Gamma`` and hence the same second moments. After normalising
columns, the admissible explanations of the group are exactly

.. math:: \{\operatorname{normalize}(M R) : R \in O(g)\}.

(That *all* explanations have this form, given that the other factors are
identified and have profiles non-collinear with ``u``, is Proposition 1 in
``docs/THEORY.md``.)

Reduction to a diagonal problem
-------------------------------
Column ``i`` of ``M R`` is ``M r_i``. Its normalised loading on neuron ``j``
reaches the threshold iff

.. math:: (m_j^\top r_i)^2 \ge \tau^2\, r_i^\top M^\top M r_i
          \iff r_i^\top Q_j r_i \ge 0, \qquad
          Q_j = m_j m_j^\top - \tau^2 M^\top M,

so the group's contribution to :math:`P_j` is the number of non-negative
diagonal entries of :math:`R^\top Q_j R`.

Schur–Horn
----------
The set of diagonals of :math:`R^\top Q R` over :math:`R \in O(g)` is exactly
the permutohedron of the eigenvalues :math:`\lambda(Q)` — the vectors
majorised by :math:`\lambda(Q)` (Schur 1923; Horn 1954). The extreme counts
therefore have closed forms (proved in ``docs/THEORY.md``, Lemma 2):

* **max**: if ``tr Q >= 0`` it is ``g``; otherwise it is the largest ``m``
  with ``(0,…,0, s/(g-m),…,s/(g-m)) ≺ λ(Q)`` (``m`` zeros, ``s = tr Q``);
* **min**: if ``tr Q < 0`` it is ``0``; otherwise ``g - n`` for the largest
  ``n`` with ``(-ε,…,-ε, (s+nε)/(g-n),…) ≺ λ(Q)`` for small ``ε > 0``.

Every intermediate integer is attained along a path in ``SO(g)`` only
generically; the functions return the interval hull ``[min, max]``.

Conservative fallback (allowed within-group correlation / mixed signs)
----------------------------------------------------------------------
If the group's latents may be correlated, or the ``lambda_k`` have mixed
signs (invariance group ``O(p, q)``, non-compact), the admissible columns
are any basis of ``span(A_G)``; then the range is ``[0, g]`` when
``||P_{span} e_j|| > tau`` and ``[0, 0]`` otherwise (``g >= 2``). This is an
outer bound (valid, wider than necessary).
"""

from __future__ import annotations

import numpy as np


def is_majorized(x: np.ndarray, lam: np.ndarray, tol: float = 1e-10) -> bool:
    """``x ≺ lam``: equal sums and dominated partial sums of decreasing rearrangements."""
    xs = np.sort(np.asarray(x, dtype=float))[::-1]
    ls = np.sort(np.asarray(lam, dtype=float))[::-1]
    scale = max(np.max(np.abs(ls)), np.max(np.abs(xs)), 1e-300)
    if abs(xs.sum() - ls.sum()) > tol * scale * len(ls):
        return False
    return bool(np.all(np.cumsum(xs)[:-1] <= np.cumsum(ls)[:-1] + tol * scale))


def nonneg_diagonal_count_range(eigs: np.ndarray, tol: float = 1e-10, eps_rel: float = 1e-9) -> tuple[int, int]:
    """Min and max of ``#{i : (R^T Q R)_ii >= 0}`` over ``R in O(g)`` given ``eig(Q)``."""
    lam = np.sort(np.asarray(eigs, dtype=float))[::-1]
    g = lam.size
    if g == 0:
        return 0, 0
    scale = max(np.max(np.abs(lam)), 1e-300)
    s = float(lam.sum())
    # ---- maximum
    if s >= -tol * scale:
        hi = g
    else:
        hi = 0
        for m in range(g - 1, 0, -1):
            x = np.concatenate([np.zeros(m), np.full(g - m, s / (g - m))])
            if is_majorized(x, lam, tol):
                hi = m
                break
    # ---- minimum
    if s < -tol * scale:
        lo = 0
    else:
        eps = eps_rel * scale
        lo = g
        for n in range(g - 1, 0, -1):
            x = np.concatenate([np.full(n, -eps), np.full(g - n, (s + n * eps) / (g - n))])
            if is_majorized(x, lam, tol):
                lo = g - n
                break
    return int(lo), int(hi)


def rotation_group_ranges(M: np.ndarray, tau: float, tol: float = 1e-10) -> tuple[np.ndarray, np.ndarray]:
    """Per-neuron ``[min, max]`` contribution of a rotation-unidentified group.

    ``M`` is ``d x g`` with ``M M^T = Gamma`` (any square root works; the
    result depends on ``Gamma`` only).
    """
    M = np.asarray(M, dtype=float)
    d, g = M.shape
    G = M.T @ M
    Q = np.einsum("ja,jb->jab", M, M) - (tau**2) * G[None]
    eigs = np.linalg.eigvalsh(Q)
    lo = np.empty(d, dtype=int)
    hi = np.empty(d, dtype=int)
    for j in range(d):
        lo[j], hi[j] = nonneg_diagonal_count_range(eigs[j], tol=tol)
    return lo, hi


def span_group_ranges(basis: np.ndarray, tau: float, tol: float = 1e-10) -> tuple[np.ndarray, np.ndarray]:
    """Outer-bound ranges when any basis of ``span(basis)`` is admissible."""
    basis = np.asarray(basis, dtype=float)
    d, g = basis.shape
    Q, _ = np.linalg.qr(basis)
    proj = np.linalg.norm(Q[:, :g], axis=1)  # ||P_S e_j||
    if g == 1:
        a = basis[:, 0] / np.linalg.norm(basis[:, 0])
        c = (np.abs(a) >= tau - tol).astype(int)
        return c, c.copy()
    hi = np.where(proj > tau + tol, g, np.where(proj >= tau - tol, 1, 0)).astype(int)
    lo = np.zeros(d, dtype=int)
    return lo, hi


def group_ranges(A_cols: np.ndarray, lambdas: np.ndarray, tau: float, allow_correlation: bool = False,
                 sign_tol: float = 0.0) -> tuple[np.ndarray, np.ndarray, str]:
    """Dispatch: exact Schur–Horn range when admissible, otherwise the span outer bound.

    Returns ``(lo, hi, mode)`` with ``mode`` in {``"rotation"``, ``"span"``}.
    ``lambdas`` are the coefficients of the group members' profiles on the
    common profile direction (``sign_tol`` absorbs estimation noise around 0).
    """
    lam = np.asarray(lambdas, dtype=float)
    if not allow_correlation:
        if np.all(lam >= -sign_tol) or np.all(lam <= sign_tol):
            M = np.asarray(A_cols, dtype=float) * np.sqrt(np.abs(lam))[None, :]
            lo, hi = rotation_group_ranges(M, tau)
            return lo, hi, "rotation"
    lo, hi = span_group_ranges(A_cols, tau)
    return lo, hi, "span"


def brute_force_rotation_counts(M: np.ndarray, tau: float, j: int, n_samples: int = 20000,
                                rng: np.random.Generator | None = None) -> tuple[int, int]:
    """Monte-Carlo check of :func:`rotation_group_ranges` (used in tests only)."""
    rng = rng or np.random.default_rng(0)
    d, g = M.shape
    best_lo, best_hi = g, 0
    for _ in range(n_samples):
        Z = rng.standard_normal((g, g))
        R, Rr = np.linalg.qr(Z)
        R = R * np.sign(np.diag(Rr))
        C = M @ R
        C = C / np.linalg.norm(C, axis=0)
        cnt = int(np.sum(np.abs(C[j]) >= tau))
        best_lo, best_hi = min(best_lo, cnt), max(best_hi, cnt)
    return best_lo, best_hi
