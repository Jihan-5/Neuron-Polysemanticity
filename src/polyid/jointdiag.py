r"""Approximate non-orthogonal joint diagonalisation.

Pipeline step **Recover** (plan p.7): "Start with a square/full-rank
joint-diagonalization baseline." This is the coroICA identification route
(Pfister et al., JMLR 2019, syllabus S11): under the declared model every
centred covariance ``D[e] = A diag(c_e) A^T``, so an unmixing matrix
``V = A^{-1}`` makes all ``V D[e] V^T`` diagonal. The solution is unique up to
row scale and permutation **iff** the centred variance profiles of distinct
factors are pairwise non-collinear (Hyvärinen & Oja tutorial, syllabus P2,
explains the scale/permutation ambiguity).

Algorithm
---------
U-WEDGE (Tichavský & Yeredor, 2009): Gauss–Newton updates
``V <- (I + H) V`` with ``H`` solving, for each pair ``i < j``, the 2x2
least-squares problem obtained by linearising the off-diagonal entries::

    minimise_{H_ij, H_ji}  sum_m ( T_m[i,j] + H_ij T_m[j,j] + H_ji T_m[i,i] )^2

where ``T_m = V M_m V^T``. Rows of ``V`` are rescaled after every step so that
``diag(V R V^T) = 1`` for a positive-definite reference ``R`` (we use the
pooled covariance), which fixes the scale ambiguity.

Step control follows FFDiag (Ziehe et al., JMLR 2004): the update is clamped
to ``||H||_F <= theta < 1`` so that ``I + H`` stays invertible. This matters:
in development we found that (i) a backtracking line search on the
normalised objective stalls because the Gauss–Newton direction is not a
descent direction for it far from the solution, and (ii) quasi-Newton
minimisation of the scale-invariant objective drives ``V`` to *degenerate*
near-singular solutions with objective ~0 (rows collapsing onto common
isotropic directions of the indefinite stack). Clamped Gauss–Newton steps
avoid both and reached the correct basin from every start in our planted
tests (see ``tests/test_jointdiag.py``).

The 2x2 normal-equation determinant for pair ``(i, j)`` normalised by
``z_ii z_jj`` equals ``1 - cos^2`` of the *estimated* diagonal profiles of
the two components. It is returned as ``pair_sin2`` — a direct, free
diagnostic of weak identification for that pair.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import linalg


@dataclass
class JDResult:
    V: np.ndarray  # unmixing, rows = components
    A: np.ndarray  # mixing = inv(V), columns = components (unnormalised)
    objective: float
    n_iter: int
    converged: bool
    history: list = field(default_factory=list)
    pair_sin2: np.ndarray | None = None  # d x d, 1 - cos^2 between estimated profiles


def offdiag_objective(V: np.ndarray, mats: np.ndarray) -> float:
    T = V @ mats @ V.T
    diag = np.einsum("mii->mi", T)
    return float(np.sum(T**2) - np.sum(diag**2))


def _row_normalize(V: np.ndarray, ref: np.ndarray) -> np.ndarray:
    s = np.sqrt(np.abs(np.einsum("ij,jk,ik->i", V, ref, V)))
    s[s == 0] = 1.0
    return V / s[:, None]


def whitening_init(ref: np.ndarray) -> np.ndarray:
    w, U = np.linalg.eigh(ref)
    w = np.maximum(w, 1e-12 * max(w.max(), 1e-300))
    return (U / np.sqrt(w)).T


def uwedge(
    mats: np.ndarray,
    ref: np.ndarray,
    init: np.ndarray | None = None,
    max_iter: int = 1000,
    tol: float = 1e-9,
    det_floor: float = 1e-12,
    theta: float = 0.5,
    record_history: bool = False,
) -> JDResult:
    """Joint-diagonalise symmetric ``mats`` (``M x d x d``) given a PD reference ``ref``."""
    mats = np.asarray(mats, dtype=float)
    mats = 0.5 * (mats + np.swapaxes(mats, 1, 2))
    d = mats.shape[1]
    V = whitening_init(ref) if init is None else np.array(init, dtype=float, copy=True)
    V = _row_normalize(V, ref)
    history = [offdiag_objective(V, mats)] if record_history else []
    converged = False
    it = 0
    eye = np.eye(d)
    for it in range(1, max_iter + 1):
        T = V @ mats @ V.T
        Dg = np.einsum("mii->mi", T)  # M x d
        z = Dg.T @ Dg  # z_ij = sum_m D_mi D_mj
        y = np.einsum("mj,mij->ij", Dg, T)  # y_ij = sum_m D_mj T_mij
        zd = np.diag(z)
        det = np.outer(zd, zd) - z**2
        floor = det_floor * np.outer(zd, zd) + 1e-300
        det = np.where(det > floor, det, floor)
        H = -(zd[:, None] * y - z * y.T) / det
        np.fill_diagonal(H, 0.0)
        nh = np.linalg.norm(H)
        if nh > theta:
            H *= theta / nh
        V = _row_normalize((eye + H) @ V, ref)
        if record_history:
            history.append(offdiag_objective(V, mats))
        if np.max(np.abs(H)) < tol:
            converged = True
            break
    obj = offdiag_objective(V, mats)
    # pair separability diagnostic at the solution
    T = V @ mats @ V.T
    Dg = np.einsum("mii->mi", T)
    z = Dg.T @ Dg
    zd = np.diag(z)
    with np.errstate(divide="ignore", invalid="ignore"):
        sin2 = 1.0 - z**2 / np.outer(zd, zd)
    sin2 = np.nan_to_num(sin2, nan=0.0)
    np.fill_diagonal(sin2, 1.0)
    return JDResult(V=V, A=np.linalg.inv(V), objective=obj, n_iter=it, converged=converged,
                    history=history, pair_sin2=sin2)


def gevd_diag(M1: np.ndarray, M2: np.ndarray) -> np.ndarray:
    """Exact joint diagonaliser of two symmetric matrices via the generalised eigenproblem.

    Returns an unmixing matrix ``V`` (rows = generalised eigenvectors). Exact when
    exactly two centred matrices carry the information (``E = 3``) and the
    generalised eigenvalues are distinct.
    """
    w, vecs = linalg.eig(M1, M2)
    return np.real(vecs).T


def random_starts(ref: np.ndarray, mats: np.ndarray, n_starts: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Diverse initialisations: whitening x random orthogonal, and GEVD of random pencils."""
    d = ref.shape[0]
    W = whitening_init(ref)
    starts = []
    for s in range(n_starts):
        if s % 2 == 0 or mats.shape[0] < 2:
            Q, R = np.linalg.qr(rng.standard_normal((d, d)))
            Q = Q * np.sign(np.diag(R))
            starts.append(Q @ W)
        else:
            a, b = rng.standard_normal(mats.shape[0]), rng.standard_normal(mats.shape[0])
            M1 = np.tensordot(a, mats, axes=1)
            M2 = np.tensordot(b, mats, axes=1) + 1e-6 * ref
            try:
                V0 = gevd_diag(M1, M2)
                if np.linalg.cond(V0) < 1e8:
                    starts.append(V0)
                    continue
            except np.linalg.LinAlgError:
                pass
            starts.append(W)
    return starts


def multistart_uwedge(mats: np.ndarray, ref: np.ndarray, n_starts: int, rng: np.random.Generator,
                      **kw) -> list[JDResult]:
    return [uwedge(mats, ref, init=V0, **kw) for V0 in random_starts(ref, mats, n_starts, rng)]


def principal_pencil_init(mats: np.ndarray, ref: np.ndarray) -> np.ndarray | None:
    """Deterministic start: GEVD of the two leading principal matrices of the stack.

    The stack ``{D[e]}`` spans at most an ``(E-1)``-dimensional space of
    symmetric matrices; its two leading principal components (in vech space)
    form a pencil whose generalised eigenvectors already jointly diagonalise
    the population stack whenever the corresponding generalised eigenvalues
    are distinct. U-WEDGE started from a plain whitening matrix can stall in
    spurious stationary points (observed on planted data), which this avoids.
    """
    M, d, _ = mats.shape
    if M < 2:
        return None
    flat = mats.reshape(M, -1)
    _, _, Vt = np.linalg.svd(flat, full_matrices=False)
    P1 = Vt[0].reshape(d, d)
    P2 = Vt[1].reshape(d, d)
    P1, P2 = 0.5 * (P1 + P1.T), 0.5 * (P2 + P2.T)
    try:
        V0 = gevd_diag(P1, P2 + 1e-9 * np.trace(np.abs(P2)) / d * np.eye(d))
    except (np.linalg.LinAlgError, ValueError):
        return None
    if not np.all(np.isfinite(V0)) or np.linalg.cond(V0) > 1e10:
        return None
    return V0


def fit_joint_diag(mats: np.ndarray, ref: np.ndarray, n_starts: int = 6,
                   rng: np.random.Generator | None = None, init: np.ndarray | None = None,
                   **kw) -> tuple[JDResult, list[JDResult]]:
    """Best-of-several U-WEDGE runs (lowest off-diagonal objective).

    With ``init`` given (warm start, e.g. inside a bootstrap) a single run is
    made. Otherwise starts are: the principal-pencil GEVD, the whitening
    matrix and ``n_starts - 2`` random starts. All runs are returned so that
    callers can inspect instability across starts (plan p.7 *Diagnose*).
    """
    if init is not None:
        r = uwedge(mats, ref, init=init, **kw)
        return r, [r]
    rng = rng or np.random.default_rng(0)
    starts: list = []
    V0 = principal_pencil_init(mats, ref)
    if V0 is not None:
        starts.append(V0)
    starts.append(whitening_init(ref))
    if n_starts > len(starts):
        starts.extend(random_starts(ref, mats, n_starts - len(starts), rng))
    runs = [uwedge(mats, ref, init=s, **kw) for s in starts]
    best = min(runs, key=lambda r: r.objective)
    return best, runs
