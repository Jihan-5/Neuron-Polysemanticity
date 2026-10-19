"""Sign- and permutation-invariant column matching.

Two uses, kept strictly apart:

1. **Evaluation** (``metrics``): match estimated columns to planted columns by
   a sign-invariant assignment "only for evaluation" (plan p.7).
2. **Estimate-to-estimate alignment** (``bounds``): align bootstrap or restart
   replicates to the full-sample estimate. This never touches the oracle.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from polyid.estimand import normalize_columns


def abs_cosine_matrix(A_est: np.ndarray, A_ref: np.ndarray) -> np.ndarray:
    a = normalize_columns(A_est)[0]
    b = normalize_columns(A_ref)[0]
    return np.abs(a.T @ b)


def match_columns(A_est: np.ndarray, A_ref: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hungarian assignment maximising total |cos|.

    Returns ``(est_idx, ref_idx, abs_cos)``: estimated column ``est_idx[i]`` is
    matched to reference column ``ref_idx[i]``. Rectangular inputs are allowed
    (unmatched columns are dropped).
    """
    G = abs_cosine_matrix(A_est, A_ref)
    r, c = linear_sum_assignment(-G)
    return r, c, G[r, c]


def align_to_reference(A_rep: np.ndarray, A_ref: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Permute and sign-flip columns of ``A_rep`` to best match ``A_ref``.

    Returns ``(A_aligned, perm, abs_cos)`` with ``A_aligned[:, k]`` matched to
    ``A_ref[:, k]``. Both inputs must have the same number of columns.
    """
    if A_rep.shape != A_ref.shape:
        raise ValueError("alignment requires equal shapes")
    an = normalize_columns(A_rep)[0]
    bn = normalize_columns(A_ref)[0]
    C = an.T @ bn
    r, c = linear_sum_assignment(-np.abs(C))
    perm = np.empty(A_ref.shape[1], dtype=int)
    perm[c] = r
    signs = np.sign(C[perm, np.arange(A_ref.shape[1])])
    signs[signs == 0] = 1.0
    return an[:, perm] * signs, perm, np.abs(C[perm, np.arange(A_ref.shape[1])])


def amari_distance(V: np.ndarray, A: np.ndarray) -> float:
    """Amari index of ``P = V A`` in [0, 1]; 0 iff P is a scaled permutation."""
    P = np.abs(V @ A)
    d = P.shape[0]
    r = (P / P.max(axis=1, keepdims=True)).sum(axis=1) - 1.0
    c = (P / P.max(axis=0, keepdims=True)).sum(axis=0) - 1.0
    return float((r.sum() + c.sum()) / (2.0 * d * (d - 1))) if d > 1 else 0.0
