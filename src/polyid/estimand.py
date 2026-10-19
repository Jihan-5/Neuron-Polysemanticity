r"""The estimand: thresholded structural loading multiplicity.

Declared model (plan p.3)
-------------------------
For environment :math:`e`::

    h^{(e)} = A z^{(e)} + \varepsilon^{(e)},   A \in R^{d x K}

* columns of ``A`` are normalised to unit Euclidean norm; a column and its
  negation describe the same feature; the column order is irrelevant
  (equivalence up to **sign and permutation**);
* the latent coordinates are conditionally independent within an environment
  (relaxed in the stress experiments, never silently);
* :math:`P_j(\tau) = \#\{k \text{ visible}: |A_{jk}| \ge \tau\}`.

Visibility (minimum-variation condition)
----------------------------------------
Only features whose variance actually changes across the observed
environments are part of the recoverable target. With :math:`v_{e,k}` the
variance of :math:`z_k` in environment :math:`e` (columns unit-norm), the
**centred profile** is :math:`c_{k,e} = v_{e,k} - \bar v_k` and the
visibility statistic is the across-environment standard deviation

.. math:: \mathrm{vis}_k = \Big(\tfrac1E\sum_e c_{k,e}^2\Big)^{1/2}.

Feature :math:`k` is visible iff :math:`\mathrm{vis}_k \ge v_{\min}`. The
centred profile is reference-free (it does not single out an environment 0)
and spans the same space as the covariance differences
:math:`C_e - C_0`, so it carries exactly the information the
covariance-difference strategy uses.

Minimality / non-duplication
----------------------------
A factor may not be split into several copies: two admissible columns must
satisfy :math:`|\cos(a_k, a_l)| \le c_{\max}` and the factor count is the
smallest that explains the moments. :func:`check_nonduplication` enforces the
first condition; the second is enforced by counting only visible factors of a
minimal decomposition (see ``polyid.bounds``).

Boundary convention
-------------------
:math:`P_j` uses a *closed* threshold, :math:`|A_{jk}| \ge \tau`, implemented
with an absolute tolerance ``atol`` so that a loading equal to ``tau`` up to
floating-point error is counted. Ties are measure-zero under every generator
in this repository except the deliberately constructed near-``tau`` stress
case, where they are the point of the experiment.

What :math:`P_j` is *not*: a global feature count, a histogram mode count, an
activation frequency, or a count of human-interpretable concepts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

ATOL = 1e-12


@dataclass(frozen=True)
class EstimandSpec:
    """Everything that must be fixed *before* looking at estimates.

    Attributes
    ----------
    tau:
        Loading threshold (primary 0.10; sensitivity 0.05 and 0.20).
    v_min:
        Visibility threshold on the across-environment SD of a factor's
        variance, in squared activation units with unit-norm columns.
    max_abs_cos:
        Non-duplication bound: admissible columns have ``|cos| <= max_abs_cos``.
    k_max:
        Declared upper bound on the factor count (``None`` = ``d``, square case).
    """

    tau: float = 0.10
    v_min: float = 6e-3
    max_abs_cos: float = 0.98
    k_max: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def normalize_columns(A: np.ndarray, eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(A / ||A_:,k||, norms)``; zero columns stay zero."""
    A = np.asarray(A, dtype=float)
    norms = np.linalg.norm(A, axis=0)
    safe = np.where(norms > eps, norms, 1.0)
    return A / safe, norms


def canonical_signs(A: np.ndarray) -> np.ndarray:
    """Flip each column so that its largest-magnitude entry is positive.

    Purely a presentation convention; :math:`P_j` is sign invariant.
    """
    A = np.array(A, dtype=float, copy=True)
    if A.size == 0:
        return A
    idx = np.argmax(np.abs(A), axis=0)
    s = np.sign(A[idx, np.arange(A.shape[1])])
    s[s == 0] = 1.0
    return A * s


def support(A: np.ndarray, tau: float, visible: np.ndarray | None = None, normalize: bool = True) -> np.ndarray:
    """Boolean ``d x K`` matrix of counted loadings ``|A_jk| >= tau`` (visible columns only)."""
    An = normalize_columns(A)[0] if normalize else np.asarray(A, dtype=float)
    S = np.abs(An) >= tau - ATOL
    if visible is not None:
        S = S & np.asarray(visible, dtype=bool)[None, :]
    return S


def multiplicity(A: np.ndarray, tau: float, visible: np.ndarray | None = None, normalize: bool = True) -> np.ndarray:
    r""":math:`P_j(\tau)` for every neuron ``j`` (integer array of length ``d``)."""
    return support(A, tau, visible=visible, normalize=normalize).sum(axis=1).astype(int)


def smooth_multiplicity(A: np.ndarray, tau: float, bandwidth: float = 0.01, visible: np.ndarray | None = None) -> np.ndarray:
    r"""Smooth surrogate :math:`\tilde P_j = \sum_k \sigma((|A_{jk}| - \tau)/h)`.

    Used only to illustrate where smooth-functional inference (calibrated
    projection, Kaido–Molinari–Stoye 2019) applies and where the discontinuous
    :math:`P_j` departs from it. It is *not* the estimand.
    """
    An = normalize_columns(A)[0]
    z = (np.abs(An) - tau) / bandwidth
    s = 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))
    if visible is not None:
        s = s * np.asarray(visible, dtype=float)[None, :]
    return s.sum(axis=1)


def centered_profiles(variances: np.ndarray) -> np.ndarray:
    """``variances`` is ``E x K``; returns the ``K x E`` centred profile matrix ``c``."""
    v = np.asarray(variances, dtype=float)
    return (v - v.mean(axis=0, keepdims=True)).T


def visibility_stat(profiles: np.ndarray) -> np.ndarray:
    """RMS of each row of a ``K x E`` centred profile matrix."""
    P = np.asarray(profiles, dtype=float)
    return np.sqrt(np.mean(P**2, axis=1))


def visible_mask(profiles: np.ndarray, v_min: float) -> np.ndarray:
    return visibility_stat(profiles) >= v_min - ATOL


def column_abs_cosines(A: np.ndarray) -> np.ndarray:
    An = normalize_columns(A)[0]
    G = np.abs(An.T @ An)
    np.fill_diagonal(G, 0.0)
    return G


def check_nonduplication(A: np.ndarray, max_abs_cos: float) -> tuple[bool, tuple[int, int] | None, float]:
    """Return ``(ok, worst_pair, worst_abs_cos)`` for the non-duplication rule."""
    if A.shape[1] < 2:
        return True, None, 0.0
    G = column_abs_cosines(A)
    i, j = np.unravel_index(np.argmax(G), G.shape)
    worst = float(G[i, j])
    return worst <= max_abs_cos, (int(min(i, j)), int(max(i, j))), worst


def profile_abs_cosines(profiles: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    """``|cos|`` between centred profiles (rows); 1 means collinear (unidentified pair)."""
    P = np.asarray(profiles, dtype=float)
    n = np.linalg.norm(P, axis=1)
    Pn = P / np.where(n > eps, n, 1.0)[:, None]
    G = np.abs(Pn @ Pn.T)
    np.fill_diagonal(G, 0.0)
    return G


def environmental_diversity(profiles: np.ndarray, visible: np.ndarray | None = None) -> float:
    r"""Min pairwise :math:`\sin` angle between visible centred profiles.

    0 means at least one exactly collinear pair (not identified by second
    moments); values near 1 mean well-separated variance profiles. Used as an
    axis of the identification map.
    """
    P = np.asarray(profiles, dtype=float)
    if visible is not None:
        P = P[np.asarray(visible, dtype=bool)]
    if P.shape[0] < 2:
        return 1.0
    G = profile_abs_cosines(P)
    iu = np.triu_indices(P.shape[0], 1)
    c = np.clip(G[iu], 0.0, 1.0)
    return float(np.sqrt(1.0 - c.max() ** 2))
