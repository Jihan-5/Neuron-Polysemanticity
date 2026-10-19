r"""ISMB — Identified-Set Multiplicity Bounds (the proposed uncertainty layer).

Plan p.7: "Construct a set of parameter explanations consistent with moment
uncertainty, factor-count bounds, nonduplication, visibility, and allowed
correlation. Seek the smallest and largest multiplicity consistent with that
set." ISMB makes that construction tractable by decomposing it into pieces
whose extremes can be computed exactly or bounded conservatively.

Procedure (all steps use only :class:`~polyid.data.ObservedData`)
------------------------------------------------------------------
1. **Fit.** Centred per-environment covariances; best-of-``n_starts``
   joint diagonalisation; unit-norm columns ``Â`` and centred profiles ``ĉ``.
2. **Bootstrap.** ``n_boot`` replicates resample rows (or clusters) within
   environments; each replicate is warm-started from the full-sample solution
   and aligned to it by sign/permutation (estimate-to-estimate only).
3. **Visibility.** Analytic (fourth-moment, cluster-robust) band for each
   component's visibility statistic, Bonferroni over components and
   environments → *sure-visible / ambiguous / invisible*. (A bootstrap
   percentile band was tried first and fails: pseudo-components spanning an
   invisible subspace are re-labelled arbitrarily across replicates.)
4. **Who is separately identified?** For every pair of *sure-visible*
   factors a Cragg–Donald rank-1 test of profile collinearity (H0 = *not
   separated*); pairs that cannot be separated at the Bonferroni level are
   merged (union–find). An approximate residual-dependence test merges pairs
   whose latents look dependent (allowed-correlation route). Merging only
   widens. Each merged group gets a joint rank-1 test: if the whole group is
   consistent with collinear profiles the exact rotation range is used,
   otherwise the span outer bound.
5. **Singletons.** Sup-t bootstrap band (robust IQR scale) for the loadings of
   separated or ambiguous-visibility factors, simultaneous over the ``K``
   loadings of each neuron (``simultaneity="row"``, matching the per-neuron
   coverage target) or over all loadings (``"all"``); an entry is *surely
   counted* if its band lies outside ``(-tau, tau)``, *surely not counted* if
   inside, ambiguous otherwise. Ambiguous-visibility factors only ever add
   to the upper bound. With ``selection="gms"`` (default) loadings further
   than ``kappa_n = sqrt(log n)`` robust SEs from ``tau`` are classified by
   the point estimate and excluded from the critical value (generalised
   moment selection, Andrews & Soares 2010), which removes the cost of
   simultaneity over loadings that cannot flip.
6. **Groups.** Exact Schur–Horn ``[min, max]`` of the group's contribution
   (rotation-unidentified, positive profile coefficients), or the span outer
   bound (correlation allowed / mixed signs / joint test rejects); widened by
   the bootstrap distribution of the range.
7. **Combine.** ``L_j`` sums surely-counted singleton entries of
   sure-visible factors and group minima; ``U_j`` sums everything that is not
   surely excluded plus group maxima. ``alpha`` is split across steps 3–6 by
   a union bound.

What the coverage argument needs (stated, not hidden)
-----------------------------------------------------
Asymptotically ``P(L_j <= P_j <= U_j for all j) >= 1 - alpha`` holds if
(a) the declared model holds with stationary noise and conditionally
independent latents *outside* merged groups; (b) every truly collinear pair is
merged (controlled by the pair-test level); (c) separated factors are strongly
identified so the bootstrap band for their loadings is valid; (d) the
bootstrap range quantiles cover the group ranges. (c) is exactly where
Stoye-type uniformity problems live (near-collinear but distinct profiles);
experiments E2/E4 measure that regime instead of assuming it away. The AR-grid
procedure (``bounds.ar_grid``) gives a weak-identification-robust reference in
``d = 3``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from scipy import stats

from polyid.bounds.rank_tests import UnionFind, group_rank1_pvalue, profile_band, test_pair
from polyid.bounds.schur_horn import group_ranges
from polyid.data import ObservedData
from polyid.estimand import EstimandSpec, multiplicity, normalize_columns
from polyid.estimators.base import Access, EstimateResult
from polyid.jointdiag import fit_joint_diag
from polyid.matching import align_to_reference, amari_distance
from polyid.moments import bootstrap_weights, centered_stack, compute_moment_stats, env_covariances


@dataclass
class ISMBConfig:
    alpha: float = 0.05
    share_pairs: float = 0.25
    share_loadings: float = 0.45
    share_visibility: float = 0.15
    share_groups: float = 0.15
    pair_correction: str = "bonferroni"  # or "none"
    dependence_merge: bool = True
    dependence_alpha: float = 0.01
    allow_within_group_correlation: bool = False
    n_boot: int = 200
    n_starts: int = 6
    simultaneity: str = "row"  # "row": per-neuron simultaneous bands; "all": over every loading
    selection: str = "gms"  # "gms": generalised moment selection of near-threshold loadings; "none"
    gms_kappa: float | None = None  # default sqrt(log(min_e n_e))
    abstain_width: int = 1

    @classmethod
    def from_dict(cls, d: dict | None) -> "ISMBConfig":
        return cls(**(d or {}))


@dataclass
class ISMBFit:
    A_hat: np.ndarray
    B_hat: np.ndarray
    profiles: np.ndarray
    vis: np.ndarray
    vis_lo: np.ndarray
    vis_hi: np.ndarray
    vis_status: np.ndarray  # 2 sure-visible, 1 ambiguous, 0 invisible
    groups: list
    group_modes: list
    pair_tests: list
    load_lo: np.ndarray
    load_hi: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    point: np.ndarray
    naive_point: np.ndarray
    diagnostics: dict = field(default_factory=dict)


def _profiles(B: np.ndarray, D: np.ndarray) -> np.ndarray:
    return np.einsum("ki,eij,kj->ke", B, D, B)


def _rms(P: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.sqrt(np.mean(P**2, axis=axis))


def _group_lambdas(P_G: np.ndarray) -> np.ndarray:
    """Coefficients of group profiles on their common (leading) direction, sign-normalised."""
    if P_G.shape[0] == 1:
        lam = np.array([_rms(P_G[0])])
        return lam
    _, _, Vt = np.linalg.svd(P_G, full_matrices=False)
    lam = P_G @ Vt[0]
    if lam.sum() < 0:
        lam = -lam
    return lam


def _offdiag_ratio(B: np.ndarray, D: np.ndarray) -> float:
    T = B @ D @ B.T
    diag = np.einsum("mii->mi", T)
    off = np.sum(T**2) - np.sum(diag**2)
    return float(np.sqrt(max(off, 0.0) / max(np.sum(T**2), 1e-300)))


def fit_ismb(data: ObservedData, spec: EstimandSpec, cfg: ISMBConfig, rng: np.random.Generator) -> ISMBFit:
    tau, v_min = spec.tau, spec.v_min
    ms = compute_moment_stats(data, with_covariance=False)
    E, d = ms.n_envs, ms.d
    best, runs = fit_joint_diag(ms.D, ms.pooled, n_starts=cfg.n_starts, rng=rng)
    A_n, norms = normalize_columns(best.A)
    B_n = norms[:, None] * best.V
    K = A_n.shape[1]
    prof = _profiles(B_n, ms.D)
    U = data.X @ B_n.T  # unmixed signals, one column per estimated component

    # --------------------------------------------- visibility (analytic, per component)
    a_vis = cfg.alpha * cfg.share_visibility
    z_vis = float(stats.norm.ppf(1 - a_vis / (2 * E * max(K, 1))))
    vis = np.empty(K)
    vis_lo = np.empty(K)
    vis_hi = np.empty(K)
    for k in range(K):
        vis[k], vis_lo[k], vis_hi[k] = profile_band(U[:, k], data.env, E, z_vis, data.cluster)
    vis_status = np.where(vis_lo >= v_min, 2, np.where(vis_hi < v_min, 0, 1))
    sure = [k for k in range(K) if vis_status[k] == 2]
    ambiguous = [k for k in range(K) if vis_status[k] == 1]

    # ------------------------------------------------- which factors are separated?
    pairs = [(sure[i], sure[j]) for i in range(len(sure)) for j in range(i + 1, len(sure))]
    n_pairs = max(len(pairs), 1)
    corr = n_pairs if cfg.pair_correction == "bonferroni" else 1
    a_pair = cfg.alpha * cfg.share_pairs / corr
    a_dep = cfg.dependence_alpha / corr
    uf = UnionFind(K)
    span_edges: set = set()
    tests = []
    for k, l in pairs:
        t = test_pair(U[:, k], U[:, l], data.env, E, k, l, data.cluster)
        tests.append(t)
        if t.collinear_p >= a_pair:
            uf.union(k, l)
        if cfg.dependence_merge and t.dependence_p < a_dep:
            uf.union(k, l)
            span_edges.add((k, l))
    comps = uf.groups(sure)
    groups = [g for g in comps if len(g) > 1]
    singletons = sorted([g[0] for g in comps if len(g) == 1] + ambiguous)
    group_joint_p = [group_rank1_pvalue(U[:, G], data.env, E, data.cluster) for G in groups]

    # ------------------------------------------------------------------- bootstrap
    nb = cfg.n_boot
    boot_A = np.empty((nb, d, K))
    boot_prof = np.empty((nb, K, E))
    boot_cos = np.empty((nb, K))
    for b in range(nb):
        w = bootstrap_weights(data, rng)
        C, _, _ = env_covariances(data.X, data.env, E, weights=w)
        D = centered_stack(C)
        jd, _ = fit_joint_diag(D, C.mean(axis=0), init=best.V, max_iter=300)
        Ab, nrm = normalize_columns(jd.A)
        pb = _profiles(nrm[:, None] * jd.V, D)
        Ab_al, perm, cosv = align_to_reference(Ab, A_n)
        boot_A[b], boot_prof[b], boot_cos[b] = Ab_al, pb[perm], cosv

    # ------------------------------------------------------------ singleton loadings
    load_lo = np.full((d, K), np.nan)
    load_hi = np.full((d, K), np.nan)
    surely_in = np.zeros((d, K), dtype=bool)
    maybe_in = np.zeros((d, K), dtype=bool)
    crit = np.full(d, np.nan)
    if singletons:
        S = np.array(singletons)
        dev = boot_A[:, :, S] - A_n[None, :, S]
        q75, q25 = np.quantile(dev, [0.75, 0.25], axis=0)
        scale = np.maximum((q75 - q25) / 1.349, 1e-6)
        tstat = np.abs(dev) / scale[None]
        a_load = cfg.alpha * cfg.share_loadings
        # generalised moment selection (Andrews & Soares 2010): loadings more than
        # kappa_n robust SEs away from the threshold are classified directly and
        # excluded from the simultaneous critical value; kappa_n -> inf slowly.
        if cfg.selection == "gms":
            kappa = cfg.gms_kappa or float(np.sqrt(np.log(max(int(data.env_counts().min()), 3))))
            relevant = np.abs(np.abs(A_n[:, S]) - tau) / scale <= kappa
        else:
            kappa = np.inf
            relevant = np.ones_like(scale, dtype=bool)
        t_rel = np.where(relevant[None], tstat, 0.0)
        if cfg.simultaneity == "all":
            c_all = float(np.quantile(t_rel.reshape(nb, -1).max(axis=1), 1 - a_load, method="higher"))
            crit[:] = c_all
        else:  # row-wise: simultaneous over the loadings that make up one P_j
            crit = np.quantile(t_rel.max(axis=2), 1 - a_load, axis=0, method="higher")
        half = np.where(relevant, crit[:, None] * scale, 0.0)
        lo = A_n[:, S] - half
        hi = A_n[:, S] + half
        load_lo[:, S], load_hi[:, S] = lo, hi
        above = (lo >= tau) | (hi <= -tau)
        below = (lo > -tau) & (hi < tau)
        surely_in[:, S] = above & (vis_status[S] == 2)[None, :]
        maybe_in[:, S] = ~below
    lower = surely_in.sum(axis=1).astype(int)
    upper = maybe_in.sum(axis=1).astype(int)

    # ------------------------------------------------------------------------ groups
    a_grp = cfg.alpha * cfg.share_groups / max(len(groups), 1)
    group_modes = []
    group_mid = np.zeros(d)
    for G, p_joint in zip(groups, group_joint_p):
        has_span = any((min(a, b), max(a, b)) in span_edges for a in G for b in G if a < b)
        # 'rotation' (exact Schur-Horn) only if the whole group is jointly consistent with
        # collinear profiles; otherwise the conservative span outer bound.
        allow = cfg.allow_within_group_correlation or has_span or p_joint < a_pair
        lam = _group_lambdas(prof[G])
        lo_hat, hi_hat, mode = group_ranges(A_n[:, G], lam, tau, allow_correlation=allow)
        lo_b = np.empty((nb, d), dtype=int)
        hi_b = np.empty((nb, d), dtype=int)
        for b in range(nb):
            lam_b = _group_lambdas(boot_prof[b][G])
            lo_b[b], hi_b[b], _ = group_ranges(boot_A[b][:, G], lam_b, tau, allow_correlation=allow)
        L_G = np.minimum(lo_hat, np.quantile(lo_b, a_grp / 2, axis=0, method="lower")).astype(int)
        U_G = np.maximum(hi_hat, np.quantile(hi_b, 1 - a_grp / 2, axis=0, method="higher")).astype(int)
        lower += L_G
        upper += U_G
        group_mid += np.floor((lo_hat + hi_hat) / 2 + 0.5)
        group_modes.append(mode)
    upper = np.minimum(upper, len(sure) + len(ambiguous))

    # ---------------------------------------------------------------- point estimate
    point_vis = vis >= v_min
    naive_point = multiplicity(A_n, tau, visible=point_vis, normalize=False)
    in_group = np.zeros(K, dtype=bool)
    for G in groups:
        in_group[G] = True
    point = multiplicity(A_n, tau, visible=point_vis & ~in_group, normalize=False) + group_mid.astype(int)
    point = np.clip(point, lower, upper)

    # ------------------------------------------------------------------- diagnostics
    amaris = [amari_distance(r.V, best.A) for r in runs if r is not best]
    iu = np.triu_indices(K, 1)
    diag = {
        "jd_objective": best.objective,
        "jd_converged": best.converged,
        "jd_iterations": best.n_iter,
        "restart_amari_max": float(max(amaris)) if amaris else 0.0,
        "offdiag_ratio": _offdiag_ratio(B_n, ms.D),
        "min_pair_sin2": float(best.pair_sin2[iu].min()) if K > 1 else 1.0,
        "boot_align_cos_median": float(np.median(boot_cos)),
        "boot_align_cos_min_col": float(np.median(boot_cos, axis=0).min()),
        "cond_A": float(np.linalg.cond(A_n)),
        "n_sure_visible": len(sure),
        "n_ambiguous_visible": len(ambiguous),
        "n_invisible": int(np.sum(vis_status == 0)),
        "n_groups": len(groups),
        "group_sizes": [len(g) for g in groups],
        "group_joint_p": group_joint_p,
        "n_span_groups": int(sum(m == "span" for m in group_modes)),
        "n_dependence_flags": len(span_edges),
        "loading_crit_median": float(np.nanmedian(crit)) if singletons else float("nan"),
        "alpha": cfg.alpha,
    }
    return ISMBFit(
        A_hat=A_n, B_hat=B_n, profiles=prof, vis=vis, vis_lo=vis_lo, vis_hi=vis_hi, vis_status=vis_status,
        groups=groups, group_modes=group_modes, pair_tests=tests, load_lo=load_lo, load_hi=load_hi,
        lower=lower, upper=upper, point=point, naive_point=naive_point, diagnostics=diag,
    )


class ISMB:
    """Estimator wrapper returning an :class:`EstimateResult` (kind ``interval``)."""

    name = "ismb"

    def __init__(self, cfg: ISMBConfig | dict | None = None):
        self.cfg = cfg if isinstance(cfg, ISMBConfig) else ISMBConfig.from_dict(cfg)

    def fit(self, data: ObservedData, spec: EstimandSpec, rng: np.random.Generator) -> EstimateResult:
        f = fit_ismb(data, spec, self.cfg, rng)
        diag = dict(f.diagnostics)
        diag["groups"] = f.groups
        diag["group_modes"] = f.group_modes
        diag["abstain"] = (f.upper - f.lower >= self.cfg.abstain_width).tolist()
        return EstimateResult(
            method=self.name, kind="interval", point=f.point.astype(float), lower=f.lower.astype(float),
            upper=f.upper.astype(float), loadings=f.A_hat,
            label=f"confidence(asymptotic, alpha={self.cfg.alpha}, conditions in bounds/ismb.py)",
            access=Access(env_labels=True, higher_moments=False, n_observations=data.n,
                          restarts=self.cfg.n_starts, notes=f"bootstrap={self.cfg.n_boot}; fourth moments used only for variances"),
            diagnostics=diag,
        )


# ---------------------------------------------------------------------------
# infinite-sample version (exact moments): the identified set of the procedure
# ---------------------------------------------------------------------------

def ismb_population(D: np.ndarray, pooled: np.ndarray, spec: EstimandSpec, n_starts: int = 8,
                    collinear_tol: float = 1e-6, residual_tol: float = 1e-6,
                    rng: np.random.Generator | None = None) -> dict:
    """Apply ISMB's logic to *exact* centred covariances (no sampling error).

    Returns per-neuron ``lower``/``upper`` — the identified set of ``P_j``
    implied by second moments under the declared model with adaptive
    allowed-correlation merges — plus the detected group structure and a
    ``misfit`` flag when the stack cannot be jointly diagonalised.
    """
    rng = rng or np.random.default_rng(0)
    tau, v_min = spec.tau, spec.v_min
    best, _ = fit_joint_diag(D, pooled, n_starts=n_starts, rng=rng, max_iter=3000, tol=1e-13)
    A_n, norms = normalize_columns(best.A)
    B_n = norms[:, None] * best.V
    K = A_n.shape[1]
    T = B_n @ D @ B_n.T
    prof = np.einsum("mkk->km", T)
    vis = _rms(prof)
    scale = max(np.max(np.abs(prof)), 1e-300)
    cand = [k for k in range(K) if vis[k] >= v_min]
    uf = UnionFind(K)
    span_edges = set()
    for i, k in enumerate(cand):
        for l in cand[i + 1:]:
            pk, pl = prof[k], prof[l]
            cos = abs(pk @ pl) / max(np.linalg.norm(pk) * np.linalg.norm(pl), 1e-300)
            if cos >= 1 - collinear_tol:
                # within a collinear pair any rotation is admissible, so a leftover
                # off-diagonal residual there carries no information
                uf.union(k, l)
                continue
            off = np.max(np.abs(T[:, k, l])) / scale
            if off > residual_tol:
                uf.union(k, l)
                span_edges.add((k, l))
    comps = uf.groups(cand)
    lower = np.zeros(A_n.shape[0], dtype=int)
    upper = np.zeros(A_n.shape[0], dtype=int)
    modes = []
    for G in comps:
        if len(G) == 1:
            c = (np.abs(A_n[:, G[0]]) >= tau - 1e-12).astype(int)
            lower += c
            upper += c
            continue
        allow = any((min(a, b), max(a, b)) in span_edges for a in G for b in G if a < b)
        lo, hi, mode = group_ranges(A_n[:, G], _group_lambdas(prof[G]), tau, allow_correlation=allow)
        lower += lo
        upper += hi
        modes.append(mode)
    return {
        "lower": lower,
        "upper": upper,
        "groups": [g for g in comps if len(g) > 1],
        "group_modes": modes,
        "misfit": _offdiag_ratio(B_n, D) > 1e-5,
        "offdiag_ratio": _offdiag_ratio(B_n, D),
        "A_hat": A_n,
        "visible": vis >= v_min,
    }
