r"""Oracle-side identification analysis: is ``P_j`` identified *even with infinite data*?

This module reads the :class:`~polyid.generators.planted.Oracle` and is
therefore only used by metrics and experiment runners, never by estimators.

It answers the plan's question "distinguish 'not enough samples' from 'not
identified even with infinite samples'" by computing, from the *true*
parameters, the identified set of ``P_j`` under the declared second-moment
model:

* **identified** — the identified set is a single integer;
* **partial** — the set is an interval of width >= 1 (structural ambiguity);
* **misspecified** — the declared independent-factor model is false for this
  data-generating process (latents dependent or noise non-stationary).
  Two sub-cases are recorded: *detectable* (no independent-factor model fits
  the population moments) and *undetectable* (another independent-factor
  model with different loadings fits them exactly, e.g. a co-occurring pair —
  the dangerous case, see ``docs/THEORY.md`` Remark 4).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from polyid.bounds.ismb import ismb_population
from polyid.bounds.schur_horn import group_ranges
from polyid.bounds.rank_tests import UnionFind
from polyid.estimand import EstimandSpec, multiplicity, profile_abs_cosines
from polyid.generators.planted import Oracle
from polyid.moments import population_stats


@dataclass
class IdentificationReport:
    lower: np.ndarray  # exact identified set under the declared model (oracle parameters)
    upper: np.ndarray
    status: np.ndarray  # per neuron: "identified" | "partial" | "misspecified"
    groups: list
    misspecified: bool
    misfit_detectable: bool | None
    procedure_lower: np.ndarray  # what ISMB's logic returns on exact moments
    procedure_upper: np.ndarray
    procedure_covers_truth: bool

    def summary(self) -> dict:
        w = self.upper - self.lower
        return {
            "frac_identified": float(np.mean(self.status == "identified")),
            "frac_partial": float(np.mean(self.status == "partial")),
            "frac_misspecified": float(np.mean(self.status == "misspecified")),
            "mean_width": float(np.mean(w)),
            "n_groups": len(self.groups),
            "misspecified": self.misspecified,
            "misfit_detectable": self.misfit_detectable,
            "procedure_covers_truth": self.procedure_covers_truth,
        }


def identified_set(oracle: Oracle, spec: EstimandSpec, collinear_tol: float = 1e-9) -> IdentificationReport:
    tau = spec.tau
    A, prof, vis = oracle.A, oracle.profiles, oracle.visible
    K = A.shape[1]
    truth = multiplicity(A, tau, visible=vis)
    misspecified = (not oracle.independent) or (not oracle.noise_stationary)

    # exact groups from the true profiles (visible factors only)
    G = profile_abs_cosines(prof)
    uf = UnionFind(K)
    idx = np.flatnonzero(vis)
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            k, l = idx[a], idx[b]
            if G[k, l] >= 1 - collinear_tol:
                uf.union(k, l)
    # latents that are correlated are only identified as a block (allowed-correlation model)
    if not oracle.independent:
        for e in range(oracle.latent_cov.shape[0]):
            C = oracle.latent_cov[e]
            for k in idx:
                for l in idx:
                    if k < l and abs(C[k, l]) > 1e-12:
                        uf.union(k, l)
    comps = uf.groups(list(idx))
    lower = np.zeros(A.shape[0], dtype=int)
    upper = np.zeros(A.shape[0], dtype=int)
    groups = []
    corr_pairs = set()
    if not oracle.independent:
        off = np.abs(oracle.latent_cov).max(axis=0)
        corr_pairs = {(k, l) for k in range(K) for l in range(K) if k < l and off[k, l] > 1e-12}
    for comp in comps:
        if len(comp) == 1:
            c = (np.abs(A[:, comp[0]]) >= tau - 1e-12).astype(int)
            lower += c
            upper += c
            continue
        groups.append(comp)
        allow = any((a, b) in corr_pairs for a in comp for b in comp if a < b)
        # coefficients on the common profile direction
        _, _, Vt = np.linalg.svd(prof[comp], full_matrices=False)
        lam = prof[comp] @ Vt[0]
        if lam.sum() < 0:
            lam = -lam
        lo, hi, _ = group_ranges(A[:, comp], lam, tau, allow_correlation=allow)
        lower += lo
        upper += hi

    # what the estimation procedure itself returns on exact moments
    ps = population_stats(oracle.pop_cov)
    proc = ismb_population(ps.D, ps.pooled, spec)
    covers = bool(np.all((proc["lower"] <= truth) & (truth <= proc["upper"])))
    detectable = None
    if misspecified:
        detectable = bool(proc["misfit"])

    status = np.where(upper > lower, "partial", "identified").astype(object)
    if misspecified:
        status[:] = "misspecified"
    return IdentificationReport(
        lower=lower, upper=upper, status=status, groups=groups, misspecified=misspecified,
        misfit_detectable=detectable, procedure_lower=proc["lower"], procedure_upper=proc["upper"],
        procedure_covers_truth=covers,
    )
