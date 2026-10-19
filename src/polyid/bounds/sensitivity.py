"""Multi-start sensitivity range — explicitly *not* a confidence interval.

Plan p.7, "Compute honesty": warm starts and multiple optimisations can find
competing explanations, but they do not certify that every explanation was
considered; such results are labelled "sensitivity range" until an outer-bound
or confidence argument exists.

The range here is the min/max of ``P_j`` over joint-diagonalisation solutions
(random restarts on the full sample plus warm-started fits on a few bootstrap
perturbations) whose off-diagonal objective is within ``rel_tol`` of the best
full-sample objective.
"""

from __future__ import annotations

import numpy as np

from polyid.data import ObservedData
from polyid.estimand import EstimandSpec, multiplicity, normalize_columns
from polyid.estimators.base import Access, EstimateResult
from polyid.jointdiag import fit_joint_diag, offdiag_objective, random_starts, uwedge, _row_normalize
from polyid.moments import bootstrap_weights, centered_stack, compute_moment_stats, env_covariances


def _count(V: np.ndarray, D: np.ndarray, spec: EstimandSpec) -> np.ndarray:
    A, nrm = normalize_columns(np.linalg.inv(V))
    B = nrm[:, None] * V
    vis = np.sqrt(np.mean(np.einsum("ki,eij,kj->ke", B, D, B) ** 2, axis=1))
    return multiplicity(A, spec.tau, visible=vis >= spec.v_min, normalize=False)


class SensitivityRange:
    name = "sensitivity"

    def __init__(self, cfg: dict | None = None):
        cfg = cfg or {}
        self.n_restarts = int(cfg.get("n_restarts", 24))
        self.n_perturb = int(cfg.get("n_perturb", 10))
        self.rel_tol = float(cfg.get("rel_tol", 0.10))

    def fit(self, data: ObservedData, spec: EstimandSpec, rng: np.random.Generator) -> EstimateResult:
        ms = compute_moment_stats(data, with_covariance=False)
        best, _ = fit_joint_diag(ms.D, ms.pooled, n_starts=6, rng=rng)
        counts = [_count(best.V, ms.D, spec)]
        accepted = 1
        for V0 in random_starts(ms.pooled, ms.D, self.n_restarts, rng):
            r = uwedge(ms.D, ms.pooled, init=V0)
            if r.objective <= best.objective * (1 + self.rel_tol) + 1e-15:
                counts.append(_count(r.V, ms.D, spec))
                accepted += 1
        for _ in range(self.n_perturb):
            w = bootstrap_weights(data, rng)
            C, _, _ = env_covariances(data.X, data.env, data.n_envs, weights=w)
            D = centered_stack(C)
            for V0 in random_starts(C.mean(axis=0), D, 2, rng) + [best.V]:
                r = uwedge(D, C.mean(axis=0), init=V0)
                # judge fit on the *full-sample* moments so all solutions face the same criterion
                Vn = _row_normalize(r.V, ms.pooled)
                if offdiag_objective(Vn, ms.D) <= best.objective * (1 + self.rel_tol) + 1e-15:
                    counts.append(_count(r.V, ms.D, spec))
                    accepted += 1
        counts = np.array(counts)
        return EstimateResult(
            method=self.name, kind="interval", point=counts[0].astype(float),
            lower=counts.min(axis=0).astype(float), upper=counts.max(axis=0).astype(float),
            label="sensitivity range (no coverage claim)",
            access=Access(env_labels=True, higher_moments=False, n_observations=data.n,
                          restarts=self.n_restarts + 3 * self.n_perturb),
            diagnostics={"n_solutions_accepted": accepted},
        )
