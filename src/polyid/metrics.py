r"""Scoring: the only place where estimates meet the oracle.

Point accuracy and uncertainty quality are reported **separately** (plan
p.10): "An interval spanning every possible count has good coverage but little
practical value; a narrow wrong interval has poor validity."

Per-neuron quantities
---------------------
``abs_err``           |point - P_j|
``covered``           L_j <= P_j <= U_j  (coverage of the *parameter*)
``covers_idset``      L_j <= L*_j and U_j >= U*_j (coverage of the *identified
                      set* — Imbens–Manski distinction, syllabus S20)
``width``             U_j - L_j
``singleton``         width == 0
``confidently_wrong`` singleton and not covered ("narrowing to a wrong
                      singleton is failure, not progress", plan p.8)

Dataset-level quantities
------------------------
``all_covered``       every neuron covered (simultaneous coverage)
support recovery      precision / recall / F1 of the thresholded support after a
                      sign-invariant Hungarian match (evaluation only)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sstats
from sklearn.metrics import roc_auc_score

from polyid.estimand import EstimandSpec, normalize_columns, support
from polyid.estimators.base import EstimateResult
from polyid.matching import match_columns
from polyid.stats import cluster_bootstrap_ci, wilson_interval


def support_recovery(A_est: np.ndarray, A_true: np.ndarray, tau: float, visible_true: np.ndarray | None = None) -> dict:
    """Precision/recall/F1 of thresholded support after sign-invariant matching."""
    if visible_true is not None:
        A_true = A_true[:, np.asarray(visible_true, dtype=bool)]
    if A_true.shape[1] == 0 or A_est is None or A_est.shape[1] == 0:
        return {"support_precision": np.nan, "support_recall": np.nan, "support_f1": np.nan, "match_cos_mean": np.nan}
    r, c, cos = match_columns(A_est, A_true)
    Se = support(A_est[:, r], tau)
    St = support(A_true[:, c], tau)
    tp = np.sum(Se & St)
    fp = np.sum(Se & ~St)
    unmatched = np.setdiff1d(np.arange(A_true.shape[1]), c)
    fn = np.sum(~Se & St) + int(support(A_true[:, unmatched], tau).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    return {"support_precision": float(prec), "support_recall": float(rec), "support_f1": float(f1),
            "match_cos_mean": float(np.mean(cos))}


def score_result(res: EstimateResult, truth: np.ndarray, *, spec: EstimandSpec, A_true: np.ndarray | None = None,
                 visible_true: np.ndarray | None = None, idset: tuple[np.ndarray, np.ndarray] | None = None,
                 ids: dict | None = None) -> list[dict]:
    """Per-neuron rows for one (dataset, method)."""
    ids = ids or {}
    d = len(truth)
    rows = []
    sup = {}
    if res.loadings is not None and A_true is not None:
        sup = support_recovery(res.loadings, A_true, spec.tau, visible_true)
    for j in range(d):
        r = {**ids, "method": res.method, "kind": res.kind, "neuron": j, "truth": int(truth[j])}
        if res.point is not None and np.isfinite(res.point[j]):
            r["point"] = float(res.point[j])
            r["abs_err"] = abs(float(res.point[j]) - truth[j])
            r["signed_err"] = float(res.point[j]) - truth[j]
        if res.score is not None:
            r["score"] = float(res.score[j])
        if res.lower is not None:
            L, U = float(res.lower[j]), float(res.upper[j])
            # flags stored as floats: pandas averages object-dtype booleans with NaNs incorrectly
            cov = float(L <= truth[j] <= U)
            r.update(lower=L, upper=U, width=U - L, covered=cov, singleton=float(U == L))
            r["confidently_wrong"] = float(U == L and not cov)
            if idset is not None:
                r["idset_lower"], r["idset_upper"] = int(idset[0][j]), int(idset[1][j])
                r["covers_idset"] = float(L <= idset[0][j] and U >= idset[1][j])
        r.update(sup)
        rows.append(r)
    return rows


def _auroc(score: np.ndarray, label: np.ndarray) -> float:
    if label.min() == label.max():
        return np.nan
    return float(roc_auc_score(label, score))


def aggregate(df: pd.DataFrame, by: list[str], dataset_key: str = "dataset") -> pd.DataFrame:
    """Aggregate per-neuron rows into one row per group with uncertainty."""
    out = []
    for keys, g in df.groupby(by, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(by, keys))
        row["n_neurons"] = len(g)
        row["n_datasets"] = g[dataset_key].nunique() if dataset_key in g else np.nan
        clusters = g[dataset_key].to_numpy() if dataset_key in g else np.arange(len(g))
        if "abs_err" in g and g["abs_err"].notna().any():
            gg = g[g["abs_err"].notna()]
            est, lo, hi = cluster_bootstrap_ci(gg["abs_err"].to_numpy(), gg[dataset_key].to_numpy(), n_boot=500)
            row.update(mae=est, mae_lo=lo, mae_hi=hi, bias=float(gg["signed_err"].mean()),
                       exact=float(np.mean(gg["abs_err"] == 0)))
        if "covered" in g and g["covered"].notna().any():
            gg = g[g["covered"].notna()]
            cov = gg["covered"].astype(float).to_numpy()
            est, lo, hi = cluster_bootstrap_ci(cov, gg[dataset_key].to_numpy(), n_boot=500)
            per_ds = gg.groupby(dataset_key)["covered"].min() > 0.5
            wl, wh = wilson_interval(int(per_ds.sum()), int(per_ds.size))
            row.update(
                coverage=est, coverage_lo=lo, coverage_hi=hi,
                all_covered=float(per_ds.mean()), all_covered_lo=wl, all_covered_hi=wh,
                width_mean=float(gg["width"].mean()), width_median=float(gg["width"].median()),
                singleton_rate=float(gg["singleton"].mean()),
                confidently_wrong=float(gg["confidently_wrong"].mean()),
            )
            if "covers_idset" in gg:
                row["covers_idset"] = float(gg["covers_idset"].astype(float).mean())
        if "score" in g and g["score"].notna().any():
            sp, au = [], []
            for _, gd in g.groupby(dataset_key):
                s, t = gd["score"].to_numpy(), gd["truth"].to_numpy()
                if np.std(s) > 0 and np.std(t) > 0:
                    sp.append(sstats.spearmanr(s, t).statistic)
                au.append(_auroc(s, (t >= 2).astype(int)))
            row.update(spearman=float(np.nanmean(sp)) if sp else np.nan, auroc_poly=float(np.nanmean(au)))
        for k in ("support_f1", "match_cos_mean"):
            if k in g and g[k].notna().any():
                row[k] = float(g.groupby(dataset_key)[k].first().mean())
        out.append(row)
    return pd.DataFrame(out)
