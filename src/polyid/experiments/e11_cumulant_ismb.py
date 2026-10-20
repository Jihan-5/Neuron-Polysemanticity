"""E11 — Cumulant-aware ISMB (``ismb_cum``) against second-moment ISMB under every stress condition.

Open item 0 of the README: the cumulant point estimator (``jointdiag_cum``) resolves the weak
identification that makes second-moment ISMB wide at ``d = 16``. This experiment asks whether
the *interval* version keeps ISMB's honesty while narrowing it:

* the 19 E2 stress conditions (collinear profiles, dependence, noise misspecification,
  overcompleteness, few environments, ...), and
* near-Gaussian latents (``latent_gaussianity`` 0.5 / 0.9 / 1.0, with and without a collinear
  pair), the regime where fourth-order information is weak or absent and ``ismb_cum`` must fall
  back to second-moment merging rather than report narrow wrong intervals.

``ismb_cum`` uses a higher-moment information set (``Access.higher_moments``), so it is a
different estimator, not a free improvement of ISMB; both are reported side by side on the same
datasets (same seeds). Primary measures: per-neuron coverage, all-neuron coverage, confidently-
wrong rate and width; paired width differences by condition.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from polyid.experiments.common import (get_methods, named_conditions, run_condition_checkpointed, save_tables,
                                       spec_from_cfg)
from polyid.metrics import aggregate
from polyid.utils.io import write_json


def plot(agg: pd.DataFrame, order: list[str], out_dir: Path) -> list[str]:
    from polyid.viz import style

    style.apply()
    import matplotlib.pyplot as plt

    iv = agg[agg["coverage"].notna()]
    if not len(iv):
        return []
    methods = sorted(iv.method.unique(), key=lambda m: (m not in style.METHOD_SLOT, m))
    fig, axes = plt.subplots(1, 3, figsize=(13, 0.32 * len(order) + 1.6), sharey=True)
    y = np.arange(len(order))
    for i, m in enumerate(methods):
        g = iv[iv.method == m].set_index("condition").reindex(order)
        off = (i - (len(methods) - 1) / 2) * 0.2
        for ax, col in zip(axes, ["coverage", "confidently_wrong", "width_mean"]):
            ax.plot(g[col], y + off, "o", color=style.method_color(m, i), label=m, ms=5)
    axes[0].axvline(0.95, color=style.INK_2, lw=1, ls=":")
    axes[0].set_yticks(y, order)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("per-neuron coverage")
    axes[1].set_xlabel("confidently-wrong rate")
    axes[2].set_xlabel("mean width")
    axes[2].legend(loc="lower right")
    p = out_dir / "e11_ismb_vs_ismb_cum.png"
    style.finish(fig, p)
    return [str(p)]


def paired_widths(df: pd.DataFrame, a: str = "ismb_cum", b: str = "ismb") -> pd.DataFrame:
    """Per condition: mean paired width difference ``a - b`` over the same neurons and datasets."""
    keys = ["condition", "rep", "neuron"]
    w = df[df.method.isin([a, b])].pivot_table(index=keys, columns="method", values="width").dropna()
    w["diff"] = w[a] - w[b]
    g = w.groupby("condition")["diff"]
    return pd.DataFrame({"width_diff": g.mean(), "se": g.std() / np.sqrt(g.count()), "n": g.count()}).reset_index()


def run(cfg: dict, out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    spec = spec_from_cfg(cfg)
    methods = get_methods(cfg)
    conds = named_conditions(cfg["generator"], cfg["conditions"])
    rows, datasets = [], []
    for cond_id, gen in conds:
        r, d = run_condition_checkpointed(out_dir, "e11", cond_id, gen, methods, spec, seed=int(cfg["seed"]),
                                          reps=int(cfg["reps"]), dev_reps=0, n_jobs=int(cfg.get("n_jobs", -1)))
        rows.append(r)
        datasets.append(d)
        print(f"[e11] {cond_id}: {len(d)} datasets", flush=True)
    df = pd.concat(rows, ignore_index=True)
    dsdf = pd.concat(datasets, ignore_index=True)
    save_tables(out_dir, "e11_rows", df)
    save_tables(out_dir, "e11_datasets", dsdf)
    agg = aggregate(df, by=["condition", "method"])
    save_tables(out_dir, "e11_by_condition", agg)
    pw = paired_widths(df) if {"ismb", "ismb_cum"} <= set(df["method"]) else pd.DataFrame()
    save_tables(out_dir, "e11_paired_width", pw)
    figs = plot(agg, [c for c, _ in conds], out_dir)
    summary = {"conditions": [c for c, _ in conds], "reps": int(cfg["reps"]), "figures": figs}
    write_json(out_dir / "e11_summary.json", summary)
    return summary
