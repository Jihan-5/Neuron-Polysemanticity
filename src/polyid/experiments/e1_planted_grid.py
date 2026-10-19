r"""E1 — Planted grid: accuracy and uncertainty on the 24 predeclared settings.

Spec doc ("Proposed synthetic generator"): ``d = K = 16``, six environments,
3 sparsity regimes x 2 noise levels x 2 environmental-diversity regimes x 2
sample sizes = 24 configurations; 500 independent datasets per configuration
for the full study (fewer, explicitly labelled pilot repetitions otherwise).
Threshold 0.10 primary with 0.05 / 0.20 sensitivity.

Outputs: per-neuron rows, per-dataset diagnostics, per-condition aggregates,
the equal-weight-across-conditions summary per method, and figures
(count error vs sample size; coverage and width by diversity).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from polyid.experiments.common import expand_grid, get_methods, run_planted_condition, save_tables, spec_from_cfg
from polyid.metrics import aggregate
from polyid.utils.io import write_json


def split_condition(df: pd.DataFrame) -> pd.DataFrame:
    """Turn ``"k1=v1|k2=v2"`` condition ids into columns."""
    parts = df["condition"].str.split("|", expand=True)
    for c in parts.columns:
        kv = parts[c].str.split("=", n=1, expand=True)
        name = kv[0].iloc[0]
        vals = pd.to_numeric(kv[1], errors="coerce")
        df[name] = vals.where(vals.notna(), kv[1])
    return df


def equal_weight_summary(agg: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    cols = [m for m in metrics if m in agg]
    return agg.groupby("method")[cols].mean().reset_index()


def plot(agg: pd.DataFrame, out_dir: Path) -> list[str]:
    from polyid.viz import style

    style.apply()
    import matplotlib.pyplot as plt

    paths = []
    agg = agg.copy()
    count_methods = [m for m in agg["method"].unique() if "@" not in m and agg.loc[agg.method == m, "mae"].notna().any()]
    count_methods = sorted(count_methods, key=lambda m: (m not in style.METHOD_SLOT, m))[:8]
    divs = sorted(agg["diversity"].dropna().unique()) if "diversity" in agg else [None]
    fig, axes = plt.subplots(1, len(divs), figsize=(4.8 * len(divs), 3.6), sharey=True, squeeze=False)
    for ax, dv in zip(axes[0], divs):
        sub = agg if dv is None else agg[agg.diversity == dv]
        for i, m in enumerate(count_methods):
            g = sub[sub.method == m].groupby("n_per_env")["mae"].mean()
            ax.plot(g.index, g.values, marker="o", color=style.method_color(m, i), label=m)
        ax.set_title(f"diversity = {dv}", fontsize=10)
        ax.set_xlabel("samples per environment")
    axes[0][0].set_ylabel("count MAE (features / neuron)")
    axes[0][-1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    p = out_dir / "e1_mae_vs_n.png"
    style.finish(fig, p)
    paths.append(str(p))

    iv = agg[agg["coverage"].notna()] if "coverage" in agg else agg.iloc[:0]
    if len(iv):
        fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
        for i, m in enumerate(sorted(iv.method.unique())):
            g = iv[iv.method == m].groupby("diversity")[["coverage", "width_mean"]].mean()
            x = np.arange(len(g))
            axes[0].plot(x, g["coverage"], marker="o", color=style.method_color(m, i), label=m)
            axes[1].plot(x, g["width_mean"], marker="o", color=style.method_color(m, i), label=m)
            for ax in axes:
                ax.set_xticks(x, g.index)
        axes[0].axhline(0.95, color=style.INK_2, lw=1, ls=":")
        axes[0].set_ylabel("per-neuron coverage")
        axes[1].set_ylabel("mean interval width")
        axes[0].set_ylim(0, 1.03)
        axes[1].legend(loc="upper left")
        p = out_dir / "e1_coverage_width_by_diversity.png"
        style.finish(fig, p)
        paths.append(str(p))
    return paths


def run(cfg: dict, out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    spec = spec_from_cfg(cfg)
    methods = get_methods(cfg)
    conds = expand_grid(cfg["generator"], cfg.get("grid"))
    rows, datasets = [], []
    for cond_id, gen in conds:
        r, d = run_planted_condition("e1", cond_id, gen, methods, spec, seed=int(cfg["seed"]), reps=int(cfg["reps"]),
                                     dev_reps=int(cfg.get("dev_reps", 0)), n_jobs=int(cfg.get("n_jobs", -1)),
                                     taus=cfg.get("taus"))
        rows += r
        datasets += d
        print(f"[e1] {cond_id}: {len(d)} datasets", flush=True)
    df = split_condition(pd.DataFrame(rows))
    save_tables(out_dir, "e1_rows", df)
    save_tables(out_dir, "e1_datasets", pd.DataFrame(datasets))
    agg = split_condition(aggregate(df, by=["condition", "method"]))
    save_tables(out_dir, "e1_by_condition", agg)
    overall = equal_weight_summary(agg, ["mae", "bias", "exact", "coverage", "all_covered", "width_mean",
                                         "singleton_rate", "confidently_wrong", "covers_idset", "spearman",
                                         "auroc_poly", "support_f1"])
    save_tables(out_dir, "e1_equal_weight_summary", overall)
    figs = plot(agg, out_dir)
    summary = {"n_conditions": len(conds), "reps": int(cfg["reps"]), "pilot": bool(cfg.get("pilot", False)),
               "equal_weight_summary": overall.to_dict(orient="records"), "figures": figs}
    write_json(out_dir / "e1_summary.json", summary)
    return summary
