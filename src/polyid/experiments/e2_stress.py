r"""E2 — Assumption stress: honesty when identification fails.

Plan p.10 test matrix rows *Nearly collinear profiles*, *Perfect
co-occurrence*, *Misspecified noise*, plus the hard cases of p.3
(near-duplicate directions, weak diversity, unknown factor count, loadings
near tau) and the overcomplete regime ("evaluate strong correlation, perfect
co-occurrence, and overcompleteness separately as assumption-stress
conditions, not as if the square independent-factor guarantee applied
there").

Each named condition is a delta on the clean base generator. Primary
measures: coverage, confidently-wrong rate, width, abstention (singleton
rate) and — for the population — whether the case is identified, partially
identified, or misspecified (detectably or not).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from polyid.experiments.common import get_methods, named_conditions, run_planted_condition, save_tables, spec_from_cfg
from polyid.metrics import aggregate
from polyid.utils.io import write_json


def plot(agg: pd.DataFrame, ds: pd.DataFrame, order: list[str], out_dir: Path) -> list[str]:
    from polyid.viz import style

    style.apply()
    import matplotlib.pyplot as plt

    paths = []
    iv = agg[agg["coverage"].notna()] if "coverage" in agg else agg.iloc[:0]
    if len(iv):
        methods = sorted(iv.method.unique(), key=lambda m: (m not in style.METHOD_SLOT, m))
        fig, axes = plt.subplots(1, 3, figsize=(13, 0.32 * len(order) + 1.6), sharey=True)
        y = np.arange(len(order))
        for i, m in enumerate(methods):
            g = iv[iv.method == m].set_index("condition").reindex(order)
            off = (i - (len(methods) - 1) / 2) * 0.16
            for ax, col in zip(axes, ["coverage", "confidently_wrong", "width_mean"]):
                ax.plot(g[col], y + off, "o", color=style.method_color(m, i), label=m, ms=5)
        axes[0].axvline(0.95, color=style.INK_2, lw=1, ls=":")
        axes[0].set_yticks(y, order)
        axes[0].invert_yaxis()
        axes[0].set_xlabel("per-neuron coverage")
        axes[1].set_xlabel("confidently-wrong rate")
        axes[2].set_xlabel("mean width")
        axes[2].legend(loc="lower right")
        p = out_dir / "e2_stress_intervals.png"
        style.finish(fig, p)
        paths.append(str(p))
    return paths


def run(cfg: dict, out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    spec = spec_from_cfg(cfg)
    methods = get_methods(cfg)
    conds = named_conditions(cfg["generator"], cfg["conditions"])
    rows, datasets = [], []
    for cond_id, gen in conds:
        r, d = run_planted_condition("e2", cond_id, gen, methods, spec, seed=int(cfg["seed"]), reps=int(cfg["reps"]),
                                     dev_reps=int(cfg.get("dev_reps", 0)), n_jobs=int(cfg.get("n_jobs", -1)))
        rows += r
        datasets += d
        print(f"[e2] {cond_id}: {len(d)} datasets", flush=True)
    df = pd.DataFrame(rows)
    dsdf = pd.DataFrame(datasets)
    save_tables(out_dir, "e2_rows", df)
    save_tables(out_dir, "e2_datasets", dsdf)
    agg = aggregate(df, by=["condition", "method"])
    save_tables(out_dir, "e2_by_condition", agg)
    pop = (dsdf.groupby("condition")[[c for c in dsdf.columns if c.startswith("id_")]]
           .agg(lambda s: s.mean() if s.dtype != object else s.iloc[0]).reset_index())
    save_tables(out_dir, "e2_population_identification", pop)
    figs = plot(agg, dsdf, [c for c, _ in conds], out_dir)
    from polyid.viz.identification_map import identification_map

    for m in ("ismb", "boot_percentile", "jointdiag"):
        if m in set(df["method"]) and "covered" in df:
            sub = df[df["covered"].notna()] if m != "jointdiag" else df.assign(
                lower=df["point"], upper=df["point"], width=0.0,
                covered=(df["point"] == df["truth"]).astype(float))
            figs.append(str(identification_map(sub, m, "condition", None, out_dir / f"e2_identification_map_{m}.png",
                                               y_order=[c for c, _ in conds], title=f"Identification map — {m}")))
    summary = {"conditions": [c for c, _ in conds], "reps": int(cfg["reps"]), "figures": figs,
               "population": pop.to_dict(orient="records")}
    write_json(out_dir / "e2_summary.json", summary)
    return summary
