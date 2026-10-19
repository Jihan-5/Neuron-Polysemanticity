r"""E4 — Statistical validation in the small certified problem (d = 3).

Plan p.8: "Run 500 independent small simulated datasets for a central
nominal-95% interval condition if feasible. Report empirical coverage with its
Monte Carlo uncertainty and mean/median interval width. Also test deliberate
nonidentification. Correct intervals may remain wide as sample size grows;
narrowing to a wrong singleton is failure, not progress."

Truth is drawn *from the AR grid* (grid-restricted statement, plan p.8), with
entries kept at least ``entry_margin`` from ``tau`` so that coverage failures
reflect identification and sampling, not rounding at the threshold.

Interval methods compared on identical datasets:

* ``jointdiag``        point estimate treated as a singleton interval;
* ``boot_percentile``  naive percentile bootstrap;
* ``sensitivity``      multi-start range (no coverage claim);
* ``ismb``             proposed decomposition bounds;
* ``ar_grid``          weak-identification–robust projection (reference).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from polyid.bounds.ar_grid import ARGridBounds
from polyid.estimators import build_estimator
from polyid.estimand import multiplicity
from polyid.experiments.common import get_methods, save_tables, spec_from_cfg
from polyid.generators.planted import PlantedConfig, generate
from polyid.identification import identified_set
from polyid.metrics import aggregate, score_result
from polyid.stats import wilson_interval
from polyid.utils.config import deep_merge
from polyid.utils.io import write_json
from polyid.utils.seeding import SeedBank


def grid_truth(ar: ARGridBounds, spec, rng: np.random.Generator, entry_margin: float, max_tries: int = 10000):
    triples = ar.triples(spec.max_abs_cos)
    for _ in range(max_tries):
        t = triples[rng.integers(len(triples))]
        A = np.stack([ar.dirs[i] for i in t], axis=1)
        absA = np.abs(A)
        if np.all(np.abs(absA - spec.tau) >= entry_margin) and np.linalg.cond(A) < 20:
            P = multiplicity(A, spec.tau)
            if P.min() != P.max():  # want variation across neurons
                return A
    raise RuntimeError("no admissible grid truth found")


def _rep(cond: str, gen: dict, n: int, rep: int, cfg: dict, methods: dict, spec) -> tuple[list, dict]:
    seed = int(cfg["seed"])
    bank = SeedBank(seed).child("e4", cond, n, "test", rep)
    ar_cfg = methods.get("ar_grid", {"n_dirs": 120})
    ar = ARGridBounds(ar_cfg)
    A = grid_truth(ar, spec, bank.child("truth").rng(), float(cfg.get("entry_margin", 0.05)))
    pc = PlantedConfig.from_dict({**gen, "n_per_env": n})
    ds = generate(pc, bank, A=A, v_min=spec.v_min)
    ident = identified_set(ds.oracle, spec)
    rows = []
    info = {"condition": cond, "n": n, "rep": rep, "diversity": ds.oracle.diversity,
            **{f"id_{k}": v for k, v in ident.summary().items()}}
    for name, mcfg in methods.items():
        est = ar if name == "ar_grid" else build_estimator(name, mcfg)
        res = est.fit(ds.observed, spec, bank.child(name).rng())
        if res.kind == "count":  # treat a point estimate as a singleton interval
            res.lower, res.upper, res.kind = res.point.copy(), res.point.copy(), "interval"
        ids = {"condition": cond, "n": n, "rep": rep, "dataset": f"{cond}|{n}#{rep}"}
        rows += score_result(res, ds.oracle.truth_at(spec.tau), spec=spec, A_true=A, visible_true=ds.oracle.visible,
                             idset=(ident.lower, ident.upper), ids=ids)
        if name == "ar_grid":
            info["ar_empty"] = res.diagnostics["empty"]
            info["ar_accepted"] = res.diagnostics["n_accepted"]
    return rows, info


def plot(agg: pd.DataFrame, conds: list[str], out_dir: Path) -> list[str]:
    from polyid.viz import style

    style.apply()
    import matplotlib.pyplot as plt

    methods = sorted(agg.method.unique(), key=lambda m: (m not in style.METHOD_SLOT, style.METHOD_SLOT.get(m, 99)))
    fig, axes = plt.subplots(2, len(conds), figsize=(3.3 * len(conds), 6.2), sharex=True, squeeze=False)
    for c, cond in enumerate(conds):
        for i, m in enumerate(methods):
            g = agg[(agg.condition == cond) & (agg.method == m)].sort_values("n")
            if g.empty:
                continue
            col = style.method_color(m, i)
            axes[0][c].plot(g["n"], g["coverage"], marker="o", color=col, label=m, ls=style.method_ls(m))
            axes[0][c].fill_between(g["n"], g["coverage_lo"], g["coverage_hi"], color=col, alpha=0.12, lw=0)
            axes[1][c].plot(g["n"], g["width_mean"], marker="o", color=col, label=m, ls=style.method_ls(m))
        axes[0][c].axhline(0.95, color=style.INK_2, lw=1, ls=":")
        axes[0][c].set_ylim(0, 1.03)
        axes[0][c].set_title(cond.replace("_", " "), fontsize=10)
        axes[1][c].set_xscale("log")
        axes[1][c].set_xlabel("samples per environment")
    axes[0][0].set_ylabel("coverage of true $P_j$")
    axes[1][0].set_ylabel("mean width")
    axes[0][-1].legend(loc="lower right", fontsize=7.5)
    p = out_dir / "e4_coverage_width_d3.png"
    style.finish(fig, p)
    return [str(p)]


def run(cfg: dict, out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    spec = spec_from_cfg(cfg)
    methods = get_methods(cfg)
    conds = {name: deep_merge(cfg["generator"], over or {}) for name, over in cfg["conditions"].items()}
    jobs = [(c, n, r) for c in conds for n in cfg["ns"] for r in range(int(cfg["reps"]))]
    out = Parallel(n_jobs=int(cfg.get("n_jobs", -1)), backend="loky")(
        delayed(_rep)(c, conds[c], int(n), r, cfg, methods, spec) for c, n, r in jobs)
    df = pd.DataFrame([row for rows, _ in out for row in rows])
    info = pd.DataFrame([i for _, i in out])
    save_tables(out_dir, "e4_rows", df)
    save_tables(out_dir, "e4_datasets", info)
    agg = aggregate(df, by=["condition", "n", "method"])
    # Wilson CI on simultaneous (all-neuron) coverage is in `all_covered_lo/hi`
    save_tables(out_dir, "e4_summary", agg)
    figs = plot(agg, list(conds), out_dir)
    from polyid.viz.identification_map import identification_map

    for m in ("ismb", "boot_percentile", "jointdiag", "ar_grid"):
        if m in set(df["method"]):
            figs.append(str(identification_map(df, m, "condition", "n", out_dir / f"e4_identification_map_{m}.png",
                                               y_order=list(conds), title=f"Identification map — {m}")))
    summary = {"conditions": list(conds), "ns": cfg["ns"], "reps": int(cfg["reps"]), "figures": figs,
               "ar_empty_rate": info.groupby("condition")["ar_empty"].mean().to_dict() if "ar_empty" in info else {}}
    write_json(out_dir / "e4_summary.json", summary)
    return summary
