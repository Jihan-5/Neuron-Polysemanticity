r"""E3 — Frequency shift with fixed ``A``: does sampling masquerade as representation?

Plan p.5 *Frequency shift*: "Hold A fixed and change activity rates."
Plan p.10: "Corpus frequency shift — count/ranking stability with fixed A —
whether sampling changes masquerade as representation changes." Syllabus S17:
"construct a case where changing frequency alters a marginal mixture while the
loading structure stays fixed."

Design: per replicate draw one ``A``. Build ``C`` corpora, each a set of
``E`` environments whose activity rates are centred at a different level
(e.g. 0.05 / 0.15 / 0.30). The structural truth ``P_j`` is identical across
corpora by construction. Every method is fitted per corpus; we measure, per
method, the fraction of neurons whose output count changes between corpora,
the Spearman correlation of scores across corpora, and the error per corpus.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

from polyid.estimators import build_estimator
from polyid.experiments.common import get_methods, save_tables, spec_from_cfg
from polyid.generators.planted import PlantedConfig, generate, sample_loading_matrix, sample_rates_with_margin
from polyid.utils.io import write_json
from polyid.utils.seeding import SeedBank


def _rep(rep: int, cfg: dict, methods: dict, spec) -> list[dict]:
    seed = int(cfg["seed"])
    base = PlantedConfig.from_dict(cfg["generator"])
    bank = SeedBank(seed).child("e3", "test", rep)
    A, _ = sample_loading_matrix(base, bank.child("loadings").rng())
    rows = []
    for c, center in enumerate(cfg["rate_centers"]):
        pc = PlantedConfig.from_dict({**cfg["generator"], "rate_center": center})
        rates = sample_rates_with_margin(pc, bank.child("corpus", c, "rates").rng())
        ds = generate(pc, bank.child("corpus", c), A=A, rates=rates, v_min=spec.v_min)
        truth = ds.oracle.truth
        for name, mcfg in methods.items():
            res = build_estimator(name, mcfg).fit(ds.observed, spec, bank.child("corpus", c, name).rng())
            for j in range(len(truth)):
                r = {"rep": rep, "corpus": c, "rate_center": center, "method": name, "neuron": j,
                     "truth": int(truth[j])}
                if res.point is not None:
                    r["point"] = float(res.point[j])
                if res.score is not None:
                    r["score"] = float(res.score[j])
                if res.lower is not None:
                    r["lower"], r["upper"] = float(res.lower[j]), float(res.upper[j])
                    r["covered"] = float(res.lower[j] <= truth[j] <= res.upper[j])
                rows.append(r)
    return rows


def stability(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for m, g in df.groupby("method"):
        changed, rhos, maes = [], [], []
        for rep, gr in g.groupby("rep"):
            piv_p = gr.pivot(index="neuron", columns="corpus", values="point") if "point" in gr and gr["point"].notna().any() else None
            piv_s = gr.pivot(index="neuron", columns="corpus", values="score") if "score" in gr and gr["score"].notna().any() else None
            cols = sorted(gr["corpus"].unique())
            for a, b in itertools.combinations(cols, 2):
                if piv_p is not None:
                    changed.append(float(np.mean(piv_p[a] != piv_p[b])))
                v = piv_s if piv_s is not None else piv_p
                if v is not None and v[a].std() > 0 and v[b].std() > 0:
                    rhos.append(stats.spearmanr(v[a], v[b]).statistic)
            if "point" in gr:
                maes.append(float(np.mean(np.abs(gr["point"] - gr["truth"]))))
        row = {"method": m, "frac_neurons_changed": np.mean(changed) if changed else np.nan,
               "cross_corpus_spearman": np.nanmean(rhos) if rhos else np.nan,
               "mae": np.nanmean(maes) if maes else np.nan}
        if "covered" in g and g["covered"].notna().any():
            row["coverage"] = float(g["covered"].mean())
        out.append(row)
    return pd.DataFrame(out)


def plot(st: pd.DataFrame, out_dir: Path) -> list[str]:
    from polyid.viz import style

    style.apply()
    import matplotlib.pyplot as plt

    st = st.sort_values("frac_neurons_changed")
    fig, ax = plt.subplots(figsize=(6.4, 0.34 * len(st) + 1.4))
    y = np.arange(len(st))
    ax.barh(y, st["frac_neurons_changed"], height=0.55,
            color=[style.method_color(m, i) for i, m in enumerate(st["method"])])
    for yi, v in zip(y, st["frac_neurons_changed"]):
        if np.isfinite(v):
            ax.text(v + 0.01, yi, f"{v:.2f}", va="center", fontsize=8, color=style.INK_2)
    ax.set_yticks(y, st["method"])
    ax.set_xlabel("fraction of neurons whose count changes between corpora (truth fixed)")
    ax.set_title("Frequency shift with fixed loadings", fontsize=10)
    p = out_dir / "e3_frequency_shift_stability.png"
    style.finish(fig, p)
    return [str(p)]


def run(cfg: dict, out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    spec = spec_from_cfg(cfg)
    methods = get_methods(cfg)
    res = Parallel(n_jobs=int(cfg.get("n_jobs", -1)), backend="loky")(
        delayed(_rep)(r, cfg, methods, spec) for r in range(int(cfg["reps"])))
    df = pd.DataFrame([row for rr in res for row in rr])
    save_tables(out_dir, "e3_rows", df)
    st = stability(df)
    save_tables(out_dir, "e3_stability", st)
    figs = plot(st, out_dir)
    summary = {"stability": st.to_dict(orient="records"), "figures": figs, "reps": int(cfg["reps"])}
    write_json(out_dir / "e3_summary.json", summary)
    return summary
