"""Regenerate the README "Results so far" section from the CSV/JSON outputs under results/.

Usage:  python scripts/summarize_results.py [--write]

Without --write the markdown is printed. With --write it replaces the block between
<!-- RESULTS:START --> and <!-- RESULTS:END --> in README.md. Every number in that block
therefore comes from a file produced by `polyid run`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"


def pick(full: str, pilot: str, key: str) -> tuple[Path, str]:
    """The full run when it has finished (its ``key`` file exists), otherwise the pilot."""
    if (R / full / key).exists():
        return R / full, "full run"
    return R / pilot, "pilot"


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT))


def cfg_of(p: Path) -> dict:
    m = p / "manifest.json"
    return json.loads(m.read_text())["config"] if m.exists() else {}


def f(x, nd=2):
    try:
        if pd.isna(x):
            return "–"
    except TypeError:
        pass
    if isinstance(x, (bool,)) or type(x).__name__ == "bool_":
        return "yes" if x else "no"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def md_table(df: pd.DataFrame, cols: list[str], headers: list[str] | None = None, nd: int = 2) -> str:
    headers = headers or cols
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    df = df.reset_index(drop=True)
    for i in range(len(df)):  # column-wise access keeps integer columns integer (iterrows upcasts)
        lines.append("| " + " | ".join(f(df[c].iloc[i], nd) for c in cols) + " |")
    return "\n".join(lines)


def section_e0() -> str:
    p = R / "e0_theory_gate"
    if not (p / "e0_summary.csv").exists():
        return ""
    s = json.loads((p / "e0_summary.json").read_text())
    ex = s["two_explanations"]
    df = pd.read_csv(p / "e0_summary.csv")
    rows = []
    for n in sorted(df.n.unique()):
        g = df[df.n == n]
        get = lambda case, m, col: g[(g.case == case) & (g.method == m)][col].iloc[0]
        rows.append({"n": int(n), "id_mae": get("identifiable", "jointdiag", "mae"),
                     "id_w": get("identifiable", "ismb", "width"),
                     "ni_mae": get("not_identified", "jointdiag", "mae"),
                     "ni_w": get("not_identified", "ismb", "width"),
                     "cov": min(get("identifiable", "ismb", "coverage"), get("not_identified", "ismb", "coverage"))})
    t = md_table(pd.DataFrame(rows), ["n", "id_mae", "id_w", "ni_mae", "ni_w", "cov"],
                 ["n / env", "identifiable: joint-diag MAE", "identifiable: ISMB width",
                  "not identified: joint-diag MAE", "not identified: ISMB width", "ISMB coverage (min of cases)"])
    idw = s["identified_set_mean_width"]["not_identified"]
    return f"""### E0 — theory gate *(full run: 40 replicates × 7 sample sizes × 2 cases)*

**Constructive non-identification.** Two explanations of the same 4-neuron model have
identical population covariances in all six environments (max |difference| =
**{ex['max_abs_covariance_difference_between_explanations']:.1e}**) but multiplicities
`{tuple(ex['P_explanation_min'])}` vs `{tuple(ex['P_explanation_max'])}`.

{t}

Identified target → error and width go to 0. Not identified → the point error stays flat and the
ISMB width converges to the identified-set width (**{idw:.2f}**) instead of collapsing onto a
wrong singleton.

<p align="center"><img src="results/e0_theory_gate/e0_width_vs_n.png" width="88%"><br>
<img src="results/e0_theory_gate/e0_amari_vs_n.png" width="44%"></p>
"""


def section_e4() -> str:
    p, lab = pick("e4_full", "e4_coverage_d3_pilot", "e4_summary.csv")
    if not (p / "e4_summary.csv").exists():
        return ""
    df = pd.read_csv(p / "e4_summary.csv")
    nmax = df.n.max()
    g = df[df.n == nmax].copy()
    g["cov"] = g.apply(lambda r: f"{r.coverage:.2f} [{r.coverage_lo:.2f}, {r.coverage_hi:.2f}]", axis=1)
    order = ["jointdiag", "sensitivity", "boot_percentile", "ismb", "ar_grid"]
    g["o"] = g.method.map({m: i for i, m in enumerate(order)})
    g = g.sort_values(["condition", "o"])
    t = md_table(g, ["condition", "method", "cov", "covers_idset", "width_mean", "confidently_wrong"],
                 ["condition", "method", f"coverage of true P_j (n={int(nmax)}) [95% CI]", "covers identified set",
                  "mean width", "confidently wrong"])
    s = json.loads((p / "e4_summary.json").read_text())
    reps = s["reps"]
    empty = ", ".join(f"{k}: {v:.2f}" for k, v in s.get("ar_empty_rate", {}).items())
    planned = "; 500 planned" if lab == "pilot" else ""
    return f"""### E4 — coverage in the certified d = 3 problem *({lab}: {reps} replicates per cell{planned})*

{t}

AR-grid empty-set rate by condition: {empty}. Full tables (all `n`) are in
`{rel(p)}/e4_summary.csv`.

<p align="center"><img src="{rel(p)}/e4_coverage_width_d3.png" width="96%"></p>
<p align="center"><img src="{rel(p)}/e4_identification_map_ismb.png" width="70%"></p>
"""


def section_e1() -> str:
    p, lab = pick("e1_full", "e1_pilot", "e1_equal_weight_summary.csv")
    if not (p / "e1_equal_weight_summary.csv").exists():
        return ""
    df = pd.read_csv(p / "e1_equal_weight_summary.csv")
    df = df[~df.method.str.contains("@")]
    cols = [c for c in ["method", "mae", "bias", "exact", "coverage", "covers_idset", "width_mean",
                        "confidently_wrong", "spearman", "auroc_poly", "support_f1"] if c in df]
    df = df.sort_values("mae", na_position="last")
    s = json.loads((p / "e1_summary.json").read_text())
    planned = "; 500 planned" if lab == "pilot" else ""
    return f"""### E1 — 24-setting planted grid *({lab}: {s['reps']} test datasets per cell{planned})*

Equal weight over the 24 cells (`d = K = 16`, `E = 6`, `τ = 0.10`). Score-only methods enter the
count columns only through development-split calibration (`+devcal`).

{md_table(df, cols)}

<p align="center"><img src="{rel(p)}/e1_mae_vs_n.png" width="96%"></p>
"""


def section_repair() -> str:
    a, b = R / "e1b_cumulant_repair_pilot", R / "e2b_cumulant_stress_pilot"
    if not (a / "e1_equal_weight_summary.csv").exists():
        return ""
    s1 = pd.read_csv(a / "e1_equal_weight_summary.csv").sort_values("mae")
    out = f"""### Repair — adding fourth-order cumulant slices *(pilots, point estimates; exploratory)*

Same 12 E1 cells at `n = 5000` (6 datasets each), equal-weight average:

{md_table(s1, ['method', 'mae', 'bias', 'exact', 'support_f1'], nd=3)}
"""
    if (b / "e2_by_condition.csv").exists():
        df = pd.read_csv(b / "e2_by_condition.csv")
        piv = df.pivot_table(index="condition", columns="method", values="mae", sort=False).reset_index()
        out += f"""
Under the E2 stress conditions (10 datasets each), count MAE:

{md_table(piv, ['condition'] + [c for c in ['jointdiag', 'fastica', 'jointdiag_cum'] if c in piv])}
"""
    return out


def section_e2() -> str:
    p, lab = pick("e2_full", "e2_pilot", "e2_by_condition.csv")
    if not (p / "e2_by_condition.csv").exists():
        return ""
    df = pd.read_csv(p / "e2_by_condition.csv")
    iv = df[df.method.isin(["ismb", "boot_percentile"])]
    piv = iv.pivot(index="condition", columns="method", values=["coverage", "confidently_wrong", "width_mean"])
    piv.columns = [f"{a}:{b}" for a, b in piv.columns]
    piv = piv.reset_index()
    jd = df[df.method == "jointdiag"].set_index("condition")["mae"]
    piv["jointdiag MAE"] = piv["condition"].map(jd)
    cols = ["condition", "jointdiag MAE", "coverage:ismb", "coverage:boot_percentile", "confidently_wrong:ismb",
            "confidently_wrong:boot_percentile", "width_mean:ismb"]
    cols = [c for c in cols if c in piv]
    s = json.loads((p / "e2_summary.json").read_text())
    return f"""### E2 — assumption stress *({lab}: {s['reps']} datasets per condition)*

{md_table(piv, cols)}

<p align="center"><img src="{rel(p)}/e2_stress_intervals.png" width="96%"></p>
"""


def section_e3() -> str:
    p, lab = pick("e3_full", "e3_pilot", "e3_stability.csv")
    if not (p / "e3_stability.csv").exists():
        return ""
    df = pd.read_csv(p / "e3_stability.csv").sort_values("frac_neurons_changed")
    cols = [c for c in ["method", "frac_neurons_changed", "cross_corpus_spearman", "mae", "coverage"] if c in df]
    reps = cfg_of(p).get("reps", "?")
    return f"""### E3 — frequency shift with fixed loadings *({lab}: {reps} replicates)*

Same `A`, three corpora with activity rates centred at 0.05 / 0.15 / 0.30; the truth is
identical across corpora by construction.

{md_table(df, cols)}

<p align="center"><img src="{rel(p)}/e3_frequency_shift_stability.png" width="60%"></p>
"""


def section_e10() -> str:
    p, lab = pick("e10_full", "e10_pilot", "e10_summary.csv")
    if not (p / "e10_summary.csv").exists():
        return ""
    df = pd.read_csv(p / "e10_summary.csv")
    from polyid.stats import paired_difference

    rows = pd.read_csv(p / "e10_rows.csv.gz").pivot(index="rep", columns="strategy", values="width")
    diffs = []
    for b in ("random", "balanced"):
        r = paired_difference(rows["adaptive"].to_numpy(), rows[b].to_numpy())
        diffs.append(f"adaptive − {b}: **{r['diff']:.2f}** [{r['lo']:.2f}, {r['hi']:.2f}]")
    return f"""### E10 — adaptive environment selection *({lab}: {len(rows)} replicates; inference on fresh data)*

{md_table(df, ['strategy', 'width', 'coverage', 'all_covered', 'n_groups'])}

Paired width differences (95 % CI, replicate-level): {'; '.join(diffs)}.
"""


def section_e5_e6() -> str:
    out = ""
    p, lab = pick("e5_full", "e5_pilot", "e5_summary.csv")
    if (p / "e5_models.csv").exists():
        c5 = cfg_of(p)
        models = pd.read_csv(p / "e5_models.csv")
        cols_m = ["n_represented", "n_target", "overcomplete"] + (["ismb__misfit_rejected"] if "ismb__misfit_rejected" in models else [])
        m = models.groupby(["d", "feature_prob"])[cols_m].mean().reset_index()
        m = m.rename(columns={"ismb__misfit_rejected": "ISMB gate fired"})
        summ = pd.read_csv(p / "e5_summary.csv")
        cols = [c for c in ["d", "feature_prob", "method", "mae", "coverage", "width_mean"] if c in summ]
        out += f"""### E5 — trained toy superposition models *({lab}: {c5.get('seeds', '?')} seeds, {c5.get('steps', '?')} steps)*

Represented features per regime (truth from the learned `W`; every model is overcomplete,
which is outside the square model, so the honest outcome is abstention):

{md_table(m, [c for c in ['d', 'feature_prob', 'n_represented', 'n_target', 'overcomplete', 'ISMB gate fired'] if c in m])}

{md_table(summ[cols], cols)}
"""
    p, lab = pick("e6_full", "e6_pilot", "e6_summary.csv")
    if (p / "e6_summary.csv").exists():
        c6 = cfg_of(p)
        s = pd.read_csv(p / "e6_summary.csv")
        corr = json.loads((p / "e6_summary.json").read_text())["corr_fve_vs_count_mae"]
        out += f"""
### E6 — TopK SAE baseline on planted data *({lab}: {c6.get('reps', '?')} datasets, {c6.get('max_steps', '?')} steps)*

{md_table(s, ['dict_mult', 'k', 'val_fve', 'dead_frac', 'mmcs_across_seeds', 'count_mae', 'count_bias'])}

Correlation between reconstruction quality (FVE) and count error: **{corr:.2f}**. Good
reconstruction does not certify multiplicity (plan p.10).
"""
    return out


def section_e11() -> str:
    p, lab = pick("e11_full", "e11_pilot", "e11_by_condition.csv")
    if not (p / "e11_by_condition.csv").exists():
        return ""
    df = pd.read_csv(p / "e11_by_condition.csv", keep_default_na=False, na_values=[""])
    order = list(dict.fromkeys(df.condition))
    iv = df[df.method.isin(["ismb", "ismb_cum"])]
    piv = iv.pivot(index="condition", columns="method", values=["coverage", "all_covered", "confidently_wrong",
                                                                 "width_mean"]).reindex(order)
    piv.columns = [f"{b}:{a}" for a, b in piv.columns]
    piv = piv.reset_index()
    cols = ["condition", "ismb:coverage", "ismb_cum:coverage", "ismb_cum:all_covered", "ismb_cum:confidently_wrong",
            "ismb:width_mean", "ismb_cum:width_mean"]
    reps = cfg_of(p).get("reps", "?")
    return f"""### E11 — cumulant-aware ISMB (`ismb_cum`) vs ISMB *({lab}: {reps} datasets per condition; exploratory, higher-moment information set)*

Same datasets for both methods. The 19 E2 stress conditions plus near-Gaussian latents
(`gaussianity_g`: a share `g` of each latent's variance replaced by a Gaussian, which leaves the truth
unchanged and weakens fourth-order information).

{md_table(piv, [c for c in cols if c in piv], nd=3)}

<p align="center"><img src="{rel(p)}/e11_ismb_vs_ismb_cum.png" width="96%"></p>
"""


def section_e12() -> str:
    a, b = R / "e12_oc_gate_toys", R / "e12_oc_gate_toys_n20000"
    if not (a / "e5_summary.csv").exists():
        return ""
    out = ["### E12 — overcompleteness gate (fourth-order cumulant rank; exploratory)\n",
           "Same 30 trained toy models as E5. `_oc` methods abstain when the rank test finds more "
           "non-Gaussian features than neurons. *gate* = share of models where it fired; *rank LB* = "
           "mean certified lower bound on the number of features (never above the truth: *LB valid*).\n"]
    for p, n in ((a, 5000), (b, 20000)):
        if not (p / "e5_summary.csv").exists():
            continue
        s = pd.read_csv(p / "e5_summary.csv")
        m = pd.read_csv(p / "e5_models.csv")
        piv = s.pivot_table(index=["d", "feature_prob"], columns="method", values="coverage", sort=False)
        g = m.groupby(["d", "feature_prob"])
        piv["gate"] = g["ismb_oc__oc_rejected"].mean()
        piv["rank_lb"] = g["ismb_oc__oc_rank_lower_bound"].mean()
        piv["n_target"] = g["n_target"].mean()
        piv["lb_valid"] = (m["ismb_oc__oc_rank_lower_bound"] <= m["n_target"]).groupby([m.d, m.feature_prob]).mean()
        piv = piv.reset_index()
        cols = ["d", "feature_prob", "ismb", "ismb_oc", "ismb_cum", "ismb_cum_oc", "gate", "rank_lb", "n_target",
                "lb_valid"]
        out.append(f"Coverage, {n:,} rows per environment ({rel(p)}):\n")
        out.append(md_table(piv, [c for c in cols if c in piv],
                            ["d", "feature prob", "ISMB", "ISMB + gate", "ISMB-cum", "ISMB-cum + gate", "gate",
                             "rank LB", "true features", "LB valid"][:len([c for c in cols if c in piv])]) + "\n")
    c = R / "e12_oc_gate_planted"
    e11 = R / "e11_full"
    if (c / "e11_by_condition.csv").exists() and (e11 / "e11_by_condition.csv").exists():
        new = pd.read_csv(c / "e11_by_condition.csv", keep_default_na=False, na_values=[""])
        old = pd.read_csv(e11 / "e11_by_condition.csv", keep_default_na=False, na_values=[""])
        ds = pd.read_csv(c / "e11_datasets.csv")
        both = pd.concat([old[old.method.isin(["ismb", "ismb_cum"])], new])
        piv = both.pivot_table(index="condition", columns="method", values=["coverage", "width_mean"], sort=False)
        piv.columns = [f"{m}:{v}" for v, m in piv.columns]
        piv["gate"] = ds.groupby("condition")["ismb_oc__oc_rejected"].mean()
        piv = piv.reindex(list(dict.fromkeys(old.condition))).reset_index()
        cols = ["condition", "gate", "ismb:coverage", "ismb_oc:coverage", "ismb:width_mean", "ismb_oc:width_mean",
                "ismb_cum:coverage", "ismb_cum_oc:coverage", "ismb_cum:width_mean", "ismb_cum_oc:width_mean"]
        out.append("False alarms on E11's exact datasets (100 per condition; only `overcomplete_24` is "
                   "overcomplete, so every other firing is a needless abstention):\n")
        out.append(md_table(piv, [c for c in cols if c in piv],
                            ["condition", "gate", "ISMB cov", "+gate cov", "ISMB width", "+gate width",
                             "ISMB-cum cov", "+gate cov", "ISMB-cum width", "+gate width"]) + "\n")
    return "\n".join(out)


def section_e13() -> str:
    p = R / "e13_ismb_cum_v2"
    if not (p / "e11_by_condition.csv").exists():
        return ""
    df = pd.read_csv(p / "e11_by_condition.csv", keep_default_na=False, na_values=[""])
    piv = df.pivot_table(index="condition", columns="method",
                         values=["coverage", "all_covered", "confidently_wrong", "width_mean"], sort=False)
    piv.columns = [f"{m}:{v}" for v, m in piv.columns]
    piv = piv.reindex(list(dict.fromkeys(df.condition))).reset_index()
    rows = pd.read_csv(p / "e11_rows.csv.gz")
    iv = rows[rows.method.isin(["ismb_cum", "ismb_cum_v2"])]
    miss = iv.assign(m=(iv.truth < iv.lower) | (iv.truth > iv.upper)).groupby("method")["m"].sum()
    cols = ["condition", "ismb_cum:coverage", "ismb_cum_v2:coverage", "ismb_cum:all_covered", "ismb_cum_v2:all_covered",
            "ismb_cum:confidently_wrong", "ismb_cum_v2:confidently_wrong", "ismb_cum:width_mean",
            "ismb_cum_v2:width_mean"]
    return f"""### E13 — `ismb_cum_v2` on fresh datasets *(full run: {cfg_of(p).get("reps", "?")} datasets per condition, new seed; exploratory)*

v2 adds the overcompleteness gate, stability widening and dependence tests on ambiguous pairs
(lab notebook §16–§17). Those fixes were designed on E11's failing datasets, so this run uses
new ones. Missed neurons in total: ISMB-cum {int(miss.get("ismb_cum", 0))}, v2 {int(miss.get("ismb_cum_v2", 0))}.

{md_table(piv, [c for c in cols if c in piv], ["condition", "cum cov", "v2 cov", "cum all", "v2 all", "cum wrong", "v2 wrong", "cum width", "v2 width"], nd=3)}
"""


def build() -> str:
    parts = [section_e0(), section_e4(), section_e1(), section_repair(), section_e2(), section_e11(), section_e13(),
             section_e3(), section_e10(), section_e5_e6(), section_e12()]
    return "\n".join(p for p in parts if p)


def main() -> int:
    md = build()
    if "--write" in sys.argv:
        readme = ROOT / "README.md"
        text = readme.read_text()
        a, b = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
        pre, rest = text.split(a)
        _, post = rest.split(b)
        readme.write_text(pre + a + "\n" + md + "\n" + b + post)
        print("README results section updated")
    else:
        print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
