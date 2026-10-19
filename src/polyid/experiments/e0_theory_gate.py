r"""E0 — Theory gate: one identifiable and one indistinguishable case.

Plan p.12, *first concrete action*: "Implement a planted sparse-factor
generator and the known-count scorer, then demonstrate one identifiable and
one indistinguishable case before fitting any real-model activations."
Plan p.8, *theory gate*: "establish one nontrivial identification,
impossibility, or finite-sample bounding result, or a compelling empirical
ambiguity-and-repair phenomenon."

Three parts
-----------
1. **Constructive non-identification.** In the case where two factors share
   their activity-rate profile, build two explanations with *identical*
   population covariances in every environment but different ``P_j`` (the
   extreme rotations given by the Schur–Horn range), and report the maximum
   moment discrepancy (should be ~1e-16).
2. **Sample-size sweep.** For both cases (fixed ``A`` and rates; only
   sampling varies) and ``n`` from 500 to 50k per environment: joint-
   diagonalisation error, naive bootstrap coverage, and ISMB coverage/width.
   Expected pattern (not a predicted number): identifiable → error and width
   fall with ``n``; not identified → ISMB width converges to the identified-
   set width (> 0) while the naive interval narrows onto arbitrary counts.
3. Figures + JSON summary.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from polyid.bounds.ismb import ISMB, ISMBConfig
from polyid.estimand import EstimandSpec, multiplicity, normalize_columns
from polyid.estimators.baselines import BootstrapPercentile, JointDiagCount
from polyid.experiments.common import save_tables, spec_from_cfg
from polyid.generators.planted import PlantedConfig, generate
from polyid.identification import identified_set
from polyid.matching import amari_distance
from polyid.utils.io import write_json
from polyid.utils.seeding import SeedBank


def two_explanations(oracle, group: list[int], tau: float, n_angles: int = 7200) -> dict:
    """Sweep the rotation angle of a 2-factor collinear group and return the extreme explanations."""
    A = oracle.A
    v = oracle.variances  # E x K (absolute variances)
    G = list(group)
    # identical rates and magnitudes => identical absolute variances s_e for the group
    s = v[:, G[0]]
    M = A[:, G]
    thetas = np.linspace(0, np.pi, n_angles, endpoint=False)
    counts = []
    for th in thetas:
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        A2 = A.copy()
        A2[:, G] = normalize_columns(M @ R)[0]
        counts.append(multiplicity(A2, tau, visible=oracle.visible))
    counts = np.array(counts)  # n_angles x d
    ranges = counts.max(axis=0) - counts.min(axis=0)
    j = int(np.argmax(ranges))
    th_lo, th_hi = thetas[np.argmin(counts[:, j])], thetas[np.argmax(counts[:, j])]

    def explanation(th):
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        raw = M @ R
        A2 = A.copy()
        A2[:, G], norms = normalize_columns(raw)
        v2 = v.copy()
        v2[:, G] = s[:, None] * norms[None, :] ** 2
        return A2, v2

    A_lo, v_lo = explanation(th_lo)
    A_hi, v_hi = explanation(th_hi)

    def cov_stack(Ax, vx):
        return np.stack([Ax @ np.diag(ve) @ Ax.T for ve in vx])

    disc = float(np.max(np.abs(cov_stack(A_lo, v_lo) - cov_stack(A_hi, v_hi))))
    disc_true = float(np.max(np.abs(cov_stack(A, v) - cov_stack(A_lo, v_lo))))
    return {
        "neuron": j,
        "P_true": multiplicity(A, tau, visible=oracle.visible).tolist(),
        "P_explanation_min": multiplicity(A_lo, tau, visible=oracle.visible).tolist(),
        "P_explanation_max": multiplicity(A_hi, tau, visible=oracle.visible).tolist(),
        "theta_min": float(th_lo),
        "theta_max": float(th_hi),
        "max_abs_covariance_difference_between_explanations": disc,
        "max_abs_covariance_difference_true_vs_min": disc_true,
        "A_true": A.tolist(),
        "A_explanation_min": A_lo.tolist(),
        "A_explanation_max": A_hi.tolist(),
        "angle_sweep_count_range_per_neuron": ranges.tolist(),
    }


def _rep(case: str, gen: dict, A, rates, n: int, rep: int, seed: int, spec: EstimandSpec, ismb_cfg: dict,
         n_boot: int, idset) -> list[dict]:
    pc = PlantedConfig.from_dict({**gen, "n_per_env": n})
    bank = SeedBank(seed).child("e0", case, n, "test", rep)
    ds = generate(pc, bank, A=A, rates=rates, v_min=spec.v_min)
    truth = ds.oracle.truth_at(spec.tau)
    rows = []
    rng = bank.child("methods").rng()
    jd = JointDiagCount({"n_starts": 6}).fit(ds.observed, spec, rng)
    am = amari_distance(np.linalg.inv(jd.loadings), A)
    for res in (jd,
                BootstrapPercentile({"n_boot": n_boot}).fit(ds.observed, spec, rng),
                ISMB(ISMBConfig(**{**ismb_cfg, "n_boot": n_boot})).fit(ds.observed, spec, rng)):
        for j in range(len(truth)):
            r = {"case": case, "n": n, "rep": rep, "method": res.method, "neuron": j, "truth": int(truth[j]),
                 "point": float(res.point[j]), "abs_err": abs(float(res.point[j]) - truth[j]), "amari_jd": am,
                 "idset_lower": int(idset.lower[j]), "idset_upper": int(idset.upper[j])}
            if res.lower is not None:
                L, U = float(res.lower[j]), float(res.upper[j])
                cov = float(L <= truth[j] <= U)
                r.update(lower=L, upper=U, width=U - L, covered=cov,
                         confidently_wrong=float(L == U and not cov),
                         covers_idset=float(L <= idset.lower[j] and U >= idset.upper[j]))
            rows.append(r)
    return rows


def plot(df: pd.DataFrame, idw: dict, out_dir: Path) -> list[str]:
    from polyid.viz import style

    style.apply()
    import matplotlib.pyplot as plt

    paths = []
    cases = [("identifiable", "Identifiable (distinct rate profiles)"),
             ("not_identified", "Not identified (factors 0, 1 share a rate profile)")]
    # 1) ISMB width vs n with identified-set width
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), sharey=True)
    for ax, (case, title) in zip(axes, cases):
        g = df[(df.case == case) & (df.method == "ismb")].groupby("n")
        w = g["width"].mean()
        ax.plot(w.index, w.values, marker="o", color=style.method_color("ismb"), label="ISMB mean width")
        b = df[(df.case == case) & (df.method == "boot_percentile")].groupby("n")["width"].mean()
        ax.plot(b.index, b.values, marker="o", color=style.method_color("boot_percentile"), label="naive bootstrap width")
        ax.axhline(idw[case], color=style.INK_2, lw=1.2, ls="--")
        ax.text(w.index.max(), idw[case] + 0.03, "identified-set width", ha="right", va="bottom", fontsize=8,
                color=style.INK_2)
        ax.set_xscale("log")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("samples per environment")
    axes[0].set_ylabel("interval width (features)")
    axes[0].legend(loc="upper right")
    p = out_dir / "e0_width_vs_n.png"
    style.finish(fig, p)
    paths.append(str(p))
    # 2) coverage vs n, both methods, both cases
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), sharey=True)
    for ax, (case, title) in zip(axes, cases):
        for m, lab in (("ismb", "ISMB"), ("boot_percentile", "naive bootstrap")):
            g = df[(df.case == case) & (df.method == m)].groupby("n")["covered"].mean()
            ax.plot(g.index, g.values, marker="o", color=style.method_color(m), label=lab)
        ax.axhline(0.95, color=style.INK_2, lw=1.0, ls=":")
        ax.set_xscale("log")
        ax.set_ylim(0, 1.03)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("samples per environment")
    axes[0].set_ylabel("per-neuron coverage of true $P_j$")
    axes[0].legend(loc="lower left")
    p = out_dir / "e0_coverage_vs_n.png"
    style.finish(fig, p)
    paths.append(str(p))
    # 3) joint-diagonalisation error vs n
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    for i, (case, title) in enumerate(cases):
        g = df[(df.method == "jointdiag") & (df.case == case)].groupby("n")["amari_jd"].median()
        ax.plot(g.index, g.values, marker="o", color=style.CATEGORICAL[i], label=title.split(" (")[0])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("samples per environment")
    ax.set_ylabel("Amari distance (median)")
    ax.set_title("Joint-diagonalisation error", fontsize=10)
    ax.legend()
    p = out_dir / "e0_amari_vs_n.png"
    style.finish(fig, p)
    paths.append(str(p))
    return paths


def run(cfg: dict, out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    spec = spec_from_cfg(cfg)
    seed = int(cfg.get("seed", 0))
    gen = dict(cfg["generator"])
    cases = {"identifiable": {}, "not_identified": {"collinear_groups": [[0, 1]]}}
    base_ds = {}
    idsets = {}
    for case, over in cases.items():
        pc = PlantedConfig.from_dict({**gen, **over})
        base_ds[case] = generate(pc, SeedBank(seed).child("e0", case, "design"), v_min=spec.v_min)
        idsets[case] = identified_set(base_ds[case].oracle, spec)
    expl = two_explanations(base_ds["not_identified"].oracle, [0, 1], spec.tau)
    ns = [int(n) for n in cfg.get("ns", [500, 1000, 2000, 5000, 10000, 20000, 50000])]
    reps = int(cfg.get("reps", 40))
    n_boot = int(cfg.get("n_boot", 200))
    jobs = [(case, n, r) for case in cases for n in ns for r in range(reps)]
    results = Parallel(n_jobs=int(cfg.get("n_jobs", -1)), backend="loky")(
        delayed(_rep)(case, {**gen, **cases[case]}, base_ds[case].oracle.A, base_ds[case].oracle.rates, n, r, seed,
                      spec, cfg.get("ismb", {}), n_boot, idsets[case])
        for case, n, r in jobs
    )
    df = pd.DataFrame([row for res in results for row in res])
    save_tables(out_dir, "e0_rows", df)
    agg = (df.groupby(["case", "method", "n"])
           .agg(mae=("abs_err", "mean"), coverage=("covered", "mean"), width=("width", "mean"),
                confidently_wrong=("confidently_wrong", "mean"), covers_idset=("covers_idset", "mean"),
                amari=("amari_jd", "median"))
           .reset_index())
    save_tables(out_dir, "e0_summary", agg)
    idw = {c: float(np.mean(idsets[c].upper - idsets[c].lower)) for c in cases}
    figs = plot(df, idw, out_dir)
    summary = {
        "two_explanations": expl,
        "identified_set": {c: {"lower": idsets[c].lower.tolist(), "upper": idsets[c].upper.tolist(),
                               "groups": idsets[c].groups, "procedure_covers_truth": idsets[c].procedure_covers_truth}
                           for c in cases},
        "identified_set_mean_width": idw,
        "figures": figs,
    }
    write_json(out_dir / "e0_summary.json", summary)
    return summary
