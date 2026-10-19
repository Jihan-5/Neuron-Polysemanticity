r"""E10 — Adaptive environment selection (optional extension, plan p.8).

"Select the next environment to maximize disagreement between candidate
decompositions. Compare against random and balanced designs. Use independent
inference data or an adaptive-valid uncertainty procedure; a naive
post-selection bootstrap is not automatically valid."

Design per replicate
--------------------
* ``E0`` initial environments in which factors 0 and 1 share an activity
  profile (so they are *not identified* from the initial design);
* a menu of ``M`` candidate environments (independent rate vectors), each
  with a small **pilot** sample;
* a budget of ``B`` environments to add, chosen by one of three strategies
  using *selection data only* (initial-environment data + pilots);
* **inference on fresh data** for the initial and chosen environments
  (sample splitting → the ISMB interval is not a post-selection object).

Strategies
----------
``random``    uniform from the menu.
``balanced``  greedily maximise the minimum pairwise sine between estimated
              component profiles (pilot covariances projected on the current
              unmixing) — a design that spreads profiles without targeting
              any particular ambiguity.
``adaptive``  for each currently merged group ``G`` (from an initial ISMB fit)
              compute, in the group's unmixed coordinates, how far the
              candidate's covariance change departs from the direction
              ``Gamma`` shared by all explanations in the identified set; a
              large departure means the candidate's moments differ across the
              competing explanations, i.e. it *disagrees* most. Falls back to
              ``balanced`` when no group is detected.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from polyid.bounds.ismb import ISMBConfig, fit_ismb
from polyid.data import ObservedData
from polyid.experiments.common import save_tables, spec_from_cfg
from polyid.generators.planted import PlantedConfig, generate, sample_loading_matrix, sample_rates_with_margin
from polyid.moments import env_covariances
from polyid.moments import vech as _vech
from polyid.utils.io import write_json
from polyid.utils.seeding import SeedBank


def _min_sine(P: np.ndarray) -> float:
    n = np.linalg.norm(P, axis=1, keepdims=True)
    Pn = P / np.maximum(n, 1e-12)
    G = np.abs(Pn @ Pn.T)
    np.fill_diagonal(G, 0.0)
    return float(np.sqrt(max(0.0, 1 - G.max() ** 2)))


def choose(strategy: str, budget: int, B: np.ndarray, C_init: np.ndarray, C_menu: np.ndarray, groups: list,
           rng: np.random.Generator) -> list[int]:
    M = C_menu.shape[0]
    if strategy == "random":
        return sorted(rng.choice(M, size=budget, replace=False).tolist())
    var_init = np.einsum("ki,eij,kj->ek", B, C_init, B)
    var_menu = np.einsum("ki,eij,kj->ek", B, C_menu, B)
    chosen: list[int] = []
    for _ in range(budget):
        best, best_score = None, -np.inf
        for c in range(M):
            if c in chosen:
                continue
            if strategy == "adaptive" and groups:
                ref = C_init.mean(axis=0)
                score = 0.0
                for G in groups:
                    BG = B[G]
                    base = np.stack([BG @ (Ce - ref) @ BG.T for Ce in C_init])
                    _, _, Vt = np.linalg.svd(_vech(base), full_matrices=False)
                    gamma = Vt[0]
                    # candidates are scored independently; greedy order = top-`budget` scores
                    delta = _vech(BG @ (C_menu[c] - ref) @ BG.T)
                    score += float(np.linalg.norm(delta - (delta @ gamma) * gamma))
            else:  # balanced (also the adaptive fallback)
                V = np.vstack([var_init, var_menu[chosen + [c]]])
                score = _min_sine((V - V.mean(axis=0)).T)
            if score > best_score:
                best, best_score = c, score
        chosen.append(int(best))
    return sorted(chosen)


def _rep(rep: int, cfg: dict, spec) -> list[dict]:
    seed = int(cfg["seed"])
    gen = dict(cfg["generator"])
    E0, M, budget = int(cfg["initial_envs"]), int(cfg["menu_size"]), int(cfg["budget"])
    n, n_pilot = int(cfg["n_per_env"]), int(cfg["n_pilot"])
    bank = SeedBank(seed).child("e10", "test", rep)
    pc0 = PlantedConfig.from_dict({**gen, "n_envs": E0, "collinear_groups": [[0, 1]]})
    A, _ = sample_loading_matrix(pc0, bank.child("loadings").rng())
    rates0 = sample_rates_with_margin(pc0, bank.child("rates0").rng())
    lo = max(pc0.rate_min, pc0.rate_center * (1 - pc0.rate_spread))
    hi = min(pc0.rate_max, pc0.rate_center * (1 + pc0.rate_spread))
    menu = bank.child("menu").rng().uniform(lo, hi, size=(M, pc0.K))
    # ---- selection data: initial environments + pilots for every menu environment
    sel_cfg = PlantedConfig.from_dict({**gen, "n_envs": E0 + M, "n_per_env": [n] * E0 + [n_pilot] * M,
                                       "min_visibility": 0.0})
    sel = generate(sel_cfg, bank.child("selection"), A=A, rates=np.vstack([rates0, menu]), v_min=spec.v_min)
    X, env = sel.observed.X, sel.observed.env
    C_all, _, _ = env_covariances(X, env, E0 + M)
    init_obs = sel.observed.subset_envs(list(range(E0)))
    fit0 = fit_ismb(init_obs, spec, ISMBConfig(n_boot=int(cfg.get("n_boot_select", 50))), bank.child("fit0").rng())
    rows = []
    for strategy in cfg["strategies"]:
        picks = choose(strategy, budget, fit0.B_hat, C_all[:E0], C_all[E0:], fit0.groups,
                       bank.child("choose", strategy).rng())
        # ---- inference on fresh, independent data
        rates = np.vstack([rates0, menu[picks]])
        inf_cfg = PlantedConfig.from_dict({**gen, "n_envs": E0 + budget, "n_per_env": n, "min_visibility": 0.0})
        ds = generate(inf_cfg, bank.child("inference", strategy), A=A, rates=rates, v_min=spec.v_min)
        f = fit_ismb(ds.observed, spec, ISMBConfig(n_boot=int(cfg.get("n_boot", 150))), bank.child("fit", strategy).rng())
        truth = ds.oracle.truth_at(spec.tau)
        covered = (f.lower <= truth) & (truth <= f.upper)
        rows.append({"rep": rep, "strategy": strategy, "picks": str(picks), "width": float(np.mean(f.upper - f.lower)),
                     "coverage": float(np.mean(covered)), "all_covered": float(np.all(covered)),
                     "n_groups": len(f.groups), "diversity_true": ds.oracle.diversity,
                     "initial_groups": len(fit0.groups)})
    return rows


def plot(df: pd.DataFrame, out_dir: Path) -> list[str]:
    from polyid.viz import style

    style.apply()
    import matplotlib.pyplot as plt

    g = df.groupby("strategy").agg(width=("width", "mean"), width_se=("width", "sem"),
                                   coverage=("coverage", "mean"), groups=("n_groups", "mean")).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2))
    x = np.arange(len(g))
    cols = [style.CATEGORICAL[i] for i in range(len(g))]
    axes[0].bar(x, g["width"], yerr=1.96 * g["width_se"], color=cols, width=0.55, capsize=3)
    axes[0].set_xticks(x, g["strategy"])
    axes[0].set_ylabel("mean ISMB width (fresh data)")
    axes[1].bar(x, g["coverage"], color=cols, width=0.55)
    axes[1].axhline(0.95, color=style.INK_2, lw=1, ls=":")
    axes[1].set_ylim(0, 1.03)
    axes[1].set_xticks(x, g["strategy"])
    axes[1].set_ylabel("per-neuron coverage")
    p = out_dir / "e10_adaptive_envs.png"
    style.finish(fig, p)
    return [str(p)]


def run(cfg: dict, out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    spec = spec_from_cfg(cfg)
    res = Parallel(n_jobs=int(cfg.get("n_jobs", -1)), backend="loky")(
        delayed(_rep)(r, cfg, spec) for r in range(int(cfg["reps"])))
    df = pd.DataFrame([row for rr in res for row in rr])
    save_tables(out_dir, "e10_rows", df)
    summ = df.groupby("strategy")[["width", "coverage", "all_covered", "n_groups"]].mean().reset_index()
    save_tables(out_dir, "e10_summary", summ)
    figs = plot(df, out_dir)
    out = {"summary": summ.to_dict(orient="records"), "figures": figs, "reps": int(cfg["reps"])}
    write_json(out_dir / "e10_summary.json", out)
    return out
