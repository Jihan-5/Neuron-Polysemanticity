r"""E5 — Trained toy superposition models: recovery under *learned* geometry.

Plan p.4/p.6: train the toy encoder/decoder from scratch (K = 32, d = 8 or
16), log all weights, compute the true loading multiplicity from the learned
matrix after normalisation, and test the estimators on activations of the
frozen hidden layer. Plan p.12 stage "Trained toys — stop or revise if the
ground truth is incorrectly defined."

Regimes: dense inputs (few features represented, often <= d — the square
model can apply) through sparse inputs (superposition: more represented
features than dimensions — the square model is misspecified and the honest
outcome is a detected misfit / wide bounds, not a precise count).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from polyid.estimators import build_estimator
from polyid.experiments.common import get_methods, save_tables, spec_from_cfg
from polyid.metrics import aggregate, score_result
from polyid.utils.io import write_json
from polyid.utils.seeding import SeedBank


def _one(d: int, prob: float, seed: int, cfg: dict, methods: dict, spec) -> tuple[list, dict]:
    from polyid.toy.superposition import ToyConfig, toy_observations, toy_truth, train_toy

    tc = ToyConfig(K=int(cfg.get("K", 32)), d=d, feature_prob=prob, steps=int(cfg.get("steps", 10000)),
                   seed=seed, importance_decay=float(cfg.get("importance_decay", 0.9)))
    trained = train_toy(tc)
    W = trained["W"]
    bank = SeedBank(int(cfg["seed"])).child("e5", d, prob, seed)
    rng = bank.child("envs").rng()
    E = int(cfg.get("n_envs", 6))
    mult = rng.uniform(cfg.get("rate_mult_low", 0.4), cfg.get("rate_mult_high", 1.6), size=(E, tc.K))
    rates = np.clip(prob * mult, 1e-4, 1.0)
    truth = toy_truth(W, rates, spec.tau, spec.v_min, float(cfg.get("r_min", 0.1)))
    obs = toy_observations(W, rates, int(cfg.get("n_per_env", 5000)), bank.child("obs").rng(),
                           noise_sd=float(cfg.get("noise_sd", 0.0)))
    ids = {"d": d, "feature_prob": prob, "seed": seed, "dataset": f"d{d}|p{prob}|s{seed}"}
    rows = []
    info = {**ids, "n_represented": truth["n_represented"], "n_target": truth["n_target"],
            "overcomplete": truth["n_target"] > d, "final_val_loss": trained["final_val_loss"],
            "steps_run": trained["steps_run"], "truth_mean": float(truth["truth"].mean())}
    for name, mcfg in methods.items():
        res = build_estimator(name, mcfg).fit(obs, spec, bank.child(name).rng())
        rows += score_result(res, truth["truth"], spec=spec, A_true=truth["A"], ids=ids)
        for k, v in (res.diagnostics or {}).items():
            if isinstance(v, (int, float, bool, np.integer, np.floating)):
                info[f"{name}__{k}"] = v
    return rows, info


def run(cfg: dict, out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    spec = spec_from_cfg(cfg)
    methods = get_methods(cfg)
    jobs = [(d, p, s) for d in cfg["d"] for p in cfg["feature_probs"] for s in range(int(cfg["seeds"]))]
    out = Parallel(n_jobs=int(cfg.get("n_jobs", -1)), backend="loky")(
        delayed(_one)(int(d), float(p), int(s), cfg, methods, spec) for d, p, s in jobs)
    df = pd.DataFrame([r for rows, _ in out for r in rows])
    info = pd.DataFrame([i for _, i in out])
    df = df.merge(info[["dataset", "overcomplete", "n_target"]], on="dataset", how="left")
    save_tables(out_dir, "e5_rows", df)
    save_tables(out_dir, "e5_models", info)
    agg = aggregate(df, by=["d", "feature_prob", "method"])
    save_tables(out_dir, "e5_summary", agg)
    summary = {"models": info.to_dict(orient="records"), "summary": agg.to_dict(orient="records")}
    write_json(out_dir / "e5_summary.json", summary)
    return summary
