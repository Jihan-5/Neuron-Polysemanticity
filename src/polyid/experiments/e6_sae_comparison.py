r"""E6 — TopK SAE baseline: reconstruction, sparsity, stability across seeds, count error.

Plan p.6 "SAE comparison" and syllabus S12/S24/X3: dictionary sizes ``2d``
and ``4d``, several fixed activity budgets ``k``, >= 3 initialisations; compare
reconstruction quality (fraction of variance explained), sparsity, dead
atoms, dictionary stability (mean max |cos| across seeds) and the count error
of the atom-to-neuron mapping. The point to document (plan p.10 "Global and
local errors"): low reconstruction loss does not validate multiplicity.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from polyid.estimand import multiplicity
from polyid.experiments.common import save_tables, spec_from_cfg
from polyid.generators.planted import PlantedConfig, generate
from polyid.utils.io import write_json
from polyid.utils.seeding import SeedBank


def _dataset(rep: int, cfg: dict, spec) -> list[dict]:
    from polyid.toy.sae import SAEConfig, mean_max_cosine, train_sae

    bank = SeedBank(int(cfg["seed"])).child("e6", "test", rep)
    ds = generate(PlantedConfig.from_dict(cfg["generator"]), bank, v_min=spec.v_min)
    X = np.array(ds.observed.X, copy=True)
    for e in range(ds.observed.n_envs):
        X[ds.observed.env == e] -= X[ds.observed.env == e].mean(0)
    truth = ds.oracle.truth_at(spec.tau)
    rows = []
    for mult, k in itertools.product(cfg["dict_mults"], cfg["ks"]):
        fits = []
        for s in range(int(cfg["sae_seeds"])):
            sc = SAEConfig(dict_mult=float(mult), k=int(k), seed=bank.child("sae", mult, k, s).int_seed(),
                           max_steps=int(cfg.get("max_steps", 6000)))
            fits.append(train_sae(X, sc))
        mmcs = [mean_max_cosine(a["W_dec"], b["W_dec"], a["alive"], b["alive"])
                for a, b in itertools.combinations(fits, 2)]
        for s, f in enumerate(fits):
            P = multiplicity(f["W_dec"], spec.tau, visible=f["alive"])
            rows.append({"rep": rep, "dict_mult": mult, "k": k, "sae_seed": s, "val_fve": f["val_fve"],
                         "dead_frac": f["dead_frac"], "mmcs_across_seeds": float(np.nanmean(mmcs)),
                         "count_mae": float(np.mean(np.abs(P - truth))), "count_bias": float(np.mean(P - truth))})
    return rows


def run(cfg: dict, out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    spec = spec_from_cfg(cfg)
    res = Parallel(n_jobs=int(cfg.get("n_jobs", -1)), backend="loky")(
        delayed(_dataset)(r, cfg, spec) for r in range(int(cfg["reps"])))
    df = pd.DataFrame([row for rr in res for row in rr])
    save_tables(out_dir, "e6_rows", df)
    summ = df.groupby(["dict_mult", "k"])[["val_fve", "dead_frac", "mmcs_across_seeds", "count_mae", "count_bias"]].mean()
    summ = summ.reset_index()
    save_tables(out_dir, "e6_summary", summ)
    corr = float(np.corrcoef(df["val_fve"], df["count_mae"])[0, 1]) if len(df) > 2 else float("nan")
    out = {"summary": summ.to_dict(orient="records"), "corr_fve_vs_count_mae": corr}
    write_json(out_dir / "e6_summary.json", out)
    return out
