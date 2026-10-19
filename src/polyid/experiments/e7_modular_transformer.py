r"""E7 — Optional controlled transformer bridge (addition mod 97).

Only after the toy estimator works (plan p.6). Train the 2-layer model;
record task accuracy and representation stability; freeze it; build
environments by shifting the operand-sum distribution over *held-out* pairs;
run the estimators on a predeclared random subset of MLP neurons (chosen by
seed before any activation is seen — no selection by activation value).

There is no planted truth here ("task-factor labels alone do not reveal the
exact internal dictionary"); the outputs are intervals, diagnostics and the
stability of each method across two disjoint environment sets.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from polyid.estimators import build_estimator
from polyid.experiments.common import get_methods, save_tables, spec_from_cfg
from polyid.utils.io import write_json
from polyid.utils.seeding import SeedBank


def run(cfg: dict, out_dir: Path) -> dict:
    from polyid.toy.modular import ModularConfig, modular_observed, operand_environments, train_modular

    out_dir = Path(out_dir)
    spec = spec_from_cfg(cfg)
    methods = get_methods(cfg)
    mc = ModularConfig(**(cfg.get("model") or {}))
    trained = train_modular(mc)
    save_tables(out_dir, "e7_training_history", trained["history"])
    bank = SeedBank(int(cfg["seed"])).child("e7")
    rng = bank.child("neurons").rng()
    neurons = np.sort(rng.choice(mc.d_mlp, size=int(cfg.get("n_neurons", 32)), replace=False))
    E = int(cfg.get("n_envs", 6))
    env_sets = {}
    for name in ("A", "B"):  # two disjoint draws of environments for stability
        idx = operand_environments(trained["val_pairs"], mc.p, E, int(cfg.get("n_per_env", 3000)),
                                   bank.child("envs", name).rng(), float(cfg.get("concentration", 0.5)))
        env_sets[name] = modular_observed(trained["model"], trained["val_pairs"], idx, int(cfg.get("layer", 1)), neurons)
    rows, diags = [], []
    for set_name, obs in env_sets.items():
        for mname, mcfg in methods.items():
            res = build_estimator(mname, mcfg).fit(obs, spec, bank.child(set_name, mname).rng())
            for j, nid in enumerate(neurons):
                r = {"env_set": set_name, "method": mname, "neuron": int(nid)}
                if res.point is not None:
                    r["point"] = float(res.point[j])
                if res.lower is not None:
                    r["lower"], r["upper"] = float(res.lower[j]), float(res.upper[j])
                if res.score is not None:
                    r["score"] = float(res.score[j])
                rows.append(r)
            diags.append({"env_set": set_name, "method": mname,
                          **{k: v for k, v in (res.diagnostics or {}).items()
                             if isinstance(v, (int, float, bool, np.integer, np.floating))}})
    df = pd.DataFrame(rows)
    save_tables(out_dir, "e7_rows", df)
    save_tables(out_dir, "e7_diagnostics", diags)
    stab = []
    for mname, g in df.groupby("method"):
        col = "point" if "point" in g and g["point"].notna().any() else "score"
        a = g[g.env_set == "A"].set_index("neuron")[col]
        b = g[g.env_set == "B"].set_index("neuron")[col]
        stab.append({"method": mname, "frac_changed_A_vs_B": float(np.mean(a != b))})
    out = {"final_history": trained["history"][-1], "stability": stab, "neurons": neurons.tolist()}
    write_json(out_dir / "e7_summary.json", out)
    return out
