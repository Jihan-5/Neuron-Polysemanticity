r"""E9 — Causal validation of candidate neurons (plan p.9; spec doc "Real-model ablation protocol").

* candidates: top ``n_candidates`` neurons of the ablation layer by the
  **development-split** ISMB lower bound written by E8 (ties broken by the dev
  point estimate) — selection never touches test items;
* controls: ``n_control_sets`` sets of random neurons matched on development
  activation magnitude and prevalence;
* tasks (held-out template families only): semantic sense decisions and an
  unrelated agreement task; both scored as logit differences and accuracy;
* outputs: per-item paired outcomes, family-clustered intervals, and the
  fraction of control sets whose drop the candidate drop exceeds.

Interpretation boundary (plan p.9): a targeted effect beyond matched controls
supports usefulness of the selected representation; it does not prove each
neuron is individually polysemantic or that its exact count is recovered.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from polyid.experiments.common import save_tables
from polyid.experiments.e8_pythia_estimation import load_lm
from polyid.utils.io import write_json
from polyid.utils.seeding import SeedBank


def run(cfg: dict, out_dir: Path) -> dict:
    from polyid.lm.ablation import matched_controls, neuron_profile, run_ablation, select_candidates
    from polyid.lm.cache import load_observed
    from polyid.lm.templates import agreement_items, semantic_choice_items, split_by_family

    out_dir = Path(out_dir)
    model, tok, info = load_lm(cfg)
    layer = int(cfg["ablation_layer"])
    e8 = Path(cfg["e8_dir"])
    cands_df = pd.read_csv(e8 / "e8_dev_candidate_scores.csv")
    cands_df = cands_df[cands_df.layer == layer]
    d_mlp = int(model.config.intermediate_size)
    score = np.full(d_mlp, -np.inf)
    score[cands_df.neuron.to_numpy()] = cands_df.dev_lower.to_numpy() + 1e-3 * cands_df.dev_point.to_numpy()
    candidates = select_candidates(score, int(cfg.get("n_candidates", 10)), eligible=np.isfinite(score))
    dev = load_observed(Path(cfg.get("cache_dir") or e8 / "cache"), layer, "dev")
    prof = neuron_profile(dev.X)
    bank = SeedBank(int(cfg["seed"])).child("e9")
    controls = matched_controls(prof, candidates, int(cfg.get("n_control_sets", 20)), bank.child("controls").rng())
    sem = split_by_family(semantic_choice_items())["test"]
    agr = split_by_family(agreement_items())["test"]
    cap = cfg.get("max_items_per_task")
    if cap:
        sem, agr = sem[:int(cap)], agr[:int(cap)]
    rep = run_ablation(model, tok, layer, candidates, controls, {"semantic": sem, "agreement": agr},
                       device=cfg.get("device", "cpu"))
    save_tables(out_dir, "e9_per_item", rep.per_item)
    save_tables(out_dir, "e9_summary", rep.summary)
    out = {"model": info, "layer": layer, "candidates": candidates.tolist(),
           "controls": [c.tolist() for c in controls], "summary": rep.summary.to_dict(orient="records"),
           "smoke": bool(cfg.get("smoke", False))}
    write_json(out_dir / "e9_summary.json", out)
    return out
