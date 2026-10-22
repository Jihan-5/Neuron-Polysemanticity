r"""E9 — Causal validation of candidate neurons (plan p.9; spec doc "Real-model ablation protocol").

* candidates: top ``n_candidates`` neurons of the ablation layer by the
  **development-split** ISMB lower bound written by E8 (ties broken by the dev
  point estimate) — selection never touches test items;
* controls: ``n_control_sets`` sets of random neurons matched on development
  activation magnitude and prevalence;
* tasks: a semantic sense decision and an unrelated agreement task, both
  scored as the log-probability difference between the correct and incorrect
  continuation (all continuation tokens) and as accuracy. Rule L7 forbids
  LLM-written items, so reported runs take items from human-validated sources:
  the agreement task from BLiMP (``agreement_source: blimp``, ``lm/blimp.py``)
  and the semantic task from SemCor swap pairs (``semantic_source: semcor``,
  ``lm/semcor.py``), scored per pair as matched-minus-swapped log-probability.
  Before anything is ablated, the intact model must beat ``semantic_min_dev_acc``
  on the SemCor dev lemmas (disjoint from the test lemmas), or the run stops:
  ablation cannot lower a task the model does not solve (why WiC was dropped).
  The template banks in ``lm/templates.py`` were drafted during development and
  are allowed only in smoke runs;
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
    from polyid.lm.ablation import choice_scores, pair_sum
    from polyid.lm.blimp import load_blimp
    from polyid.lm.semcor import semcor_choice_items
    from polyid.lm.templates import agreement_items, semantic_choice_items, split_by_family

    out_dir = Path(out_dir)
    sem_src, agr_src = cfg.get("semantic_source", "none"), cfg.get("agreement_source", "blimp")
    if not cfg.get("smoke") and "templates" in (sem_src, agr_src):
        raise RuntimeError("E9 on a real model needs human-validated items (rule L7): the template banks in "
                           "lm/templates.py were drafted during development and are allowed only in smoke runs.")
    model, tok, info = load_lm(cfg)
    device = cfg.get("device", "cpu")
    precheck = None
    if sem_src == "semcor":  # fail fast, before any ablation: is the task solved at all?
        dev_items = semcor_choice_items("dev")
        dev_scores = pair_sum(choice_scores(model, tok, dev_items, device=device),
                              np.array([it.family for it in dev_items]))[1]
        thr = float(cfg.get("semantic_min_dev_acc", 0.65))
        precheck = {"dev_pairs": int(len(dev_scores)), "dev_accuracy": float(np.mean(dev_scores > 0)),
                    "threshold": thr, "passed": bool(np.mean(dev_scores > 0) >= thr)}
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        write_json(out_dir / "e9_semantic_precheck.json", precheck)
        if not precheck["passed"]:
            raise RuntimeError(f"SemCor pre-check failed: intact accuracy {precheck['dev_accuracy']:.3f} on "
                               f"{precheck['dev_pairs']} dev pairs is below {thr}. The semantic task cannot show an "
                               "ablation effect; fall back to human-written items (README open item 3).")
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
    tasks = {}
    if sem_src == "templates":
        tasks["semantic"] = split_by_family(semantic_choice_items())["test"]
    elif sem_src == "semcor":
        tasks["semantic"] = semcor_choice_items("test")
    elif sem_src != "none":
        raise ValueError(f"unknown semantic_source {sem_src!r}")
    if agr_src == "blimp":
        # no BLiMP item is used to select neurons (selection uses E8's dev split), so all pairs are test items
        tasks["agreement"] = load_blimp(cfg.get("blimp_paradigms"), cfg.get("blimp_dir"))
    elif agr_src == "templates":
        tasks["agreement"] = split_by_family(agreement_items())["test"]
    else:
        raise ValueError(f"unknown agreement_source {agr_src!r}")
    paired = frozenset({"semantic"}) if sem_src == "semcor" else frozenset()
    cap = cfg.get("max_items_per_task")
    if cap:  # paired tasks keep whole pairs (two consecutive items each)
        tasks = {k: v[:2 * (int(cap) // 2)] if k in paired else v[:int(cap)] for k, v in tasks.items()}
    rep = run_ablation(model, tok, layer, candidates, controls, tasks, device=device, paired_tasks=paired)
    save_tables(out_dir, "e9_per_item", rep.per_item)
    save_tables(out_dir, "e9_summary", rep.summary)
    out = {"model": info, "layer": layer, "candidates": candidates.tolist(),
           "controls": [c.tolist() for c in controls], "summary": rep.summary.to_dict(orient="records"),
           "smoke": bool(cfg.get("smoke", False)), "semantic_source": sem_src, "agreement_source": agr_src,
           "n_items": {k: len(v) for k, v in tasks.items()}, "semantic_precheck": precheck}
    write_json(out_dir / "e9_summary.json", out)
    return out
