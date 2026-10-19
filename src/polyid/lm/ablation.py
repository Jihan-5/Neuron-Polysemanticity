r"""Causal validation: candidate-neuron ablation versus matched random controls.

Plan p.9 "Causal validation": "Select candidate neurons using the development
split. On held-out examples, ablate or patch them and compare with random
neurons matched on activation magnitude and prevalence. Measure targeted
behavior and unrelated-task degradation. A neuron whose ablation harms
everything is not selectively validated as carrying multiple semantic
features." Syllabus S26 (Zhang & Nanda): report a logit-difference metric
as well as accuracy, and keep the intervention type explicit (here: zero
ablation of MLP post-activations at every position).

Statistics: per-item paired outcomes are stored; effects are summarised as
paired differences with family-clustered intervals (syllabus S04), and the
candidate-vs-control contrast uses the *distribution over control sets*, not
a single control draw.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from polyid.lm.hooks import ablate_neurons
from polyid.lm.templates import ChoiceItem
from polyid.stats import paired_difference

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def neuron_profile(X: np.ndarray, active_thresh: float = 0.0) -> pd.DataFrame:
    """Per-neuron matching covariates computed on development activations."""
    return pd.DataFrame({"neuron": np.arange(X.shape[1]),
                         "log_mean_abs": np.log(np.abs(X).mean(axis=0) + 1e-8),
                         "prevalence": (X > active_thresh).mean(axis=0)})


def select_candidates(scores: np.ndarray, n: int, eligible: np.ndarray | None = None) -> np.ndarray:
    s = np.asarray(scores, dtype=float).copy()
    if eligible is not None:
        s[~eligible] = -np.inf
    return np.argsort(-s)[:n]


def matched_controls(profile: pd.DataFrame, candidates: np.ndarray, n_sets: int, rng: np.random.Generator,
                     pool_size: int = 10) -> list[np.ndarray]:
    """For each candidate draw one non-candidate among its ``pool_size`` nearest neighbours
    in (standardised log-magnitude, prevalence) space; repeated ``n_sets`` times."""
    Z = profile[["log_mean_abs", "prevalence"]].to_numpy()
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-12)
    cand = set(int(c) for c in candidates)
    others = np.array([i for i in range(len(Z)) if i not in cand])
    sets = []
    for _ in range(n_sets):
        used: set = set()
        chosen = []
        for c in candidates:
            d = np.linalg.norm(Z[others] - Z[c], axis=1)
            order = others[np.argsort(d)]
            pool = [o for o in order[: pool_size + len(used)] if o not in used][:pool_size]
            pick = int(rng.choice(pool))
            used.add(pick)
            chosen.append(pick)
        sets.append(np.array(chosen))
    return sets


@torch.no_grad() if torch is not None else (lambda f: f)
def choice_scores(model, tokenizer, items: list[ChoiceItem], batch_size: int = 32, device: str = "cpu") -> np.ndarray:
    """Logit difference (first token of correct minus first token of incorrect) at the last prompt token."""
    out = np.empty(len(items))
    for s in range(0, len(items), batch_size):
        batch = items[s:s + batch_size]
        enc = tokenizer([it.prompt for it in batch], return_tensors="pt", padding=True)
        logits = model(input_ids=enc["input_ids"].to(device), attention_mask=enc["attention_mask"].to(device)).logits
        last = enc["attention_mask"].sum(1) - 1
        for b, it in enumerate(batch):
            row = logits[b, last[b]] if tokenizer.padding_side == "right" else logits[b, -1]
            good = tokenizer(it.correct, add_special_tokens=False)["input_ids"][0]
            bad = tokenizer(it.incorrect, add_special_tokens=False)["input_ids"][0]
            out[s + b] = float(row[good] - row[bad])
    return out


@dataclass
class AblationReport:
    per_item: pd.DataFrame
    summary: pd.DataFrame


def run_ablation(model, tokenizer, layer: int, candidates: np.ndarray, control_sets: list[np.ndarray],
                 tasks: dict[str, list[ChoiceItem]], device: str = "cpu") -> AblationReport:
    rows = []
    for task, items in tasks.items():
        fam = np.array([it.family for it in items])
        base = choice_scores(model, tokenizer, items, device=device)
        with ablate_neurons(model, layer, candidates):
            cand = choice_scores(model, tokenizer, items, device=device)
        ctrl = []
        for ci, cs in enumerate(control_sets):
            with ablate_neurons(model, layer, cs):
                ctrl.append(choice_scores(model, tokenizer, items, device=device))
        ctrl = np.stack(ctrl)  # sets x items
        for i in range(len(items)):
            rows.append({"task": task, "item": i, "family": fam[i], "intact": base[i], "candidate": cand[i],
                         "control_mean": ctrl[:, i].mean(), **{f"control_{k}": ctrl[k, i] for k in range(len(ctrl))}})
    df = pd.DataFrame(rows)
    summ = []
    for task, g in df.groupby("task"):
        acc = lambda x: (x > 0).astype(float)
        intact_acc, cand_acc = acc(g["intact"]), acc(g["candidate"])
        ctrl_cols = [c for c in g.columns if c.startswith("control_") and c != "control_mean"]
        ctrl_acc = np.mean([acc(g[c]).mean() for c in ctrl_cols])
        drop_cand = paired_difference(intact_acc, cand_acc, g["family"].to_numpy())
        drop_ctrl_sets = np.array([intact_acc.mean() - acc(g[c]).mean() for c in ctrl_cols])
        ld_c = paired_difference(g["intact"], g["candidate"], g["family"].to_numpy())
        ld_k = paired_difference(g["intact"], g["control_mean"], g["family"].to_numpy())
        summ.append({
            "task": task, "n_items": len(g), "n_families": g["family"].nunique(),
            "acc_intact": intact_acc.mean(), "acc_candidate_ablation": cand_acc.mean(), "acc_control_ablation": ctrl_acc,
            "acc_drop_candidate": drop_cand["diff"], "acc_drop_candidate_lo": drop_cand["lo"],
            "acc_drop_candidate_hi": drop_cand["hi"],
            "acc_drop_control_mean": drop_ctrl_sets.mean(),
            "acc_drop_control_q05": np.quantile(drop_ctrl_sets, 0.05), "acc_drop_control_q95": np.quantile(drop_ctrl_sets, 0.95),
            "candidate_exceeds_controls_frac": float(np.mean(drop_cand["diff"] > drop_ctrl_sets)),
            "logitdiff_drop_candidate": ld_c["diff"], "logitdiff_drop_candidate_lo": ld_c["lo"],
            "logitdiff_drop_candidate_hi": ld_c["hi"], "logitdiff_drop_control": ld_k["diff"],
        })
    return AblationReport(per_item=df, summary=pd.DataFrame(summ))
