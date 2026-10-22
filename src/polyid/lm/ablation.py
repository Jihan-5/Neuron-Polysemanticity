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

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

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
def continuation_logprobs(model, tokenizer, prompts: list[str], continuations: list[str], batch_size: int = 32,
                          device: str = "cpu") -> np.ndarray:
    """``sum_t log p(continuation token t | prompt, earlier continuation tokens)`` for each pair."""
    seqs = []
    for p, c in zip(prompts, continuations):
        ids_p = list(tokenizer(p)["input_ids"])
        ids_c = list(tokenizer(c, add_special_tokens=False)["input_ids"])
        seqs.append((ids_p + ids_c, len(ids_p), len(ids_c)))
    out = np.empty(len(seqs))
    for s in range(0, len(seqs), batch_size):
        batch = seqs[s:s + batch_size]
        L = max(len(ids) for ids, _, _ in batch)
        ids = torch.zeros(len(batch), L, dtype=torch.long)  # right padding; padded positions are never read
        mask = torch.zeros(len(batch), L, dtype=torch.long)
        for b, (seq, _, _) in enumerate(batch):
            ids[b, :len(seq)] = torch.tensor(seq)
            mask[b, :len(seq)] = 1
        logp = torch.log_softmax(model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits.float(), -1)
        for b, (seq, n_p, n_c) in enumerate(batch):
            pos = torch.arange(n_p - 1, n_p + n_c - 1)
            out[s + b] = float(logp[b, pos, torch.tensor(seq[n_p:n_p + n_c])].sum())
    return out


def choice_scores(model, tokenizer, items: list[ChoiceItem], batch_size: int = 32, device: str = "cpu") -> np.ndarray:
    """Log-probability of the correct continuation minus that of the incorrect one, summed over all
    continuation tokens. For single-token continuations this equals the logit difference at the
    last prompt token (the softmax normaliser cancels); multi-token continuations such as BLiMP's
    `` hasn't`` / `` haven't`` are scored in full instead of by their first token."""
    prompts = [it.prompt for it in items]
    good = continuation_logprobs(model, tokenizer, prompts, [it.correct for it in items], batch_size, device)
    bad = continuation_logprobs(model, tokenizer, prompts, [it.incorrect for it in items], batch_size, device)
    return good - bad


@dataclass
class AblationReport:
    per_item: pd.DataFrame
    summary: pd.DataFrame


def pair_sum(scores: np.ndarray, families: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sum item scores within each family (SemCor swap pairs: the two items of a pair sum to
    matched-minus-swapped log-probability). Returns ``(family_ids, summed)`` in first-seen order."""
    ids, inv = np.unique(families, return_inverse=True)
    order = np.argsort([np.flatnonzero(inv == k)[0] for k in range(len(ids))])
    summed = np.bincount(inv, weights=scores, minlength=len(ids))
    return ids[order], summed[order]


def run_ablation(model, tokenizer, layer: int, candidates: np.ndarray, control_sets: list[np.ndarray],
                 tasks: dict[str, list[ChoiceItem]], device: str = "cpu",
                 paired_tasks: frozenset[str] = frozenset()) -> AblationReport:
    """Score every task intact, with the candidates ablated, and with each control set ablated.
    For ``paired_tasks`` the unit is a family (a SemCor swap pair): item scores are summed within
    it, and intervals cluster on the family prefix before ``#`` (the lemma)."""
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
        cluster = fam
        if task in paired_tasks:
            fam, base = pair_sum(base, np.array([it.family for it in items]))
            _, cand = pair_sum(cand, np.array([it.family for it in items]))
            ctrl = np.stack([pair_sum(c, np.array([it.family for it in items]))[1] for c in ctrl])
            cluster = np.array([f.split("#")[0] for f in fam])
        for i in range(len(fam)):
            rows.append({"task": task, "item": i, "family": cluster[i], "unit": fam[i], "intact": base[i],
                         "candidate": cand[i], "control_mean": ctrl[:, i].mean(),
                         **{f"control_{k}": ctrl[k, i] for k in range(len(ctrl))}})
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
