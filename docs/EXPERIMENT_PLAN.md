# Experiment plan (E0–E10)

This document turns the research plan ("How Many Features Does a Neuron Mix? Partial
Identification under Co-occurrence and Distribution Shift") into runnable experiments. Every
experiment has a config under `configs/`, a runner under `src/polyid/experiments/`, and writes
a manifest plus tidy tables and figures under `results/<name>/`.

> **No numbers in this document are predictions.** Expected patterns are conditional, in the
> sense of plan p.11: estimates should improve with data where the target is identifiable,
> and ambiguity should stay visible where it is not. The worked numbers in the
> "Hypothetical Results" document are not targets and are not used anywhere in this
> repository.

## Ordered stages and gates (plan p.12)

| Stage | Experiments | Required output | Stop or revise if |
|---|---|---|---|
| Definition | — (`estimand.py`, `docs/THEORY.md` §1) | Target, equivalence, visibility, threshold rules | Count changes under arbitrary factor duplication |
| Generator | E0, E1 | Planted-data tests and baseline recovery | Easy identifiable cases fail |
| Theory / bounds | E0, E4, E2 | Nontrivial result or informative valid intervals | Only standard ICA is reproduced |
| Trained toys | E5, E6, (E7) | Recovery under learned geometry | Ground truth incorrectly defined |
| Language pilot | E8 | Controlled corpus, independent labels | Estimates track lexical artefacts only |
| Validation | E9 | Intervention controls and uncertainty | Semantic claims exceed evidence |
| Manuscript | all | Reproducible identification map | Important failure regimes omitted |

**Theory gate** (plan p.8, syllabus p.04). Real-model runs of E8/E9 refuse to start unless
the config sets `theory_gate_passed: true`. The gate criteria are fixed in
`docs/PREREGISTRATION.md` §3.

## Common protocol

* **Estimand:** `P_j(τ)` with `τ = 0.10` (sensitivity 0.05, 0.20), `v_min = 0.006`, non-duplication `|cos| ≤ 0.98`.
* **Seeds:** `SeedBank(20261018).child(experiment, condition, split, rep)`; splits `dev` / `test` never overlap.
* **Methods and access** (all receive the same rows; `Access` records are written with each result):

| Method | Env labels | Higher moments | Output | Notes |
|---|---|---|---|---|
| `moments` | no | yes | score | kurtosis; bimodality coefficient in diagnostics |
| `gmm_bic` | no | yes | count proxy (`k*−1`) | 1-D GMM per neuron, BIC over `k ≤ 6` |
| `dpmi_style` | no | yes | score | **approximation** of DPMI (decision D6) |
| `pca`, `factor_analysis` | no | no | count | `k_max` components; FA with varimax |
| `fastica` | no | yes | count | pooled, environment-centred data |
| `dict_learning` | no | yes | count | L1 sparse coding, `k_max` atoms |
| `topk_sae` | no | yes | count | TopK SAE (E5/E6) |
| `jointdiag` | yes | no | count | coroICA-style point estimate with visibility filter |
| `jointdiag_cum` | yes | yes | count | **exploratory repair**: covariance differences + per-environment 4th-order cumulant slices (JADE-style); added after the E1 pilot, lab notebook §9 |
| `boot_percentile` | yes | no | interval | naive bootstrap of the joint-diag count |
| `sensitivity` | yes | no | interval | multi-start range, **no coverage claim** |
| `ismb` (proposed) | yes | no¹ | interval + point | identified-set bounds (THEORY §4, §7) with a global misfit gate (abstains to `[0, k_max]` when the declared model is rejected) |
| `ar_grid` (`d=3`) | yes | no¹ | interval | weak-ID-robust projection (THEORY §7) |

¹ fourth moments are used only to estimate the variance of second-moment statistics.

* **Primary measures:** count MAE and bias (counts); per-neuron coverage of the true `P_j`, coverage of the identified set, mean/median width, singleton rate, **confidently-wrong rate** (intervals); Spearman / AUROC(`P_j ≥ 2`) (scores); support F1 after sign-invariant matching (evaluation only).
* **Uncertainty of the evaluation itself:** cluster bootstrap over datasets for MAE and coverage; Wilson interval for all-neuron (simultaneous) coverage.

---

## E0 — Theory gate: identifiable vs indistinguishable (`configs/planted/e0_theory_gate.yaml`)

**Objective.** The plan's first concrete action. Show one identifiable and one
indistinguishable case before touching any real model.

**Design.** `d = K = 4`, 6 environments, `τ = 0.10`. Two cases with *fixed* `A` and rates,
so only sampling varies: (a) distinct rate profiles; (b) factors 0 and 1 share their rate
profile. Part 1 constructs two explanations of case (b) with identical population
covariances and different `P_j` (an angle sweep over the Schur–Horn extremes). Part 2 sweeps
`n ∈ {500, …, 50 000}` with 40 replicates per cell.

**Expected pattern.** (a) Error and ISMB width → 0. (b) Joint-diag error stays flat; ISMB width
→ the identified-set width (> 0); coverage stays nominal.

**Status: run in full.** Results in `results/e0_theory_gate/` and the README.

## E1 — The 24-setting planted grid (`configs/planted/e1_planted_grid.yaml`)

**Objective.** Accuracy and uncertainty quality on the predeclared main grid.

**Design.** `d = K = 16`, `E = 6`. Activation-rate centre {0.05, 0.15, 0.30} × noise
{0.05, 0.20} × diversity {well-separated, near-collinear} × `n` {2000, 5000} = 24 conditions.
500 test datasets per condition (full study) plus 50 dev datasets for score calibration.
Threshold sensitivity at 0.05 and 0.20 for every loading-based method.

**Measures.** Equal-weight summary across the 24 conditions per method (MAE, coverage, width,
confidently-wrong, support F1). Per-condition tables. Figures: MAE vs `n` by diversity;
coverage and width by diversity.

**Comparison fairness.** Identical rows; disclosed environment access; `k_max = d` for all;
dev-only calibration for scores.

**Stop / revise.** If `jointdiag` and ISMB fail the *well-separated, low-noise, n = 5000*
cells (MAE not near 0, or ISMB coverage clearly below nominal), stop and debug before
anything else.

**Compute.** About 10 s per dataset for the full method list on one core. Full study:
24 × 550 datasets ≈ 37 CPU-hours (≈ 9 h on 4 cores). Pilot (`reps=8 dev_reps=4`) ≈ 10–15 min.

```bash
polyid run configs/planted/e1_planted_grid.yaml --out results/e1_planted_grid            # full
polyid run configs/planted/e1_planted_grid.yaml --out results/e1_pilot reps=8 dev_reps=4 pilot=true
```

## E2 — Assumption stress (`configs/planted/e2_stress.yaml`)

**Objective.** Honesty when assumptions fail. Coverage, abstention, and especially the
confidently-wrong rate.

**Conditions** (each a delta on the clean base):

| Condition | What it tests | Population status (by construction) |
|---|---|---|
| `clean` | reference | identified |
| `weak_diversity` | near-collinear but distinct profiles (Stoye regime) | identified, weakly |
| `collinear_pair`, `collinear_triple` | exact profile collinearity | partially identified (Prop. 1) |
| `near_duplicate_columns` | `|cos(a_0, a_1)| = 0.95` | identified, ill-conditioned |
| `loadings_near_tau` | 6 loadings within ±0.02 of τ | identified; count fragile |
| `invisible_factors` | two factors with constant variance | target excludes them |
| `unknown_K_8` | `K = 8 < d` | identified; 8 invisible pseudo-components |
| `cooccurrence_partial` | switch copula ρ = 0.5 | misspecified (possibly detectable) |
| `cooccurrence_perfect` | identical switches | misspecified, **undetectable** (Prop. 2) |
| `coupled_magnitudes` | magnitude copula ρ = 0.8 | misspecified |
| `noise_correlated_stationary` | correlated but stationary noise | allowed (cancels) |
| `noise_env_scaled`, `noise_env_heteroscedastic` | non-stationary noise | misspecified |
| `noise_student_t` | heavy-tailed noise | allowed (second moments fine; variance estimates stressed) |
| `few_envs_3`, `few_envs_4`, `many_envs_8` | environmental diversity budget | — |
| `overcomplete_24` | `K = 24 > d = 16` | outside the square model |

**Expected pattern.** Valid methods widen or abstain where identification fails. Narrow,
wrong singletons are the failure signature to look for. In `cooccurrence_perfect` every
second-moment method is expected to be able to fail, which is the empirical counterpart of
Proposition 2. Reporting that is the point.

## E3 — Frequency shift with fixed `A` (`configs/planted/e3_frequency_shift.yaml`)

**Objective.** Whether sampling changes masquerade as representation changes (plan p.10;
syllabus S17).

**Design.** One `A` per replicate. Three "corpora" with rate centres 0.05 / 0.15 / 0.30.
The truth is identical by construction.

**Measures.** Fraction of neurons whose output count changes between corpora; cross-corpus
Spearman of scores; MAE per corpus; ISMB coverage per corpus.

**Expected pattern.** Mixture-based per-neuron scores (GMM, DPMI-style, moments) depend on
activation frequency and should move. Structural estimators should move only through
sampling noise.

## E4 — Coverage in the certified `d = 3` problem (`configs/planted/e4_coverage_d3.yaml`)

**Objective.** The plan's statistical-validation experiment: 500 independent datasets per
cell, nominal 95 %, deliberate non-identification included.

**Design.** `d = K = 3`, `E = 4`, `τ = 0.30` (D8). The truth is drawn from the AR grid (120
hemisphere directions; admissible triples with `|cos| ≤ 0.95`, `|det| ≥ 0.05`; entries ≥ 0.05
from τ). Conditions: identifiable, weak diversity, collinear pair, perfect co-occurrence.
`n ∈ {1000, 4000, 16000}`.

**Methods.** jointdiag (as a singleton interval), naive bootstrap, sensitivity range, ISMB,
AR-grid.

**Measures.** Coverage with cluster-bootstrap CI; all-neuron coverage with Wilson CI; width;
confidently-wrong rate; AR-set emptiness rate (a misspecification signal).

**Expected pattern.** In the collinear cell, point and naive intervals narrow onto wrong
singletons in some replicates, while ISMB and AR-grid do not. In the co-occurrence cell all
methods are exposed (Prop. 2).

## E5 — Trained toy superposition (`configs/toy/e5_toy_superposition.yaml`)

**Design.** `K = 32` features, `d ∈ {8, 16}`, feature probability {0.3, 0.1, 0.03}
(dense → sparse), 5 seeds, Adam 1e-3, batch 512, 10k steps, validation every 250 steps with a
separate generator. Truth from the learned `W` (represented ∧ visible; THEORY, `toy/superposition.py`).

**Measures.** Represented features vs `d`. Count error and coverage when `n_target ≤ d`.
Misfit diagnostics, abstention and SAE behaviour when overcomplete.

**Stop / revise.** If the ground-truth definition gives absurd counts (e.g. every neuron
loading on every input feature at the chosen τ), revise τ or `r_min` *on dev seeds* and
record the change.

## E6 — TopK SAE comparison (`configs/toy/e6_sae_comparison.yaml`)

**Design.** Planted data with known truth. Dictionary {2d, 4d} × `k` {1, 2, 4} × 3 seeds.

**Measures.** Validation FVE, dead fraction, mean max-cosine across seeds, count MAE and
bias, and the correlation between reconstruction quality and count error ("low
reconstruction loss does not validate multiplicity").

## E7 — Controlled transformer bridge (optional; `configs/toy/e7_modular_transformer.yaml`)

Two-layer decoder, width 128, 4 heads, MLP 512, addition mod 97, 80/20 split, AdamW 3e-4,
batch 256, ≤ 20k steps, validation every 250. Record task accuracy and checkpoint-to-checkpoint
CKA. The frozen model is probed with operand-sum-shifted environments on a predeclared random
32-neuron subset. There is no ground truth, so the outputs are intervals, diagnostics and
stability across two disjoint environment draws. GPU recommended.

## E8 — Pythia estimation (`configs/lm/e8_pythia160m.yaml`, `e8_pythia410m.yaml`)

**Design.** Frozen `EleutherAI/pythia-160m` and `-410m` at revision `step143000`, fp32. MLP
post-activations (input of `dense_4h_to_h`) at blocks 2/5/8 (160M) and 4/10/16 (410M). Six
declared environments (`lm/corpus.py`). Generic tokens: ≤ 8 per document at positions ≥ 16,
seeded by document id. Controlled items: final subword of the target word. Splits are
60/20/20 by document family.

**Analyses per layer (fit split).** Per-neuron score baselines on a predeclared neuron sample,
pooled and per topic environment (frequency-shift stability). ISMB and joint diagonalisation
on predeclared random 32-neuron blocks, with block diagnostics (off-diagonal ratio, dependence
flags, invisible counts). **Dev split:** ISMB lower bounds as candidate scores for E9.

**Storage.** 50k contexts/env × 6 envs × 3 layers × 3072 × 2 bytes ≈ 5.5 GB (160M); ≈ 7.4 GB (410M).

**Smoke mode** (`configs/lm/e8_smoke.yaml`) runs the whole pipeline offline on a random tiny
GPT-NeoX. It is plumbing only.

## E9 — Causal validation (`configs/lm/e9_pythia160m.yaml`, `e9_pythia410m.yaml`)

**Design.** 10 candidates = top dev-split ISMB lower bounds at the ablation layer (5 / 10).
20 control sets, each neuron matched to a candidate on dev log-mean-|activation| and
prevalence. Zero ablation at all positions. Tasks: semantic sense decisions and subject–verb
agreement (held-out template families). Metrics: accuracy and logit difference.

**Measures.** Targeted drop vs the control-set distribution; unrelated-task drop;
family-clustered paired intervals.

**Claim boundary.** A targeted effect beyond controls supports the usefulness of the
selected representations. It does not prove that each neuron is individually polysemantic
(plan p.9; spec doc). Single-neuron tests across independent semantic families are needed for
that narrower claim.

**Open item.** The text banks currently give 768 semantic and 384 agreement items in total.
The plan asks for 2,000 held-out items per task, so the banks must be extended (and
human-checked) before E9 is run for real.

## E10 — Adaptive environment selection (optional; `configs/planted/e10_adaptive_envs.yaml`)

**Design.** `d = K = 8`. Three initial environments in which factors 0 and 1 share a profile.
A menu of 12 candidate environments with 1000-sample pilots. Budget: 2 new environments.
Strategies: random, balanced (max-min profile spread), adaptive (largest departure from the
unidentified class's shared direction, i.e. maximal disagreement across the identified set).
**Inference uses fresh data** (sample splitting), never the selection data.

**Measures.** ISMB width and coverage after adding the chosen environments; groups remaining.

---

## Required figures (plan p.10) → where they come from

| Figure | Source |
|---|---|
| Count error vs sample size | `e0_width_vs_n.png`, `e0_amari_vs_n.png`, `e1_mae_vs_n.png` |
| Coverage vs correlation | `e2_stress_intervals.png` (co-occurrence rows), `e4_coverage_width_d3.png` |
| Width vs environmental diversity | `e1_coverage_width_by_diversity.png`, `e2_stress_intervals.png` (`few_envs_*`, `weak_diversity`) |
| Frequency-shift stability | `e3_frequency_shift_stability.png` |
| Targeted intervention effects with controls | `e9_summary.csv` (+ plot once real runs exist) |
| **Identification map** | `polyid.viz.identification_map` over E1/E2/E4 dataset tables |

## Compute summary

| Experiment | CPU (this repo's defaults) | GPU |
|---|---|---|
| E0 | ≈ 8 min on 4 cores (run) | — |
| E1 full | ≈ 9 h on 4 cores | — |
| E2 full | ≈ 6 h on 4 cores | — |
| E3 | ≈ 1 h | — |
| E4 full (500 reps) | ≈ 4 h on 4 cores | — |
| E5 | ≈ 1–2 h | optional |
| E6 | ≈ 1 h | optional |
| E7 | days on CPU | ≈ 1 h |
| E8 | infeasible on CPU at full size | ≈ 2–4 h per model incl. caching |
| E9 | slow on CPU | ≈ 1 h per model |
