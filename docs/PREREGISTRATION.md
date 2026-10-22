# Preregistration (locked before any test-split result is read)

Changes to this file after a test-split result exists must be appended as dated amendments.
Do not edit the original text.

## 1. Locked analysis choices

| Item | Value | Source |
|---|---|---|
| Estimand | `P_j(τ)` = number of visible unit-norm columns with `|A_jk| ≥ τ` | plan p.3 |
| Primary threshold | `τ = 0.10`; sensitivity 0.05, 0.20 (planted); `τ = 0.30` in E4 (`d=3`) and E5 (toy) | plan p.5; D8, D11 |
| Visibility | `v_min = 0.006` (SD across environments of factor variance) | D1 |
| Non-duplication | admissible `|cos| ≤ 0.98` (AR-grid: 0.95) | M7 |
| Nominal level | `α = 0.05`, split: pairs 25 %, loadings 45 %, visibility 15 %, groups 15 % | S6 |
| GMS constant | `κ_n = sqrt(log min_e n_e)` | S5 |
| Bootstrap | 200 replicates (rows or clusters within environment) | — |
| Abstention | any interval of width ≥ 1 counts as abstention for exact-count claims | D12 |
| Pythia layers | 160M: 2, 5, 8; 410M: 4, 10, 16; ablation layers 5 / 10 | plan p.4; D7 |
| LM splits | 60 / 20 / 20 by document family; candidates on dev; ablation on test | plan p.9 |

## 2. Primary comparisons (one per question; all others are secondary)

1. **Validity (E4, collinear-pair cell, n = 16000):** per-neuron coverage of ISMB vs the naive
   bootstrap and the joint-diag singleton, with cluster-bootstrap CIs. Secondary: the
   confidently-wrong rate.
2. **Informativeness (E1, equal weight over the 24 cells):** ISMB mean width and singleton
   rate where the population target is identified.
3. **Point accuracy (E1):** count MAE of the ISMB point estimate vs joint diagonalisation,
   FastICA, the TopK SAE and dev-calibrated GMM/DPMI-style, reported per cell as well as in
   aggregate.
4. **Frequency shift (E3):** fraction of neurons whose output changes across corpora.
5. **Causal validation (E9):** candidate-ablation accuracy drop on the semantic task minus the
   mean matched-control drop, family-clustered CI; unrelated-task drop reported alongside.

Multiple comparisons: Holm over the secondary method contrasts within each experiment.

## 3. Theory-gate criteria (must hold before `theory_gate_passed: true`)

* E0: the constructive non-identification holds (moment discrepancy < 1e-10), and ISMB width
  converges to the identified-set width with nominal coverage. **Status: met** (see the
  README "Results so far").
* E4 (≥ 100 replicates per cell): ISMB coverage of the true `P_j` ≥ 0.90 in the
  identifiable and collinear cells at the largest `n`, with a confidently-wrong rate ≤ 0.05.
  Failures in the `weak_diversity` and `cooccurrence_perfect` cells are reported, not
  gated (Proposition 2 makes the latter impossible to pass in general).
* Proposition 1 / Lemma 2 / Proposition 2 checked by a statistician, or explicitly downgraded
  to "numerically verified claims" in any manuscript.

## 4. What is *not* allowed

* Choosing `τ`, layers, dictionary sizes or semantic labels on test data.
* Selecting high-activation examples and then claiming population estimates.
* Treating SAE features, or SAE agreement across seeds, as ground truth.
* Using one LLM to name features and the same LLM's agreement as validation.
* Reporting the sensitivity range as a confidence interval.

## Amendments

**2026-10-19 — provenance note (please read).** This file was drafted during development, *after*
the first E4 pilot had been run. That pilot was void because of the threshold bug in
`docs/LAB_NOTEBOOK.md` §7. The file was written *before* the corrected E4 pilot. It is therefore
not a preregistration in the strict sense for E4. Freeze it (commit hash in the manuscript)
before the 500-replicate E4 and the full E1/E2 runs, which have not been run.

**2026-10-19 — theory-gate status after the corrected E4 pilot (100 replicates per cell).** At
`n = 16000`, ISMB coverage of the true `P_j` was 0.99 (identifiable) and 0.96 (collinear pair),
with confidently-wrong rates of 0.010 and 0.037. Both meet the §3 criterion *at pilot scale*.
Statistician review (the third bullet) is outstanding, so `theory_gate_passed` stays `false` in
the LM configs.

**2026-10-20 — theory-gate status after the full E4 run (500 replicates per cell).** At
`n = 16000`, ISMB coverage of the true `P_j` was 0.998 [0.992, 1.000] (identifiable) and 0.988
[0.981, 0.995] (collinear pair), with confidently-wrong rates of 0.002 and 0.011. The E4
criterion of §3 is met at full scale. The statistician review (third bullet) is still
outstanding, so `theory_gate_passed` stays `false`. The full E1/E2 runs used the configs as
committed before they started (commits `0bc3f14` and earlier); no setting was changed after
seeing their results.

**2026-10-22 — theory gate opened by the downgrade route of §3 (third bullet).** The E0 and E4
criteria are met (amendments above). A statistician review is planned but not yet done. §3
allows the alternative, which is taken here: Proposition 1, Lemma 2, Proposition 2 and the ISMB
coverage argument are labelled "numerically verified claims (statistician review pending)" in
`docs/THEORY.md`, the README and any manuscript until the review is complete. On that basis
`theory_gate_passed: true` is set in `configs/lm/e8_pythia160m.yaml` and
`configs/lm/e9_pythia160m.yaml` (the 410M and smoke configs inherit it). A positive review
restores the theorem wording. A negative one keeps the affected claims downgraded, and any
E8/E9 conclusion that depends on them is reported as conditional on them. Nothing in §1–§2
changes. Also recorded: Proposition 1 specialises known second-order blind-source-separation
identifiability results (Matsuoka et al. 1995; Pham & Cardoso 2001; Lahat, Cardoso & Messer
2012), now cited in `docs/THEORY.md` §2. This changes the novelty claim, not any result. No
E8/E9 run had been made when this was decided.

