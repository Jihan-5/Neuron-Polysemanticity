<div align="center">

# How many features does a neuron mix?

### Partial identification of thresholded feature multiplicity under co-occurrence and distribution shift

<p>
<img alt="python" src="https://img.shields.io/badge/python-3.10%2B-2a78d6">
<img alt="tests" src="https://img.shields.io/badge/tests-107%20passing-0ca30c">
<img alt="status" src="https://img.shields.io/badge/status-research%20code%20%C2%B7%20pilots%20only-eda100">
<img alt="theory gate" src="https://img.shields.io/badge/theory%20gate-E0%20met%20%C2%B7%20E4%20pilot-6250d6">
</p>

*Research B / statistical interpretability. Built from the research plan, the technical
specification and the core reading syllabus (18 Oct 2026).*

</div>

> [!IMPORTANT]
> **Every number in this README was produced by the code in this repository**, and each one is
> labelled *full run* or *pilot*. The illustrative numbers in the "Hypothetical Results"
> specification are not reproduced or targeted anywhere. Language-model experiments (E8/E9)
> are implemented and pass offline smoke tests on a random tiny GPT-NeoX. They have **not**
> been run on Pythia: this environment cannot reach Hugging Face, and the plan's theory gate
> comes first.

---

## Contents

1. [The question](#1-the-question)
2. [The estimand](#2-the-estimand)
3. [What is new here](#3-what-is-new-here)
4. [Method: ISMB in one picture](#4-method-ismb-in-one-picture)
5. [Results so far](#5-results-so-far)
6. [Experiments E0–E10](#6-experiments-e0e10)
7. [Quickstart](#7-quickstart)
8. [Repository map](#8-repository-map)
9. [Assumptions and honesty rules](#9-assumptions-and-honesty-rules)
10. [Reading map](#10-reading-map)
11. [Open items and how to help](#11-open-items)

---

## 1. The question

> *When does a polysemanticity measurement deserve confidence, and what observations make it
> more identifiable?* (plan p.2)

A neuron's activation histogram can have several peaks without the neuron mixing several
features. A sparse autoencoder is not ground truth. More data cannot separate features that
always co-occur. This project measures **how many latent features a neuron loads on**,
**reports when the data cannot determine that number**, and separates *"not enough samples"*
from *"not identified even with infinite samples."*

## 2. The estimand

For environment $e$, with a frozen network and only the input distribution varying:

$$h^{(e)} = A\,z^{(e)} + \varepsilon^{(e)}, \qquad
P_j(\tau) = \#\{\,k \text{ visible} : |A_{jk}| \ge \tau\,\},$$

with unit-norm columns, equivalence up to sign and permutation, and a **visibility** rule:
a factor counts only if its variance actually changes across environments. Stationary noise
cancels in centred covariances $D_e = A\,\mathrm{diag}(c_e)\,A^\top$.

$P_j$ is *thresholded structural loading multiplicity*. It is **not** a global feature count,
not a histogram mode count, and not a count of human concepts. Full definitions are in
[`docs/THEORY.md`](docs/THEORY.md) and [`src/polyid/estimand.py`](src/polyid/estimand.py).

## 3. What is new here

Covariance-difference identification (coroICA), mixture polysemanticity scores (DPMI) and
SAE recovery theory (GBA) already exist. The contribution targeted here is **honest inference
about $P_j$ when several decompositions fit**:

| | Result | Status |
|---|---|---|
| **Prop. 1** | Under second moments, a class of factors with collinear variance profiles is identified exactly up to $\{\mathrm{normalize}(A_G\Lambda^{1/2}R): R \in O(g)\}$ | proved ([THEORY §3](docs/THEORY.md#3-proposition-1--the-exact-second-moment-identified-set)), tested |
| **Lemma 2** | The **exact** min/max of a class's contribution to $P_j$ follows from the Schur–Horn theorem as closed-form majorisation checks, so no search over rotations is needed | proved (§4); exact against a 20k-angle sweep |
| **Prop. 2** | **Perfect co-occurrence cannot be detected from second moments.** A shared-switch block of any size is exactly fit by an uncorrelated model with *different* loadings, so every second-moment method can be confidently wrong | proved (§5); regression test |
| **ISMB** | A tractable bounding procedure combining the parts above: rank tests decide which factors are separated, an analytic visibility band, GMS-selected sup-t loading bands, and Schur–Horn group ranges | implemented; asymptotic argument with stated conditions (§7); **valid but not yet informative at `d = 16`** in pilots |
| **AR-grid** | A weak-identification-robust confidence set (Anderson–Rubin logic) for the certified $d=3$ problem; an empty set signals misspecification | implemented; grid-restricted |

## 4. Method: ISMB in one picture

```mermaid
flowchart LR
    X["activations h<br/>E environments"] --> M["per-environment centred<br/>covariances D_e"]
    M --> J["joint diagonalisation<br/>(U-WEDGE + FFDiag steps,<br/>GEVD starts)"]
    J --> V{"visibility band<br/>(analytic, 4th moments)"}
    V -- invisible --> X0["excluded from target"]
    V -- sure / ambiguous --> P{"pairwise rank-1 test<br/>(Cragg–Donald):<br/>separated?"}
    P -- separated --> S["sup-t bootstrap band<br/>on loadings (GMS)"]
    P -- not separated --> G["group: exact Schur–Horn<br/>range of P_j contribution"]
    S --> C["[L_j, U_j]"]
    G --> C
    C --> D["diagnostics: misfit,<br/>dependence flags, restarts"]
```

Merging only ever widens an interval. Every step that can fail non-uniformly (near-collinear
profiles) is measured in E2/E4 rather than assumed away.

## 5. Results so far

### Key findings (read these first; all pilot-scale except E0)

1. **Non-identification is real and constructible (E0, full run).** Two loading matrices with
   identical covariances in every environment (difference 5.6 × 10⁻¹⁷) give different
   multiplicities. When the target is identified, ISMB's width shrinks to 0. When it is not, the
   width converges to the identified-set width with coverage 1.0.
2. **Point estimates are confidently wrong under non-identification, and more data does not
   fix it (E4, `d = 3`).** Joint-diagonalisation singletons are wrong 21–30 % of the time in
   the collinear cell at every `n`, and 34–41 % under weak diversity. ISMB covers the truth
   96–100 % in those cells with ≤ 3.7 % confidently wrong.
3. **Perfect, or exchangeable partial, co-occurrence defeats every second-moment method
   (Proposition 2 and its corollary).** In E4, ISMB coverage there *falls* with `n` (0.97 → 0.83),
   exactly as the theory predicts. This is an impossibility result, not a tuning problem.
4. **At `d = 16`, ISMB is valid where naive uncertainty collapses, but it is not yet
   informative (E1, E2).** Across 19 stress conditions ISMB covered the truth every time with
   zero confidently wrong, widening where identification weakens (2.8 with 8 environments, about
   16 under weak diversity or misspecification). The naive bootstrap's coverage fell to 0.06
   (weak diversity), 0.22 (3 environments) and 0.56 (heteroscedastic noise). ISMB's mean width
   on the 24-cell grid is 13 of 16, which is honest but not yet useful.
5. **Repair: fourth-order cumulant slices.** Adding them to the joint-diagonalisation stack
   (`jointdiag_cum`) gives count MAE **0.030** on the 24-cell grid (exact 97 %, support F1
   0.99), versus 0.63 for FastICA and 2.71 for second moments alone. It is ≈ 0 in 13 of 19
   stress conditions and the most stable under frequency shift (0.3 % of neurons change vs 33 %
   for second moments; E3). It degrades where its assumptions fail: co-occurrence,
   non-stationary or heavy-tailed noise, overcompleteness. It is a point estimator with
   established prior art (JADE; Hyvärinen 2001; Tichavský et al.). The interval version is the
   next step.
6. **Misfit gate.** Added after the first E5 pilot; all pilots were then re-run. It abstains
   whenever the declared model is rejected: 100 % for overcomplete and non-stationary-noise
   designs, 0 % for correctly specified ones, and 10–20 % false abstentions under
   correlated-stationary or Student-t noise.
7. **Open failure: learned overcomplete geometry (E5).** Every trained toy model represents
   more features than dimensions. The gate catches some (d = 8, p ≤ 0.1) but misses the others,
   where ISMB undercovers (0.31 at `d = 8` dense; 0.78–0.94 at `d = 16`). This matters most for
   real language models.
8. **SAE counts are not multiplicity (E6).** TopK-SAE decoder counts over-count by 7–24
   features per neuron, worse with larger dictionaries, even at 0.96 FVE.
9. **Adaptive environment choice helps a little (E10).** Width −0.5 [−0.98, −0.03] vs random,
   at full coverage. Suggestive only.

<!-- RESULTS:START -->
### E0 — theory gate *(full run: 40 replicates × 7 sample sizes × 2 cases)*

**Constructive non-identification.** Two explanations of the same 4-neuron model have
identical population covariances in all six environments (max |difference| =
**5.6e-17**) but multiplicities
`(1, 2, 3, 1)` vs `(1, 3, 4, 1)`.

| n / env | identifiable: joint-diag MAE | identifiable: ISMB width | not identified: joint-diag MAE | not identified: ISMB width | ISMB coverage (min of cases) |
|---|---|---|---|---|---|
| 500 | 0.51 | 3.49 | 0.44 | 2.63 | 1.00 |
| 1000 | 0.24 | 3.01 | 0.41 | 1.62 | 1.00 |
| 2000 | 0.03 | 1.96 | 0.37 | 1.07 | 1.00 |
| 5000 | 0.00 | 0.33 | 0.33 | 0.53 | 1.00 |
| 10000 | 0.00 | 0.12 | 0.34 | 0.59 | 1.00 |
| 20000 | 0.00 | 0.00 | 0.39 | 0.50 | 1.00 |
| 50000 | 0.00 | 0.00 | 0.35 | 0.50 | 1.00 |

Identified target → error and width go to 0. Not identified → the point error stays flat and the
ISMB width converges to the identified-set width (**0.50**) instead of collapsing onto a
wrong singleton.

<p align="center"><img src="results/e0_theory_gate/e0_width_vs_n.png" width="88%"><br>
<img src="results/e0_theory_gate/e0_amari_vs_n.png" width="44%"></p>

### E4 — coverage in the certified d = 3 problem *(pilot: 100 replicates per cell; 500 planned)*

| condition | method | coverage of true P_j (n=16000) [95% CI] | covers identified set | mean width | confidently wrong |
|---|---|---|---|---|---|
| collinear_pair | jointdiag | 0.71 [0.65, 0.78] | 0.05 | 0.00 | 0.29 |
| collinear_pair | sensitivity | 0.71 [0.65, 0.78] | 0.05 | 0.00 | 0.29 |
| collinear_pair | boot_percentile | 1.00 [1.00, 1.00] | 0.93 | 0.94 | 0.00 |
| collinear_pair | ismb | 0.96 [0.93, 0.99] | 0.95 | 1.15 | 0.04 |
| collinear_pair | ar_grid | 1.00 [1.00, 1.00] | 0.21 | 0.28 | 0.00 |
| cooccurrence_perfect | jointdiag | 0.55 [0.49, 0.62] | 0.07 | 0.00 | 0.45 |
| cooccurrence_perfect | sensitivity | 0.55 [0.49, 0.62] | 0.07 | 0.00 | 0.45 |
| cooccurrence_perfect | boot_percentile | 0.88 [0.84, 0.91] | 0.11 | 0.93 | 0.01 |
| cooccurrence_perfect | ismb | 0.83 [0.78, 0.88] | 0.24 | 1.14 | 0.07 |
| cooccurrence_perfect | ar_grid | 0.96 [0.92, 0.98] | 0.92 | 2.75 | 0.04 |
| identifiable | jointdiag | 0.98 [0.96, 1.00] | 0.98 | 0.00 | 0.02 |
| identifiable | sensitivity | 0.98 [0.96, 1.00] | 0.98 | 0.00 | 0.02 |
| identifiable | boot_percentile | 1.00 [1.00, 1.00] | 1.00 | 0.12 | 0.00 |
| identifiable | ismb | 0.99 [0.97, 1.00] | 0.99 | 0.24 | 0.01 |
| identifiable | ar_grid | 1.00 [1.00, 1.00] | 1.00 | 0.22 | 0.00 |
| weak_diversity | jointdiag | 0.66 [0.60, 0.73] | 0.66 | 0.00 | 0.34 |
| weak_diversity | sensitivity | 0.66 [0.60, 0.73] | 0.66 | 0.00 | 0.34 |
| weak_diversity | boot_percentile | 1.00 [1.00, 1.00] | 1.00 | 1.51 | 0.00 |
| weak_diversity | ismb | 0.99 [0.97, 1.00] | 0.99 | 2.04 | 0.00 |
| weak_diversity | ar_grid | 0.99 [0.97, 1.00] | 0.99 | 0.90 | 0.01 |

AR-grid empty-set rate by condition: collinear_pair: 0.03, cooccurrence_perfect: 0.56, identifiable: 0.05, weak_diversity: 0.01. Full tables (all `n`) are in
`results/e4_coverage_d3_pilot/e4_summary.csv`.

<p align="center"><img src="results/e4_coverage_d3_pilot/e4_coverage_width_d3.png" width="96%"></p>
<p align="center"><img src="results/e4_coverage_d3_pilot/e4_identification_map_ismb.png" width="70%"></p>

### E1 — 24-setting planted grid *(pilot: 6 test datasets per cell; 500 planned)*

Equal weight over the 24 cells (`d = K = 16`, `E = 6`, `τ = 0.10`). Score-only methods enter the
count columns only through development-split calibration (`+devcal`).

| method | mae | bias | exact | coverage | covers_idset | width_mean | confidently_wrong | spearman | auroc_poly | support_f1 |
|---|---|---|---|---|---|---|---|---|---|---|
| jointdiag_cum | 0.03 | 0.02 | 0.97 | – | – | – | – | – | – | 0.99 |
| dpmi_style+devcal | 0.60 | 0.01 | 0.49 | – | – | – | – | – | – | – |
| fastica | 0.63 | 0.62 | 0.65 | – | – | – | – | – | – | 0.90 |
| gmm_bic+devcal | 0.76 | 0.03 | 0.37 | – | – | – | – | – | – | – |
| gmm_bic | 0.80 | -0.17 | 0.42 | – | – | – | – | 0.51 | 0.85 | – |
| moments+devcal | 0.90 | 0.05 | 0.32 | – | – | – | – | – | – | – |
| factor_analysis | 1.21 | 0.42 | 0.27 | – | – | – | – | – | – | 0.44 |
| jointdiag | 2.71 | 2.71 | 0.32 | – | – | – | – | – | – | 0.69 |
| boot_percentile | 2.71 | 2.71 | 0.32 | 0.50 | 0.50 | 4.85 | 0.00 | – | – | 0.69 |
| sensitivity | 2.71 | 2.71 | 0.32 | 0.33 | 0.33 | 0.27 | 0.52 | – | – | – |
| ismb | 3.44 | 3.43 | 0.22 | 1.00 | 1.00 | 13.27 | 0.00 | – | – | 0.69 |
| dict_learning | 5.12 | 5.11 | 0.05 | – | – | – | – | – | – | 0.46 |
| pca | 6.09 | 6.09 | 0.03 | – | – | – | – | – | – | 0.38 |
| dpmi_style | – | – | – | – | – | – | – | 0.61 | 0.89 | – |
| moments | – | – | – | – | – | – | – | -0.46 | 0.25 | – |

<p align="center"><img src="results/e1_pilot/e1_mae_vs_n.png" width="96%"></p>

### Repair — adding fourth-order cumulant slices *(pilots, point estimates; exploratory)*

Same 12 E1 cells at `n = 5000` (6 datasets each), equal-weight average:

| method | mae | bias | exact | support_f1 |
|---|---|---|---|---|
| jointdiag_cum | 0.016 | 0.001 | 0.984 | 0.990 |
| fastica | 0.605 | 0.605 | 0.655 | 0.897 |
| jointdiag | 2.217 | 2.217 | 0.396 | 0.735 |

Under the E2 stress conditions (10 datasets each), count MAE:

| condition | jointdiag | fastica | jointdiag_cum |
|---|---|---|---|
| clean | 0.13 | 0.09 | 0.00 |
| collinear_pair | 0.35 | 0.04 | 0.00 |
| collinear_triple | 0.60 | 0.04 | 0.00 |
| cooccurrence_partial | 0.27 | 0.09 | 0.10 |
| cooccurrence_perfect | 0.43 | 0.34 | 0.25 |
| coupled_magnitudes | 0.26 | 0.04 | 0.00 |
| few_envs_3 | 3.17 | 0.01 | 0.00 |
| few_envs_4 | 1.38 | 0.07 | 0.00 |
| invisible_factors | 0.95 | 0.31 | 0.05 |
| loadings_near_tau | 0.23 | 0.04 | 0.06 |
| many_envs_8 | 0.05 | 0.00 | 0.00 |
| near_duplicate_columns | 0.10 | 0.03 | 0.00 |
| noise_correlated_stationary | 0.41 | 0.23 | 0.01 |
| noise_env_heteroscedastic | 1.78 | 0.24 | 0.26 |
| noise_env_scaled | 1.01 | 0.24 | 0.13 |
| noise_student_t | 0.61 | 0.37 | 0.20 |
| overcomplete_24 | 1.68 | 0.86 | 0.73 |
| unknown_K_8 | 0.57 | 5.31 | 0.00 |
| weak_diversity | 6.61 | 0.01 | 0.00 |

### E2 — assumption stress *(pilot: 10 datasets per condition)*

| condition | jointdiag MAE | coverage:ismb | coverage:boot_percentile | confidently_wrong:ismb | confidently_wrong:boot_percentile | width_mean:ismb |
|---|---|---|---|---|---|---|
| clean | 0.13 | 1.00 | 0.99 | 0.00 | 0.00 | 7.08 |
| collinear_pair | 0.35 | 1.00 | 0.99 | 0.00 | 0.00 | 6.05 |
| collinear_triple | 0.60 | 1.00 | 0.92 | 0.00 | 0.00 | 7.97 |
| cooccurrence_partial | 0.27 | 1.00 | 0.97 | 0.00 | 0.00 | 6.77 |
| cooccurrence_perfect | 0.43 | 1.00 | 0.96 | 0.00 | 0.00 | 6.63 |
| coupled_magnitudes | 0.26 | 1.00 | 0.99 | 0.00 | 0.00 | 7.36 |
| few_envs_3 | 3.17 | 1.00 | 0.23 | 0.00 | 0.00 | 15.97 |
| few_envs_4 | 1.38 | 1.00 | 0.71 | 0.00 | 0.00 | 14.10 |
| invisible_factors | 0.95 | 1.00 | 0.80 | 0.00 | 0.00 | 8.78 |
| loadings_near_tau | 0.23 | 1.00 | 0.99 | 0.00 | 0.01 | 6.17 |
| many_envs_8 | 0.05 | 1.00 | 1.00 | 0.00 | 0.00 | 2.79 |
| near_duplicate_columns | 0.10 | 1.00 | 1.00 | 0.00 | 0.00 | 5.58 |
| noise_correlated_stationary | 0.41 | 1.00 | 0.96 | 0.00 | 0.00 | 9.57 |
| noise_env_heteroscedastic | 1.78 | 1.00 | 0.56 | 0.00 | 0.01 | 16.00 |
| noise_env_scaled | 1.01 | 1.00 | 0.77 | 0.00 | 0.00 | 16.00 |
| noise_student_t | 0.61 | 1.00 | 0.95 | 0.00 | 0.00 | 11.76 |
| overcomplete_24 | 1.68 | 1.00 | 0.78 | 0.00 | 0.00 | 16.00 |
| unknown_K_8 | 0.57 | 1.00 | 1.00 | 0.00 | 0.00 | 7.81 |
| weak_diversity | 6.61 | 1.00 | 0.06 | 0.00 | 0.00 | 15.06 |

<p align="center"><img src="results/e2_pilot/e2_stress_intervals.png" width="96%"></p>

### E3 — frequency shift with fixed loadings *(pilot)*

Same `A`, three corpora with activity rates centred at 0.05 / 0.15 / 0.30; the truth is
identical across corpora by construction.

| method | frac_neurons_changed | cross_corpus_spearman | mae | coverage |
|---|---|---|---|---|
| jointdiag_cum | 0.00 | 1.00 | 0.00 | – |
| jointdiag | 0.33 | 0.89 | 0.28 | – |
| ismb | 0.51 | 0.82 | 0.49 | 1.00 |
| fastica | 0.53 | 0.86 | 0.48 | – |
| gmm_bic | 0.69 | 0.30 | 0.83 | – |
| dict_learning | 0.90 | 0.52 | 4.76 | – |
| dpmi_style | – | 0.55 | – | – |
| moments | – | 0.18 | – | – |

<p align="center"><img src="results/e3_pilot/e3_frequency_shift_stability.png" width="60%"></p>

### E10 — adaptive environment selection *(pilot, 30 replicates; inference on fresh data)*

| strategy | width | coverage | all_covered | n_groups |
|---|---|---|---|---|
| adaptive | 2.92 | 1.00 | 1.00 | 0.67 |
| balanced | 3.48 | 1.00 | 1.00 | 0.53 |
| random | 3.51 | 1.00 | 1.00 | 1.00 |

Paired width differences (95 % CI, replicate-level): adaptive − random: **-0.59** [-1.11, -0.07]; adaptive − balanced: **-0.56** [-1.05, -0.07]. Suggestive only: two
comparisons, and each interval barely excludes 0.

### E5 — trained toy superposition models *(pilot: 2 seeds, 4k steps)*

Represented features per regime (truth from the learned `W`; every model is overcomplete,
which is outside the square model, so the honest outcome is abstention):

| d | feature_prob | n_represented | n_target | overcomplete | ISMB gate fired |
|---|---|---|---|---|---|
| 8 | 0.03 | 24.50 | 23.00 | 1.00 | 0.50 |
| 8 | 0.10 | 17.50 | 17.50 | 1.00 | 1.00 |
| 8 | 0.30 | 15.50 | 15.50 | 1.00 | 0.00 |
| 16 | 0.03 | 32.00 | 30.00 | 1.00 | 0.00 |
| 16 | 0.10 | 30.50 | 30.00 | 1.00 | 0.00 |
| 16 | 0.30 | 23.00 | 23.00 | 1.00 | 0.00 |

| d | feature_prob | method | mae | coverage | width_mean |
|---|---|---|---|---|---|
| 8 | 0.03 | fastica | 7.12 | – | – |
| 8 | 0.03 | gmm_bic | 9.06 | – | – |
| 8 | 0.03 | ismb | 7.62 | 0.56 | 19.75 |
| 8 | 0.03 | jointdiag | 7.50 | – | – |
| 8 | 0.03 | topk_sae | 1.81 | – | – |
| 8 | 0.10 | fastica | 4.75 | – | – |
| 8 | 0.10 | gmm_bic | 6.19 | – | – |
| 8 | 0.10 | ismb | 4.62 | 1.00 | 32.00 |
| 8 | 0.10 | jointdiag | 4.62 | – | – |
| 8 | 0.10 | topk_sae | 4.00 | – | – |
| 8 | 0.30 | fastica | 3.50 | – | – |
| 8 | 0.30 | gmm_bic | 7.12 | – | – |
| 8 | 0.30 | ismb | 3.12 | 0.31 | 3.56 |
| 8 | 0.30 | jointdiag | 3.50 | – | – |
| 8 | 0.30 | topk_sae | 6.06 | – | – |
| 16 | 0.03 | fastica | 3.50 | – | – |
| 16 | 0.03 | gmm_bic | 5.44 | – | – |
| 16 | 0.03 | ismb | 3.50 | 0.78 | 8.56 |
| 16 | 0.03 | jointdiag | 3.62 | – | – |
| 16 | 0.03 | topk_sae | 3.97 | – | – |
| 16 | 0.10 | fastica | 3.62 | – | – |
| 16 | 0.10 | gmm_bic | 6.12 | – | – |
| 16 | 0.10 | ismb | 2.06 | 0.94 | 10.19 |
| 16 | 0.10 | jointdiag | 3.56 | – | – |
| 16 | 0.10 | topk_sae | 4.19 | – | – |
| 16 | 0.30 | fastica | 1.84 | – | – |
| 16 | 0.30 | gmm_bic | 5.72 | – | – |
| 16 | 0.30 | ismb | 1.78 | 0.94 | 6.50 |
| 16 | 0.30 | jointdiag | 1.84 | – | – |
| 16 | 0.30 | topk_sae | 5.12 | – | – |

### E6 — TopK SAE baseline on planted data *(pilot)*

| dict_mult | k | val_fve | dead_frac | mmcs_across_seeds | count_mae | count_bias |
|---|---|---|---|---|---|---|
| 2 | 1 | 0.65 | 0.01 | 0.93 | 13.51 | 13.51 |
| 2 | 2 | 0.81 | 0.04 | 0.88 | 7.27 | 7.27 |
| 2 | 4 | 0.96 | 0.11 | 0.86 | 6.86 | 6.86 |
| 4 | 1 | 0.72 | 0.05 | 0.93 | 23.89 | 23.89 |
| 4 | 2 | 0.83 | 0.04 | 0.88 | 18.98 | 18.98 |
| 4 | 4 | 0.96 | 0.16 | 0.82 | 20.48 | 20.48 |

Correlation between reconstruction quality (FVE) and count error: **-0.20**. Good
reconstruction does not certify multiplicity (plan p.10).

<!-- RESULTS:END -->

## 6. Experiments E0–E10

Full designs, expected patterns, stop rules and compute estimates are in
[`docs/EXPERIMENT_PLAN.md`](docs/EXPERIMENT_PLAN.md).

| | Experiment | Plan section | Config | Status |
|---|---|---|---|---|
| E0 | Theory gate: identifiable vs indistinguishable | p.12 first action | `configs/planted/e0_theory_gate.yaml` | ✅ full run |
| E1 | 24-setting planted grid | spec doc | `configs/planted/e1_planted_grid.yaml` | pilot |
| E2 | Assumption stress (19 conditions) | p.3, p.10 | `configs/planted/e2_stress.yaml` | pilot |
| E3 | Frequency shift with fixed $A$ | p.5, p.10 | `configs/planted/e3_frequency_shift.yaml` | pilot |
| E4 | Coverage in the certified $d=3$ problem | p.8 | `configs/planted/e4_coverage_d3.yaml` | pilot (100 reps; 500 planned) |
| E5 | Trained toy superposition models | p.6 | `configs/toy/e5_toy_superposition.yaml` | pilot |
| E6 | TopK SAE comparison | p.6 | `configs/toy/e6_sae_comparison.yaml` | pilot |
| E7 | Modular-addition transformer bridge | p.6 (optional) | `configs/toy/e7_modular_transformer.yaml` | implemented; needs GPU |
| E8 | Pythia-160M / 410M estimation | p.4, p.9 | `configs/lm/e8_pythia*.yaml` | implemented; smoke-tested offline |
| E9 | Causal validation (matched-control ablation) | p.9 | `configs/lm/e9_pythia*.yaml` | implemented; smoke-tested offline |
| E10 | Adaptive environment selection | p.8 (optional) | `configs/planted/e10_adaptive_envs.yaml` | pilot |

## 7. Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"          # numpy/scipy/sklearn + torch + transformers + pytest
pytest -q                         # 107 tests (≈ 6 min on 4 CPU cores)

polyid list                                            # all experiments
polyid run configs/planted/e0_theory_gate.yaml         # theory gate (≈ 8 min, 4 cores)
polyid run configs/planted/e4_coverage_d3.yaml reps=100 --out results/e4_pilot
polyid run configs/planted/e1_planted_grid.yaml --dry-run reps=3   # inspect a resolved config
bash scripts/run_pilots.sh                             # every pilot in this README
```

Every run writes `manifest.json` (config hash, git commit, library versions, seeds), tidy CSVs
and figures. Overrides are dotted keys (`generator.n_per_env=2000`).

**Language models** (needs a GPU and Hugging Face access). First record the theory-gate
decision in the config (`theory_gate_passed: true`; criteria in
[`docs/PREREGISTRATION.md`](docs/PREREGISTRATION.md)), then:

```bash
polyid run configs/lm/e8_pythia160m.yaml --out results/e8_pythia160m
polyid run configs/lm/e9_pythia160m.yaml --out results/e9_pythia160m
```

Use the API directly:

```python
import numpy as np
from polyid import EstimandSpec
from polyid.generators import PlantedConfig, generate
from polyid.bounds.ismb import ISMB
from polyid.utils.seeding import SeedBank

ds = generate(PlantedConfig(collinear_groups=[[0, 1]]), SeedBank(0))      # oracle stays in ds.oracle
res = ISMB({"n_boot": 200}).fit(ds.observed, EstimandSpec(tau=0.10), np.random.default_rng(0))
print(res.lower, res.upper, res.diagnostics["groups"])
```

## 8. Repository map

```
src/polyid/
├── estimand.py          P_j(τ), normalisation, visibility, non-duplication, smooth surrogate
├── data.py              ObservedData: the ONLY object estimators ever see
├── generators/planted.py  planted generator + exact population moments + every stress switch
├── moments.py           per-environment covariances, fourth-moment / cluster-robust variances, bootstrap weights
├── jointdiag.py         U-WEDGE with FFDiag clamping, GEVD starts, multistart
├── matching.py          sign/permutation matching (evaluation) and replicate alignment
├── estimators/          baselines (moments, GMM/BIC, DPMI-style, PCA, FA, FastICA, joint diag,
│                        dictionary learning, naive bootstrap) + registry + Access records;
│                        cumulant.py = exploratory 2nd+4th-order repair (jointdiag_cum)
├── bounds/
│   ├── schur_horn.py    Lemma 2: exact group ranges; span outer bound
│   ├── rank_tests.py    Cragg–Donald rank-1 tests, dependence test, visibility bands
│   ├── ismb.py          the proposed ISMB procedure (+ infinite-sample version)
│   ├── ar_grid.py       AR-type confidence set for d = 3 (grid-restricted)
│   └── sensitivity.py   multi-start sensitivity range (no coverage claim)
├── identification.py    oracle-side identified set: identified / partial / misspecified
├── metrics.py, stats.py coverage, width, confidently-wrong, Wilson, clustered SEs, McNemar, Holm
├── toy/                 superposition toy, TopK SAE, modular-addition transformer
├── lm/                  Pythia hooks (verified), corpus + splits, templates, cache, ablation
├── experiments/         E0–E10 runners
├── viz/                 figure style + identification map
└── cli.py               `polyid run|list`
configs/                 one YAML per experiment (+ smoke configs)
docs/                    THEORY · ASSUMPTIONS · EXPERIMENT_PLAN · PREREGISTRATION · READING_MAP
tests/                   107 tests incl. AST leakage scan, Schur–Horn exactness, rank-test size
results/                 manifests, summary CSVs and figures of every run reported above
```

## 9. Assumptions and honesty rules

All assumptions are numbered in [`docs/ASSUMPTIONS.md`](docs/ASSUMPTIONS.md): model M1–M9,
statistics S1–S9, evaluation E1–E8, language model L1–L7, and decisions D1–D12 where the
source documents were silent or in conflict. The rules enforced in code:

* **No leakage.** Estimators receive `ObservedData` only. `tests/test_leakage_and_seeds.py`
  parses every estimator module and fails if it imports the generator, the identification
  module or the metrics, or touches an oracle name.
* **Dev / test separation.** Seed namespaces `dev` and `test` never coincide. Score→count
  calibration maps are fitted on dev datasets only.
* **Fair comparisons.** Same rows for all methods; every result carries an `Access` record
  (environment labels? higher moments? restarts? model-selection budget?).
* **Labels mean something.** `confidence` = has a coverage argument; `identified_set` = exact
  population object; `sensitivity` = optimiser spread, **no coverage claim**.
* **Failure is a result.** A method that crashes is recorded, not dropped. "Confidently wrong"
  is a first-class metric.

## 10. Reading map

[`docs/READING_MAP.md`](docs/READING_MAP.md) maps each Research-B reading (S02, P1, S04, T1,
P2, S11–S12, S17–S27, X2–X4, X8–X9) to the module, test or experiment that applies it.

## 11. Open items

0. **Cumulant-aware ISMB** (top priority). The point estimator `jointdiag_cum` already
   resolves the weak-identification failures. The interval version needs pair tests that use
   cumulant slices and a coverage argument for the higher-moment information set. Check
   whether dependent latents (Prop. 2) become *detectable* with cumulants: a cumulant tensor of
   dependent latents is not diagonal.
1. **Statistician review** of Proposition 1, Lemma 2, Proposition 2 and the ISMB coverage
   argument (plan p.8 and syllabus S23 both flag this).
2. **Full-scale runs:** E1 (500 reps × 24 cells), E2 (200 reps), E4 (500 reps). About a day on
   4 CPU cores.
3. **Language-model phase:** Hugging Face access and a GPU for E8/E9; extend the text banks to
   ≥ 2 000 held-out items per task with human checking (D10).
4. **Uniformity near collinearity** (Stoye regime): E2/E4 measure it; a repair is open.
5. **Overcomplete extension:** not claimed. SOBIUM (X9) is the prior art to engage first.
6. **DPMI:** replace the DPMI-style approximation with a verified reimplementation (D6).

---

<sub>Built on the research plan *Statistical Interpretability Research Plan* (Jihan Mankani,
18 Oct 2026). License: not yet chosen; add one before making the repository public.</sub>
