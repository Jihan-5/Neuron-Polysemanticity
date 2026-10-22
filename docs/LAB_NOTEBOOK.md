# Lab notebook: development findings, failures and fixes

A chronological record of what went wrong while building the pipeline and what was changed.
Nothing here is a result about neurons. It documents why the code is shaped the way it is,
and which earlier outputs are void.

## 2026-10-18/19

### 1. U-WEDGE stalls; quasi-Newton finds degenerate optima → FFDiag step clamping
* **Observed.** U-WEDGE from a whitening start with a backtracking line search hit the
  1000-iteration cap from most starts on planted `d = 16` sample data (Amari 0.19–0.64 vs
  0.03–0.07 from a truth start). L-BFGS on the scale-invariant off-diagonal objective drove the
  objective to ~0 at *degenerate* near-singular unmixing matrices: rows collapse onto common
  isotropic directions of the indefinite stack.
* **Fix.** Gauss–Newton steps clamped to `||H||_F ≤ 0.5` (FFDiag, Ziehe et al. 2004), no line
  search, plus a deterministic principal-pencil GEVD start. Every start then reached the correct
  basin. Population recovery is exact (Amari < 1e-14) across 30 random designs.
  `tests/test_jointdiag.py` locks this in.

### 2. Bootstrap visibility bands fail for invisible directions → analytic bands
* **Observed.** With `K = 8 < d = 16`, the 8 pseudo-components spanning the invisible subspace
  are relabelled arbitrarily across bootstrap replicates (alignment swaps). Their percentile
  bands therefore reached visible levels, nothing was declared invisible, and the pair tests
  then merged everything into one 16-component group (width 16).
* **Fix.** Per-component analytic band from per-environment fourth moments of the unmixed
  signal. This is first-order exact for invisible directions because `D_e b = 0`. Pair tests
  now run only among *sure-visible* components.

### 3. Simultaneity over all loadings is uninformative → row-wise sup-t with GMS
* **Observed.** A sup-t band over all 256 loadings (critical value ≈ 4.0) made zero loadings
  "ambiguous" at `τ = 0.10` (mean width 9 of 16 in a clean design).
* **Check.** The bootstrap scale is calibrated: robust bootstrap SD / Monte Carlo SD across 40
  independent datasets had median 1.01 (IQR 0.90–1.16).
* **Fix.** Simultaneity per neuron (the coverage target) and generalised moment selection
  (`κ_n = sqrt(log n)`). Loadings far from `τ` no longer inflate the critical value. Clean-design
  width fell to 5.0 at `n = 5000` and 0.5 at `n = 20 000`. **The remaining width at
  `n = 5000` is honest:** loading SEs of 0.02–0.04 against `τ = 0.1`.

### 4. Visibility threshold at the noise floor
* `v_min = 1e-3` sat below the sampling noise of an invisible component's profile (≈ 0.003).
  Set `v_min = 0.006` and generator `min_visibility = 0.012` (decision D1).

### 5. Perfect co-occurrence is undetectable (Proposition 2)
* A smoke test showed the infinite-sample procedure returning a narrow wrong answer for a
  shared-switch pair, with *zero* misfit. Working out why gave Proposition 2 (THEORY §5): the
  moments are exactly those of an uncorrelated model with different loadings. Regression test:
  `tests/test_bounds.py::test_proposition2_perfect_cooccurrence_is_undetectable`.

### 6. pandas averages object-dtype booleans with NaNs incorrectly
* The E0 summary first showed coverage 0.0625 where every interval covered.
  `groupby().agg(mean)` on an object column mixing `True` and `NaN` returned a wrong value under
  pandas 3. All coverage flags are now stored as floats.

### 7. **E4 bug: truth computed at the wrong threshold (first E4 pilot void)**
* **Observed.** In the first E4 pilot every method, including the AR-grid set and the
  identified-set oracle, showed about 60 % coverage in the *identifiable* cell, while coverage
  of the identified set was about 1.0.
* **Cause.** Experiments scored against `oracle.truth`, which the generator computes at its own
  `tau` field (default 0.10). E4 declares `spec.tau = 0.30`. E0, E1, E2, E3, E6 and E10 use
  matching values (0.10) and were unaffected. E5 computes its own truth.
* **Fix.** Every runner now scores against `oracle.truth_at(spec.tau)`. The E4 config pins the
  generator `tau` to 0.30. `tests/test_generator.py::test_truth_at_uses_the_requested_threshold`
  guards against regressions. The void output was deleted and E4 was re-run.

### 8. E1 pilot: second moments are weak at `d = 16, E = 6`; non-Gaussianity is strong
* **Observed (E1 pilot, 6 test datasets per cell).** In the `near_collinear` half of the grid
  (population min pairwise profile sine 0.05–0.09), joint-diagonalisation count MAE was 2.5–6.5
  with large positive bias, and ISMB merged nearly everything (mean width ≈ 16, coverage 1.0).
  FastICA, which identifies through non-Gaussianity, had MAE 0–1.7 in the same cells.
* **Checked.** Joint diagonalisation on the *population* moments of five near-collinear designs
  recovers `A` exactly (Amari ≤ 1e-14, count MAE 0). The failure is statistical weak
  identification at `n ≤ 5000`, not the optimiser.
* **Implication for the plan.** The declared second-moment information set leaves most of the
  identifying information in this generator unused: sparse Bernoulli × magnitude latents are
  strongly non-Gaussian. The most promising repair is to add fourth-order cumulant conditions
  per environment to the identified-set construction. Collinear-profile classes would then
  be identified whenever the latents are independent and non-Gaussian. Proposition 2's
  co-occurrence case (dependent latents) would remain unidentified. This is recommended
  next step 1 in the README.

### 9. Repair: add fourth-order cumulant slices to the joint-diagonalisation stack
* **Implemented** `estimators/cumulant.py` (`jointdiag_cum`, exploratory, point estimate only).
  Per-environment slices `cum(x_p, x_p, x_i, x_j)` are diagonalised by the same unmixing
  under independent latents, and Gaussian noise contributes zero.
* **E1b pilot** (`results/e1b_cumulant_repair_pilot`, 12 cells at `n = 5000`, 6 datasets each):
  count MAE 0.016 (exact 0.98, support F1 0.99), vs 2.22 for second-moment joint
  diagonalisation and 0.61 for pooled FastICA. The near-collinear failure disappears (MAE ≤ 0.06
  in every cell), and it beats FastICA at higher activation rates where the latents are less
  sparse.
* **Prior art, not a novelty claim.** Combining non-stationarity of variance with
  non-Gaussianity is established: JADE (Cardoso & Souloumiac 1993); Hyvärinen (2001),
  *Blind source separation by nonstationarity of variance: a cumulant-based approach*;
  Tichavský, Koldovský and colleagues' combined second- and fourth-order methods. Check them
  before writing anything up.
* **Not yet tested:** dependent latents (Prop. 2 still applies; cumulant tensors of dependent
  latents are not diagonal, so the misfit may become *detectable*, which is worth testing),
  heavy-tailed or non-Gaussian noise (noise cumulants do not vanish), and an interval version
  (cumulant-aware pair tests inside ISMB).

### 10. E2 pilot (`d = 16`): validity holds, naive intervals collapse, labels fixed
* **ISMB** covered the truth in all 19 stress conditions (coverage 1.00, zero confidently
  wrong). Its width tracks difficulty: 2.8 with 8 environments, 7.1 clean, 15–16 under weak
  diversity, 3 environments, overcompleteness or heteroscedastic noise. It is valid and honest,
  but at `n = 5000` it is not informative enough.
* **The naive bootstrap fails at `d = 16`** where it had not at `d = 3`: coverage 0.06 (weak
  diversity), 0.22 (3 environments), 0.56 (heteroscedastic noise). The sensitivity range is
  confidently wrong 5–74 % of the time.
* **Label fix.** The oracle had labelled `overcomplete_24` "identified" (it only checked profile
  collinearity). Proposition 1 needs `K_visible ≤ d`, so overcomplete designs are now
  "misspecified" (M5) with the trivial range. The pilot's population labels for that condition
  were recomputed deterministically from the same seed paths; no estimator output changed.
* **Partial co-occurrence is undetectable too** when rates are shared: see the corollary in
  THEORY §5.

### 11. E5/E6 first pilots → an ISMB misfit gate; every pilot re-run
* **E5 (trained toy superposition, first pilot).** Every trained model was overcomplete:
  15–31 represented, visible features in `d = 8` or `16` dimensions, even at feature
  probability 0.3. All `d`-component methods undercounted (bias −2 to −9). ISMB *undercovered*
  (0.06–0.31 at `d = 8`), although its own off-diagonal misfit ratio was high (0.14–0.56).
  Reporting a narrow interval under a visibly failing model violates plan p.7 ("bad fit is
  evidence against assumptions, not permission to report a precise count anyway").
* **E6 (TopK SAE, first pilot).** SAE-derived counts over-count massively (MAE 6.9–23.9, all
  positive bias). Doubling the dictionary makes it worse (feature splitting), even at FVE 0.96.
* **Fix: misfit gate.** The pairwise residual-dependence statistics are pooled into one
  approximate chi-square test. On rejection at 1e-3, ISMB abstains to `[0, k_max]`.
  Calibration on 6 datasets per condition: 0 % rejection for clean, weak-diversity, collinear,
  unknown-`K` and 4-environment designs; 100 % for overcomplete and environment-scaled noise;
  17 % for Student-t noise (a stationary design the model allows, so these are conservative
  abstentions); 0 % for perfect co-occurrence (undetectable, Prop. 2).
* **Consequence.** ISMB changed, so *every* pilot was re-run from scratch with the gate
  (`scripts/run_pilots.sh`). Earlier pilot outputs remain in git history (commits before
  `0e15e2d`). The README numbers come from the re-run.

## 2026-10-20

### 12. Cumulant-aware ISMB (`ismb_cum`): two failures before it worked
* **Design** (THEORY §8). Fourth-order cumulant slices are added to the joint-diagonalisation
  stack. A collinear pair is merged only if neither member is detectably non-Gaussian. A
  cross-cumulant test span-merges dependent pairs.
* **Failure 1: the misfit gate fired on every dataset.** On 4 datasets each of clean,
  collinear-pair, near-collinear and perfect co-occurrence data (`d = 16`, `n = 5000`), it
  abstained to `[0, 16]` every time. ISMB's second-moment pair, dependence and misfit tests
  assume a basis fitted to the covariance stack. The combined basis is a compromise with the
  noisier cumulant slices, and leaves second-moment residuals those tests read as misfit.
  *Fix:* run them on a second-moment refinement warm-started from the combined solution (D14).
  Widths then fell to 0.0–1.2, with every interval covering.
* **Failure 2: near-Gaussian latents made it far wider than ISMB.** With a collinear pair and
  `latent_gaussianity` 0.9 or 1.0, width was ≈ 15.4 against ISMB's 5.0–6.6, although coverage
  held. Each cumulant slice had been scaled to unit norm, so 96 near-pure-noise slices
  outweighed 15 informative covariance differences. *Fix:* precision weighting (D13). Widths at
  g = 0.5 / 0.9 / 1.0 fell from 5.4 / 15.3 / 15.6 to 1.0 / 5.0 / 5.3: graceful fallback to ISMB.
* **Checks.** With `cumulants` off, ISMB output is byte-identical to the committed version on
  three seeded datasets (clean, collinear, perfect co-occurrence), so the full-scale E2/E1 runs
  that import the edited module are unaffected. `latent_gaussianity = 0` leaves generated data
  identical. Both new tests have size 2–9 % at the 5 % level and strong power
  (`tests/test_ismb_cum.py`).
* **Status.** Exploratory, and validated so far only on these 28 hand-run datasets (7 settings ×
  4). E11 (`configs/planted/e11_cumulant_ismb.yaml`) compares it with ISMB on the same datasets,
  across all 19 E2 stress conditions plus six near-Gaussian ones.


### 13. E11 pilot (6 datasets × 25 conditions): `ismb_cum` narrows sharply, and one narrow-and-wrong case
* **Coverage** of the true $P_j$ was 1.00 in 24 of 25 conditions for both methods.
* **Width** fell from ISMB's 6–16 to 0.1–1.2 in every condition where latents are non-Gaussian:
  clean 6.1 → 0.08, weak diversity 15.1 → 0.26, 3 environments 15.9 → 0.42, collinear pair
  6.8 → 0.18, perfect co-occurrence 7.4 → 1.19.
* **Near-Gaussian latents:** width returned to ISMB's level (g = 0.9: 5.8 vs 6.3; g = 1.0: 4.2 vs
  4.9), which is the intended fallback.
* **Misspecified noise and overcomplete models:** both methods abstain to `[0, 16]` (environment-
  scaled and heteroscedastic noise, `overcomplete_24`). Student-t noise: 11.8 vs 13.8.
* **The exception: `noise_correlated_stationary`, replicate 3.** Two neurons got point intervals
  `[4, 4]` and `[3, 3]` for true counts 5 and 4: confidently wrong. ISMB gave `[2, 16]` for both.
  In that dataset only 15 of 16 components were sure-visible. The likely mechanism is a weakly
  visible factor poorly estimated by the combined fit, with a bootstrap band too narrow for its
  error. This is the failure the method must not have. The full E11 (100 datasets per condition)
  measures its rate before any claim is made.


## 2026-10-21

### 14. E1 at full size (500 datasets × 24 cells): the pilot holds
* **Ranking unchanged.** Equal-weight count MAE: `jointdiag_cum` 0.030 (97 % exact), FastICA
  0.61, DPMI-style + dev calibration 0.64, second-moment joint diagonalisation 2.75. Every
  method's MAE is within 0.06 of the pilot (6 datasets per cell) except GMM-BIC's raw count
  (0.80 → 1.05).
* **ISMB** covered every neuron in all 24 cells, with no confidently-wrong interval, at mean
  width 13.3 of 16: 10.8 well separated, 15.7 near collinear. Valid but rarely informative at
  `d = 16` with second moments, as in E2.
* **Naive uncertainty collapses near collinearity.** Bootstrap-percentile coverage was 0.83 well
  separated and 0.17 near collinear (0.02 in the worst cell). The sensitivity range was
  confidently wrong on 38 % and 75 % of neurons.
* **Infrastructure.** The container restarted three times during the run (roughly every
  1.5 h). Per-condition checkpoints lost up to an hour each time, so conditions now save every
  40 datasets (`run_planted_condition(chunk_dir=...)`). Chunked, resumed and uninterrupted runs
  give identical rows on a test condition; only timing columns differ.

### 15. E11 at full size (100 datasets × 25 conditions): `ismb_cum` holds up
* **Coverage.** Per-neuron coverage ≥ 0.994 in all 25 conditions (1.000 in 21). All 16 neurons
  of a dataset were covered at once in ≥ 95 % of datasets in every condition; the lowest was
  perfect co-occurrence (0.95).
* **Width.** With non-Gaussian latents and a correctly specified model, 0.2–2.3 against ISMB's
  2.8–16 on the same datasets (clean 6.2 → 0.47, collinear pair 6.4 → 0.18, weak diversity
  15.1 → 0.45, 3 environments 15.9 → 0.44, perfect co-occurrence 7.2 → 1.2, correlated
  stationary noise 10.2 → 2.3).
* **Gaussian fallback works.** At `g = 0.9` and `g = 1.0`, width 5.0–5.6 against ISMB's
  5.5–6.2, coverage 1.000, for both the base and the collinear-pair design. The cumulant point
  estimate fails in exactly these cells (MAE 5.7–6.1, exact ≤ 2 %), so the interval method is
  what makes the higher-moment information set safe to use. At `g = 0.5` width is 1.0–1.6.
* **Abstention** to `[0, 16]` on every dataset with environment-scaled or heteroscedastic noise
  and with 24 factors in 16 neurons, as for ISMB. Student-t noise: 12.0 → 8.4.
* **The §13 failure mode, measured.** Confidently wrong neurons: 0.6 % under correlated
  stationary noise, 0.4 % under perfect co-occurrence, 0.2 % with Student-t noise, 0.1 % with
  loadings near `τ`, 0 in the other 21 conditions. ISMB had none. These sit inside the
  nominal 5 % simultaneous error, but they are the misses to explain before the method is
  claimed beyond "exploratory".


## 2026-10-22

### 16. Why ISMB undercovers on trained toy models (E5), and the overcompleteness gate (E12)
* **Every miss is on the high side.** In E5 the true count exceeds ISMB's upper bound; ISMB
  never overcounts. At `d = 8` most misses are structural: the true count exceeds `d` for
  25–96 % of neurons, and a square model's upper bound can never exceed its number of
  components.
* **Second moments cannot count past `d`**, so the misfit gate misses many of these models.
  New diagnostic (`bounds/overcomplete.py`): the averaged within-environment fourth-order
  cumulant flattening has rank equal to the number of non-Gaussian latents (the idea behind
  FOOBI, De Lathauwer, Castaing & Cardoso 2007). A Weyl bound with a bootstrap noise floor makes
  the test of `rank <= d` conservative. The count of singular values above the floor is a
  lower bound on the number of features in the block.
* **Feasibility.** The test caught 8 of 12 trained toys (every model with feature probability
  ≤ 0.1), with no false alarm in 50 planted `K = d` datasets including heavy-tailed and
  correlated noise. A sharper projected variant had more power but 20 % false alarms under
  collinearity and Student-t noise, so it was rejected.
* **E12a/b (same 30 trained models; 5,000 and 20,000 rows per environment).** With the gate
  (`ismb_oc`) coverage is 1.00 at feature probability 0.03 for both `d`, where it was 0.43 and
  0.80. The gate fires on 80 % of models at probability 0.1 and never at 0.3. The rank lower
  bound never exceeded the true number of features. Without the gate, cumulant ISMB is badly
  wrong on these models (coverage 0.08–0.59), so the gate is essential for it. Plain ISMB gets
  *worse* with more data at `d = 16` (0.95 → 0.53 at 0.1; 0.93 → 0.55 at 0.3): narrower intervals
  under a wrong model.
* **The dense regime is antipodal superposition, not weak features.** At probability 0.3 every
  target feature has norm ≈ 1, visibility 4–13 × `v_min` and a fourth-order signal comparable to
  the strongest feature. But the 15–16 target features of the `d = 8` models form 7–8 antipodal
  pairs (`w`, `−w`, cos < −0.95), leaving exactly 8 distinct directions. Every even-order
  statistic depends on `w wᵀ`, which is identical for `w` and `−w`, so second- and fourth-order
  methods see one feature per pair at any sample size (the gate's signal stayed at 0.7–0.8 of
  the noise floor when the data were quadrupled). The ground truth counts both, so `P_j` is not
  identified from even-order moments when features are stored antipodally: each pair
  contributes 1 or 2. Sign-sensitive statistics (positive and negative parts of the projection,
  for non-negative features) are the candidate fix (§16a).

### 16a. Antipodal pairs: census and a sign-split check (feasibility only)
* **Census.** `d = 8`, probability 0.3: 16 and 15 target features in 8 and 7 antipodal pairs,
  leaving exactly 8 directions. `d = 16`, probability 0.3: 23 features, 7 pairs, exactly 16
  directions. So in the dense regime the number of distinct directions equals `d`, which is why
  neither the misfit gate nor the fourth-order rank test can see the extra features.
* **Sign split.** Project the data on a feature direction `w` (`u = wᵀh`, 20,000 rows per
  environment). For the four single (non-antipodal) features checked, the negative side carries
  essentially nothing (`E[(u⁻)²]` ≈ 1e-7 to 1e-5 against 0.04–0.15 for the positive side). For
  the four antipodal directions both sides carry 0.03–0.13. A two-sidedness ratio separates them
  completely here. The profile correlation between the two sides does not (it is negative for
  both kinds).
* **What it would take.** Counting antipodal pairs needs a declared assumption that latents are
  non-negative. That holds for the toys and for SAE-style features, but not automatically for
  raw MLP neurons (GELU outputs can be negative), and Gaussian components or noise also make a
  projection two-sided. A sign-split extension of ISMB (widen `U_j` by one for each two-sided
  component that loads on neuron `j`, with a calibrated threshold) is the next method step. It
  is not implemented. Until then, the honest statement for the dense regime is that `P_j` is
  identified only up to antipodal pairs.
* **Census across regimes (seed 0).** `d = 8`: probability 0.03 → 22 features, 0 pairs, 22
  directions; 0.1 → 18 features, 5 pairs, 13 directions. `d = 16`: 0.1 → 30 features, 11
  pairs, 19 directions; 0.3 → 23 features, 7 pairs, 16 directions. The overcompleteness gate
  fires exactly where distinct directions exceed `d`.
* **E12c (false alarms, E11's 2,500 datasets).** The gate fired on 1 of 2,400 non-overcomplete
  datasets (0.04 %, one `cooccurrence_partial` dataset) and widths are unchanged. On planted
  `overcomplete_24` it fired on only 5 %, but the second-moment misfit gate already abstains on
  every one of those datasets.

### 17. The 27 confidently-wrong cumulant-ISMB neurons in E11, and `ismb_cum_v2`
* **What failed.** 27 of 40,000 neuron intervals in 15 of 2,500 datasets, every one off by
  exactly one feature, most of them zero-width. Reproduced exactly from the same seeds.
* **Correlated or heavy-tailed noise (16 neurons).** Two true factors with similar columns
  merge into one estimated column that is stable under the bootstrap, while the spare column
  fits noise and is declared invisible. Two features are then counted as one. Every such
  dataset has an unstable column (median bootstrap alignment 0.65–0.95). Across 1,008
  estimated columns, "median bootstrap cosine < 0.97" flagged 21 columns, all badly
  estimated, with no false flag. But it cannot flag the merged column itself, which looks
  stable.
* **Perfect co-occurrence (9 neurons).** The co-occurring pair is hugely significant
  (cross-cumulant p ≈ 1e-70 to 1e-97) in four of the five failing datasets, yet was never
  flagged. Pair tests ran only among *sure*-visible components, and one member had been
  classed ambiguous. The fifth dataset is a genuine power failure (smallest p 3.6e-3 against a
  Bonferroni level of 8.3e-5).
* **Loadings near `τ` (2 neurons).** Ordinary sampling error at the nominal level.
* **Fixes (opt-in flags; `ismb_cum_v2` bundles them with the overcompleteness gate).**
  `stab_widen` raises `U_j` by the number of unstable columns, capped at `K`. `dep_ambiguous`
  runs the dependence tests on pairs with ambiguous members and span-merges flagged pairs; the
  span bound adds nothing to `L_j`. On the 15 failing datasets misses fell from 27 to 7
  (correlated noise 10 → 0, Student-t 6 → 0, co-occurrence 9 → 5, near-`τ` 2 → 2). Width did not
  change on datasets whose columns are all stable (clean, weak diversity); the main cost is
  under Student-t noise (mean width 5.3 → 8.1). The overcompleteness gate fired on none of the
  39 planted datasets checked.
* **Out of sample (E13: new seed, 100 datasets × 10 conditions).** `ismb_cum` missed 7 neurons
  (perfect co-occurrence 4, correlated noise 2, Student-t 1); v2 missed 4, all in one
  perfect-co-occurrence dataset (rep 65) that had one ambiguous component whose dependence test
  stayed above the Bonferroni level. That is a power limit, not the skipped-test bug.
  Simultaneous coverage was ≥ 0.99 in every condition for v2. Width changes: Student-t 8.50 →
  10.34, correlated noise 2.68 → 2.91, perfect co-occurrence 1.41 → 1.58, others within ±0.15.
  The overcompleteness gate fired on 0 of 900 non-overcomplete datasets and on 4 % of
  `overcomplete_24`, which the misfit gate already covers.

### 18. E9 items: BLiMP for agreement, a provenance correction, and WiC ruled out
* **Provenance correction.** The template banks in `lm/templates.py` were drafted by the AI
  coding assistant during development. Their docstring said they were hand-written with no LLM
  involved, which was wrong. Under rule L7 they are smoke-test placeholders only, and E9 now
  refuses them on a real model.
* **Agreement task.** 4,000 BLiMP minimal pairs (Warstadt et al. 2020; CC-BY; 96.4 % human
  agreement) from the four subject–verb agreement paradigms that support the one-prefix
  method, two of them with attractor nouns. Vendored with checksums in `data/blimp`.
* **Scoring.** Choices are now scored by the full continuation log-probability. For
  single-token continuations this equals the previous last-position logit difference
  (tested); multi-token words such as " hasn't" used to be scored by their first token only.
* **WiC is ruled out for the semantic task.** Even 175B–530B models score near chance on it
  (about 49 % zero-shot, 55–58 % few-shot), so an ablation cannot lower Pythia's accuracy.
* **SemCor swap pairs instead (decided 2026-10-22).** 4,011 pairs over 558 nouns built
  deterministically from SemCor 3.0 (`scripts/build_semcor_items.py`; two runs give identical
  bytes): same noun, different WordNet supersense, real sentences cut after the noun.
  CoarseWSD-20 was considered and rejected: its labels come mainly from Wikipedia links, its
  repository has no licence, and choosing between sense names is likely too hard for a 160M
  model. SemCor's labels were hand-annotated against WordNet 1.6 and mapped to 3.0
  automatically. Its licence allows redistribution with the Princeton notice, which is
  vendored. Splits are by noun: 727 dev pairs (103 nouns) for the pre-check, 3,284 test pairs
  (455 nouns) for E9. On the offline smoke model the pre-check gives 0.52, chance, as it
  should for random weights. Whether Pythia clears 0.65 is the first thing the GPU run checks.
