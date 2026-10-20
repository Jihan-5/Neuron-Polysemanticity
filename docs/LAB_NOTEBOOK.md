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

