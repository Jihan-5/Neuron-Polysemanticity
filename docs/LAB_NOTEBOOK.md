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
