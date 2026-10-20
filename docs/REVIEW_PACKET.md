# Review packet for a statistician

This is what to send for item 4, the last unmet item of the theory gate (`docs/PREREGISTRATION.md`
§3). It lists every mathematical claim the study rests on, where its argument is written, and
the specific questions a reviewer should answer. Reading time is roughly 2–4 hours. Everything
cited is in [`docs/THEORY.md`](THEORY.md).

## The setting in one paragraph

Neurons `h` (dimension `d`) are a linear mix of latent factors, `h^(e) = A z^(e) + ε^(e)`, in
several environments `e` that change only the factors' variances. `A` is shared, and the noise
covariance is stationary. Columns of `A` have unit norm. The target is the **thresholded loading
multiplicity** `P_j(τ)`: the number of *visible* factors `k` with `|A_jk| ≥ τ`, where a factor
is visible if its centred variance profile across environments has RMS at least `v_min`. The
information set is the stack of centred covariances `D_e = C_e − mean_e C_e`, i.e. second
moments only. The question is what `P_j` can be learned from that stack, and how to report
honest uncertainty when it cannot be pinned down.

## Claims to check

| # | Claim | Where | Status in the code |
|---|---|---|---|
| 1 | **Proposition 1**: the exact second-moment identified set. Singleton classes of collinear profiles are identified up to sign; a class of size `g ≥ 2` with positive coefficients is identified exactly up to `normalize(A_G Λ^{1/2} R)`, `R ∈ O(g)` | THEORY §3 | Numerically verified (E0: two members agree to 5.6 × 10⁻¹⁷ and give different `P_j`) |
| 2 | **Lemma 2**: exact min and max of a class's contribution to `P_j` over `R ∈ O(g)`, via Schur–Horn majorisation | THEORY §4 | Exact against a 20 000-angle sweep for `g = 2`; containment for `g = 3` (`tests/test_schur_horn.py`) |
| 3 | **Proposition 2 and corollary**: perfect co-occurrence (shared activity switch), and any exchangeable within-block dependence with shared rates, is observationally equivalent under second moments to an uncorrelated model with different loadings | THEORY §5 | Numerically confirmed (E2, E4: ISMB coverage falls with `n` in this cell, 0.96 → 0.87) |
| 4 | **Remark 3**: the pairwise Cragg–Donald rank-1 test of profile collinearity is first-order robust to estimating the pair subspace | THEORY §6 | Empirical size 5.5 % at nominal 5 %, power 0.95 |
| 5 | **ISMB coverage**: `P(L_j ≤ P_j ≤ U_j ∀ j) → ≥ 1 − α` under conditions (a)–(e) | THEORY §7 | Full runs: coverage ≥ 0.988 in identified and collinear cells (E4, 500 reps); 1.00 in all 19 stress conditions (E2, 200 reps) |
| 6 | **ISMB-cum** (exploratory): adding fourth-order cumulant slices; collinear pairs are identified unless both members are Gaussian; a cross-cumulant test detects dependence | THEORY §8 | Pilot only (E11) |
| 7 | **AR-grid** (`d = 3`): grid-restricted confidence set from a profiled efficient criterion | THEORY §7 | E4: coverage 0.99–1.00 in the identifiable, collinear and weak-diversity cells; 0.89–0.94 under perfect co-occurrence, where the model is misspecified |

## Specific questions

**Proofs (claims 1–3).**
1. Proposition 1: is the genericity argument for the weights `w, w'` complete? Is the use of
   Sylvester's law of inertia to force `λ' ≻ 0` correct? Remark (i) on mixed signs (`O(p, q)`)
   and remark (ii) on invisible factors: are they right as stated?
2. Lemma 2: is the minimum formula (with the small `ε`) correct? Is the interval hull
   `[Σ min, Σ max]` the right object to report, given that classes rotate independently?
3. Proposition 2: the proof relies on simultaneous diagonalisation of a positive-definite form
   and a symmetric form on `span(A_G)`, and on non-negativity of the implied variances. Is
   anything missing? The claim that *partial* co-occurrence with unequal rates is "generally
   detectable" is heuristic: should it be stated as a conjecture?

**Inference (claims 4–5).**
4. Remark 3: does the tangent-space argument suffice for the Cragg–Donald statistic to keep its
   `χ²` limit with an estimated subspace?
5. ISMB groups factors using data-dependent tests, then splits `α` across steps by a union
   bound. Is the post-selection structure handled correctly, at least pointwise? Which further
   conditions are needed?
6. Loading bands use a row-wise sup-t bootstrap with generalised moment selection
   (`κ_n = √log n`). Is pointwise validity under strong identification of separated factors
   enough to state the result, given that uniformity fails near collinearity (Stoye 2009)? How
   should that limitation be worded?
7. Group ranges are widened by bootstrap quantiles of the range. Is that a valid way to cover
   the identified-set component?

**Wording.**
8. For a manuscript, which of claims 1–5 can be called theorems as written, which need
   changes, and which should be presented as "numerically verified claims"?

## What the answer changes

If claims 1–3 hold and the inference argument (claims 4–5) is judged sound pointwise with
stated conditions, `theory_gate_passed` can be set to `true` with a dated amendment, and the
Pythia experiments (E8/E9) may run. If not, the gate stays closed, or the affected claims are
downgraded to "numerically verified" in every write-up, as the preregistration requires.
