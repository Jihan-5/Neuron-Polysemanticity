# Reading map: syllabus → code

Each Research-B reading in the *Revised Core Reading Syllabus* (18 Oct 2026) is mapped to the
code, test or experiment that applies it, plus its "done when" exercise. The monitoring
readings (Research A) are mapped in `docs/monitoring/READING_MAP.md`. Override Lab readings are
out of scope for this repository.

| ID | Reading | Applied in | "Done when" exercise → artefact |
|---|---|---|---|
| S02 | Toy Models of Superposition | `toy/superposition.py`, E5 | Learned-`W` ground truth instead of "every neuron = K features"; fixed-`A` frequency shift separate from retraining (E3 vs E5) |
| P1 | Wasserman, *All of Statistics* ch. 5, 7–10 | `stats.py`, bootstrap in `moments.py` / `bounds/ismb.py` | Bootstrap invalid at boundaries → visibility uses analytic bands (`rank_tests.profile_band`); Holm in `stats.holm` |
| S04 | Miller, *Adding Error Bars to Evals* | `stats.clustered_mean_se`, `paired_difference`, `metrics.aggregate` | Clustered paired intervals over datasets / template families |
| T1 | ARENA, TransformerLens / nnsight, SAELens | `lm/hooks.py` | Hook on `dense_4h_to_h` input, asserted MLP width, verified against a manual forward (`tests/test_torch_lm.py`) |
| P2 | Hyvärinen & Oja, ICA tutorial | `jointdiag.py`, FastICA baseline | Scale and permutation ambiguity → `matching.py`; non-Gaussianity (FastICA) vs environment variation (joint diag) disclosed in `Access` |
| S11 | coroICA (Pfister et al.) | `moments.centered_stack`, `jointdiag.py` | Covariance-difference identity; non-collinearity condition → `rank_tests` + Proposition 1 |
| S12 | Gao et al., Scaling and evaluating SAEs | `toy/sae.py`, E6 | TopK SAE, dead latents, dictionary sizes 2d/4d; reconstruction vs count error (E6 correlation) |
| S17 | DPMI | `estimators/baselines.DPMIStyle` (approximation), E3 | Frequency change alters mixture scores with loadings fixed (E3) |
| S18 | Taming Polysemanticity (GBA) | E2 stress conditions | Assumption comparison: low co-occurrence ↔ `cooccurrence_*`; separation ↔ `near_duplicate_columns` |
| S19 | Nonlinear multi-study factor analysis | `docs/THEORY.md` §1–§3 | Support recovery vs interval for a thresholded count are reported separately (`metrics.support_recovery` vs coverage) |
| S20 | Tamer; Imbens & Manski | `identification.py`, `metrics.covers_idset` | Coverage of the parameter (`covered`) vs of the identified set (`covers_idset`) |
| S21 | Stoye (2009) | E2 `weak_diversity`, E4 `weak_diversity` | Near-point-identified regime measured, not assumed |
| S22 | Chernozhukov, Hong & Tamer | `bounds/ar_grid.py` | `C_n` as a criterion level set in `d = 3`; projection = first honest range |
| S23 | Kaido, Molinari & Stoye | `estimand.smooth_multiplicity`, AR-grid smooth range | Smooth surrogate vs discontinuous `P_j`; the assumption `P_j` violates (continuity) is documented in THEORY §7 |
| S24 | Towards Monosemanticity | E6 (dictionary 2d vs 4d) | Feature splitting stresses non-duplication; atom counts per neuron at two sizes |
| S25 | Finding Neurons in a Haystack | E8 layer choice, E9 | Relative depths 2/5/8; sparse-probe corroboration listed as future work |
| S26 | Zhang & Nanda, activation patching | `lm/ablation.py` | Logit difference and accuracy; zero ablation stated explicitly; matched controls |
| S27 | Polysemanticity or Polysemy? | `lm/templates.py` | Four-way same/different word × sense controls (`four_way_pairs`) |
| X2 | WSAE | — | Conditional: only if claiming SAE recovery |
| X3 | SAE seed instability | E6 `mmcs_across_seeds` | SAE agreement is never treated as ground truth |
| X4 | PSI | — | Conditional: only if adding activation-cluster scoring |
| X8 | Dietterich (McNemar) | `stats.mcnemar_exact` | Reproduces the spec doc's worked p ≈ 1.87e-4 (`tests/test_metrics_stats_cli.py`) |
| X9 | Kolda & Bader; SOBIUM | E2 `overcomplete_24` | The overcomplete extension is *not* claimed; SOBIUM is the prior art to cite before any tensor method |
