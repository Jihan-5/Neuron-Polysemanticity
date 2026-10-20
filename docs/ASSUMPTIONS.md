# Assumptions register

Every assumption the code makes, where it is enforced, what breaks if it is false, and which
experiment stresses it. IDs are stable and referenced from code and docs.

Legend: **M** = model, **S** = statistical procedure, **E** = evaluation and fairness,
**L** = language-model phase, **G** = engineering, **D** = a decision where the source
documents were silent, ambiguous or in conflict. Each D-item records the alternative,
because a reviewer may prefer it.

## M — the declared model

| ID | Assumption | Enforced / encoded in | If false | Stressed by |
|---|---|---|---|---|
| M1 | Linear mixing: `h^(e) = A z^(e) + ε^(e)` with the **same `A` in every environment** (frozen network, only the sampling distribution varies) | `generators/planted.py`; LM: weights frozen, `requires_grad=False` | Moments of different environments describe different representations | E3 (fixed `A`), E8 (frozen Pythia) |
| M2 | Columns unit-norm; equivalence up to sign and permutation | `estimand.normalize_columns`, `matching.py` | — (convention) | `tests/test_estimand.py` |
| M3 | Latents uncorrelated within an environment (independent in the generator) | generator default; ISMB pair dependence test merges violations | Joint diagonalisation finds a different basis; perfect co-occurrence is **undetectable** (Proposition 2) | E2 `cooccurrence_*`, `coupled_magnitudes`; E4 `cooccurrence_perfect` |
| M4 | Noise uncorrelated with latents and **stationary** (same covariance in every environment); it may be correlated across neurons | centred covariances cancel it; generator `noise_mode` | Non-cancelling residual looks like signal | E2 `noise_env_scaled`, `noise_env_heteroscedastic`; allowed case `noise_correlated_stationary` |
| M5 | Square or under-complete: visible factor count `K_V ≤ d` (factor-count bound `k_max`, default `d`) | joint diagonaliser has `d` components; extra components should come out invisible | Overcomplete mixtures cannot be represented; misfit expected | E2 `overcomplete_24`, E5 sparse regimes |
| M6 | Visibility: a factor is in the target only if `vis_k ≥ v_min` | `estimand.visible_mask`, ISMB visibility bands | Invisible factors are unidentifiable by differences and are excluded by definition | E2 `invisible_factors`, `unknown_K_8` |
| M7 | Minimality / non-duplication: no factor split into copies; admissible columns have `|cos| ≤ max_abs_cos` | count of visible factors of a minimal decomposition; AR-grid admissibility | A split factor would inflate `P_j` arbitrarily | E2 `near_duplicate_columns` |
| M8 | Threshold is closed: `|A_jk| ≥ τ` counts | `estimand.ATOL` | Ties are measure-zero except in the near-τ stress case | E2 `loadings_near_tau` |
| M9 | Information set = second moments (centred covariances). Fourth moments are used only for *variances* of estimates, except in the exploratory `jointdiag_cum` and `ismb_cum`, which also diagonalise 4th-order cumulant slices (requires independent latents and Gaussian, or at least zero-4th-cumulant, noise; THEORY §8) | `moments.py`, `bounds/*`, `estimators/cumulant.py` | Higher-moment methods (FastICA, `jointdiag_cum`) identify much more in the planted generator (lab notebook §8–9); that is a different information set and is disclosed in `Access` | E1b/E2b pilots |

## S — statistical procedures

| ID | Assumption | Where | Notes |
|---|---|---|---|
| S1 | Rows i.i.d. within an environment, **or** cluster labels given (documents, template families), and then resampled as clusters | `moments.bootstrap_weights`, `rank_tests.env_second_moments` (sandwich) | LM contexts from one document are clustered (plan p.9) |
| S2 | Environments are independent samples | `MomentStats.moment_covariance` (block-diagonal across `e`) | Shared documents across environments would violate this. AG News topic environments draw from the same pool, so overlap is possible; recorded per document id |
| S3 | Pair collinearity: Cragg–Donald rank-1 statistic `→ χ²_{2(E-2)}` under H0, first-order robust to basis estimation | `bounds/rank_tests.py`; THEORY §6 | Needs `E ≥ 3`. With `E = 2` everything merges (honest) |
| S4 | Bootstrap valid for loadings of **strongly** identified factors | `bounds/ismb.py` | Fails non-uniformly near collinearity (Stoye); measured in E2/E4 |
| S5 | GMS threshold `κ_n = sqrt(log n_min)` | `ISMBConfig.gms_kappa` | Pointwise valid; a predeclared tuning constant |
| S6 | Union-bound split of α across steps: pairs 25 %, loadings 45 %, visibility 15 %, groups 15 % | `ISMBConfig.share_*` | Predeclared in `docs/PREREGISTRATION.md` |
| S7 | Residual-dependence test is approximate (dof `E-3`) and is used only to **merge** (widen) | `rank_tests.test_pair` | Can never narrow an interval |
| S10 | Global misfit gate: the sum of pairwise dependence statistics is treated as `χ²` with the summed dof (pairs are not independent, so this is approximate). On rejection at `1e-3`, abstain to `[0, k_max]` | `ISMBConfig.misfit_gate` | Calibrated empirically (lab notebook §11); it can only widen |
| S8 | AR-grid: truth on the grid; `K = d = 3`, all visible; efficient weight from fourth moments; `χ²_{3(E-1)}` | `bounds/ar_grid.py` | Grid-restricted statement (plan p.8) |
| S11 | `ismb_cum` Gaussianity test: per environment, the delta-method $z$-score of the excess fourth cumulant of an unmixed component, squared and summed over environments, is $\chi^2_E$ under "Gaussian in every environment". The unmixing is treated as known and clusters are ignored | `bounds/ismb._fourth_cumulant_test` | Size checked in `tests/test_ismb_cum.py` (2–9 % at the 5 % level). Its level is `share_gauss`, Bonferroni over components |
| S12 | `ismb_cum` cross-cumulant test of `cum(u_k,u_k,u_l,u_l)=0`: same construction, $\chi^2_E$. First-order robust to unmixing error | `bounds/ismb._cross_cumulant_test` | Used only to **merge** (widen). Size checked in `tests/test_ismb_cum.py` |
| S9 | Coverage target is **per neuron** (row-wise simultaneous over a neuron's loadings). Simultaneous-over-all-neurons coverage is also reported (`all_covered`) | `ISMBConfig.simultaneity` | `"all"` available for a family-wise claim |

## E — evaluation and fairness

| ID | Rule | Where |
|---|---|---|
| E1 | Latent labels and planted `A` reach only `metrics.py` / `identification.py`. Estimators receive `ObservedData` only | `tests/test_leakage_and_seeds.py` (AST scan) |
| E2 | Development datasets (seed split `dev`) are used only to fit score→count calibration maps and to choose hyperparameters; test datasets (`test`) are only scored | `experiments/common.py`, `utils/seeding.py` |
| E3 | All methods see the same rows. Single-environment methods get pooled, environment-centred data and are labelled `env_labels=False` | `estimators/baselines.py` `Access` records |
| E4 | Methods needing a component count get the declared bound `k_max` (default `d`), never the true `K` | `_n_factors` |
| E5 | Score-only methods (moments, DPMI-style) are compared by ranking metrics; count MAE only via `+devcal` maps | `metrics.aggregate`, `common.run_planted_condition` |
| E6 | Point accuracy and uncertainty quality are reported separately; "confidently wrong" (narrow and wrong) is its own metric | `metrics.py` |
| E7 | Main-grid summaries weight the 24 conditions equally; per-condition tables are always written too | `e1_planted_grid.equal_weight_summary` |
| E8 | Coverage uncertainty: cluster bootstrap over datasets (neurons within a dataset are dependent), plus a Wilson interval for all-neuron coverage | `metrics.aggregate` |

## L — language-model phase

| ID | Assumption | Where |
|---|---|---|
| L1 | Tensor = input of `gpt_neox.layers[i].mlp.dense_4h_to_h` (post-GELU, width 3072 / 4096), verified against a manual forward pass | `lm/hooks.py::verify_hook` |
| L2 | Checkpoint revision pinned (`step143000`), fp32 by default, tokenizer from the same revision | `lm/hooks.load_model`, cache manifest |
| L3 | Environments vary only the sampling distribution; the six are declared in advance | `lm/corpus.py` |
| L4 | Splits by document / template family, 60/20/20, salted hash, so no family crosses splits | `lm/corpus.doc_split`, `lm/templates.split_by_family` |
| L5 | Token positions chosen by a predeclared rule, never by activation value | `lm/corpus.select_positions`, `target_position` |
| L6 | Candidate neurons chosen on the dev split; ablation evaluated on held-out test families; controls matched on dev magnitude and prevalence | `experiments/e9_causal_validation.py` |
| L7 | No LLM writes or labels items; the text banks are small and hand-written | `lm/templates.py` |

## G — engineering

| ID | Item |
|---|---|
| G1 | All randomness flows from `SeedBank(root).child(...)` namespaces; changing the method list never changes the data |
| G2 | Every run writes an immutable `manifest.json` (config hash, git commit, package versions). Rerunning a different config into the same directory fails |
| G3 | Joint diagonalisation = U-WEDGE with FFDiag step clamping and GEVD starts. Line search and L-BFGS were rejected after observed failures (see `jointdiag.py` docstring) |

## D — decisions where the sources were silent or in conflict

| ID | Decision | Alternative | Why |
|---|---|---|---|
| D1 | `v_min = 0.006` and generator `min_visibility = 0.012` (SD across environments of a factor's variance) | A relative threshold | Planted profiles have RMS ≈ 0.013–0.05. Sampling noise of an invisible component's band at `n = 5000` is ≈ 0.005, so 0.006 separates them with margin |
| D2 | Centred profiles (subtract the environment mean) rather than differences to a reference environment 0 | `C_e − C_0` (plan p.3) | Same span, but reference-free; symmetric in environments |
| D3 | Signal scale = average per-neuron latent-driven variance; noise SD = `noise_rel_sd × signal_scale` | Per-neuron signal scale | The plan asks only that the convention be declared |
| D4 | Rates: `well_separated` = i.i.d. uniform per (environment, factor); `near_collinear` = shared pattern + `η`-jitter | Other parameterisations | Gives one knob for diversity. Population diversity (min pairwise sine) is recorded per dataset |
| D5 | Activation-rate regimes centred at 0.05 / 0.15 / 0.30 with ±60 % spread | The plan's "0.05–0.30 varied by environment" | Spec-doc grid; the spread keeps visibility |
| D6 | DPMI baseline is an **approximation** ("DPMI-style"), not a verified reimplementation | Reimplement from the paper | The paper's exact formula was not available offline. Labelled in every output |
| D7 | Pythia-410M layers 4, 10, 16 (exact relative depth of 2, 5, 8 / 12); ablation at 10 | Spec doc suggested block 11 for 410M | The plan says choose analogous depths in advance by a rule. 11 is kept as an alternative config line |
| D8 | E4 uses `τ = 0.30` | `τ = 0.10` | In `d = 3` almost every grid direction has all entries ≥ 0.10, so counts would not vary |
| D9 | LM estimator applied to predeclared random neuron **blocks** of 32 | The full 3072-wide square model | The full model is intractable and implausible (`K ≫ d`). Block stationarity is checked by diagnostics, not assumed |
| D10 | Semantic decision task = forced choice between sense-consistent and inconsistent associates; unrelated task = subject–verb agreement with an attractor | Other tasks | Deterministic, auditable, family-split. **The banks are below the plan's 2,000 held-out items per task; expanding them is an open item** |
| D11 | E5 toy uses `τ = 0.30` | `τ = 0.10` | Trained toy columns are dense: at 0.10 nearly every neuron loads on every represented feature |
| D13 | `ismb_cum` weights every matrix in the joint-diagonalisation stack by the inverse of its estimated sampling-noise scale | Unit-norm weighting (as in the `jointdiag_cum` point estimator) | Unit-norm weighting gives pure-noise cumulant slices full weight when latents are near-Gaussian. Widths went to ≈ 15 of 16 where ISMB had ≈ 5 (lab notebook §12) |
| D14 | `ismb_cum` runs ISMB's second-moment pair, dependence and misfit tests on a second-moment refinement of the basis, warm-started from the combined solution | Run them on the combined basis | Those tests assume a basis fitted to the covariance stack. On the combined basis the misfit gate fired on every clean dataset (lab notebook §12) |
| D15 | Near-Gaussian stress `latent_gaussianity = g`: each latent becomes $\sqrt{1-g}$ × (original, centred) + $\sqrt g$ × an independent Gaussian of the same per-environment variance | Gaussian magnitudes; lower activity rates | Leaves variance profiles, loadings and $P_j$ exactly unchanged, and scales fourth cumulants by $(1-g)^2$. Isolates the information `ismb_cum` adds. Off by default, with no extra random draws |
| D12 | ISMB α split and `abstain_width = 1` (any non-singleton interval counts as abstention for exact-count claims) | Other splits | Predeclared; changing them requires a new preregistration entry |
