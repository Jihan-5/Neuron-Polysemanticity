#!/usr/bin/env bash
# Reproduce every CPU pilot reported in the README (4 cores, roughly 1.5 h). Outputs are labelled pilot.
# Baseline row budgets are reduced *for pilots only* via overrides; the full-study configs are unchanged.
set -euo pipefail
cd "$(dirname "$0")/.."
P=${P:-.venv/bin/polyid}
FAST="methods.gmm_bic.max_rows=1500 methods.gmm_bic.n_init=1 methods.dpmi_style.max_rows=1000"
DL="methods.dict_learning.max_iter=10 methods.dict_learning.max_rows=4000"

[ -f results/e0_theory_gate/e0_summary.json ] || $P run configs/planted/e0_theory_gate.yaml --out results/e0_theory_gate
$P run configs/planted/e4_coverage_d3.yaml     --out results/e4_coverage_d3_pilot reps=100
$P run configs/planted/e1_planted_grid.yaml    --out results/e1_pilot reps=6 dev_reps=3 pilot=true $FAST $DL
$P run configs/planted/e2_stress.yaml          --out results/e2_pilot reps=10 dev_reps=4 $FAST
$P run configs/planted/e3_frequency_shift.yaml --out results/e3_pilot reps=12 $FAST $DL
$P run configs/planted/e10_adaptive_envs.yaml  --out results/e10_pilot reps=30
$P run configs/toy/e5_toy_superposition.yaml   --out results/e5_pilot seeds=2 steps=4000 methods.gmm_bic.max_rows=1500 methods.gmm_bic.n_init=1
$P run configs/toy/e6_sae_comparison.yaml      --out results/e6_pilot reps=4 max_steps=3000
$P run configs/lm/e8_smoke.yaml                --out results/smoke/e8
$P run configs/lm/e9_smoke.yaml                --out results/smoke/e9
