#!/usr/bin/env bash
# Reproduce every CPU pilot reported in the README. Pilot outputs are labelled as such.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/polyid}
$PY run configs/planted/e0_theory_gate.yaml      --out results/e0_theory_gate
$PY run configs/planted/e4_coverage_d3.yaml      --out results/e4_coverage_d3_pilot reps=100
$PY run configs/planted/e1_planted_grid.yaml     --out results/e1_pilot reps=8 dev_reps=4 pilot=true
$PY run configs/planted/e2_stress.yaml           --out results/e2_pilot reps=12 dev_reps=6
$PY run configs/planted/e3_frequency_shift.yaml  --out results/e3_pilot reps=15
$PY run configs/planted/e10_adaptive_envs.yaml   --out results/e10_pilot reps=30
$PY run configs/toy/e5_toy_superposition.yaml    --out results/e5_pilot seeds=2 steps=4000
$PY run configs/toy/e6_sae_comparison.yaml       --out results/e6_pilot reps=4 max_steps=3000
$PY run configs/lm/e8_smoke.yaml                 --out results/smoke/e8
$PY run configs/lm/e9_smoke.yaml                 --out results/smoke/e9
