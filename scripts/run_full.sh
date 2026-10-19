#!/usr/bin/env bash
# Full-scale runs at the replicate counts and baseline budgets declared in each config
# (no pilot overrides). Shortest first. E1 and E2 checkpoint every condition under
# results/<exp>_full/_partial/, so rerunning this script resumes where it stopped.
# Roughly 18-24 h on 4 CPU cores, about 14-20 h of it E1.
set -euo pipefail
cd "$(dirname "$0")/.."
P=${P:-.venv/bin/polyid}
run() { echo "[run_full] $(date -u +%FT%TZ) start $1"; $P run "$2" --out "results/$1"; echo "[run_full] $(date -u +%FT%TZ) done $1"; }

run e10_full configs/planted/e10_adaptive_envs.yaml
run e5_full  configs/toy/e5_toy_superposition.yaml
run e6_full  configs/toy/e6_sae_comparison.yaml
run e3_full  configs/planted/e3_frequency_shift.yaml
run e4_full  configs/planted/e4_coverage_d3.yaml
run e2_full  configs/planted/e2_stress.yaml
run e1_full  configs/planted/e1_planted_grid.yaml
echo "[run_full] all done"
