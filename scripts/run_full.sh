#!/usr/bin/env bash
# Full-scale runs at the replicate counts and baseline budgets declared in each config
# (no pilot overrides). Shortest first. Safe to rerun at any time:
#   * an experiment whose summary file exists is skipped;
#   * E1, E2 and E4 checkpoint every condition under results/<exp>_full/_partial/ and resume;
#   * each finished experiment is committed and pushed immediately (PUSH=0 to disable).
# Roughly a day on 4 CPU cores, about 20 h of it E1.
set -euo pipefail
cd "$(dirname "$0")/.."
P=${P:-.venv/bin/polyid}
PUSH=${PUSH:-1}
BRANCH=$(git rev-parse --abbrev-ref HEAD)

publish() {  # commit and push one result directory
  [ "$PUSH" = 1 ] || return 0
  git add "results/$1"
  git diff --cached --quiet && return 0
  git commit -q -m "Full-scale results: $1

Run at the replicate counts and baseline budgets declared in its config.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Eh23LNGMGuEVsg8tLV7S5u"
  for wait in 2 4 8 16; do git push -q origin "$BRANCH" && return 0; sleep "$wait"; done
  echo "[run_full] push failed for $1 (committed locally)"
}

run() {  # name, config, summary file that marks completion
  if [ -f "results/$1/$3" ]; then echo "[run_full] skip $1 (done)"; publish "$1"; return; fi
  echo "[run_full] $(date -u +%FT%TZ) start $1"
  $P run "$2" --out "results/$1"
  echo "[run_full] $(date -u +%FT%TZ) done $1"
  publish "$1"
}

run e10_full configs/planted/e10_adaptive_envs.yaml   e10_summary.csv
run e5_full  configs/toy/e5_toy_superposition.yaml    e5_summary.csv
run e6_full  configs/toy/e6_sae_comparison.yaml       e6_summary.csv
run e3_full  configs/planted/e3_frequency_shift.yaml  e3_stability.csv
run e4_full  configs/planted/e4_coverage_d3.yaml      e4_summary.csv
run e2_full  configs/planted/e2_stress.yaml           e2_by_condition.csv
run e1_full  configs/planted/e1_planted_grid.yaml     e1_equal_weight_summary.csv
echo "[run_full] all done"
