#!/usr/bin/env bash
# E13 — out-of-sample check of ismb_cum_v2 on fresh datasets (new seed), 100 x 25 conditions.
# Resumes from checkpoints; commits and pushes when done (PUSH=0 to disable).
set -euo pipefail
cd "$(dirname "$0")/.."
P=${P:-.venv/bin/polyid}
N=e13_ismb_cum_v2
if [ ! -f "results/$N/e11_by_condition.csv" ]; then
  $P run configs/planted/e13_ismb_cum_v2.yaml --out "results/.running/$N"
  rm -rf "results/$N" && mv "results/.running/$N" "results/$N"
fi
if [ "${PUSH:-1}" = 1 ]; then
  git add "results/$N"
  git diff --cached --quiet || git commit -q -m "Results: $N (ismb_cum vs ismb_cum_v2 on fresh datasets, 100 x 25)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Eh23LNGMGuEVsg8tLV7S5u"
  for wait in 2 4 8 16; do git push -q origin "$(git rev-parse --abbrev-ref HEAD)" && break; sleep "$wait"; done
fi
echo "[run_e13] done"
