#!/usr/bin/env bash
# E11 at full size (100 datasets per condition, about 2 h on 4 cores). Run after scripts/run_full.sh.
# Resumes from results/.running/e11_full/_partial and pushes the result when done (PUSH=0 to disable).
set -euo pipefail
cd "$(dirname "$0")/.."
P=${P:-.venv/bin/polyid}
if [ ! -f results/e11_full/e11_by_condition.csv ]; then
  $P run configs/planted/e11_cumulant_ismb.yaml --out results/.running/e11_full
  rm -rf results/e11_full && mv results/.running/e11_full results/e11_full
fi
if [ "${PUSH:-1}" = 1 ]; then
  git add results/e11_full
  git diff --cached --quiet || git commit -q -m "Full-scale results: e11_full (cumulant-aware ISMB, 100 datasets x 25 conditions)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Eh23LNGMGuEVsg8tLV7S5u"
  for wait in 2 4 8 16; do git push -q origin "$(git rev-parse --abbrev-ref HEAD)" && break; sleep "$wait"; done
fi
echo "[run_e11] done"
