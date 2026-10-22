#!/usr/bin/env bash
# E12 — overcompleteness gate: toys at n = 5000 and 20000 per environment, then the planted
# false-alarm check on E11's datasets. Each part runs into results/.running/<name>, moves to
# results/<name> when finished, and is committed and pushed (PUSH=0 to disable). Finished parts
# are skipped, and the planted part resumes from its checkpoints.
set -euo pipefail
cd "$(dirname "$0")/.."
P=${P:-.venv/bin/polyid}

publish() {
  [ "${PUSH:-1}" = 1 ] || return 0
  git add "results/$1"
  git diff --cached --quiet || git commit -q -m "Results: $1 ($2)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01Eh23LNGMGuEVsg8tLV7S5u"
  for wait in 2 4 8 16; do git push -q origin "$(git rev-parse --abbrev-ref HEAD)" && break; sleep "$wait"; done
}

run() {  # name config done-file description
  if [ -f "results/$1/$3" ]; then echo "[run_e12] skip $1 (done)"; return; fi
  echo "[run_e12] $(date -u +%FT%TZ) start $1"
  $P run "$2" --out "results/.running/$1"
  rm -rf "results/$1" && mv "results/.running/$1" "results/$1"
  echo "[run_e12] $(date -u +%FT%TZ) done $1"
  publish "$1" "$4"
}

run e12_oc_gate_toys configs/toy/e12_oc_gate_toys.yaml e5_summary.csv "overcompleteness gate on E5 toys, n=5000/env"
run e12_oc_gate_toys_n20000 configs/toy/e12_oc_gate_toys_n20000.yaml e5_summary.csv "overcompleteness gate on E5 toys, n=20000/env"
run e12_oc_gate_planted configs/planted/e12_oc_gate_planted.yaml e11_by_condition.csv "gate false alarms on E11 datasets, 100 x 25"
echo "[run_e12] all done"
