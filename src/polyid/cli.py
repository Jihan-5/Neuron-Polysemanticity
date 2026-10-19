"""Command-line entry point.

Examples
--------
    polyid list
    polyid run configs/planted/e0_theory_gate.yaml --out results/e0
    polyid run configs/planted/e1_planted_grid.yaml --out results/e1_pilot reps=10 dev_reps=5
    polyid run configs/planted/e4_coverage_d3.yaml --out results/e4 --dry-run
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

from polyid.experiments import EXPERIMENTS
from polyid.utils.config import load_config
from polyid.utils.manifest import build_manifest, write_manifest


def _cmd_list(_args) -> int:
    for name, mod in EXPERIMENTS.items():
        doc = (importlib.import_module(mod).__doc__ or "").strip().splitlines()
        print(f"{name:26s} {doc[0] if doc else ''}")
    return 0


def _cmd_run(args) -> int:
    cfg = load_config(args.config, args.overrides)
    exp = cfg.get("experiment")
    if exp not in EXPERIMENTS:
        print(f"config must set experiment to one of {sorted(EXPERIMENTS)}", file=sys.stderr)
        return 2
    out = Path(args.out or f"results/{exp}")
    if args.dry_run:
        print(json.dumps(cfg, indent=2))
        return 0
    manifest = build_manifest(cfg, experiment=exp, seeds={"root": cfg.get("seed")})
    write_manifest(out, manifest)
    t0 = time.time()
    mod = importlib.import_module(EXPERIMENTS[exp])
    summary = mod.run(cfg, out)
    print(f"[polyid] {exp} finished in {time.time() - t0:.1f}s -> {out}")
    if isinstance(summary, dict):
        keys = [k for k in summary if k != "figures"]
        print(f"[polyid] summary keys: {keys}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="polyid", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="list experiments").set_defaults(func=_cmd_list)
    r = sub.add_parser("run", help="run an experiment from a YAML config")
    r.add_argument("config")
    r.add_argument("--out", default=None)
    r.add_argument("--dry-run", action="store_true", help="print the resolved config and exit")
    r.add_argument("overrides", nargs="*", help="dotted overrides, e.g. reps=10 generator.n_per_env=2000")
    r.set_defaults(func=_cmd_run)
    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
