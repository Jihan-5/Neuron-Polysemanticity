"""Shared machinery for planted-data experiments.

* ``expand_grid``       — Cartesian product of grid axes into named conditions.
* ``run_planted_condition`` — draw ``reps`` test datasets (and ``dev_reps``
  development datasets for score calibration), run every method, score
  against the oracle, return tidy rows. Replicates run in parallel processes.

Seed discipline: dataset ``r`` of condition ``c`` uses
``SeedBank(seed).child(experiment, c, "test", r)``; development datasets use
the ``"dev"`` split; method randomness uses a separate child of the dataset
node. Dev and test datasets therefore never coincide, and changing the
method list does not change the data.
"""

from __future__ import annotations

import itertools
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from polyid.estimand import EstimandSpec
from polyid.estimators import build_estimator
from polyid.estimators.baselines import DevCalibratedCountMap
from polyid.generators.planted import PlantedConfig, generate
from polyid.identification import identified_set
from polyid.metrics import score_result
from polyid.utils.config import deep_merge
from polyid.utils.seeding import SeedBank, assert_disjoint_splits


def spec_from_cfg(cfg: dict) -> EstimandSpec:
    return EstimandSpec(**(cfg.get("spec") or {}))


def expand_grid(base: dict, grid: dict | None) -> list[tuple[str, dict]]:
    """``grid`` maps generator keys to lists; returns ``[(condition_id, generator_cfg)]``."""
    if not grid:
        return [("base", dict(base))]
    keys = sorted(grid)
    out = []
    for vals in itertools.product(*(grid[k] for k in keys)):
        g = dict(base)
        parts = []
        for k, v in zip(keys, vals):
            g[k] = v
            parts.append(f"{k}={v}")
        out.append(("|".join(parts), g))
    return out


def named_conditions(base: dict, conditions: dict) -> list[tuple[str, dict]]:
    return [(name, deep_merge(base, over or {})) for name, over in conditions.items()]


def _one_dataset(exp: str, cond_id: str, gen_cfg: dict, methods: dict, spec: EstimandSpec, seed: int,
                 split: str, rep: int, taus: list[float]) -> dict:
    bank = SeedBank(seed).child(exp, cond_id, split, rep)
    pc = PlantedConfig.from_dict(gen_cfg)
    ds = generate(pc, bank, v_min=spec.v_min)
    out = {"rows": [], "datasets": [], "scores": []}
    ident = identified_set(ds.oracle, spec)
    ds_row = {"experiment": exp, "condition": cond_id, "split": split, "rep": rep,
              "dataset": f"{cond_id}#{split}{rep}", "diversity": ds.oracle.diversity,
              "independent": ds.oracle.independent, "noise_stationary": ds.oracle.noise_stationary,
              "truth_mean": float(ds.oracle.truth_at(spec.tau).mean()), **{f"id_{k}": v for k, v in ident.summary().items()}}
    for name, mcfg in methods.items():
        t0 = time.time()
        try:
            est = build_estimator(name, mcfg)
            res = est.fit(ds.observed, spec, bank.child("method", name).rng())
        except Exception as exc:  # a failing method is a result, not a crash
            ds_row[f"{name}_error"] = f"{type(exc).__name__}: {exc}"
            ds_row[f"{name}_traceback"] = traceback.format_exc(limit=3)
            continue
        ds_row[f"{name}_seconds"] = time.time() - t0
        for key, val in (res.diagnostics or {}).items():
            if isinstance(val, (int, float, bool, np.integer, np.floating)):
                ds_row[f"{name}__{key}"] = val
        ids = {"experiment": exp, "condition": cond_id, "split": split, "rep": rep, "dataset": ds_row["dataset"]}
        out["rows"].extend(score_result(res, ds.oracle.truth_at(spec.tau), spec=spec, A_true=ds.oracle.A,
                                        visible_true=ds.oracle.visible, idset=(ident.lower, ident.upper), ids=ids))
        if res.kind == "score" or (res.score is not None and res.kind == "count"):
            out["scores"].append({"method": name, "score": res.score, "truth": ds.oracle.truth_at(spec.tau), **ids})
        # threshold sensitivity for count methods with loadings
        if res.loadings is not None and res.point is not None and res.kind == "count":
            for t in taus:
                if t == spec.tau:
                    continue
                from polyid.estimand import multiplicity

                truth_t = ds.oracle.truth_at(t)
                pt = multiplicity(res.loadings, t, visible=None)
                for j in range(len(truth_t)):
                    out["rows"].append({**ids, "method": f"{name}@tau={t}", "kind": "count", "neuron": j,
                                        "truth": int(truth_t[j]), "point": float(pt[j]),
                                        "abs_err": abs(float(pt[j]) - truth_t[j]),
                                        "signed_err": float(pt[j]) - truth_t[j]})
    out["datasets"].append(ds_row)
    return out


def run_planted_condition(exp: str, cond_id: str, gen_cfg: dict, methods: dict, spec: EstimandSpec, *,
                          seed: int, reps: int, dev_reps: int = 0, n_jobs: int = -1, taus: list[float] | None = None
                          ) -> tuple[list[dict], list[dict]]:
    taus = taus or [spec.tau]
    assert_disjoint_splits(SeedBank(seed).child(exp, cond_id, "dev", 0), SeedBank(seed).child(exp, cond_id, "test", 0))
    jobs = [("test", r) for r in range(reps)] + [("dev", r) for r in range(dev_reps)]
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_one_dataset)(exp, cond_id, gen_cfg, methods, spec, seed, split, rep, taus) for split, rep in jobs
    )
    rows = [r for res in results for r in res["rows"]]
    datasets = [r for res in results for r in res["datasets"]]
    scores = [s for res in results for s in res["scores"]]
    # development-split calibration of score-only methods (never fitted on test data)
    for name in {s["method"] for s in scores}:
        dev = [s for s in scores if s["method"] == name and s["split"] == "dev"]
        test = [s for s in scores if s["method"] == name and s["split"] == "test"]
        if not dev or not test:
            continue
        k_max = int(max(np.max(s["truth"]) for s in dev + test))
        cmap = DevCalibratedCountMap(k_max).fit(np.concatenate([s["score"] for s in dev]),
                                                np.concatenate([s["truth"] for s in dev]))
        for s in test:
            pred = cmap.predict(s["score"])
            for j in range(len(pred)):
                rows.append({"experiment": exp, "condition": cond_id, "split": "test", "rep": s["rep"],
                             "dataset": s["dataset"], "method": f"{name}+devcal", "kind": "count", "neuron": j,
                             "truth": int(s["truth"][j]), "point": float(pred[j]),
                             "abs_err": abs(float(pred[j]) - s["truth"][j]),
                             "signed_err": float(pred[j]) - s["truth"][j]})
    rows = [r for r in rows if r.get("split") == "test"]
    return rows, datasets


def save_tables(out_dir: Path, name: str, rows: list[dict] | pd.DataFrame) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    path = out_dir / f"{name}.csv"
    df.to_csv(path, index=False)
    return path


def get_methods(cfg: dict) -> dict[str, Any]:
    return {k: (v or {}) for k, v in (cfg.get("methods") or {}).items()}
