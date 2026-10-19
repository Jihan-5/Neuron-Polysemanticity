import json
import subprocess
import sys

import numpy as np
import pandas as pd

from polyid.estimand import EstimandSpec
from polyid.estimators import build_estimator
from polyid.estimators.base import EstimateResult
from polyid.experiments.e0_theory_gate import two_explanations
from polyid.generators.planted import PlantedConfig, generate
from polyid.metrics import aggregate, score_result, support_recovery
from polyid.stats import holm, mcnemar_exact, paired_difference, wilson_interval
from polyid.utils.config import apply_overrides
from polyid.utils.seeding import SeedBank


def test_wilson_and_mcnemar_reference_values():
    lo, hi = wilson_interval(475, 500)
    assert 0.92 < lo < 0.95 < hi < 0.97
    # spec doc worked example: discordant 60 vs 25 -> exact two-sided p ~ 1.87e-4
    assert abs(mcnemar_exact(250, 60, 25, 165) - 0.000187) < 2e-5


def test_holm_monotone():
    adj = holm([0.01, 0.04, 0.03])
    assert np.all(adj >= np.array([0.01, 0.04, 0.03])) and adj[0] == 0.03


def test_paired_difference_clusters():
    a = np.array([1, 1, 0, 1, 0, 1.0])
    b = np.array([0, 1, 0, 0, 0, 1.0])
    r = paired_difference(a, b, np.array([0, 0, 1, 1, 2, 2]))
    assert r["n_clusters"] == 3 and abs(r["diff"] - 1 / 3) < 1e-12


def test_score_result_flags():
    res = EstimateResult(method="x", kind="interval", point=np.array([1.0, 2.0]), lower=np.array([1.0, 2.0]),
                         upper=np.array([1.0, 3.0]))
    rows = score_result(res, np.array([2, 2]), spec=EstimandSpec(), ids={"dataset": "d0"})
    assert rows[0]["confidently_wrong"] == 1.0 and rows[1]["covered"] == 1.0
    agg = aggregate(pd.DataFrame(rows), by=["method"])
    assert abs(agg["coverage"].iloc[0] - 0.5) < 1e-12


def test_support_recovery_perfect():
    A = np.eye(3) + 0.0
    r = support_recovery(A[:, [2, 0, 1]] * -1, A, 0.5)
    assert r["support_f1"] == 1.0


def test_all_registry_methods_run_on_small_data():
    spec = EstimandSpec()
    ds = generate(PlantedConfig(d=5, K=5, n_per_env=1500, n_envs=4), SeedBank(0))
    for name, cfg in [("moments", {}), ("gmm_bic", {"k_max": 3}), ("dpmi_style", {"k_max": 4}), ("pca", {}),
                      ("factor_analysis", {}), ("fastica", {}), ("jointdiag", {}), ("jointdiag_nofilter", {}),
                      ("dict_learning", {"max_iter": 5}), ("boot_percentile", {"n_boot": 10}),
                      ("sensitivity", {"n_restarts": 3, "n_perturb": 1}), ("ismb", {"n_boot": 10})]:
        res = build_estimator(name, cfg).fit(ds.observed, spec, np.random.default_rng(0))
        vals = res.point if res.point is not None else res.score
        assert vals.shape == (5,), name
        assert res.access is not None, name


def test_two_explanations_are_observationally_equivalent():
    ds = generate(PlantedConfig(d=4, K=4, collinear_groups=[[0, 1]], col_support_max=3), SeedBank(20261018).child("t"))
    out = two_explanations(ds.oracle, [0, 1], 0.1, n_angles=720)
    assert out["max_abs_covariance_difference_between_explanations"] < 1e-12


def test_overrides_parse_yaml_scalars():
    cfg = apply_overrides({"a": {"b": 1}}, ["a.b=2.5", "c=[1,2]", "d=true"])
    assert cfg == {"a": {"b": 2.5}, "c": [1, 2], "d": True}


def test_cli_list_and_dry_run(tmp_path):
    out = subprocess.run([sys.executable, "-m", "polyid.cli", "list"], capture_output=True, text=True, check=True)
    assert "e0_theory_gate" in out.stdout and "e10_adaptive_envs" in out.stdout
    out = subprocess.run([sys.executable, "-m", "polyid.cli", "run", "configs/planted/e1_planted_grid.yaml",
                          "--dry-run", "reps=3"], capture_output=True, text=True, check=True)
    cfg = json.loads(out.stdout)
    assert cfg["reps"] == 3 and cfg["generator"]["d"] == 16 and cfg["experiment"] == "e1_planted_grid"
