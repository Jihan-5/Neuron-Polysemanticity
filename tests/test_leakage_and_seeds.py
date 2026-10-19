"""Leakage rules (plan p.10) and seed separation (plan p.5), enforced structurally."""

import ast
from pathlib import Path

import numpy as np
import pytest

from polyid.generators.planted import PlantedConfig, generate
from polyid.utils.seeding import SeedBank, assert_disjoint_splits

PKG = Path(__file__).resolve().parents[1] / "src" / "polyid"
ESTIMATOR_DIRS = [PKG / "estimators", PKG / "bounds"]
FORBIDDEN_MODULES = ("polyid.generators", "polyid.identification", "polyid.metrics")
FORBIDDEN_NAMES = {"oracle", "Oracle", "latents", "A_true"}


def _py_files():
    for d in ESTIMATOR_DIRS:
        yield from d.glob("*.py")
    yield PKG / "jointdiag.py"
    yield PKG / "moments.py"


@pytest.mark.parametrize("path", list(_py_files()), ids=lambda p: p.name)
def test_estimators_cannot_see_the_oracle(path):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert not node.module.startswith(FORBIDDEN_MODULES), f"{path.name} imports {node.module}"
        if isinstance(node, ast.Import):
            for a in node.names:
                assert not a.name.startswith(FORBIDDEN_MODULES), f"{path.name} imports {a.name}"
        if isinstance(node, ast.Name):
            assert node.id not in FORBIDDEN_NAMES, f"{path.name} uses name {node.id}"
        if isinstance(node, ast.Attribute):
            assert node.attr not in {"oracle", "latents"}, f"{path.name} touches .{node.attr}"


def test_observed_data_carries_no_truth():
    ds = generate(PlantedConfig(d=4, K=4, n_per_env=100, store_latents=True), SeedBank(0))
    fields = set(vars(ds.observed))
    assert fields <= {"X", "env", "cluster", "neuron_ids", "env_names", "meta"}
    assert "A" not in ds.observed.meta and "truth" not in ds.observed.meta


def test_same_path_same_stream_and_dev_test_differ():
    a = SeedBank(5).child("e1", "c", "test", 0)
    assert np.array_equal(a.rng().random(5), SeedBank(5).child("e1", "c", "test", 0).rng().random(5))
    dev = SeedBank(5).child("e1", "c", "dev", 0)
    assert not np.array_equal(a.rng().random(5), dev.rng().random(5))
    assert_disjoint_splits(dev, a)
    with pytest.raises(ValueError):
        assert_disjoint_splits(a, SeedBank(5).child("e1", "c", "test", 0))


def test_dev_and_test_datasets_differ():
    cfg = PlantedConfig(d=4, K=4, n_per_env=200)
    d1 = generate(cfg, SeedBank(1).child("x", "dev", 0))
    d2 = generate(cfg, SeedBank(1).child("x", "test", 0))
    assert not np.allclose(d1.observed.X, d2.observed.X)


def test_generation_is_reproducible():
    cfg = PlantedConfig(d=4, K=4, n_per_env=300)
    a = generate(cfg, SeedBank(2).child("r"))
    b = generate(cfg, SeedBank(2).child("r"))
    assert np.array_equal(a.observed.X, b.observed.X) and np.array_equal(a.oracle.A, b.oracle.A)
