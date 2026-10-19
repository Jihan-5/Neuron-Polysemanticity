import numpy as np
import pytest

from polyid.estimand import EstimandSpec
from polyid.generators.planted import PlantedConfig, generate
from polyid.utils.seeding import SeedBank


@pytest.fixture
def spec():
    return EstimandSpec(tau=0.10, v_min=0.006, max_abs_cos=0.98)


@pytest.fixture
def small_clean(spec):
    cfg = PlantedConfig(d=6, K=6, n_envs=5, n_per_env=20000, noise_rel_sd=0.05)
    return generate(cfg, SeedBank(11).child("test", "small_clean"), v_min=spec.v_min)


@pytest.fixture
def small_collinear(spec):
    cfg = PlantedConfig(d=6, K=6, n_envs=5, n_per_env=20000, noise_rel_sd=0.05, collinear_groups=[[0, 1]])
    return generate(cfg, SeedBank(11).child("test", "small_collinear"), v_min=spec.v_min)


def pytest_collection_modifyitems(config, items):
    try:
        import torch  # noqa: F401
    except ImportError:
        skip = pytest.mark.skip(reason="torch not installed")
        for it in items:
            if "torch" in it.keywords or "lm" in it.keywords:
                it.add_marker(skip)
    try:
        import transformers  # noqa: F401
    except ImportError:
        skip = pytest.mark.skip(reason="transformers not installed")
        for it in items:
            if "lm" in it.keywords:
                it.add_marker(skip)
