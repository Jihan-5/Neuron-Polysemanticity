"""Name -> constructor registry so experiment configs can list methods declaratively."""

from __future__ import annotations

from typing import Callable

from polyid.estimators import baselines as B

_LAZY: dict[str, Callable] = {}


def _ismb(cfg):
    from polyid.bounds.ismb import ISMB

    return ISMB(cfg)


def _sensitivity(cfg):
    from polyid.bounds.sensitivity import SensitivityRange

    return SensitivityRange(cfg)


def _ar_grid(cfg):
    from polyid.bounds.ar_grid import ARGridBounds

    return ARGridBounds(cfg)


def _topk_sae(cfg):
    from polyid.toy.sae import TopKSAECount

    return TopKSAECount(cfg)


REGISTRY: dict[str, Callable] = {
    "moments": B.UnivariateMoments,
    "gmm_bic": B.GMMBIC,
    "dpmi_style": B.DPMIStyle,
    "pca": B.PCACount,
    "factor_analysis": B.FactorAnalysisCount,
    "fastica": B.FastICACount,
    "jointdiag": B.JointDiagCount,
    "jointdiag_nofilter": lambda cfg=None: B.JointDiagCount({**(cfg or {}), "visibility_filter": False}),
    "dict_learning": B.DictLearningCount,
    "boot_percentile": B.BootstrapPercentile,
    "ismb": _ismb,
    "sensitivity": _sensitivity,
    "ar_grid": _ar_grid,
    "topk_sae": _topk_sae,
}


def build_estimator(name: str, cfg: dict | None = None):
    if name not in REGISTRY:
        raise KeyError(f"unknown method {name!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[name](cfg or {})
