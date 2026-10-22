"""Name -> constructor registry so experiment configs can list methods declaratively."""

from __future__ import annotations

from typing import Callable

from polyid.estimators import baselines as B

_LAZY: dict[str, Callable] = {}


def _ismb(cfg):
    from polyid.bounds.ismb import ISMB

    return ISMB(cfg)


def _ismb_cum(cfg):
    from polyid.bounds.ismb import ISMB

    return ISMB({**(cfg or {}), "cumulants": True})


def _ismb_oc(cfg):
    from polyid.bounds.ismb import ISMB

    return ISMB({**(cfg or {}), "oc_gate": True})


def _ismb_cum_oc(cfg):
    from polyid.bounds.ismb import ISMB

    return ISMB({**(cfg or {}), "cumulants": True, "oc_gate": True})


def _ismb_cum_v2(cfg):
    from polyid.bounds.ismb import ISMB

    return ISMB({**(cfg or {}), "cumulants": True, "oc_gate": True, "stab_widen": True, "dep_ambiguous": True,
                 "name": "ismb_cum_v2"})


def _sensitivity(cfg):
    from polyid.bounds.sensitivity import SensitivityRange

    return SensitivityRange(cfg)


def _ar_grid(cfg):
    from polyid.bounds.ar_grid import ARGridBounds

    return ARGridBounds(cfg)


def _topk_sae(cfg):
    from polyid.toy.sae import TopKSAECount

    return TopKSAECount(cfg)


def _jd_cum(cfg):
    from polyid.estimators.cumulant import JointDiagCumulant

    return JointDiagCumulant(cfg)


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
    "jointdiag_cum": _jd_cum,
    "ismb": _ismb,
    "ismb_cum": _ismb_cum,
    "ismb_oc": _ismb_oc,
    "ismb_cum_oc": _ismb_cum_oc,
    "ismb_cum_v2": _ismb_cum_v2,
    "sensitivity": _sensitivity,
    "ar_grid": _ar_grid,
    "topk_sae": _topk_sae,
}


def build_estimator(name: str, cfg: dict | None = None):
    if name not in REGISTRY:
        raise KeyError(f"unknown method {name!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[name](cfg or {})
