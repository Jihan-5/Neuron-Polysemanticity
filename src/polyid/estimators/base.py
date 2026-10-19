"""Common estimator interface and result container.

Every method — baseline or proposed — returns an :class:`EstimateResult` and
declares an :class:`Access` record, so that the comparison-fairness rules
of the plan (p.10: "match observations, environment access, model-selection
budget, and preprocessing ... information advantage being disclosed") are
auditable from the results table itself.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Protocol

import numpy as np

from polyid.data import ObservedData
from polyid.estimand import EstimandSpec


@dataclass
class Access:
    env_labels: bool
    higher_moments: bool
    n_observations: int
    model_selection_budget: int = 0
    restarts: int = 1
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EstimateResult:
    """Per-neuron output of one method on one dataset.

    ``kind`` is one of:

    * ``"count"``    — ``point`` is an integer multiplicity estimate;
    * ``"score"``    — ``score`` is a polysemanticity score on an arbitrary scale
      (count MAE only after development-split calibration);
    * ``"interval"`` — ``lower``/``upper`` bound P_j; ``point`` optional;
      ``label`` says whether it is a confidence, identified-set or sensitivity
      object.
    """

    method: str
    kind: str
    point: np.ndarray | None = None
    lower: np.ndarray | None = None
    upper: np.ndarray | None = None
    score: np.ndarray | None = None
    loadings: np.ndarray | None = None
    label: str = ""
    access: Access | None = None
    diagnostics: dict = field(default_factory=dict)

    def as_rows(self, **ids) -> list[dict]:
        d = len(next(x for x in (self.point, self.lower, self.score) if x is not None))
        rows = []
        for j in range(d):
            r = {"method": self.method, "kind": self.kind, "label": self.label, "neuron": j, **ids}
            if self.point is not None:
                r["point"] = float(self.point[j])
            if self.lower is not None:
                r["lower"] = float(self.lower[j])
                r["upper"] = float(self.upper[j])
            if self.score is not None:
                r["score"] = float(self.score[j])
            rows.append(r)
        return rows


class Estimator(Protocol):
    name: str

    def fit(self, data: ObservedData, spec: EstimandSpec, rng: np.random.Generator) -> EstimateResult:
        ...
