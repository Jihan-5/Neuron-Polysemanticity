"""The observation container passed to every estimator.

``ObservedData`` is deliberately minimal: activations, environment labels and
optional cluster labels (e.g. source document IDs). It carries no latent
features, no loading matrix and no truth. Estimators receive only this object,
which is how the plan's leakage rule ("keep latent labels and planted A
unavailable to estimator code except in scoring") is enforced in code.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class ObservedData:
    X: np.ndarray
    env: np.ndarray
    cluster: np.ndarray | None = None
    neuron_ids: np.ndarray | None = None
    env_names: tuple[str, ...] | None = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        X = np.asarray(self.X)
        env = np.asarray(self.env)
        if X.ndim != 2:
            raise ValueError(f"X must be (N, d), got {X.shape}")
        if env.shape != (X.shape[0],):
            raise ValueError("env must have one label per row of X")
        if self.cluster is not None and np.asarray(self.cluster).shape != (X.shape[0],):
            raise ValueError("cluster must have one label per row of X")
        if env.min() < 0:
            raise ValueError("environment labels must be non-negative integers")
        if len(np.unique(env)) != env.max() + 1:
            raise ValueError("environment labels must be 0..E-1 with every environment present")

    @property
    def n(self) -> int:
        return int(self.X.shape[0])

    @property
    def d(self) -> int:
        return int(self.X.shape[1])

    @property
    def n_envs(self) -> int:
        return int(np.asarray(self.env).max() + 1)

    def env_counts(self) -> np.ndarray:
        return np.bincount(self.env, minlength=self.n_envs)

    def subset_rows(self, mask: np.ndarray) -> "ObservedData":
        mask = np.asarray(mask)
        env = self.env[mask]
        # relabel environments so that labels stay contiguous
        uniq, env_new = np.unique(env, return_inverse=True)
        names = None if self.env_names is None else tuple(self.env_names[u] for u in uniq)
        return ObservedData(
            X=self.X[mask],
            env=env_new,
            cluster=None if self.cluster is None else self.cluster[mask],
            neuron_ids=self.neuron_ids,
            env_names=names,
            meta=dict(self.meta),
        )

    def subset_envs(self, envs: list[int]) -> "ObservedData":
        return self.subset_rows(np.isin(self.env, envs))

    def subset_neurons(self, idx: np.ndarray) -> "ObservedData":
        ids = np.arange(self.d) if self.neuron_ids is None else self.neuron_ids
        return ObservedData(
            X=self.X[:, idx],
            env=self.env,
            cluster=self.cluster,
            neuron_ids=np.asarray(ids)[idx],
            env_names=self.env_names,
            meta=dict(self.meta),
        )

    def pooled(self) -> "ObservedData":
        """Same rows with a single environment label (for single-environment baselines)."""
        return ObservedData(
            X=self.X, env=np.zeros(self.n, dtype=int), cluster=self.cluster,
            neuron_ids=self.neuron_ids, env_names=("pooled",), meta=dict(self.meta),
        )
