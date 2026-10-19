r"""Toy model of superposition (Elhage et al. 2022, syllabus S02) and its ground truth.

Recipe (plan p.6)
-----------------
* input ``x in R^K`` with ``K = 32`` known features, ``x_k = b_k u_k``,
  ``b_k ~ Bernoulli(p)``, ``u_k ~ U[0, 1]``;
* hidden width ``d = 8`` or ``16``: ``h = W x``;
* reconstruction ``x_hat = ReLU(W^T h + b)``;
* loss ``sum_k I_k (x_k - x_hat_k)^2`` with importance ``I_k = decay^k``;
* Adam, lr 1e-3, batch 512, 10 000 steps; a *separate validation generator*
  every 250 steps; record reconstruction and feature-representation stability
  (cosine between successive normalised ``W``). Five seeds per condition.

Ground truth after training (plan p.6 "Ground truth after training")
--------------------------------------------------------------------
"Distinguish the number of input features from features actually
represented." A feature is **represented** iff ``||W_:,k|| >= r_min``.
When the trained model is probed with environments that shift feature
frequencies, the latent ``z_k = ||W_:,k|| x_k`` has variance
``v_{e,k} = ||W_:,k||^2 Var_e(x_k)``; the feature is **visible** iff the
across-environment SD of ``v_{e,k}`` is at least ``v_min`` (the same rule as
the estimand). Then

.. math:: P_j = \#\{k \text{ represented and visible}: |W_{jk}|/\|W_{:,k}\| \ge \tau\}.

The hidden layer of this toy has no privileged basis (the loss is invariant
to rotations of ``h``); the target is therefore defined in the realised basis
of the trained ``W`` and the estimator sees ``h`` in that same basis — a
statement about *that* basis, exactly as the plan requires.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from polyid.data import ObservedData

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None


@dataclass
class ToyConfig:
    K: int = 32
    d: int = 8
    feature_prob: float = 0.1
    importance_decay: float = 0.9
    lr: float = 1e-3
    batch_size: int = 512
    steps: int = 10000
    eval_every: int = 250
    val_size: int = 8192
    patience_evals: int = 12  # stop when val loss and W-stability plateau for this many evals
    min_steps: int = 2000
    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def sample_features(n: int, K: int, prob, generator) -> "torch.Tensor":
    prob = torch.as_tensor(prob, dtype=torch.float32).expand(K)
    on = torch.rand(n, K, generator=generator) < prob
    return on.float() * torch.rand(n, K, generator=generator)


if nn is not None:

    class ToyModel(nn.Module):
        def __init__(self, K: int, d: int, generator=None):
            super().__init__()
            W = torch.empty(d, K)
            nn.init.xavier_normal_(W, generator=generator)
            self.W = nn.Parameter(W)
            self.b = nn.Parameter(torch.zeros(K))

        def hidden(self, x):
            return x @ self.W.T

        def forward(self, x):
            return torch.relu(self.hidden(x) @ self.W + self.b)


def train_toy(cfg: ToyConfig) -> dict:
    if torch is None:
        raise ImportError("PyTorch is required for the toy superposition model")
    g = torch.Generator().manual_seed(cfg.seed)
    gval = torch.Generator().manual_seed(cfg.seed + 10_000_019)  # separate validation stream
    model = ToyModel(cfg.K, cfg.d, generator=g)
    imp = torch.tensor([cfg.importance_decay**k for k in range(cfg.K)], dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    xval = sample_features(cfg.val_size, cfg.K, cfg.feature_prob, gval)
    hist = []
    prev_W = None
    best, stall = np.inf, 0
    for step in range(1, cfg.steps + 1):
        x = sample_features(cfg.batch_size, cfg.K, cfg.feature_prob, g)
        loss = ((imp * (x - model(x)) ** 2).sum(dim=1)).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % cfg.eval_every == 0 or step == cfg.steps:
            with torch.no_grad():
                vl = float(((imp * (xval - model(xval)) ** 2).sum(dim=1)).mean())
                W = model.W.detach().numpy().copy()
                Wn = W / np.maximum(np.linalg.norm(W, axis=0, keepdims=True), 1e-12)
                stab = float(np.mean(np.sum(Wn * prev_W, axis=0))) if prev_W is not None else np.nan
                prev_W = Wn
            hist.append({"step": step, "train_loss": float(loss.detach()), "val_loss": vl, "W_stability": stab})
            if vl < best * (1 - 1e-3):
                best, stall = vl, 0
            else:
                stall += 1
            if step >= cfg.min_steps and stall >= cfg.patience_evals and np.isfinite(stab) and stab > 0.999:
                break
    return {"W": model.W.detach().numpy().copy(), "b": model.b.detach().numpy().copy(), "history": hist,
            "config": cfg.to_dict(), "final_val_loss": hist[-1]["val_loss"], "steps_run": hist[-1]["step"]}


def feature_variances(rates: np.ndarray) -> np.ndarray:
    """``Var(x_k)`` for ``x_k = b u``, ``u ~ U[0,1]``: ``p/3 - p^2/4``."""
    p = np.asarray(rates, dtype=float)
    return p / 3.0 - p**2 / 4.0


def toy_truth(W: np.ndarray, rates: np.ndarray, tau: float, v_min: float, r_min: float = 0.1) -> dict:
    norms = np.linalg.norm(W, axis=0)
    represented = norms >= r_min
    v = feature_variances(rates) * norms[None, :] ** 2  # E x K variances of z_k = ||W_k|| x_k
    vis = np.sqrt(np.mean((v - v.mean(axis=0)) ** 2, axis=0))
    visible = vis >= v_min
    Wn = W / np.maximum(norms[None, :], 1e-12)
    S = (np.abs(Wn) >= tau) & (represented & visible)[None, :]
    return {"truth": S.sum(axis=1).astype(int), "represented": represented, "visible": visible,
            "n_represented": int(represented.sum()), "n_target": int((represented & visible).sum()),
            "A": Wn[:, represented & visible], "norms": norms}


def toy_observations(W: np.ndarray, rates: np.ndarray, n_per_env: int, rng: np.random.Generator,
                     noise_sd: float = 0.0) -> ObservedData:
    """Probe the *frozen* hidden layer with environments that shift feature frequencies."""
    E, K = rates.shape
    Xs, envs = [], []
    for e in range(E):
        x = (rng.random((n_per_env, K)) < rates[e]) * rng.random((n_per_env, K))
        h = x @ W.T
        if noise_sd > 0:
            h = h + rng.normal(0.0, noise_sd, size=h.shape)
        Xs.append(h)
        envs.append(np.full(n_per_env, e))
    return ObservedData(X=np.concatenate(Xs), env=np.concatenate(envs), neuron_ids=np.arange(W.shape[0]),
                        meta={"source": "toy_superposition"})
