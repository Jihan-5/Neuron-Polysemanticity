r"""TopK sparse autoencoder baseline (Gao et al. 2024, syllabus S12).

Plan p.6 "SAE comparison": train a small TopK SAE on the same activation
split, dictionary sizes ``2d`` and ``4d`` and several fixed activity budgets,
MSE objective, AdamW from 1e-3, validation early stopping; compare
reconstruction, sparsity and feature stability across >= 3 initialisations.
"Do not interpret SAE agreement as ground truth" (Paulo & Belrose 2025, X3).

Mapping dictionary atoms back to neurons (syllabus S12 "define how
dictionary atoms map back to the original neuron coordinates"): decoder
column ``a`` is a direction in neuron space; normalise it and count, for
neuron ``j``, the *alive* atoms with ``|W_dec[j, a]| >= tau``. A larger
dictionary can split one direction into several atoms, which inflates this
count — reported, not corrected, because that is the baseline's behaviour.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from polyid.data import ObservedData
from polyid.estimand import EstimandSpec, multiplicity
from polyid.estimators.base import Access, EstimateResult

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None


@dataclass
class SAEConfig:
    dict_mult: float = 2.0
    k: int = 2
    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 512
    max_steps: int = 6000
    eval_every: int = 200
    patience_evals: int = 8
    val_frac: float = 0.2
    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


if nn is not None:

    class TopKSAE(nn.Module):
        def __init__(self, d: int, m: int, k: int, generator=None, b_dec_init=None):
            super().__init__()
            W = torch.randn(d, m, generator=generator)
            W = W / W.norm(dim=0, keepdim=True)
            self.W_dec = nn.Parameter(W.clone())
            self.W_enc = nn.Parameter(W.T.clone())  # tied init, untied training (Gao et al.)
            self.b_enc = nn.Parameter(torch.zeros(m))
            self.b_dec = nn.Parameter(torch.zeros(d) if b_dec_init is None else torch.as_tensor(b_dec_init).float())
            self.k = k

        def encode(self, x):
            pre = torch.relu((x - self.b_dec) @ self.W_enc.T + self.b_enc)
            vals, idx = torch.topk(pre, self.k, dim=-1)
            z = torch.zeros_like(pre)
            return z.scatter(-1, idx, vals)

        def forward(self, x):
            z = self.encode(x)
            return z @ self.W_dec.T + self.b_dec, z

        @torch.no_grad()
        def normalize_decoder(self):
            self.W_dec.data /= self.W_dec.data.norm(dim=0, keepdim=True).clamp_min(1e-8)


def train_sae(X: np.ndarray, cfg: SAEConfig) -> dict:
    if torch is None:
        raise ImportError("PyTorch is required for the TopK SAE")
    g = torch.Generator().manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    X = np.asarray(X, dtype=np.float32)
    perm = rng.permutation(X.shape[0])
    n_val = max(1, int(cfg.val_frac * X.shape[0]))
    Xv, Xt = torch.from_numpy(X[perm[:n_val]]), torch.from_numpy(X[perm[n_val:]])
    d = X.shape[1]
    m = int(round(cfg.dict_mult * d))
    model = TopKSAE(d, m, min(cfg.k, m), generator=g, b_dec_init=Xt.mean(0))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best, best_state, stall, hist = np.inf, None, 0, []
    var_v = float(((Xv - Xv.mean(0)) ** 2).sum(1).mean())
    for step in range(1, cfg.max_steps + 1):
        idx = torch.randint(0, Xt.shape[0], (cfg.batch_size,), generator=g)
        xb = Xt[idx]
        xh, _ = model(xb)
        loss = ((xb - xh) ** 2).sum(1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        model.normalize_decoder()
        if step % cfg.eval_every == 0:
            with torch.no_grad():
                xh, z = model(Xv)
                vl = float(((Xv - xh) ** 2).sum(1).mean())
            hist.append({"step": step, "val_mse": vl, "val_fve": 1 - vl / max(var_v, 1e-12)})
            if vl < best * (1 - 1e-4):
                best, stall = vl, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                stall += 1
                if stall >= cfg.patience_evals:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    with torch.no_grad():
        xh, z = model(Xv)
        alive = (z > 0).any(dim=0).numpy()
        freq = (z > 0).float().mean(0).numpy()
        vl = float(((Xv - xh) ** 2).sum(1).mean())
    return {"W_dec": model.W_dec.detach().numpy().copy(), "alive": alive, "freq": freq, "val_mse": vl,
            "val_fve": 1 - vl / max(var_v, 1e-12), "dead_frac": float(1 - alive.mean()), "history": hist,
            "config": cfg.to_dict()}


def mean_max_cosine(Wa: np.ndarray, Wb: np.ndarray, alive_a=None, alive_b=None) -> float:
    """Mean over atoms of A of the max |cos| with any atom of B (dictionary stability)."""
    if alive_a is not None:
        Wa = Wa[:, alive_a]
    if alive_b is not None:
        Wb = Wb[:, alive_b]
    if Wa.shape[1] == 0 or Wb.shape[1] == 0:
        return float("nan")
    a = Wa / np.linalg.norm(Wa, axis=0, keepdims=True)
    b = Wb / np.linalg.norm(Wb, axis=0, keepdims=True)
    return float(np.mean(np.abs(a.T @ b).max(axis=1)))


class TopKSAECount:
    """Estimator wrapper: per-neuron count of alive decoder atoms loading >= tau."""

    name = "topk_sae"

    def __init__(self, cfg: dict | None = None):
        self.cfg = dict(cfg or {})

    def fit(self, data: ObservedData, spec: EstimandSpec, rng: np.random.Generator) -> EstimateResult:
        X = np.array(data.X, dtype=float, copy=True)
        for e in range(data.n_envs):
            X[data.env == e] -= X[data.env == e].mean(axis=0)
        c = SAEConfig(**{**self.cfg, "seed": int(rng.integers(2**31 - 1))})
        out = train_sae(X, c)
        P = multiplicity(out["W_dec"], spec.tau, visible=out["alive"])
        return EstimateResult(
            method=self.name, kind="count", point=P.astype(float), loadings=out["W_dec"][:, out["alive"]],
            label=f"TopK SAE m={c.dict_mult}d k={c.k}",
            access=Access(env_labels=False, higher_moments=True, n_observations=data.n,
                          model_selection_budget=1, notes="dictionary size and k fixed by config"),
            diagnostics={"val_fve": out["val_fve"], "dead_frac": out["dead_frac"], "n_alive": int(out["alive"].sum())},
        )
