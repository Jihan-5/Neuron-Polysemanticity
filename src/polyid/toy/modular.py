r"""Optional controlled transformer bridge: addition modulo 97 (plan p.6).

"Use a two-layer decoder on a small modular task only if the toy estimator
works. A starter configuration is width 128, four attention heads, and MLP
width 512, trained on addition modulo 97 with an 80/20 split of operand
pairs. Try AdamW at 3e-4, batch 256, and a 20,000-step cap with validation
every 250 steps. Record both task accuracy and representation stability;
task-factor labels alone do not reveal the exact internal dictionary."

Activations for the estimator are the post-ReLU MLP activations (width 512)
of a chosen block at the final ('=') position. Environments are built by
shifting the *operand distribution* over held-out pairs (the model is frozen;
only the sampling distribution changes — the same logic as the language-model
corpus shift).
"""

from __future__ import annotations

import math
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
class ModularConfig:
    p: int = 97
    d_model: int = 128
    n_heads: int = 4
    d_mlp: int = 512
    n_layers: int = 2
    train_frac: float = 0.8
    lr: float = 3e-4
    weight_decay: float = 1.0
    batch_size: int = 256
    max_steps: int = 20000
    eval_every: int = 250
    target_val_acc: float = 0.995  # stop early once generalisation is reached and stable
    stable_evals: int = 4
    seed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


if nn is not None:

    class Block(nn.Module):
        def __init__(self, d_model: int, n_heads: int, d_mlp: int):
            super().__init__()
            self.ln1 = nn.LayerNorm(d_model)
            self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
            self.ln2 = nn.LayerNorm(d_model)
            self.fc_in = nn.Linear(d_model, d_mlp)
            self.fc_out = nn.Linear(d_mlp, d_model)

        def forward(self, x, mask, cache: dict | None = None, ablate: "torch.Tensor | None" = None):
            a = self.ln1(x)
            x = x + self.attn(a, a, a, attn_mask=mask, need_weights=False)[0]
            post = torch.relu(self.fc_in(self.ln2(x)))
            if ablate is not None:
                post = post * ablate
            if cache is not None:
                cache.setdefault("mlp_post", []).append(post)
            return x + self.fc_out(post)

    class ModularTransformer(nn.Module):
        def __init__(self, cfg: ModularConfig):
            super().__init__()
            self.cfg = cfg
            self.embed = nn.Embedding(cfg.p + 1, cfg.d_model)  # token p is '='
            self.pos = nn.Parameter(torch.randn(3, cfg.d_model) * 0.02)
            self.blocks = nn.ModuleList([Block(cfg.d_model, cfg.n_heads, cfg.d_mlp) for _ in range(cfg.n_layers)])
            self.ln_f = nn.LayerNorm(cfg.d_model)
            self.unembed = nn.Linear(cfg.d_model, cfg.p, bias=False)
            self.register_buffer("mask", torch.triu(torch.full((3, 3), float("-inf")), diagonal=1))

        def forward(self, tokens, cache: dict | None = None, ablate: dict | None = None):
            x = self.embed(tokens) + self.pos
            for i, blk in enumerate(self.blocks):
                x = blk(x, self.mask, cache, None if ablate is None else ablate.get(i))
            return self.unembed(self.ln_f(x[:, -1]))


def all_pairs(p: int) -> np.ndarray:
    a, b = np.meshgrid(np.arange(p), np.arange(p), indexing="ij")
    return np.stack([a.ravel(), b.ravel()], axis=1)


def split_pairs(cfg: ModularConfig) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    pairs = all_pairs(cfg.p)
    perm = rng.permutation(len(pairs))
    n_tr = int(cfg.train_frac * len(pairs))
    return pairs[perm[:n_tr]], pairs[perm[n_tr:]]


def _tokens(pairs: np.ndarray, p: int) -> "torch.Tensor":
    t = np.concatenate([pairs, np.full((len(pairs), 1), p)], axis=1)
    return torch.as_tensor(t, dtype=torch.long)


def train_modular(cfg: ModularConfig) -> dict:
    if torch is None:
        raise ImportError("PyTorch is required")
    torch.manual_seed(cfg.seed)
    g = torch.Generator().manual_seed(cfg.seed)
    tr, va = split_pairs(cfg)
    Xtr, Xva = _tokens(tr, cfg.p), _tokens(va, cfg.p)
    ytr = torch.as_tensor((tr[:, 0] + tr[:, 1]) % cfg.p)
    yva = torch.as_tensor((va[:, 0] + va[:, 1]) % cfg.p)
    model = ModularTransformer(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.98))
    hist, prev_act, stable = [], None, 0
    probe = Xva[:512]
    for step in range(1, cfg.max_steps + 1):
        idx = torch.randint(0, len(Xtr), (cfg.batch_size,), generator=g)
        loss = nn.functional.cross_entropy(model(Xtr[idx]), ytr[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % cfg.eval_every == 0:
            model.eval()
            with torch.no_grad():
                acc_tr = float((model(Xtr[:4096]).argmax(-1) == ytr[:4096]).float().mean())
                acc_va = float((model(Xva).argmax(-1) == yva).float().mean())
                cache: dict = {}
                model(probe, cache=cache)
                act = torch.cat([c[:, -1] for c in cache["mlp_post"]], dim=1).numpy()
                # representation stability: linear CKA between successive checkpoints
                cka = linear_cka(prev_act, act) if prev_act is not None else float("nan")
                prev_act = act
            model.train()
            hist.append({"step": step, "loss": float(loss.detach()), "train_acc": acc_tr, "val_acc": acc_va, "cka_prev": cka})
            stable = stable + 1 if acc_va >= cfg.target_val_acc else 0
            if stable >= cfg.stable_evals:
                break
    return {"model": model, "history": hist, "train_pairs": tr, "val_pairs": va, "config": cfg.to_dict()}


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    num = np.linalg.norm(X.T @ Y) ** 2
    den = np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y)
    return float(num / max(den, 1e-12))


@torch.no_grad() if torch is not None else (lambda f: f)
def mlp_activations(model, pairs: np.ndarray, layer: int, batch: int = 2048) -> np.ndarray:
    model.eval()
    out = []
    for s in range(0, len(pairs), batch):
        cache: dict = {}
        model(_tokens(pairs[s:s + batch], model.cfg.p), cache=cache)
        out.append(cache["mlp_post"][layer][:, -1].numpy())
    return np.concatenate(out)


def operand_environments(pairs: np.ndarray, p: int, n_envs: int, n_per_env: int, rng: np.random.Generator,
                         concentration: float = 0.5) -> list[np.ndarray]:
    """Environments = different non-uniform distributions over the *sum* residue (a+b) mod p.

    Each environment draws Dirichlet(concentration) weights over residues and
    resamples held-out pairs accordingly (with replacement). The frozen model
    is unchanged; only which inputs are seen changes.
    """
    s = (pairs[:, 0] + pairs[:, 1]) % p
    envs = []
    for _ in range(n_envs):
        w_res = rng.dirichlet(np.full(p, concentration))
        w = w_res[s]
        w = w / w.sum()
        envs.append(rng.choice(len(pairs), size=n_per_env, replace=True, p=w))
    return envs


def modular_observed(model, pairs: np.ndarray, env_idx: list[np.ndarray], layer: int, neurons: np.ndarray | None = None
                     ) -> ObservedData:
    acts = mlp_activations(model, pairs, layer)
    if neurons is not None:
        acts = acts[:, neurons]
    X = np.concatenate([acts[idx] for idx in env_idx])
    env = np.concatenate([np.full(len(idx), e) for e, idx in enumerate(env_idx)])
    cluster = np.concatenate([idx for idx in env_idx])  # repeated pairs are not independent draws
    return ObservedData(X=X, env=env, cluster=cluster,
                        neuron_ids=np.arange(acts.shape[1]) if neurons is None else np.asarray(neurons),
                        meta={"source": "modular_transformer", "layer": layer})
