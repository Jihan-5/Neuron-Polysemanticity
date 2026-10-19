r"""Pythia loading, MLP post-activation hooks, hook verification and ablation.

Tensor (plan p.4): "post-nonlinearity MLP activations entering the output
projection" of GPT-NeoX — i.e. the *input* of
``gpt_neox.layers[i].mlp.dense_4h_to_h``. We hook it with a forward
**pre**-hook on that ``Linear`` so the definition does not depend on the
name of the activation module. Shape is asserted to be the MLP intermediate
width (3072 for 160M, 4096 for 410M), never the residual width (syllabus T1:
"assert the intermediate width, not the residual width").

Verification (syllabus T1 "Done when"): :func:`verify_hook` recomputes the
same tensor manually from ``output_hidden_states`` —
``act(dense_h_to_4h(post_attention_layernorm(x_in)))`` for the parallel-residual
GPT-NeoX blocks used by Pythia (and the sequential variant otherwise) — and
returns the max absolute difference.

Layers (plan p.4): fixed relative depths chosen in advance. 160M: blocks
2, 5, 8 (zero-based). 410M: the same relative depths, ``round(l * 24/12)`` =
4, 10, 16. The ablation layer is the middle one (5 and 10). The spec
document suggested block 11 for 410M; we use 10 to keep the relative-depth
rule exact (recorded as decision D7 in ``docs/ASSUMPTIONS.md``).
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

PYTHIA_REVISION = "step143000"  # final checkpoint branch of every Pythia model

MODELS = {
    "EleutherAI/pythia-160m": {"n_layers": 12, "d_model": 768, "d_mlp": 3072, "layers": [2, 5, 8], "ablation_layer": 5},
    "EleutherAI/pythia-410m": {"n_layers": 24, "d_model": 1024, "d_mlp": 4096, "layers": [4, 10, 16], "ablation_layer": 10},
}

TENSOR_NAME = "gpt_neox.layers.{layer}.mlp.dense_4h_to_h:input"


def relative_depth_layers(n_layers: int, reference: tuple[int, ...] = (2, 5, 8), reference_depth: int = 12) -> list[int]:
    return [int(round(l * n_layers / reference_depth)) for l in reference]


def load_model(name: str, revision: str = PYTHIA_REVISION, dtype: str = "float32", device: str = "cpu",
               cache_dir: str | None = None):
    """Load a frozen GPT-NeoX causal LM and its tokenizer at a pinned revision."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name, revision=revision, cache_dir=cache_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, revision=revision, cache_dir=cache_dir,
                                                 torch_dtype=getattr(torch, dtype))
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


def tiny_random_gptneox(n_layers: int = 3, d_model: int = 64, d_mlp: int = 256, vocab: int = 50304, seed: int = 0):
    """Randomly initialised GPT-NeoX for offline plumbing tests (no download, no claims)."""
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

    torch.manual_seed(seed)
    cfg = GPTNeoXConfig(vocab_size=vocab, hidden_size=d_model, num_hidden_layers=n_layers, num_attention_heads=4,
                        intermediate_size=d_mlp, max_position_embeddings=256, use_parallel_residual=True,
                        rotary_pct=0.25)
    model = GPTNeoXForCausalLM(cfg).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _layers(model):
    return model.gpt_neox.layers


def mlp_width(model) -> int:
    return int(model.config.intermediate_size)


@dataclass
class MLPRecorder:
    """Records the input of ``dense_4h_to_h`` for selected layers.

    ``positions`` (list of 1-D index tensors, one per batch row) selects token
    positions to keep; ``None`` keeps all positions.
    """

    model: object
    layers: list[int]
    store: dict = field(default_factory=dict)
    _handles: list = field(default_factory=list)
    positions: list | None = None

    def _make_hook(self, layer: int):
        width = mlp_width(self.model)

        def hook(module, inputs):
            x = inputs[0]
            if x.shape[-1] != width or x.shape[-1] == self.model.config.hidden_size:
                raise RuntimeError(f"hooked tensor has width {x.shape[-1]}, expected MLP width {width}")
            if self.positions is None:
                self.store.setdefault(layer, []).append(x.detach().float().cpu())
            else:
                rows = [x[b, pos].detach().float().cpu() for b, pos in enumerate(self.positions)]
                self.store.setdefault(layer, []).append(torch.cat(rows, dim=0) if rows else x[:0, 0])
        return hook

    def __enter__(self):
        for l in self.layers:
            mod = _layers(self.model)[l].mlp.dense_4h_to_h
            self._handles.append(mod.register_forward_pre_hook(self._make_hook(l)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()


@contextmanager
def ablate_neurons(model, layer: int, neurons, value: float = 0.0, positions: list | None = None):
    """Zero-ablate (or set to ``value``) MLP neurons of one layer at all or selected positions."""
    idx = torch.as_tensor(np.asarray(neurons, dtype=np.int64))

    def hook(module, inputs):
        x = inputs[0].clone()
        if positions is None:
            x[..., idx] = value
        else:
            for b, pos in enumerate(positions):
                x[b, pos[:, None], idx[None, :]] = value
        return (x,) + tuple(inputs[1:])

    h = _layers(model)[layer].mlp.dense_4h_to_h.register_forward_pre_hook(hook)
    try:
        yield
    finally:
        h.remove()


@torch.no_grad() if torch is not None else (lambda f: f)
def verify_hook(model, input_ids, layer: int, attention_mask=None) -> float:
    """Max |hooked - manual| for the MLP post-activation of ``layer`` on one batch."""
    rec = MLPRecorder(model, [layer])
    with rec:
        out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    hooked = rec.store[layer][0]
    blk = _layers(model)[layer]
    x_in = out.hidden_states[layer]
    if getattr(model.config, "use_parallel_residual", True):
        mlp_in = blk.post_attention_layernorm(x_in)
    else:  # pragma: no cover - not used by Pythia
        attn = blk.attention(blk.input_layernorm(x_in), attention_mask=attention_mask)[0]
        mlp_in = blk.post_attention_layernorm(x_in + attn)
    manual = blk.mlp.act(blk.mlp.dense_h_to_4h(mlp_in)).float().cpu()
    return float((hooked - manual).abs().max())
