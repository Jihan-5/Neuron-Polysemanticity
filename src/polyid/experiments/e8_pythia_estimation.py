r"""E8 — Language-model estimation on frozen Pythia-160M / 410M.

Plan p.4 and p.9. Steps:

1. build the six declared environments (or offline stand-ins in smoke mode);
2. cache MLP post-activations at the predeclared layers for fit/dev/test;
3. per layer, on the **fit** split:
   * per-neuron score baselines (moments, GMM/BIC, DPMI-style) on a
     predeclared neuron sample;
   * ISMB / joint diagonalisation on predeclared neuron **blocks** (random
     partition fixed by seed before any activation is seen; see decision D9 in
     docs/ASSUMPTIONS.md — the full ``d_mlp`` square model is neither
     tractable nor plausible, and the block model's stationarity assumption is
     itself checked by ISMB's dependence and off-diagonal diagnostics);
   * frequency-shift stability: per-neuron outputs on the three topic
     environments separately (fixed weights, different sampling);
4. on the **dev** split only: candidate scores for E9 (ISMB lower bounds), so
   that causal validation never selects on test data.

Guard: real-model runs require ``theory_gate_passed: true`` in the config
(plan: theory gate before language-model activations). ``smoke: true``
uses a randomly initialised tiny GPT-NeoX and a byte tokenizer, offline —
for plumbing only, never for claims.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from polyid.estimators import build_estimator
from polyid.experiments.common import save_tables, spec_from_cfg
from polyid.utils.io import write_json
from polyid.utils.seeding import SeedBank


def load_lm(cfg: dict):
    from polyid.lm import hooks

    if cfg.get("smoke"):
        from polyid.lm.smoke_tokenizer import ByteTokenizer

        model = hooks.tiny_random_gptneox(**(cfg.get("smoke_model") or {}))
        return model, ByteTokenizer(), {"name": "tiny_random_gptneox", "smoke": True}
    if not cfg.get("theory_gate_passed", False):
        raise RuntimeError("E8/E9 on real models require `theory_gate_passed: true` (plan p.8 theory gate). "
                           "Record that decision explicitly in the config.")
    name = cfg["model"]
    model, tok = hooks.load_model(name, revision=cfg.get("revision", hooks.PYTHIA_REVISION),
                                  dtype=cfg.get("dtype", "float32"), device=cfg.get("device", "cpu"),
                                  cache_dir=cfg.get("hf_cache"))
    info = {"name": name, "revision": cfg.get("revision", hooks.PYTHIA_REVISION), "dtype": cfg.get("dtype", "float32"),
            **hooks.MODELS.get(name, {})}
    return model, tok, info


def neuron_blocks(d_mlp: int, block_size: int, n_blocks: int, rng: np.random.Generator) -> list[np.ndarray]:
    perm = rng.permutation(d_mlp)
    return [np.sort(perm[i * block_size:(i + 1) * block_size]) for i in range(min(n_blocks, d_mlp // block_size))]


def run(cfg: dict, out_dir: Path) -> dict:
    from polyid.lm.cache import build_cache, load_observed
    from polyid.lm.corpus import build_environments, corpus_manifest

    out_dir = Path(out_dir)
    spec = spec_from_cfg(cfg)
    model, tok, info = load_lm(cfg)
    layers = [int(l) for l in cfg["layers"]]
    envs = build_environments(n_docs=int(cfg.get("n_docs", 2000)), seed=int(cfg["seed"]),
                              offline=bool(cfg.get("offline", cfg.get("smoke", False))))
    cache_dir = Path(cfg.get("cache_dir") or out_dir / "cache")
    if not (cache_dir / "cache_manifest.json").exists():
        build_cache(model, tok, envs, layers, cache_dir, max_length=int(cfg.get("max_length", 128)),
                    batch_size=int(cfg.get("batch_size", 16)), max_per_doc=int(cfg.get("max_per_doc", 8)),
                    device=cfg.get("device", "cpu"), model_info=info,
                    corpus_info=corpus_manifest(envs, bool(cfg.get("offline", cfg.get("smoke", False)))))
    bank = SeedBank(int(cfg["seed"])).child("e8")
    d_mlp = int(model.config.intermediate_size)
    neuron_sample = np.sort(bank.child("neuron_sample").rng().choice(d_mlp, size=min(int(cfg.get("n_score_neurons", 256)), d_mlp), replace=False))
    blocks = neuron_blocks(d_mlp, int(cfg.get("block_size", 32)), int(cfg.get("n_blocks", 4)), bank.child("blocks").rng())
    rows, diags, cands = [], [], []
    topic_envs = ["topics_balanced", "topics_world", "topics_scitech"]
    for layer in layers:
        fit_all = load_observed(cache_dir, layer, "fit")
        # ---- per-neuron score baselines, pooled and per topic environment (frequency shift)
        for name, mcfg in (cfg.get("score_methods") or {}).items():
            for scope in ["pooled"] + topic_envs:
                obs = (load_observed(cache_dir, layer, "fit", neurons=neuron_sample) if scope == "pooled"
                       else load_observed(cache_dir, layer, "fit", envs=[scope], neurons=neuron_sample))
                res = build_estimator(name, mcfg).fit(obs, spec, bank.child(layer, name, scope).rng())
                vals = res.point if res.point is not None else res.score
                for j, nid in enumerate(neuron_sample):
                    rows.append({"layer": layer, "method": name, "scope": scope, "neuron": int(nid),
                                 "value": float(vals[j])})
        # ---- block-wise multi-environment methods
        for b, blk in enumerate(blocks):
            obs = load_observed(cache_dir, layer, "fit", neurons=blk)
            for name, mcfg in (cfg.get("block_methods") or {}).items():
                res = build_estimator(name, mcfg).fit(obs, spec, bank.child(layer, b, name).rng())
                for j, nid in enumerate(blk):
                    r = {"layer": layer, "method": name, "scope": f"block{b}", "neuron": int(nid),
                         "value": float(res.point[j])}
                    if res.lower is not None:
                        r["lower"], r["upper"] = float(res.lower[j]), float(res.upper[j])
                    rows.append(r)
                diags.append({"layer": layer, "block": b, "method": name, "n": obs.n, "n_envs": obs.n_envs,
                              **{k: v for k, v in (res.diagnostics or {}).items()
                                 if isinstance(v, (int, float, bool, np.integer, np.floating))}})
            # ---- development split: candidate scores for E9 (never test data)
            dev = load_observed(cache_dir, layer, "dev", neurons=blk)
            if dev.n_envs >= 3:
                res = build_estimator("ismb", (cfg.get("block_methods") or {}).get("ismb", {})).fit(
                    dev, spec, bank.child(layer, b, "dev").rng())
                for j, nid in enumerate(blk):
                    cands.append({"layer": layer, "neuron": int(nid), "dev_lower": float(res.lower[j]),
                                  "dev_upper": float(res.upper[j]), "dev_point": float(res.point[j])})
    df = pd.DataFrame(rows)
    save_tables(out_dir, "e8_rows", df)
    save_tables(out_dir, "e8_block_diagnostics", diags)
    save_tables(out_dir, "e8_dev_candidate_scores", cands)
    stab = []
    if len(df):
        for (layer, m), g in df[df.scope.isin(topic_envs)].groupby(["layer", "method"]):
            piv = g.pivot(index="neuron", columns="scope", values="value")
            ch = [float(np.mean(piv[a] != piv[b])) for a in piv for b in piv if a < b]
            stab.append({"layer": layer, "method": m, "frac_changed_across_topic_envs": float(np.mean(ch))})
    save_tables(out_dir, "e8_frequency_shift_stability", stab)
    out = {"model": info, "layers": layers, "smoke": bool(cfg.get("smoke", False)), "n_blocks": len(blocks),
           "cache_dir": str(cache_dir), "stability": stab}
    write_json(out_dir / "e8_summary.json", out)
    return out
