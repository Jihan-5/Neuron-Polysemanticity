r"""Activation cache: selected token contexts -> MLP post-activations on disk.

Plan p.12: "Storage scales with contexts x selected layers x retained neuron
width x bytes. Cache selected tokens, store immutable manifests, and make a
small end-to-end corpus pass before collecting the full dataset."

Layout under ``out_dir``::

    index.csv                one row per cached context (env, doc_id, family, split, position, token_id)
    layer{L}.npy             float16 array [n_contexts, d_mlp] for every requested layer
    cache_manifest.json      model / revision / tokenizer / dtype / tensor name / corpus hash / rule / versions

Model weights are never modified; neuron coordinates are kept exactly (no
projection), as the plan requires for the per-neuron target.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from polyid.data import ObservedData
from polyid.lm.corpus import Document, doc_split, select_positions, target_position
from polyid.lm.hooks import TENSOR_NAME, MLPRecorder
from polyid.utils.manifest import package_versions

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def storage_bytes(n_contexts: int, n_layers: int, width: int, bytes_per: int = 2) -> int:
    return n_contexts * n_layers * width * bytes_per


@torch.no_grad() if torch is not None else (lambda f: f)
def build_cache(model, tokenizer, envs: dict[str, list[Document]], layers: list[int], out_dir: str | Path, *,
                max_length: int = 128, batch_size: int = 16, min_position: int = 16, max_per_doc: int = 8,
                max_contexts_per_env: int | None = None, device: str = "cpu", model_info: dict | None = None,
                corpus_info: dict | None = None) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    index_rows: list[dict] = []
    acts: dict[int, list[np.ndarray]] = {l: [] for l in layers}
    for env_name in sorted(envs):
        docs = envs[env_name]
        n_env = 0
        for s in range(0, len(docs), batch_size):
            batch = docs[s:s + batch_size]
            enc = tokenizer([d.text for d in batch], return_tensors="pt", padding=True, truncation=True,
                            max_length=max_length)
            positions = []
            for b, d in enumerate(batch):
                ids = enc["input_ids"][b][enc["attention_mask"][b].bool()].tolist()
                if d.target_word is not None:
                    p = target_position(tokenizer, d.text, d.target_word)
                    pos = [p] if p is not None and p < len(ids) else []
                else:
                    pos = select_positions(ids, d.doc_id, min_position, max_per_doc)
                if tokenizer.padding_side == "left":
                    pad = enc["input_ids"].shape[1] - len(ids)
                    pos = [p + pad for p in pos]
                positions.append(torch.as_tensor(pos, dtype=torch.long))
                for p in pos:
                    index_rows.append({"env": env_name, "doc_id": d.doc_id, "family": d.family,
                                       "split": doc_split(d.family), "position": int(p),
                                       "token_id": int(enc["input_ids"][b, p]), "target_word": d.target_word,
                                       "sense": d.sense})
                n_env += len(pos)
            rec = MLPRecorder(model, layers, positions=positions)
            with rec:
                model(input_ids=enc["input_ids"].to(device), attention_mask=enc["attention_mask"].to(device))
            for l in layers:
                acts[l].append(torch.cat(rec.store[l], dim=0).numpy().astype(np.float16))
            if max_contexts_per_env is not None and n_env >= max_contexts_per_env:
                break
    idx = pd.DataFrame(index_rows)
    idx.to_csv(out / "index.csv", index=False)
    for l in layers:
        arr = np.concatenate(acts[l]) if acts[l] else np.zeros((0, model.config.intermediate_size), np.float16)
        assert arr.shape[0] == len(idx), "activation rows must align with the index"
        np.save(out / f"layer{l}.npy", arr)
    manifest = {
        "tensor_name": TENSOR_NAME, "layers": layers, "dtype_on_disk": "float16", "n_contexts": len(idx),
        "token_rule": {"generic": f"uniform positions >= {min_position}, <= {max_per_doc} per doc, seeded by doc id",
                       "controlled": "final subword of target word"},
        "max_length": max_length, "model": model_info or {}, "corpus": corpus_info or {},
        "storage_bytes": storage_bytes(len(idx), len(layers), int(model.config.intermediate_size)),
        "versions": package_versions(),
    }
    (out / "cache_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return out


def load_observed(cache_dir: str | Path, layer: int, split: str, envs: list[str] | None = None,
                  neurons: np.ndarray | None = None) -> ObservedData:
    """Load one split of one layer as :class:`ObservedData` (clusters = document families)."""
    cache_dir = Path(cache_dir)
    idx = pd.read_csv(cache_dir / "index.csv")
    X = np.load(cache_dir / f"layer{layer}.npy", mmap_mode="r")
    mask = (idx["split"] == split).to_numpy()
    if envs is not None:
        mask &= idx["env"].isin(envs).to_numpy()
    sub = idx[mask]
    env_names = sorted(sub["env"].unique())
    env = sub["env"].map({e: i for i, e in enumerate(env_names)}).to_numpy()
    Xs = np.asarray(X[np.flatnonzero(mask)], dtype=np.float64)
    if neurons is not None:
        Xs = Xs[:, neurons]
    _, cluster = np.unique(sub["family"].to_numpy(), return_inverse=True)
    return ObservedData(X=Xs, env=env, cluster=cluster,
                        neuron_ids=np.arange(X.shape[1]) if neurons is None else np.asarray(neurons),
                        env_names=tuple(env_names), meta={"source": str(cache_dir), "layer": layer, "split": split})
