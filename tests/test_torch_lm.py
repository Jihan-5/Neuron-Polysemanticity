import numpy as np
import pytest

from polyid.estimand import EstimandSpec


@pytest.mark.torch
def test_toy_model_trains_and_truth_is_consistent():
    from polyid.toy.superposition import ToyConfig, toy_observations, toy_truth, train_toy

    out = train_toy(ToyConfig(K=8, d=3, steps=300, eval_every=100, feature_prob=0.2, seed=0))
    assert out["W"].shape == (3, 8) and out["history"][-1]["val_loss"] < out["history"][0]["val_loss"] * 1.5
    rates = np.random.default_rng(0).uniform(0.1, 0.3, size=(4, 8))
    tr = toy_truth(out["W"], rates, 0.3, 1e-4)
    assert tr["truth"].shape == (3,) and tr["n_target"] <= 8
    obs = toy_observations(out["W"], rates, 200, np.random.default_rng(1))
    assert obs.X.shape == (800, 3) and obs.n_envs == 4


@pytest.mark.torch
def test_topk_sae_trains_and_counts():
    from polyid.toy.sae import SAEConfig, mean_max_cosine, train_sae
    from polyid.estimators import build_estimator
    from polyid.generators.planted import PlantedConfig, generate
    from polyid.utils.seeding import SeedBank

    ds = generate(PlantedConfig(d=6, K=6, n_per_env=1000, n_envs=3), SeedBank(0))
    a = train_sae(ds.observed.X, SAEConfig(max_steps=300, eval_every=100, seed=0))
    b = train_sae(ds.observed.X, SAEConfig(max_steps=300, eval_every=100, seed=1))
    assert a["W_dec"].shape == (6, 12) and 0 <= mean_max_cosine(a["W_dec"], b["W_dec"]) <= 1
    res = build_estimator("topk_sae", {"max_steps": 200, "eval_every": 100}).fit(ds.observed, EstimandSpec(),
                                                                              np.random.default_rng(0))
    assert res.point.shape == (6,)


@pytest.mark.torch
def test_modular_transformer_forward_and_activations():
    import torch

    from polyid.toy.modular import ModularConfig, ModularTransformer, mlp_activations, operand_environments, split_pairs

    cfg = ModularConfig(p=11, d_model=32, n_heads=4, d_mlp=64)
    m = ModularTransformer(cfg)
    tr, va = split_pairs(cfg)
    acts = mlp_activations(m, va[:7], layer=1)
    assert acts.shape == (7, 64) and np.all(acts >= 0)
    envs = operand_environments(va, cfg.p, 3, 20, np.random.default_rng(0))
    assert len(envs) == 3 and all(len(e) == 20 for e in envs)


@pytest.mark.lm
def test_hook_matches_manual_forward_and_ablation_changes_output():
    import torch

    from polyid.lm.hooks import ablate_neurons, tiny_random_gptneox, verify_hook

    m = tiny_random_gptneox(n_layers=2, d_model=32, d_mlp=128)
    ids = torch.randint(0, 50000, (2, 12))
    for layer in (0, 1):
        assert verify_hook(m, ids, layer) < 1e-5
    base = m(input_ids=ids).logits
    with ablate_neurons(m, 1, np.arange(64)):
        abl = m(input_ids=ids).logits
    assert not torch.allclose(base, abl)
    assert torch.allclose(m(input_ids=ids).logits, base)  # hook removed


@pytest.mark.lm
def test_cache_roundtrip_and_family_splits(tmp_path):
    from polyid.lm.cache import build_cache, load_observed
    from polyid.lm.corpus import build_environments, doc_split
    from polyid.lm.hooks import tiny_random_gptneox
    from polyid.lm.smoke_tokenizer import ByteTokenizer
    from polyid.lm.templates import agreement_items, semantic_choice_items, split_by_family

    m = tiny_random_gptneox(n_layers=2, d_model=32, d_mlp=64)
    envs = build_environments(n_docs=6, offline=True)
    build_cache(m, ByteTokenizer(), envs, [1], tmp_path, max_length=64, batch_size=4, max_per_doc=2)
    obs = load_observed(tmp_path, 1, "fit")
    assert obs.d == 64 and obs.cluster is not None and obs.n_envs >= 2
    for items in (semantic_choice_items(), agreement_items()):
        sp = split_by_family(items)
        fams = [set(i.family for i in sp[k]) for k in ("fit", "dev", "test")]
        assert not (fams[0] & fams[1]) and not (fams[0] & fams[2]) and not (fams[1] & fams[2])
    assert doc_split("abc") == doc_split("abc")
