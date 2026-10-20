"""Experiment runners E0–E11. Each module exposes ``run(cfg: dict, out_dir: Path) -> dict``."""

EXPERIMENTS = {
    "e0_theory_gate": "polyid.experiments.e0_theory_gate",
    "e1_planted_grid": "polyid.experiments.e1_planted_grid",
    "e2_stress": "polyid.experiments.e2_stress",
    "e3_frequency_shift": "polyid.experiments.e3_frequency_shift",
    "e4_coverage_d3": "polyid.experiments.e4_coverage_d3",
    "e5_toy_superposition": "polyid.experiments.e5_toy_superposition",
    "e6_sae_comparison": "polyid.experiments.e6_sae_comparison",
    "e7_modular_transformer": "polyid.experiments.e7_modular_transformer",
    "e8_pythia_estimation": "polyid.experiments.e8_pythia_estimation",
    "e9_causal_validation": "polyid.experiments.e9_causal_validation",
    "e10_adaptive_envs": "polyid.experiments.e10_adaptive_envs",
    "e11_cumulant_ismb": "polyid.experiments.e11_cumulant_ismb",
}
