r"""Planted sparse-factor generator with exact population moments.

Implements plan p.5 ("Planted data: exact implementation"):

* a loading matrix ``A`` with unit-norm columns and a sparse support pattern
  chosen so that neuron-level multiplicity varies, with degenerate duplicate
  columns rejected;
* per environment, latent activity indicators ``b ~ Bernoulli(p_{e,k})`` times
  bounded positive magnitudes ``s ~ Uniform[mag_low, mag_high]``;
* zero-mean noise independent of the latents.

Every assumption-stress condition of the plan is a *switch* of this one
generator, so that a stress result is always comparable to its clean
counterpart:

==========================  ==================================================
condition                   config fields
==========================  ==================================================
weak environmental          ``diversity="near_collinear"``, ``collinear_eta``
diversity
exactly collinear profiles  ``collinear_groups``
invisible factors           ``invisible_factors``
co-occurrence /             ``switch_corr_blocks``, ``switch_corr`` (1.0 with
perfect co-occurrence       shared rates = identical switches)
coupled magnitudes          ``magnitude_corr_blocks``, ``magnitude_corr``
nearly duplicate columns    ``near_duplicate_pairs``
loadings near tau           ``near_tau_columns``, ``near_tau_width``
misspecified noise          ``noise_mode`` in {env_scaled, env_heteroscedastic,
                            student_t}; ``correlated_stationary`` is *allowed*
overcomplete                ``K > d``
unknown factor count        ``K < d`` (the estimator is not told ``K``)
==========================  ==================================================

Signal / noise convention (reported in every manifest)
------------------------------------------------------
``signal_scale**2 = mean_e trace(A Sigma_z^{(e)} A^T) / d`` — the average
per-neuron latent-driven variance. Noise has per-coordinate SD
``noise_rel_sd * signal_scale`` (before any environment-specific scaling in the
misspecified modes).

Population moments are computed *exactly* (closed forms for the Gaussian
copula below), so the identification map can separate "not enough samples"
from "not identified even with infinite samples".
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields

import numpy as np
from scipy import stats

from polyid.data import ObservedData
from polyid.estimand import (
    centered_profiles,
    check_nonduplication,
    environmental_diversity,
    multiplicity,
    normalize_columns,
    visibility_stat,
)
from polyid.utils.seeding import SeedBank


@dataclass
class PlantedConfig:
    # sizes
    d: int = 16
    K: int = 16
    n_envs: int = 6
    n_per_env: int | list = 5000
    # loading matrix
    col_support_min: int = 1
    col_support_max: int = 4
    loading_low: float = 0.3
    background_sd: float = 0.0
    tau: float = 0.10
    tau_margin: float = 0.03
    near_tau_columns: int = 0
    near_tau_width: float = 0.02
    near_duplicate_pairs: list = field(default_factory=list)
    max_abs_cos: float = 0.9
    max_cond: float = 100.0
    max_attempts: int = 2000
    # activity rates
    rate_center: float = 0.15
    rate_spread: float = 0.6
    rate_min: float = 0.01
    rate_max: float = 0.6
    min_visibility: float = 0.012
    diversity: str = "well_separated"
    collinear_gamma: float = 0.6
    collinear_eta: float = 0.15
    collinear_groups: list = field(default_factory=list)
    invisible_factors: list = field(default_factory=list)
    # magnitudes
    mag_low: float = 0.5
    mag_high: float = 1.5
    # dependence between latents (departures from conditional independence)
    switch_corr_blocks: list = field(default_factory=list)
    switch_corr: float = 0.0
    shared_rates_in_switch_blocks: bool = True
    magnitude_corr_blocks: list = field(default_factory=list)
    magnitude_corr: float = 0.0
    # noise
    noise_rel_sd: float = 0.05
    noise_mode: str = "iid"
    noise_env_scale_min: float = 0.5
    noise_env_scale_max: float = 2.0
    student_df: float = 3.0
    # bookkeeping
    store_latents: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> "PlantedConfig":
        names = {f.name for f in fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise KeyError(f"unknown PlantedConfig keys: {sorted(unknown)}")
        return cls(**d)

    def to_dict(self) -> dict:
        return asdict(self)

    def n_list(self) -> list[int]:
        if isinstance(self.n_per_env, (list, tuple)):
            if len(self.n_per_env) != self.n_envs:
                raise ValueError("n_per_env list must have one entry per environment")
            return [int(x) for x in self.n_per_env]
        return [int(self.n_per_env)] * self.n_envs


@dataclass
class Oracle:
    """Ground truth. Only ``polyid.metrics`` / ``polyid.identification`` read this."""

    A: np.ndarray  # d x K, unit-norm columns
    rates: np.ndarray  # E x K activation probabilities
    latent_cov: np.ndarray  # E x K x K population latent covariance
    noise_cov: np.ndarray  # E x d x d population noise covariance
    pop_cov: np.ndarray  # E x d x d population observed covariance
    variances: np.ndarray  # E x K diag(latent_cov)
    profiles: np.ndarray  # K x E centred variance profiles
    visible: np.ndarray  # K bool
    truth: np.ndarray  # d int, P_j(tau) at the generator's tau
    tau: float
    v_min: float
    independent: bool  # latent covariances diagonal in every environment
    noise_stationary: bool  # noise covariance identical across environments
    diversity: float  # min pairwise sin angle between visible profiles
    signal_scale: float
    noise_sd: float
    exempt_entries: list  # (j, k) entries deliberately placed near tau
    latents: np.ndarray | None = None

    def truth_at(self, tau: float) -> np.ndarray:
        return multiplicity(self.A, tau, visible=self.visible)


@dataclass
class PlantedDataset:
    observed: ObservedData
    oracle: Oracle
    config: PlantedConfig


# --------------------------------------------------------------------------
# loading matrix
# --------------------------------------------------------------------------

def _loading_ok(A: np.ndarray, S: np.ndarray, cfg: PlantedConfig, exempt_cols: set, exempt_entries: set) -> bool:
    d, K = A.shape
    if K <= d:
        sv = np.linalg.svd(A, compute_uv=False)
        if sv[-1] <= 0 or sv[0] / sv[-1] > cfg.max_cond:
            return False
    # non-duplication, except pairs that are deliberately near-duplicate
    G = np.abs(A.T @ A)
    np.fill_diagonal(G, 0.0)
    for k, l, _ in cfg.near_duplicate_pairs:
        G[k, l] = G[l, k] = 0.0
    if G.max(initial=0.0) > cfg.max_abs_cos:
        return False
    # threshold margin: supported entries clearly above tau, others clearly below
    absA = np.abs(A)
    for k in range(K):
        if k in exempt_cols:
            continue
        for j in range(d):
            if (j, k) in exempt_entries:
                continue
            if S[j, k] and absA[j, k] < cfg.tau + cfg.tau_margin:
                return False
            if not S[j, k] and absA[j, k] > cfg.tau - cfg.tau_margin:
                return False
    return True


def sample_loading_matrix(cfg: PlantedConfig, rng: np.random.Generator) -> tuple[np.ndarray, list]:
    """Sample ``A`` (unit-norm columns) satisfying the declared regularity conditions.

    Returns ``(A, exempt_entries)`` where ``exempt_entries`` lists (j, k) loadings
    that were deliberately placed within ``near_tau_width`` of ``tau``.
    """
    d, K = cfg.d, cfg.K
    near_dup_cols = {int(l) for _, l, _ in cfg.near_duplicate_pairs}
    for _attempt in range(cfg.max_attempts):
        S = np.zeros((d, K), dtype=bool)
        anchors = rng.permutation(d)
        for k in range(K):
            S[anchors[k % d], k] = True
            s_k = int(rng.integers(cfg.col_support_min, cfg.col_support_max + 1))
            extra = rng.choice(np.setdiff1d(np.arange(d), [anchors[k % d]]), size=max(0, min(s_k - 1, d - 1)), replace=False)
            S[extra, k] = True
        A = np.where(S, rng.uniform(cfg.loading_low, 1.0, size=(d, K)) * rng.choice([-1.0, 1.0], size=(d, K)), 0.0)
        if cfg.background_sd > 0:
            A = A + np.where(S, 0.0, rng.normal(0.0, cfg.background_sd, size=(d, K)))
        A = normalize_columns(A)[0]

        exempt_entries: set = set()
        # loadings near tau: one entry per selected column, renormalising the rest
        if cfg.near_tau_columns > 0:
            cols = rng.choice(K, size=min(cfg.near_tau_columns, K), replace=False)
            for k in cols:
                rows = np.flatnonzero(S[:, k])
                if rows.size < 2:
                    continue  # a single-entry column cannot hold a near-tau entry and keep unit norm
                j = int(rng.choice(rows))
                target = cfg.tau + rng.uniform(-cfg.near_tau_width, cfg.near_tau_width)
                rest = np.ones(d, dtype=bool)
                rest[j] = False
                A[j, k] = math.copysign(target, A[j, k] if A[j, k] != 0 else 1.0)
                A[rest, k] *= math.sqrt(1.0 - target**2) / np.linalg.norm(A[rest, k])
                exempt_entries.add((j, int(k)))
        # nearly duplicate directions
        for k, l, c in cfg.near_duplicate_pairs:
            k, l = int(k), int(l)
            u = A[:, l] - (A[:, l] @ A[:, k]) * A[:, k]
            u /= np.linalg.norm(u)
            A[:, l] = c * A[:, k] + math.sqrt(1.0 - c**2) * u
            S[:, l] = np.abs(A[:, l]) >= cfg.tau

        if _loading_ok(A, S, cfg, near_dup_cols, exempt_entries):
            return A, sorted(exempt_entries)
    raise RuntimeError(
        f"could not sample a loading matrix satisfying the constraints in {cfg.max_attempts} attempts; "
        "relax max_abs_cos / max_cond / tau_margin"
    )


# --------------------------------------------------------------------------
# activity rates
# --------------------------------------------------------------------------

def sample_rates(cfg: PlantedConfig, rng: np.random.Generator) -> np.ndarray:
    """``E x K`` activation probabilities encoding environmental diversity."""
    E, K = cfg.n_envs, cfg.K
    lo = max(cfg.rate_min, cfg.rate_center * (1.0 - cfg.rate_spread))
    hi = min(cfg.rate_max, cfg.rate_center * (1.0 + cfg.rate_spread))
    if cfg.diversity == "well_separated":
        p = rng.uniform(lo, hi, size=(E, K))
    elif cfg.diversity == "near_collinear":
        base = rng.uniform(lo, hi, size=K)
        pattern = rng.uniform(-1.0, 1.0, size=E)
        pattern -= pattern.mean()
        jitter = rng.uniform(-1.0, 1.0, size=(E, K))
        p = base[None, :] * (1.0 + cfg.collinear_gamma * pattern[:, None]) + cfg.collinear_eta * cfg.rate_center * jitter
    elif cfg.diversity == "constant":
        p = np.repeat(rng.uniform(lo, hi, size=(1, K)), E, axis=0)
    else:
        raise ValueError(f"unknown diversity regime {cfg.diversity!r}")
    p = np.clip(p, cfg.rate_min, cfg.rate_max)
    p = _apply_rate_structure(cfg, p)
    return p


def _apply_rate_structure(cfg: PlantedConfig, p: np.ndarray) -> np.ndarray:
    for group in cfg.collinear_groups:
        group = [int(g) for g in group]
        p[:, group] = p[:, [group[0]]]
    if cfg.shared_rates_in_switch_blocks:
        for block in cfg.switch_corr_blocks:
            block = [int(b) for b in block]
            p[:, block] = p[:, [block[0]]]
    for k in cfg.invisible_factors:
        p[:, int(k)] = p[:, int(k)].mean()
    return p


def _independent_visibility(cfg: PlantedConfig, p: np.ndarray) -> np.ndarray:
    lo, hi = cfg.mag_low, cfg.mag_high
    mu, m2 = (lo + hi) / 2.0, (lo * lo + lo * hi + hi * hi) / 3.0
    v = p * m2 - (p * mu) ** 2
    return visibility_stat(centered_profiles(v))


def sample_rates_with_margin(cfg: PlantedConfig, rng: np.random.Generator) -> np.ndarray:
    """Rates such that every factor meant to be visible has ``vis >= min_visibility``.

    Keeps the visibility decision away from the threshold ``v_min`` so that the
    primary experiments test identification, not the visibility boundary
    (which gets its own stress condition via ``invisible_factors`` and
    ``min_visibility = 0``).
    """
    meant_visible = np.ones(cfg.K, dtype=bool)
    meant_visible[[int(k) for k in cfg.invisible_factors]] = False
    for _ in range(cfg.max_attempts):
        p = sample_rates(cfg, rng)
        vis = _independent_visibility(cfg, p)
        if cfg.min_visibility <= 0 or np.all(vis[meant_visible] >= cfg.min_visibility):
            return p
    raise RuntimeError("could not draw rates with the requested visibility margin; lower min_visibility")


# --------------------------------------------------------------------------
# population latent covariance (exact)
# --------------------------------------------------------------------------

def _block_corr(K: int, blocks: list, r: float) -> np.ndarray:
    R = np.eye(K)
    for b in blocks:
        b = [int(x) for x in b]
        for i in b:
            for j in b:
                if i != j:
                    R[i, j] = r
    return R


def _bvn_cdf(x: float, y: float, r: float) -> float:
    """P(X <= x, Y <= y) for standard bivariate normal with correlation r."""
    if r >= 1.0 - 1e-12:
        return float(stats.norm.cdf(min(x, y)))
    if r <= -1.0 + 1e-12:
        return float(max(0.0, stats.norm.cdf(x) - stats.norm.cdf(-y)))
    if abs(r) < 1e-15:
        return float(stats.norm.cdf(x) * stats.norm.cdf(y))
    return float(stats.multivariate_normal(mean=[0.0, 0.0], cov=[[1.0, r], [r, 1.0]]).cdf([x, y]))


def latent_covariance(cfg: PlantedConfig, p_e: np.ndarray) -> np.ndarray:
    r"""Exact ``Cov(z)`` for one environment.

    ``z_k = b_k s_k``, ``b_k = 1[g_k < Phi^{-1}(p_k)]``,
    ``s_k = lo + (hi-lo) Phi(h_k)``, with ``g`` and ``h`` independent Gaussian
    vectors with block-equicorrelation. Uses
    ``E[Phi(X)Phi(Y)] = 1/4 + arcsin(r/2)/(2 pi)`` for magnitudes and the
    bivariate normal CDF for switches.
    """
    K = cfg.K
    lo, hi = cfg.mag_low, cfg.mag_high
    w = hi - lo
    mu = (lo + hi) / 2.0
    m2 = (lo * lo + lo * hi + hi * hi) / 3.0
    Rs = _block_corr(K, cfg.switch_corr_blocks, cfg.switch_corr)
    Rm = _block_corr(K, cfg.magnitude_corr_blocks, cfg.magnitude_corr)
    t = stats.norm.ppf(p_e)
    C = np.zeros((K, K))
    for i in range(K):
        C[i, i] = p_e[i] * m2 - (p_e[i] * mu) ** 2
        for j in range(i + 1, K):
            Ebb = _bvn_cdf(t[i], t[j], Rs[i, j]) if Rs[i, j] != 0 else p_e[i] * p_e[j]
            if Rm[i, j] != 0:
                Ephiphi = 0.25 + math.asin(Rm[i, j] / 2.0) / (2.0 * math.pi)
                Ess = lo * lo + lo * w + w * w * Ephiphi
            else:
                Ess = mu * mu
            C[i, j] = C[j, i] = Ebb * Ess - p_e[i] * p_e[j] * mu * mu
    return C


def _random_correlation(d: int, rng: np.random.Generator, strength: float = 0.5) -> np.ndarray:
    W = rng.normal(size=(d, d))
    S = (1 - strength) * np.eye(d) + strength * (W @ W.T) / d
    s = np.sqrt(np.diag(S))
    return S / np.outer(s, s)


def noise_covariances(cfg: PlantedConfig, sigma: float, rng: np.random.Generator) -> tuple[np.ndarray, bool]:
    E, d = cfg.n_envs, cfg.d
    mode = cfg.noise_mode
    if mode in ("iid", "student_t"):
        N = np.repeat((sigma**2 * np.eye(d))[None], E, axis=0)
        return N, True
    if mode == "correlated_stationary":
        R = _random_correlation(d, rng)
        return np.repeat((sigma**2 * R)[None], E, axis=0), True
    lo, hi = math.log(cfg.noise_env_scale_min), math.log(cfg.noise_env_scale_max)
    if mode == "env_scaled":
        f = np.exp(rng.uniform(lo, hi, size=E))
        return np.stack([(sigma * fe) ** 2 * np.eye(d) for fe in f]), False
    if mode == "env_heteroscedastic":
        f = np.exp(rng.uniform(lo, hi, size=(E, d)))
        return np.stack([np.diag((sigma * fe) ** 2) for fe in f]), False
    raise ValueError(f"unknown noise_mode {mode!r}")


# --------------------------------------------------------------------------
# sampling
# --------------------------------------------------------------------------

def _equicorrelated_normals(n: int, K: int, blocks: list, r: float, rng: np.random.Generator) -> np.ndarray:
    g = rng.standard_normal((n, K))
    if r == 0 or not blocks:
        return g
    for b in blocks:
        b = [int(x) for x in b]
        shared = rng.standard_normal((n, 1))
        g[:, b] = math.sqrt(r) * shared + math.sqrt(max(0.0, 1.0 - r)) * g[:, b]
    return g


def sample_latents(cfg: PlantedConfig, p_e: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    K = cfg.K
    g = _equicorrelated_normals(n, K, cfg.switch_corr_blocks, cfg.switch_corr, rng)
    b = g < stats.norm.ppf(p_e)[None, :]
    h = _equicorrelated_normals(n, K, cfg.magnitude_corr_blocks, cfg.magnitude_corr, rng)
    s = cfg.mag_low + (cfg.mag_high - cfg.mag_low) * stats.norm.cdf(h)
    return b * s


def generate(
    cfg: PlantedConfig,
    bank: SeedBank,
    *,
    A: np.ndarray | None = None,
    rates: np.ndarray | None = None,
    v_min: float = 6e-3,
) -> PlantedDataset:
    """Draw one planted dataset.

    Separate child streams are used for the loading matrix, the rates, the
    noise structure and the samples, so an experiment can hold ``A`` fixed and
    change only activity rates (frequency shift) by passing ``A=`` explicitly
    or by reusing the ``loadings`` stream.
    """
    exempt: list = []
    if A is None:
        A, exempt = sample_loading_matrix(cfg, bank.child("loadings").rng())
    else:
        A = normalize_columns(np.asarray(A, dtype=float))[0]
        if A.shape != (cfg.d, cfg.K):
            raise ValueError(f"A has shape {A.shape}, config says ({cfg.d}, {cfg.K})")
    if rates is None:
        rates = sample_rates_with_margin(cfg, bank.child("rates").rng())
    rates = np.asarray(rates, dtype=float)

    E, d, K = cfg.n_envs, cfg.d, cfg.K
    lat_cov = np.stack([latent_covariance(cfg, rates[e]) for e in range(E)])
    variances = np.stack([np.diag(lat_cov[e]) for e in range(E)])
    signal_scale = math.sqrt(float(np.mean([np.trace(A @ lat_cov[e] @ A.T) for e in range(E)])) / d)
    sigma = cfg.noise_rel_sd * signal_scale
    noise_cov, noise_stationary = noise_covariances(cfg, sigma, bank.child("noise_structure").rng())
    pop_cov = np.stack([A @ lat_cov[e] @ A.T + noise_cov[e] for e in range(E)])

    profiles = centered_profiles(variances)
    visible = visibility_stat(profiles) >= v_min
    independent = all(np.allclose(lat_cov[e], np.diag(np.diag(lat_cov[e])), atol=1e-14) for e in range(E))

    rng = bank.child("samples").rng()
    Xs, envs, Zs = [], [], []
    for e, n_e in enumerate(cfg.n_list()):
        Z = sample_latents(cfg, rates[e], n_e, rng)
        if cfg.noise_mode == "student_t":
            df = cfg.student_df
            eps = rng.standard_t(df, size=(n_e, d)) * sigma * math.sqrt((df - 2.0) / df)
        else:
            L = np.linalg.cholesky(noise_cov[e] + 1e-18 * np.eye(d)) if sigma > 0 else np.zeros((d, d))
            eps = rng.standard_normal((n_e, d)) @ L.T
        Xs.append(Z @ A.T + eps)
        envs.append(np.full(n_e, e, dtype=int))
        if cfg.store_latents:
            Zs.append(Z)
    X = np.concatenate(Xs)
    env = np.concatenate(envs)

    oracle = Oracle(
        A=A,
        rates=rates,
        latent_cov=lat_cov,
        noise_cov=noise_cov,
        pop_cov=pop_cov,
        variances=variances,
        profiles=profiles,
        visible=visible,
        truth=multiplicity(A, cfg.tau, visible=visible),
        tau=cfg.tau,
        v_min=v_min,
        independent=independent,
        noise_stationary=noise_stationary,
        diversity=environmental_diversity(profiles, visible),
        signal_scale=signal_scale,
        noise_sd=sigma,
        exempt_entries=exempt,
        latents=np.concatenate(Zs) if cfg.store_latents else None,
    )
    observed = ObservedData(X=X, env=env, neuron_ids=np.arange(d), meta={"source": "planted"})
    return PlantedDataset(observed=observed, oracle=oracle, config=cfg)


def population_moments(oracle: Oracle) -> np.ndarray:
    """Exact ``E x d x d`` observed covariances (infinite-sample limit)."""
    return oracle.pop_cov.copy()


def check_two_explanations(A1: np.ndarray, v1: np.ndarray, A2: np.ndarray, v2: np.ndarray) -> float:
    """Max abs difference between the centred covariance stacks implied by two explanations.

    Zero (to rounding) means the two explanations are observationally
    equivalent at the level of second moments, whatever their multiplicities.
    """
    def stack(A, v):
        C = np.stack([A @ np.diag(ve) @ A.T for ve in v])
        return C - C.mean(axis=0, keepdims=True)

    return float(np.max(np.abs(stack(A1, v1) - stack(A2, v2))))
