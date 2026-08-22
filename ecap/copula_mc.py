"""ECap Monte Carlo: Gaussian/t-copula for credit and stress simulations."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm, t

from config.params import PORTFOLIO_SEGMENTS


def build_factor_correlation_matrix(
    segments: List[str],
    macro_conditions: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """Segment-level systematic factor correlation matrix with sovereign nexus amplifier."""
    n = len(segments)
    R = np.zeros((n, n), dtype=float)
    ls = float((macro_conditions or {}).get("load_shedding_stage", 2)) / 8.0
    sov = float((macro_conditions or {}).get("sovereign_cds_change_bps", 0)) / 300.0
    stress = 1.0 + 0.3 * ls + 0.5 * sov

    for i, si in enumerate(segments):
        for j, sj in enumerate(segments):
            if i == j:
                R[i, j] = 1.0
                continue
            base = 0.25
            if "Retail" in si and "Retail" in sj:
                base = 0.45
            if si == "Sovereign_Bank" or sj == "Sovereign_Bank":
                base = 0.40 if si != sj else 0.55
                base *= (1.0 + 0.5 * sov)
            if "SME" in si or "SME" in sj:
                base += 0.05
            if "Corporate" in si and "Corporate" in sj:
                base = 0.50
            R[i, j] = float(np.clip(base * stress, 0.05, 0.95))
    return R


def simulate_copula_defaults(
    portfolio_df: pd.DataFrame,
    n_sims: int = 5000,
    copula_type: str = "t",
    t_df: int = 6,
    confidence_levels: Optional[List[float]] = None,
    macro_conditions: Optional[Dict[str, float]] = None,
    seed: int = 2024,
) -> Dict:
    """Simulate correlated defaults with distribution-consistent thresholds.

    Both copula choices use the same Gaussian factor model. The Gaussian path
    compares the latent variable with ``Phi^-1(PD)``. The t path applies one
    common chi-square scale to the full latent vector and compares it with
    ``t_df^-1(PD)``. Using a shared scale is what creates t-copula tail
    dependence; transforming a t variate with the normal CDF does not.
    """
    copula_name = copula_type.lower()
    if copula_name not in {"gaussian", "t"}:
        raise ValueError("copula_type must be 'Gaussian' or 't'")
    if n_sims < 1:
        raise ValueError("n_sims must be positive")
    if copula_name == "t" and t_df <= 2:
        raise ValueError("t_df must be greater than 2 for stable tail simulation")

    rng = np.random.default_rng(seed)
    cls = confidence_levels or [0.90, 0.95, 0.975, 0.99, 0.999]

    seg_list = list(PORTFOLIO_SEGMENTS.keys())
    segments = portfolio_df["segment"].values
    eads = portfolio_df["ead"].values.astype(float)
    lgds = portfolio_df["lgd"].values.astype(float)
    pds = portfolio_df["pit_pd_12m"].values.astype(float)
    n_loans = len(portfolio_df)
    seg_idx = np.array([seg_list.index(s) if s in seg_list else 3 for s in segments], dtype=int)

    R = build_factor_correlation_matrix(seg_list, macro_conditions)
    try:
        L = np.linalg.cholesky(R)
    except np.linalg.LinAlgError:
        R = 0.5 * (R + R.T) + np.eye(len(seg_list)) * 0.001
        L = np.linalg.cholesky(R)

    seg_corr_values = np.array([PORTFOLIO_SEGMENTS[s]["corr"] for s in segments], dtype=float)
    systematic_loading = np.sqrt(np.clip(seg_corr_values, 0.0, 0.99))
    idio_vol = np.sqrt(np.clip(1.0 - seg_corr_values, 0.01, 1.0))
    pd_clip = np.clip(pds, 1e-8, 1 - 1e-8)
    thresholds = (
        t.ppf(pd_clip, t_df)
        if copula_name == "t"
        else norm.ppf(pd_clip)
    )

    losses = np.zeros(n_sims, dtype=float)
    for sim in range(n_sims):
        systematic = (L @ rng.standard_normal(len(seg_list)))[seg_idx]
        latent = systematic_loading * systematic + idio_vol * rng.standard_normal(n_loans)
        if copula_name == "t":
            latent = latent / np.sqrt(max(rng.chisquare(t_df) / t_df, 1e-12))
        defaults = latent < thresholds
        lgd_rand = rng.beta(4, 6, n_loans) * 0.4 + 0.8
        losses[sim] = float(np.sum(eads * lgds * lgd_rand * defaults))

    sorted_losses = np.sort(losses)
    var_dict: Dict[float, float] = {}
    es_dict: Dict[float, float] = {}
    for cl in cls:
        idx = int(np.clip(np.floor(n_sims * cl), 0, n_sims - 1))
        var_dict[cl] = float(sorted_losses[idx])
        tail = sorted_losses[idx:]
        es_dict[cl] = float(np.mean(tail)) if len(tail) > 0 else var_dict[cl]

    el = float(np.mean(losses))
    ul999 = float(var_dict.get(0.999, el + 3.09 * np.std(losses)))
    ecap = max(ul999 - el, 0.0)
    tail_count_999 = max(1, int(np.ceil(n_sims * (1.0 - 0.999))))
    return {
        "simulated_losses": losses,
        "expected_loss": el,
        "VaR": var_dict,
        "Expected_Shortfall": es_dict,
        "credit_ecap_999": float(ecap),
        "credit_ecap_999_pct_ead": float(ecap / max(eads.sum(), 1.0)),
        "loss_std": float(np.std(losses)),
        "n_sims": int(n_sims),
        "copula_type": copula_name,
        "tail_observations_999": int(tail_count_999),
        "tail_estimate_warning": bool(tail_count_999 < 100),
    }


def compute_market_risk_ecap(
    total_assets: float,
    macro_conditions: Optional[Dict[str, float]] = None,
    n_sims: int = 2000,
    seed: int = 2025,
) -> Dict[str, float]:
    """FRTB-style Expected Shortfall 97.5% across 6 correlated market risk factors."""
    rng = np.random.default_rng(seed)
    mc = macro_conditions or {}
    ls = float(mc.get("load_shedding_stage", 2)) / 8.0
    sov = float(mc.get("sovereign_cds_change_bps", 0)) / 300.0
    gdp = float(mc.get("gdp_yoy", 0.012))
    stress_mult = 1.0 + 0.5 * ls + 0.4 * sov + 0.6 * max(0.0, 0.012 - gdp) * 10.0

    weights = np.array([0.30, 0.15, 0.15, 0.15, 0.12, 0.13], dtype=float)
    vols = np.array([0.18, 0.12, 0.20, 0.22, 0.25, 0.10], dtype=float) * stress_mult
    corr = np.full((6, 6), 0.35, dtype=float)
    np.fill_diagonal(corr, 1.0)
    L = np.linalg.cholesky(corr)
    exposure = total_assets * 0.30

    pnls = np.zeros(n_sims, dtype=float)
    for s in range(n_sims):
        z = L @ rng.standard_normal(6)
        ret = z * vols
        pnls[s] = -exposure * float(np.sum(ret * weights))

    sorted_pnls = np.sort(pnls)
    idx975 = int(np.floor(n_sims * 0.975))
    idx99 = int(np.floor(n_sims * 0.99))
    return {
        "market_ecap": float(max(np.mean(sorted_pnls[idx975:]), 0.0)),
        "market_pnl_sims": pnls,
        "market_VaR_99": float(sorted_pnls[idx99]),
        "market_ES_975": float(np.mean(sorted_pnls[idx975:])),
    }


def compute_operational_ecap(
    total_assets: float,
    n_scenarios: int = 50,
    ls_stage: int = 2,
    seed: int = 2026,
) -> Dict[str, float]:
    """Prototype operational-risk scenario simulation with loadshedding uplift.

    This is not a claim to implement the Basel Standardised Measurement
    Approach or any current supervisory operational-risk framework.
    """
    rng = np.random.default_rng(seed)
    gi = total_assets * 0.035
    factor = 1.0 + (float(ls_stage) / 8.0) * 0.6

    losses = []
    for _ in range(n_scenarios):
        freq = rng.poisson(lam=4 * factor)
        sev = rng.pareto(a=1.8, size=freq) * gi * 0.008 if freq > 0 else np.zeros(1)
        losses.append(float(sev.sum()))
    losses_arr = np.array(losses, dtype=float)
    sorted_loss = np.sort(losses_arr)
    idx999 = int(np.clip(np.floor(n_scenarios * 0.999), 0, n_scenarios - 1))
    var999 = float(sorted_loss[idx999])
    ecap = max(var999 - float(np.mean(losses_arr)), 0.0)
    return {
        "operational_ecap": float(ecap),
        "op_loss_sims": losses_arr,
        "op_VaR_999": var999,
    }
