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


def repair_correlation_matrix(
    matrix: np.ndarray,
    eigenvalue_floor: float = 1e-8,
) -> tuple[np.ndarray, Dict[str, object]]:
    """Project a symmetric matrix to a positive-definite correlation matrix.

    The returned diagnostics quantify the adjustment so model inputs are not
    silently repaired. Large corrections should trigger model review.
    """
    original = np.asarray(matrix, dtype=float)
    if original.ndim != 2 or original.shape[0] != original.shape[1]:
        raise ValueError("correlation matrix must be square")
    if not np.isfinite(original).all():
        raise ValueError("correlation matrix must contain only finite values")
    if eigenvalue_floor <= 0:
        raise ValueError("eigenvalue_floor must be positive")

    symmetric = 0.5 * (original + original.T)
    np.fill_diagonal(symmetric, 1.0)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    minimum_before = float(eigenvalues.min())
    clipped_eigenvalues = np.maximum(eigenvalues, eigenvalue_floor)
    projected = (eigenvectors * clipped_eigenvalues) @ eigenvectors.T
    scale = np.sqrt(np.maximum(np.diag(projected), eigenvalue_floor))
    repaired = projected / np.outer(scale, scale)
    repaired = np.clip(0.5 * (repaired + repaired.T), -1.0, 1.0)
    np.fill_diagonal(repaired, 1.0)
    minimum_after = float(np.linalg.eigvalsh(repaired).min())
    delta = repaired - original
    frobenius = float(np.linalg.norm(delta, ord="fro"))
    max_change = float(np.max(np.abs(delta)))
    diagnostics: Dict[str, object] = {
        "matrix_repaired": bool(frobenius > 1e-12),
        "minimum_eigenvalue_before": minimum_before,
        "minimum_eigenvalue_after": minimum_after,
        "frobenius_adjustment": frobenius,
        "max_element_adjustment": max_change,
        "warning": bool(frobenius > 0.05),
    }
    return repaired, diagnostics


def compute_convergence_diagnostics(
    losses: np.ndarray,
    confidence_level: float = 0.999,
    checkpoints: Optional[List[float]] = None,
    tolerance: float = 0.05,
) -> Dict[str, object]:
    """Report exploratory prefix stability, which is not tail validation.

    Checkpoints reuse nested samples from one seed. Passing this diagnostic
    therefore cannot establish independent replication stability or justify a
    formal interval.
    """
    values = np.asarray(losses, dtype=float)
    if values.size == 0:
        raise ValueError("losses must contain at least one observation")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")
    points = checkpoints or [0.25, 0.50, 0.75, 1.0]
    estimates = []
    counts = []
    for point in points:
        count = max(1, int(np.ceil(values.size * point)))
        sample = np.sort(values[:count])
        index = min(int(np.floor(count * confidence_level)), count - 1)
        estimates.append(float(sample[index]))
        counts.append(count)
    relative_changes = [
        abs(estimates[index] - estimates[index - 1]) / max(abs(estimates[index - 1]), 1.0)
        for index in range(1, len(estimates))
    ]
    checkpoint_stable = bool(relative_changes and relative_changes[-1] <= tolerance)
    return {
        "confidence_level": float(confidence_level),
        "checkpoints": [float(point) for point in points],
        "simulation_counts": counts,
        "tail_estimates": estimates,
        "relative_changes": relative_changes,
        "checkpoint_stable": checkpoint_stable,
        "converged": False,
        "independent_replicates": 1,
        "formal_interval_available": False,
        "interpretation": "Nested prefix stability only; not evidence of extreme-tail accuracy",
        "tolerance": float(tolerance),
    }


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

    raw_correlation = build_factor_correlation_matrix(seg_list, macro_conditions)
    R, correlation_diagnostics = repair_correlation_matrix(raw_correlation)
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
    expected_tail_count_999 = n_sims * (1.0 - 0.999)
    tail_count_999 = max(1, n_sims - int(np.floor(n_sims * 0.999)))
    convergence = compute_convergence_diagnostics(losses)
    warnings = []
    if correlation_diagnostics["warning"]:
        warnings.append(
            "Factor correlation matrix required a material positive-definite repair; "
            "review the stated adjustment before interpreting results."
        )
    if tail_count_999 < 1000:
        warnings.append("Empirical 99.9% tail is inadequately sampled for decision-grade interpretation.")
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
        "expected_tail_observations_999": float(expected_tail_count_999),
        "independent_replicates": 1,
        "tail_estimate_warning": bool(tail_count_999 < 1000),
        "tail_estimate_status": "Exploratory; no formal uncertainty interval or validated convergence claim",
        "convergence": convergence,
        "correlation_matrix_diagnostics": correlation_diagnostics,
        "warnings": warnings,
    }


def compute_market_risk_ecap(
    total_assets: float,
    macro_conditions: Optional[Dict[str, float]] = None,
    n_sims: int = 2000,
    seed: int = 2025,
    scenario_shocks: Optional[Dict[str, float]] = None,
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
    shocks = scenario_shocks or {}
    commodity_return = float(np.mean([
        float(shocks.get("gold_price_change", 0.0)),
        float(shocks.get("platinum_price_change", 0.0)),
        float(shocks.get("coal_price_change", 0.0)),
    ]))
    scenario_returns = np.array([
        0.0,
        0.0,
        0.0,
        float(shocks.get("jse_property_change", 0.0)),
        commodity_return,
        -float(shocks.get("sovereign_cds_change_bps", 0.0)) / 10000.0,
    ])

    pnls = np.zeros(n_sims, dtype=float)
    for s in range(n_sims):
        z = L @ rng.standard_normal(6)
        ret = z * vols
        pnls[s] = -exposure * float(np.sum((ret + scenario_returns) * weights))

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
    operational_shock_multiplier: float = 1.0,
) -> Dict[str, float]:
    """Prototype operational-risk scenario simulation with loadshedding uplift.

    This is not a claim to implement the Basel Standardised Measurement
    Approach or any current supervisory operational-risk framework.
    """
    if not np.isfinite(operational_shock_multiplier) or operational_shock_multiplier < 1.0:
        raise ValueError("operational_shock_multiplier must be finite and at least 1")
    rng = np.random.default_rng(seed)
    gi = total_assets * 0.035
    factor = (1.0 + (float(ls_stage) / 8.0) * 0.6) * operational_shock_multiplier

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
        "operational_shock_multiplier": float(operational_shock_multiplier),
    }
