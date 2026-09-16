"""RegCap RWA engine: Basel III IRB Vasicek ASRF, standardised approach,
output floor mechanics, and stress-sensitive market/operational RWA."""

from __future__ import annotations

from datetime import datetime
from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import norm

from config.params import (
    BANK_BENCHMARK_PROFILES,
    DEFAULT_BANK_PROFILE,
    OUTPUT_FLOOR,
    PORTFOLIO_SEGMENTS,
    VASICEK_CORRELATION_PARAMS,
)

# Segment classes for regulatory asset-correlation formulas (Basel CRE31).
_SEGMENT_CLASS: Dict[str, str] = {
    "Retail_Mortgage": "mortgage",
    "Retail_Vehicle": "other_retail",
    "Retail_CreditCard": "qrre",
    "Retail_Overdraft": "other_retail",
    "SME_Corporate": "sme",
    "Corporate_Large": "corporate",
    "Sovereign_Bank": "corporate",
}

# Maturity adjustment applies to corporate/sovereign/bank exposures only.
_MATURITY_ADJ_CLASSES = {"corporate", "sme"}


def basel_asset_correlation(pd_arr: np.ndarray, segment_class: str) -> np.ndarray:
    """Regulatory asset correlation R per Basel CRE31 formulas."""
    pd_arr = np.asarray(pd_arr, dtype=float)
    if segment_class == "mortgage":
        return np.full_like(pd_arr, 0.15)
    if segment_class == "qrre":
        return np.full_like(pd_arr, 0.04)
    if segment_class == "other_retail":
        w = (1.0 - np.exp(-35.0 * pd_arr)) / (1.0 - np.exp(-35.0))
        return 0.03 * w + 0.16 * (1.0 - w)
    # Corporate / sovereign / bank (SME gets a size adjustment of -0.04
    # as a mid-range firm-size proxy).
    w = (1.0 - np.exp(-50.0 * pd_arr)) / (1.0 - np.exp(-50.0))
    r = 0.12 * w + 0.24 * (1.0 - w)
    if segment_class == "sme":
        r = r - 0.04
    return r


def vasicek_asrf(
    pd: np.ndarray,
    lgd: np.ndarray,
    corr: np.ndarray | float = 0.15,
    alpha: float = 0.999,
    scaling: float = 1.0,
    maturity_adj: bool = True,
    maturity: np.ndarray | None = None,
) -> np.ndarray:
    """Basel III IRB ASRF capital requirement K (unexpected loss only).

    K = LGD * [Phi((Phi^-1(PD) + sqrt(R) Phi^-1(alpha)) / sqrt(1-R)) - PD] * MA

    Expected loss is excluded from K (covered by provisions), and the legacy
    1.06 scaling factor is removed under the final Basel III framework.
    """
    vcp = VASICEK_CORRELATION_PARAMS
    pd_arr = np.clip(np.asarray(pd, dtype=float), vcp["pd_min"], vcp["pd_max"])
    lgd_arr = np.clip(np.asarray(lgd, dtype=float), 0.0, 1.0)
    corr_arr = np.clip(np.asarray(corr, dtype=float), 0.03, 0.30)
    if corr_arr.shape == ():
        corr_arr = np.full_like(pd_arr, float(corr_arr))
    z_alpha = norm.ppf(alpha)
    z_pd = norm.ppf(pd_arr)
    cond_pd = norm.cdf((z_pd + np.sqrt(corr_arr) * z_alpha) / np.sqrt(1.0 - corr_arr))
    k_ul = (cond_pd - pd_arr) * lgd_arr
    if maturity_adj:
        mat = maturity if maturity is not None else np.full_like(pd_arr, 2.5)
        mat = np.clip(np.asarray(mat, dtype=float), 1.0, 5.0)
        b = (0.11852 - 0.05478 * np.log(np.clip(pd_arr, 1e-6, 1.0))) ** 2
        k_ul = k_ul * (1.0 + (mat - 2.5) * b) / (1.0 - 1.5 * b)
    return np.clip(k_ul * scaling, 0.0, None)


def compute_credit_rwa(
    portfolio_df: pd.DataFrame,
    use_pit_for_irb: bool = False,
    lgd_calibration_factor: float = 1.0,
) -> pd.DataFrame:
    """Return portfolio dataframe with per-account IRB K, IRB RWA, and
    standardised RWA amounts.

    ``lgd_calibration_factor`` is a documented central-tendency calibration
    scalar applied to regulatory (downturn) LGD so that portfolio RWA density
    matches the D-SIB benchmark; it is solved in data/calibration.py and
    reported in the run metadata for auditability.
    """
    result = portfolio_df.copy()
    pds = result["pit_pd_12m"].values if use_pit_for_irb else result["ttc_pd"].values
    lgds = result["lgd"].values.astype(float) * float(lgd_calibration_factor)
    lgds = np.clip(lgds, 0.01, 1.0)
    eads = result["ead"].values.astype(float)
    mats = np.clip(result["tenure_years"].replace(0, 2.0).values.astype(float), 1.0, 5.0)

    seg_class = result["segment"].map(_SEGMENT_CLASS).fillna("corporate")
    corrs = np.empty(len(result), dtype=float)
    mat_adj_mask = np.zeros(len(result), dtype=bool)
    for cls in seg_class.unique():
        mask = (seg_class == cls).values
        corrs[mask] = basel_asset_correlation(pds[mask], cls)
        if cls in _MATURITY_ADJ_CLASSES:
            mat_adj_mask[mask] = True

    k_ratios = np.where(
        mat_adj_mask,
        vasicek_asrf(pds, lgds, corrs, alpha=0.999, scaling=1.0,
                     maturity_adj=True, maturity=mats),
        vasicek_asrf(pds, lgds, corrs, alpha=0.999, scaling=1.0,
                     maturity_adj=False),
    )
    result["regulatory_corr"] = corrs
    result["irb_k_ratio"] = np.clip(k_ratios, 0.0, 0.99)
    result["credit_rwa_unit"] = result["irb_k_ratio"] * 12.5
    result["credit_rwa"] = eads * result["credit_rwa_unit"]

    sa_risk_weights = {seg: cfg["sa_rw"] for seg, cfg in PORTFOLIO_SEGMENTS.items()}
    result["sa_risk_weight"] = result["segment"].map(sa_risk_weights).fillna(1.0).astype(float)
    result["standardised_rwa"] = eads * result["sa_risk_weight"]
    result["concentration_hhi"] = float(((eads / max(eads.sum(), 1.0)) ** 2).sum())
    return result


def compute_market_rwa(
    total_credit_rwa: float,
    market_share: float,
    credit_share: float,
    macro_conditions: Dict[str, float] | None = None,
) -> Dict[str, float]:
    """Market risk RWA scaled off the credit RWA base to match the benchmark
    composition (market_share / credit_share), with ZAR/loadshedding stress."""
    ls_factor = 0.0
    zar_factor = 0.0
    if macro_conditions:
        ls_factor = max(0.0, (float(macro_conditions.get("load_shedding_stage", 2)) - 2.0) / 8.0) * 0.5
        zar_factor = max(0.0, float(macro_conditions.get("zar_usd_vol_change", 0.0))) * 1.0
    base = total_credit_rwa * (market_share / max(credit_share, 1e-9))
    total = base * (1.0 + ls_factor + zar_factor)
    fx = total * 0.35
    rates = total * 0.25
    eq = total * 0.20
    comm = total * 0.20
    return {
        "market_rwa_total": float(total),
        "fx_rwa": float(fx), "rates_rwa": float(rates),
        "equity_rwa": float(eq), "commodity_rwa": float(comm),
    }


def compute_operational_rwa(
    total_credit_rwa: float,
    oprisk_share: float,
    credit_share: float,
    ls_stage: int = 2,
    op_stress_impact: float = 0.10,
) -> Dict[str, float]:
    """Operational risk RWA scaled to the benchmark composition with a
    loadshedding business-disruption uplift."""
    disruption = 1.0 + max(0.0, (float(ls_stage) - 2.0) / 8.0) * op_stress_impact * 3.0
    rwa = total_credit_rwa * (oprisk_share / max(credit_share, 1e-9)) * disruption
    return {
        "operational_rwa": float(rwa),
        "op_capital_req": float(rwa * 0.08),
        "ls_disruption_factor": float(disruption),
    }


def current_output_floor_pct(year: int | None = None) -> float:
    """Output floor percentage under the phase-in schedule for a given year."""
    y = int(year if year is not None else datetime.now().year)
    phase_in = OUTPUT_FLOOR["phase_in"]
    years = sorted(phase_in)
    if y <= years[0]:
        return float(phase_in[years[0]])
    if y >= years[-1]:
        return float(phase_in[years[-1]])
    return float(phase_in[y])


def apply_output_floor(
    modelled_total_rwa: float,
    standardised_total_rwa: float,
    floor_pct: float | None = None,
) -> Dict[str, float]:
    """Basel III output floor: final RWA = max(modelled, floor_pct x standardised)."""
    fp = float(floor_pct if floor_pct is not None else current_output_floor_pct())
    floor_rwa = fp * standardised_total_rwa
    final_rwa = max(modelled_total_rwa, floor_rwa)
    return {
        "floor_pct": fp,
        "modelled_rwa": float(modelled_total_rwa),
        "standardised_rwa": float(standardised_total_rwa),
        "floor_rwa": float(floor_rwa),
        "final_rwa": float(final_rwa),
        "floor_applied": bool(floor_rwa > modelled_total_rwa),
        "floor_headroom": float(modelled_total_rwa - floor_rwa),
    }


def compute_total_rwa(
    portfolio_df: pd.DataFrame,
    macro_conditions: Dict[str, float] | None = None,
    bank_profile: str = DEFAULT_BANK_PROFILE,
    lgd_calibration_factor: float = 1.0,
) -> Dict:
    """Compute total RWA: IRB credit + benchmark-composition market/op/other,
    with the Basel III output floor applied against the standardised total.

    Market, operational, and 'other' RWA are scaled to the benchmark RWA
    composition of the selected bank profile (default: D-SIB average of
    credit 72% / operational 14% / market 4% / other 10%).
    """
    profile = BANK_BENCHMARK_PROFILES.get(bank_profile, BANK_BENCHMARK_PROFILES[DEFAULT_BANK_PROFILE])
    credit_share = float(profile["credit_rwa_share"])
    market_share = float(profile["market_rwa_share"])
    oprisk_share = float(profile["oprisk_rwa_share"])
    other_share = float(profile.get("other_rwa_share", 0.0))

    credit_df = compute_credit_rwa(portfolio_df, lgd_calibration_factor=lgd_calibration_factor)
    tcr = float(credit_df["credit_rwa"].sum())
    sa_credit = float(credit_df["standardised_rwa"].sum())
    ls_stage = int((macro_conditions or {}).get("load_shedding_stage", 2))

    market = compute_market_rwa(tcr, market_share, credit_share, macro_conditions=macro_conditions)
    op = compute_operational_rwa(tcr, oprisk_share, credit_share, ls_stage=ls_stage)
    other_rwa = tcr * (other_share / max(credit_share, 1e-9))

    modelled_total = tcr + market["market_rwa_total"] + op["operational_rwa"] + other_rwa
    # Standardised total: standardised credit plus the same non-credit RWA
    # (market/operational are already standardised-style measures).
    standardised_total = sa_credit + market["market_rwa_total"] + op["operational_rwa"] + other_rwa
    floor = apply_output_floor(modelled_total, standardised_total)
    total = floor["final_rwa"]

    ead_total = float(credit_df["ead"].sum())
    return {
        "methodology": {
            "credit": "Basel III IRB Vasicek ASRF (UL-only, regulatory correlations)",
            "market": "Benchmark-composition scaling with FX/loadshedding stress",
            "operational": "Benchmark-composition scaling with disruption uplift",
            "output_floor": f"max(modelled, {floor['floor_pct']:.1%} x standardised)",
            "regulatory_use": "Prototype only; not a regulatory reporting calculation",
        },
        "bank_profile": bank_profile,
        "credit_rwa_df": credit_df,
        "total_credit_rwa": tcr,
        "standardised_credit_rwa": sa_credit,
        "market": market,
        "operational": op,
        "other_rwa": float(other_rwa),
        "output_floor": floor,
        "total_rwa": float(total),
        "rwa_density": float(total / ead_total) if ead_total > 0 else float("nan"),
        "breakdown": {
            "Credit RWA": tcr,
            "Market RWA": float(market["market_rwa_total"]),
            "Operational RWA": float(op["operational_rwa"]),
            "Other RWA": float(other_rwa),
        },
    }
