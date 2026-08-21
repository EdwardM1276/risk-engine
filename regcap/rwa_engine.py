"""RegCap RWA engine: Vasicek IRB, credit/market/operational RWA, HHI concentration."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

from config.params import VASICEK_CORRELATION_PARAMS


def vasicek_asrf(
    pd: np.ndarray,
    lgd: np.ndarray,
    corr: np.ndarray | float = 0.15,
    alpha: float = 0.999,
    scaling: float = 1.06,
    maturity_adj: bool = True,
    maturity: np.ndarray | None = None,
) -> np.ndarray:
    """Basel III IRB Asymptotic Single Risk Factor (ASRF)."""
    vcp = VASICEK_CORRELATION_PARAMS
    pd_arr = np.clip(np.asarray(pd, dtype=float), vcp["pd_min"], vcp["pd_max"])
    corr_arr = np.clip(np.asarray(corr, dtype=float), 0.04, 0.30)
    if corr_arr.shape == ():
        corr_arr = np.full_like(pd_arr, float(corr_arr))
    z_alpha = norm.ppf(alpha)
    z_pd = norm.ppf(pd_arr)
    cond_pd = norm.cdf((z_pd + np.sqrt(corr_arr) * z_alpha) / np.sqrt(1.0 - corr_arr))
    k_ul = (cond_pd - pd_arr) * lgd
    if maturity_adj:
        mat = maturity if maturity is not None else np.full_like(pd_arr, 2.5)
        mat = np.clip(np.asarray(mat, dtype=float), 1.0, 5.0)
        b = (0.11852 - 0.05478 * np.log(np.clip(pd_arr, 1e-6, 1.0))) ** 2
        k_ul = k_ul * (1.0 + (mat - 2.5) * b) / (1.0 - 1.5 * b)
    return np.clip((lgd * pd_arr + k_ul) * scaling, 0.0, None)


def compute_credit_rwa(
    portfolio_df: pd.DataFrame,
    use_pit_for_irb: bool = False,
) -> pd.DataFrame:
    """Return portfolio dataframe with per-account IRB K ratio and RWA amounts."""
    result = portfolio_df.copy()
    pds = result["pit_pd_12m"].values if use_pit_for_irb else result["ttc_pd"].values
    lgds = result["downturn_lgd"].values if use_pit_for_irb else result["lgd"].values
    corrs = result["base_segment_corr"].values.astype(float)
    eads = result["ead"].values.astype(float)
    mats = np.clip(result["tenure_years"].replace(0, 2.0).values.astype(float), 1.0, 5.0)

    k_ratios = vasicek_asrf(pds, lgds, corrs, alpha=0.999, scaling=1.06,
                             maturity_adj=True, maturity=mats)
    result["irb_k_ratio"] = np.clip(k_ratios, 0.0, 0.99)
    result["credit_rwa_unit"] = result["irb_k_ratio"] * 12.5
    result["credit_rwa"] = eads * result["credit_rwa_unit"]

    sa_risk_weights: Dict[str, float] = {
        "Retail_Mortgage": 0.35, "Retail_Vehicle": 0.75, "Retail_CreditCard": 0.85,
        "Retail_Overdraft": 0.85, "SME_Corporate": 1.00, "Corporate_Large": 1.00,
        "Sovereign_Bank": 0.20,
    }
    result["sa_risk_weight"] = result["segment"].map(sa_risk_weights).fillna(1.0).astype(float)
    result["standardised_rwa"] = eads * result["sa_risk_weight"]
    result["concentration_hhi"] = float(((eads / max(eads.sum(), 1.0)) ** 2).sum())
    return result


def compute_market_rwa(
    total_credit_rwa: float,
    market_risk_pct: float = 0.07,
    macro_conditions: Dict[str, float] | None = None,
) -> Dict[str, float]:
    """FRTB-style market risk RWA sensitive to ZAR/sovereign stress."""
    ls_factor = 0.0
    zar_factor = 0.0
    if macro_conditions:
        ls_factor = (float(macro_conditions.get("load_shedding_stage", 2)) / 8.0) * 1.2
        zar_factor = float(macro_conditions.get("zar_usd_vol_change", 0.0)) * 1.5
    market_pct = market_risk_pct * (1.0 + ls_factor + zar_factor)
    base = total_credit_rwa * market_pct
    fx = base * 0.35 * (1.0 + zar_factor)
    rates = base * 0.25
    eq = base * 0.20
    comm = base * 0.20
    total = fx + rates + eq + comm
    return {
        "market_rwa_total": float(total),
        "fx_rwa": float(fx), "rates_rwa": float(rates),
        "equity_rwa": float(eq), "commodity_rwa": float(comm),
    }


def compute_operational_rwa(
    total_credit_rwa: float,
    total_assets: float,
    ls_stage: int = 2,
    gross_income_pct: float = 0.018,
    op_stress_impact: float = 0.10,
) -> Dict[str, float]:
    """Standardised operational risk RWA with loadshedding disruption uplift."""
    disruption = 1.0 + (float(ls_stage) / 8.0) * op_stress_impact * 3.0
    req = max(total_credit_rwa * 0.035, total_assets * gross_income_pct * 1.1) * disruption
    return {
        "operational_rwa": float(req * 12.5),
        "op_capital_req": float(req),
        "ls_disruption_factor": float(disruption),
    }


def compute_total_rwa(
    portfolio_df: pd.DataFrame,
    macro_conditions: Dict[str, float] | None = None,
) -> Dict:
    """Compute total RWA across credit (IRB), market (FRTB) and operational (SA-OR) risks."""
    credit_df = compute_credit_rwa(portfolio_df)
    tcr = float(credit_df["credit_rwa"].sum())
    market = compute_market_rwa(tcr, macro_conditions=macro_conditions)
    op = compute_operational_rwa(
        tcr, float(credit_df["ead"].sum() * 1.1),
        ls_stage=int((macro_conditions or {}).get("load_shedding_stage", 2)),
    )
    total = tcr + market["market_rwa_total"] + op["operational_rwa"]
    return {
        "credit_rwa_df": credit_df,
        "total_credit_rwa": tcr,
        "market": market,
        "operational": op,
        "total_rwa": float(total),
        "breakdown": {
            "Credit RWA": tcr,
            "Market RWA": float(market["market_rwa_total"]),
            "Operational RWA": float(op["operational_rwa"]),
        },
    }
