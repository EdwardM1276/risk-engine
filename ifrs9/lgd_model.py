"""IFRS 9 LGD model: SA-adjusted with GN 3/2016 compliant collateral haircuts.

Segment-specific adjustments include:
- Retail Mortgage: LS-induced property depreciation + JSE property index,
  province-specific haircuts, LTV surcharges above 90%.
- Retail Vehicle: MFC recovery rate reference curve (age-adjusted).
- SME/Corp: LS depreciation, commodity exposure, sovereign CDS spread.
- Unsecured: Elevated recovery volatility tied to unemployment gap.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


def calculate_sa_adjusted_lgd(
    portfolio_df: pd.DataFrame,
    macro_conditions: Dict[str, float],
    market_data: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Return DataFrame with `lgd`, `downturn_lgd`, `collateral_haircut_pct`, `recovery_discount_rate`."""
    result = portfolio_df.copy()
    market_data = market_data or {}

    ls_stage = float(macro_conditions.get("load_shedding_stage", 2))
    gdp_yoy = float(macro_conditions.get("gdp_yoy", 0.0))
    unemp = float(macro_conditions.get("unemployment_rate", 0.32))
    jse_prop_change = float(market_data.get("jse_property_change", 0.0))
    sov_cds_change_bps = float(market_data.get("sovereign_cds_change_bps", 0))
    sovereign_cds_change = sov_cds_change_bps / 10000.0
    coal_change = float(market_data.get("coal_price_change", 0.0))
    platinum_change = float(market_data.get("platinum_price_change", 0.0))

    n = len(result)
    lgd_out = np.zeros(n, dtype=float)
    haircut = np.zeros(n, dtype=float)
    disc = np.zeros(n, dtype=float)

    seg = result["segment"].values
    base_lgd = result["base_segment_lgd"].values.astype(float)
    ltv = result["loan_to_value"].values.astype(float)
    ltv[~np.isfinite(ltv)] = 5.0
    col_val = result["collateral_value"].values.astype(float)
    ls_vuln = result["loadshedding_vulnerability_score"].values.astype(float) / 5.0
    dpd = result["dpd"].values.astype(int)
    tenure = result["tenure_years"].values.astype(float)
    province = result["province"].values
    principal = result["principal_outstanding"].values.astype(float)
    undrawn = result["undrawn_limit"].values.astype(float)
    base_ccf = result["base_segment_ccf"].values.astype(float)

    for i in range(n):
        if seg[i] == "Retail_Mortgage":
            ls_dep = (ls_stage / 8.0) * 0.08 * ls_vuln[i]
            prop_dep = ls_dep - 0.6 * jse_prop_change
            h = 0.12 + prop_dep
            if ltv[i] > 0.90:
                h += (ltv[i] - 0.90) * 1.5
            if province[i] in ("Eastern Cape", "Limpopo"):
                h += 0.04
            col_adj = max(col_val[i] * (1 - h), 0.0)
            loan = principal[i] + undrawn[i] * base_ccf[i]
            loss_id = max(loan - col_adj, 0.0) / max(loan, 1.0)
            econ = np.clip(base_lgd[i] * 0.45 + loss_id * 0.55, 0.05, 0.70)
            econ *= 1.0 + 0.3 * (unemp - 0.32)
            lgd_out[i] = np.clip(econ, 0.05, 0.75)
            haircut[i] = h
            disc[i] = np.clip(0.15 + ls_stage * 0.01, 0.15, 0.30)

        elif seg[i] == "Retail_Vehicle":
            age_factor = min(tenure[i] / 7.0, 1.0)
            h = 0.35 + age_factor * 0.30
            h += (ls_stage / 8.0) * 0.05 * ls_vuln[i]
            mfc_recovery = 1.0 - 0.55 - age_factor * 0.15
            h += max(0.0, 0.55 - mfc_recovery)
            col_adj = max(col_val[i] * (1 - h), 0.0)
            loan = principal[i] + undrawn[i] * base_ccf[i]
            loss_id = max(loan - col_adj, 0.0) / max(loan, 1.0)
            econ = np.clip(base_lgd[i] * 0.35 + loss_id * 0.65, 0.15, 0.70)
            econ *= 1.0 + 0.2 * (unemp - 0.32)
            lgd_out[i] = np.clip(econ, 0.15, 0.75)
            haircut[i] = h
            disc[i] = np.clip(0.20 + ls_stage * 0.01, 0.20, 0.35)

        elif seg[i] in ("Retail_CreditCard", "Retail_Overdraft"):
            u = 1.0 + (unemp - 0.32) * 1.5
            u += (ls_stage / 8.0) * 0.15 * ls_vuln[i]
            u += (gdp_yoy - 0.015) * (-2.0)
            lgd_out[i] = np.clip(base_lgd[i] * u, 0.30, 0.85)
            haircut[i] = 1.0
            disc[i] = np.clip(0.35 + ls_stage * 0.015, 0.35, 0.55)

        elif seg[i] == "SME_Corporate":
            ls_dep = (ls_stage / 8.0) * 0.18 * ls_vuln[i]
            coal_exposure = 1.0 if province[i] in ("Mpumalanga", "Limpopo") else 0.2
            comm = ls_dep - 0.3 * platinum_change - 0.2 * coal_change * coal_exposure
            h = 0.25 + comm + 0.5 * max(0.0, ltv[i] - 0.70)
            if dpd[i] >= 90:
                h += 0.10
            col_adj = max(col_val[i] * (1 - h), 0.0)
            loan = principal[i] + undrawn[i] * base_ccf[i]
            loss_id = max(loan - col_adj, 0.0) / max(loan, 1.0)
            econ = np.clip(base_lgd[i] * 0.40 + loss_id * 0.60, 0.15, 0.75)
            econ *= 1.0 + 0.4 * (unemp - 0.32) + 0.5 * (0.015 - gdp_yoy)
            lgd_out[i] = np.clip(econ, 0.15, 0.80)
            haircut[i] = h
            disc[i] = np.clip(0.25 + ls_stage * 0.02, 0.25, 0.45)

        elif seg[i] == "Corporate_Large":
            ls_impact = (ls_stage / 8.0) * 0.08 * ls_vuln[i]
            sov = sovereign_cds_change * 0.5
            h = 0.20 + ls_impact + sov + 0.4 * max(0.0, ltv[i] - 0.60)
            col_adj = max(col_val[i] * (1 - h), 0.0)
            loan = principal[i] + undrawn[i] * base_ccf[i]
            loss_id = max(loan - col_adj, 0.0) / max(loan, 1.0)
            econ = np.clip(base_lgd[i] * 0.45 + loss_id * 0.55, 0.10, 0.70)
            econ *= 1.0 + 0.25 * (0.015 - gdp_yoy)
            lgd_out[i] = np.clip(econ, 0.10, 0.75)
            haircut[i] = h
            disc[i] = np.clip(0.20 + ls_stage * 0.01, 0.20, 0.35)

        else:  # Sovereign_Bank
            h = 0.10 + sovereign_cds_change * 0.8
            lgd_out[i] = np.clip(base_lgd[i] * (1 + 0.5 * sovereign_cds_change * 100), 0.02, 0.60)
            haircut[i] = h
            disc[i] = np.clip(0.10 + ls_stage * 0.005, 0.10, 0.20)

        if dpd[i] >= 180:
            lgd_out[i] = np.clip(lgd_out[i] * 1.15, lgd_out[i], 0.95)
        elif dpd[i] >= 90:
            lgd_out[i] = np.clip(lgd_out[i] * 1.08, lgd_out[i], 0.95)

    result["lgd"] = np.clip(lgd_out, 0.01, 0.95)
    result["collateral_haircut_pct"] = haircut
    result["recovery_discount_rate"] = disc
    result["downturn_lgd"] = np.clip(result["lgd"] * 1.15, result["lgd"], 0.98)
    return result
