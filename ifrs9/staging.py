"""IFRS 9 staging and ECL pipeline (SICR + provision calculation).

Implements:
    - Backstop triggers: NCA 30/90 dpd, debt review, judgement, admin order
    - Quantitative SICR: 3x, 2.5x, +225bps thresholds per rating bucket
    - Qualitative SICR: SME loadshedding cashflow indicator, sub-investment rating
    - ECL: 12-month for Stage 1, lifetime for Stage 2 and Stage 3
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from config.params import NCA_THRESHOLDS, SICR_TRIGGERS


def assign_ifrs9_staging(
    portfolio_df: pd.DataFrame,
    macro_conditions: Dict[str, float],
) -> pd.DataFrame:
    """Assign IFRS9 stages: Stage1=Performing, Stage2=SICR, Stage3=Credit Impaired."""
    result = portfolio_df.copy()
    ls_stage = float(macro_conditions.get("load_shedding_stage", 2))

    dpd = result["dpd"].values.astype(int)
    debt = result["debt_review_flag"].values.astype(bool)
    judgement = result["judgement_flag"].values.astype(bool)
    admin = result["administration_order"].values.astype(bool)
    pit_pd = result["pit_pd_12m"].values.astype(float)
    ttc_pd = result["ttc_pd"].values.astype(float)
    seg = result["segment"].values
    ls_vuln = result["loadshedding_vulnerability_score"].values.astype(float) / 5.0
    rating = result["internal_rating"].values

    n = len(result)
    stage_arr = np.ones(n, dtype=int)
    basis_arr = np.empty(n, dtype=object)
    qs_arr = np.zeros(n, dtype=bool)
    reasons = [[] for _ in range(n)]

    for i in range(n):
        pd_ratio = pit_pd[i] / max(ttc_pd[i], 1e-6)
        pd_inc = pit_pd[i] - ttc_pd[i]

        qsicr = False
        if ttc_pd[i] <= 0.005 and pd_ratio >= 3.0:
            qsicr = True
            reasons[i].append("Low-PD 3x increase")
        elif ttc_pd[i] <= 0.02 and pd_ratio >= 2.5:
            qsicr = True
            reasons[i].append("Mid-PD 2.5x increase")
        elif ttc_pd[i] > 0.02 and pd_inc >= 0.0225:
            qsicr = True
            reasons[i].append("High-PD +225bps increase")

        if SICR_TRIGGERS["loadshedding_sme_cashflow_stage2"] and "SME" in seg[i]:
            if (ls_stage / 8.0) * ls_vuln[i] > 0.35 and pd_ratio > 1.4:
                qsicr = True
                reasons[i].append("SME loadshedding cashflow stress")

        if rating[i] in ("B", "CCC") and not debt[i]:
            qsicr = True
            reasons[i].append("Sub-investment grade")

        dpd_s3 = (dpd[i] >= NCA_THRESHOLDS["dpd_stage3"]) and SICR_TRIGGERS["dpd_90_plus_stage3"]
        debt_s3 = bool(debt[i]) and SICR_TRIGGERS["debt_review_stage3"]
        admin_s3 = bool(admin[i]) and SICR_TRIGGERS["administration_order_stage3"]
        j_s3 = bool(judgement[i]) and SICR_TRIGGERS["judgement_stage3"]

        dpd_s2 = (dpd[i] >= NCA_THRESHOLDS["dpd_stage2"]) and SICR_TRIGGERS["dpd_30_plus_stage2"]

        if dpd_s3 or debt_s3 or admin_s3 or j_s3:
            stage_arr[i] = 3
            if dpd_s3:
                reasons[i].append("NCA 90+ dpd")
            if debt_s3:
                reasons[i].append("Debt review")
            if admin_s3:
                reasons[i].append("Administration order")
            if j_s3:
                reasons[i].append("Judgement")
            basis_arr[i] = "Credit-impaired (backstop)"
        elif dpd_s2 or qsicr:
            stage_arr[i] = 2
            if dpd_s2:
                reasons[i].append("NCA 30+ dpd backstop")
            basis_arr[i] = "SICR detected"
        else:
            basis_arr[i] = "Performing"
            if not reasons[i]:
                reasons[i].append("No SICR indicators")

        qs_arr[i] = qsicr

    result["ifrs9_stage"] = stage_arr
    result["staging_basis"] = basis_arr
    result["sicr_reason"] = ["; ".join(r) for r in reasons]
    result["quantitative_sicr_flag"] = qs_arr
    return result


def calculate_ecl(
   
    portfolio_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute 12m, lifetime, and stage-applied ECL columns."""
    result = portfolio_df.copy()
    ead = result["ead"].values.astype(float)
    pit = result["pit_pd_12m"].values.astype(float)
    life = result["lifetime_pd"].values.astype(float)
    lgd = result["lgd"].values.astype(float)
    dlgd = result["downturn_lgd"].values.astype(float)
    stage = result["ifrs9_stage"].values.astype(int)

    ecl_12 = ead * pit * lgd
    ecl_life = ead * life * lgd
    ecl_applied = np.where(stage >= 2, ecl_life, ecl_12)
    hor = np.where(stage >= 2, "Lifetime", "12-month PIT")
    downturn_applied = np.where(stage >= 2, ead * life * dlgd, ead * pit * dlgd)

    result["12m_ecl"] = ecl_12
    result["lifetime_ecl"] = ecl_life
    result["ecl"] = ecl_applied
    result["ecl_horizon_applied"] = hor
    result["downturn_ecl"] = downturn_applied
    return result


def run_full_ifrs9_pipeline(
   
    portfolio_df: pd.DataFrame,
    macro_conditions: Dict[str, float],
    market_data: Dict[str, float] | None = None,
    forecast_horizon_months: int = 12,
) -> pd.DataFrame:
    """Complete IFRS 9 pipeline: PD -> LGD -> EAD -> Staging -> ECL."""
    from ifrs9.ead_model import calculate_elevated_ccf_ead
    from ifrs9.lgd_model import calculate_sa_adjusted_lgd
    from ifrs9.pd_model import convert_ttc_to_pit

    df = convert_ttc_to_pit(portfolio_df, macro_conditions, forecast_horizon_months)
    df = calculate_sa_adjusted_lgd(df, macro_conditions, market_data)
    df = calculate_elevated_ccf_ead(df, macro_conditions)
    df = assign_ifrs9_staging(df, macro_conditions)
    df = calculate_ecl(df)
    return df
