"""Full RegCap stack: Pillar 1 minima, buffers, capital resources, ratios, MDA triggers."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from config.params import CAPITAL_BUFFERS, PILLAR1_MINIMA


def compute_full_capital_stack(
    total_rwa: float,
    total_assets: float,
    ifrs9_ecl_total: float,
    institution_size: str = "Large_D-SIB",
    ccyb_rate: float | None = None,
    d_sib_score: float = 0.020,
    include_pillar2: bool = True,
) -> Dict:
    """Return full breakdown of percentage + absolute capital requirements."""
    ccyb_rate = CAPITAL_BUFFERS["CCyB"] if ccyb_rate is None else float(ccyb_rate)

    cet1_min = PILLAR1_MINIMA["CET1"]
    t1_min = PILLAR1_MINIMA["Tier1"]
    tot_min = PILLAR1_MINIMA["Total_Capital"]

    ccb = CAPITAL_BUFFERS["CCB"]
    ccyb = ccyb_rate

    if institution_size.startswith("Large") or "D-SIB" in institution_size:
        d_sib = float(np.clip(d_sib_score, CAPITAL_BUFFERS["D_SIB_MIN"], CAPITAL_BUFFERS["D_SIB_MAX"]))
    elif institution_size == "Medium":
        d_sib = CAPITAL_BUFFERS["D_SIB_MIN"] * 0.5
    else:
        d_sib = 0.0
    hla = CAPITAL_BUFFERS["HLA"] if institution_size in ("Large_D-SIB", "Large") else 0.0
    pillar2 = 0.015 if include_pillar2 else 0.0
    combined = ccb + ccyb + d_sib + hla + pillar2

    cet1_req = cet1_min + combined
    t1_req = t1_min + combined
    tot_req = tot_min + combined

    leverage_denom = total_assets * 1.05
    t1_lv_min = 0.030
    t1_lv_buf = (0.020 * (d_sib / CAPITAL_BUFFERS["D_SIB_MAX"])) if d_sib > 0 else 0.0
    lv_req = leverage_denom * (t1_lv_min + t1_lv_buf)

    return {
        "percentages": {
            "CET1 Min (Pillar 1)": cet1_min,
            "Tier1 Min (Pillar 1)": t1_min,
            "Total Cap Min (Pillar 1)": tot_min,
            "CCB (Capital Conservation)": ccb,
            "CCyB (Countercyclical 2026)": ccyb,
            "D-SIB Buffer": d_sib,
            "HLA Buffer": hla,
            "Pillar 2 Add-on": pillar2,
            "Total Combined Buffer": combined,
            "CET1 + Buffers": cet1_req,
            "T1 + Buffers": t1_req,
            "Total + Buffers": tot_req,
        },
        "absolute_amounts_rwa_based": {
            "CET1 Required": total_rwa * cet1_req,
            "Tier1 Required": total_rwa * t1_req,
            "Total Capital Required": total_rwa * tot_req,
        },
        "leverage_ratio": {
            "Denominator (Total Assets adj)": float(leverage_denom),
            "Minimum Tier1 Leverage": t1_lv_min,
            "D-SIB Leverage Buffer": t1_lv_buf,
            "Tier1 Leverage + Buffer": t1_lv_min + t1_lv_buf,
            "Required Tier1 (LR)": float(lv_req),
        },
        "total_conservation_requirement": float(max(total_rwa * tot_req, lv_req)),
    }


def generate_default_capital_resources(
    total_rwa: float,
    ifrs9_ecl_total: float,
    car_target: float = 0.175,
    cet1_ratio_target: float = 0.130,
) -> Dict:
    """Build a realistic capital resources breakdown consistent with SA large-bank Pillar 3 disclosures."""
    tot_cap = total_rwa * car_target
    cet1 = total_rwa * cet1_ratio_target
    at1 = max(total_rwa * 0.015, 0.0)
    t2_cap = max(tot_cap - cet1 - at1, total_rwa * 0.020)
    at1 = max(tot_cap - cet1 - t2_cap, 0.0)

    gross_cet1 = cet1 / (1.0 - 0.08)
    ded = gross_cet1 - cet1

    return {
        "CET1": {
            "Gross CET1 before deductions": float(gross_cet1),
            "Retained Earnings": float(gross_cet1 * 0.65),
            "Share Premium": float(gross_cet1 * 0.15),
            "Other Reserves": float(gross_cet1 * 0.12),
            "Minority Interest (CET1)": float(gross_cet1 * 0.03),
            "AOCI / Fair Value Reserves": float(gross_cet1 * 0.05),
            "CET1 Deductions (Goodwill, DTAs, etc.)": float(ded),
            "Net CET1 Available": float(cet1),
        },
        "Additional Tier 1 (AT1)": {
            "AT1 Capital Instruments": float(at1),
            "Net AT1 Available": float(at1),
        },
        "Tier 2": {
            "Subordinated Debt": float(t2_cap * 0.60),
            "General Provisions in Excess of EL":
                float(max(0.0, total_rwa * 0.0125 - ifrs9_ecl_total * 0.40)),
            "Revaluation Reserves (55% haircut)": float(t2_cap * 0.15),
            "Other T2": float(t2_cap * 0.25),
            "Net T2 Available": float(t2_cap),
        },
        "Total Capital Available": float(cet1 + at1 + t2_cap),
        "CET1 Available": float(cet1),
        "Tier1 Available (CET1+AT1)": float(cet1 + at1),
        "T2 Available": float(t2_cap),
    }


def compute_capital_ratios_and_erosion(
    capital_resources: Dict,
    capital_stack: Dict,
    total_rwa: float,
    total_assets: float,
) -> Dict:
    """Return all ratios + conservation buffer utilisation level."""
    cet1 = float(capital_resources["CET1 Available"])
    t1 = float(capital_resources["Tier1 Available (CET1+AT1)"])
    tot_cap = float(capital_resources["Total Capital Available"])
    pct = capital_stack["percentages"]
    abs_req = capital_stack["absolute_amounts_rwa_based"]

    cet1_ratio = cet1 / max(total_rwa, 1.0)
    t1_ratio = t1 / max(total_rwa, 1.0)
    car = tot_cap / max(total_rwa, 1.0)
    lv_ratio = t1 / max(capital_stack["leverage_ratio"]["Denominator (Total Assets adj)"], 1.0)

    buf_req = total_rwa * pct["Total Combined Buffer"]
    cet1_above = cet1 - total_rwa * pct["CET1 Min (Pillar 1)"]
    buf_util = float(np.clip(1.0 - (cet1_above / max(buf_req, 1.0)), 0.0, 1.5))

    if buf_util < 0.5:
        level = "0 - No restrictions"
        payout = 0.0
    elif buf_util < 0.6325:
        level = "1 - Up to 40% restrictions"
        payout = 0.20
    elif buf_util < 0.7975:
        level = "2 - Up to 60% restrictions"
        payout = 0.50
    elif buf_util < 1.0:
        level = "3 - Up to 80% restrictions"
        payout = 0.80
    else:
        level = "4 - 100% restrictions (MDA trigger)"
        payout = 1.0

    return {
        "CET1 Ratio": float(cet1_ratio),
        "Tier1 Ratio": float(t1_ratio),
        "Total CAR": float(car),
        "Leverage Ratio": float(lv_ratio),
        "CET1 Surplus (vs CET1+Buffers)": float(cet1 - abs_req["CET1 Required"]),
        "T1 Surplus (vs T1+Buffers)": float(t1 - abs_req["Tier1 Required"]),
        "Total Capital Surplus (vs Total+Buffers)": float(tot_cap - abs_req["Total Capital Required"]),
        "CET1 Minimum Reached": bool(cet1_ratio < pct["CET1 Min (Pillar 1)"]),
        "Combined Buffer Utilisation %": buf_util,
        "Conservation Level": level,
        "Payout Distribution Restriction %": float(payout),
    }


def run_full_regcap_analysis(
    portfolio_df: pd.DataFrame,
    rwa_result: Dict,
    ifrs9_ecl_total: float,
    macro_conditions: Dict | None = None,
    institution_size: str = "Large_D-SIB",
) -> Dict:
    """Complete RegCap analysis: stack, resources, ratios."""
    total_rwa = float(rwa_result["total_rwa"])
    total_assets = float(portfolio_df["ead"].sum() * 1.1)
    stack = compute_full_capital_stack(total_rwa, total_assets, ifrs9_ecl_total, institution_size)
    resources = generate_default_capital_resources(total_rwa, ifrs9_ecl_total)
    ratios = compute_capital_ratios_and_erosion(resources, stack, total_rwa, total_assets)
    return {
        "capital_stack": stack,
        "capital_resources": resources,
        "capital_ratios": ratios,
        "total_rwa": total_rwa,
        "total_assets": total_assets,
    }
