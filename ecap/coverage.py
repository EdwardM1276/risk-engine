"""ECap coverage analysis and stress erosion paths -- regulator-ready views."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from config.params import NEDBANK_ECAP_BENCHMARK_2024


def compute_afr_coverage_ratio(
    available_financial_resources: float,
    total_ecap_required: float,
    total_ecap_components: Dict[str, float],
) -> Dict:
    """Return AFR vs ECap coverage vs Nedbank 2024 170% benchmark."""
    afr = float(available_financial_resources)
    req = float(total_ecap_required)
    coverage = afr / max(req, 1.0)
    bench = float(NEDBANK_ECAP_BENCHMARK_2024["afr_coverage_ratio"])
    surplus = afr - req
    headroom = (coverage - 1.0) * 100.0 if coverage >= 1.0 else -(1.0 - coverage) * 100.0

    if coverage >= 1.5:
        rating, color = "Strong - Well above Nedbank 2024 benchmark", "green"
    elif coverage >= 1.2:
        rating, color = "Healthy - Above minimum requirement", "lime"
    elif coverage >= 1.0:
        rating, color = "Adequate - At or just above minimum", "yellow"
    elif coverage >= 0.85:
        rating, color = "Warning - Approaching minimum threshold", "orange"
    else:
        rating, color = "Critical - Below minimum ECap requirement", "red"

    return {
        "AFR": afr,
        "Total ECap Required": req,
        "ECap Components": dict(total_ecap_components),
        "Coverage Ratio": float(coverage),
        "Nedbank 2024 Benchmark": bench,
        "Coverage vs Benchmark %": float((coverage / bench) * 100.0),
        "Surplus / (Shortfall)": float(surplus),
        "Headroom vs 100% (pp)": float(headroom),
        "Coverage Rating": rating,
        "Coverage Color": color,
    }


def stress_coverage_erosion(
    base_coverage: Dict,
    scenario_shocks: Dict[str, float],
    total_ecap_components: Dict[str, float],
) -> pd.DataFrame:
    """Project coverage ratio for 8 increasing shock intensities."""
    afr_base = float(base_coverage["AFR"])
    ecap_base = float(base_coverage["Total ECap Required"])
    shock_x = [0.0, 0.25, 0.50, 0.75, 1.0, 1.5, 2.0, 2.5]
    rows = []
    for s in shock_x:
        afr_sf = 1.0 - s * 0.18
        ecap_sf = 1.0 + s * 0.35
        for k, factor in scenario_shocks.items():
            if k == "gdp_shock":
                ecap_sf *= 1.0 + s * abs(float(factor)) * 3.0
            elif k == "ls_stage":
                ecap_sf *= 1.0 + s * (float(factor) / 8.0) * 0.6
            elif k == "sovereign_cds_bps":
                afr_sf *= 1.0 - s * min(float(factor) / 1000.0, 0.25)
        afr_new = afr_base * afr_sf
        ecap_new = ecap_base * ecap_sf
        cov = afr_new / max(ecap_new, 1.0)
        rows.append({
            "Shock Intensity (x)": float(s),
            "AFR After Shock": float(afr_new),
            "ECap After Shock": float(ecap_new),
            "Coverage Ratio": float(cov),
            "AFR Erosion %": float((1 - afr_sf) * 100.0),
            "ECap Expansion %": float((ecap_sf - 1) * 100.0),
            "Surplus / (Shortfall)": float(afr_new - ecap_new),
            "Coverage Status": "Breached" if cov < 1.0 else "Maintained",
        })
    return pd.DataFrame(rows)


def decompose_coverage_drivers(
    coverage_result: Dict,
    regcap_result: Dict,
    ifrs9_total_ecl: float,
) -> pd.DataFrame:
    """Decompose AFR composition (CET1/AT1/T2) side-by-side with ECap requirement by type."""
    afr = float(coverage_result["AFR"])
    comp = coverage_result["ECap Components"]
    rows = []
    for name, amt in comp.items():
        rows.append({
            "Category": "ECap Requirement",
            "Item": name,
            "Amount": float(amt),
            "% of Total": float(amt / max(coverage_result["Total ECap Required"], 1.0)) * 100.0,
        })
    cr = regcap_result["capital_resources"]
    cet1 = float(cr["CET1 Available"])
    at1 = float(cr["Additional Tier 1 (AT1)"]["Net AT1 Available"])
    t2 = float(cr["T2 Available"])
    for item, amount in [("Net CET1 Capital", cet1), ("Additional Tier 1", at1), ("Tier 2 Capital", t2)]:
        rows.append({
            "Category": "AFR Composition",
            "Item": item,
            "Amount": amount,
            "% of Total": amount / max(afr, 1.0) * 100.0,
        })
    other = afr - cet1 - at1 - t2
    rows.append({
        "Category": "AFR Composition",
        "Item": "Other Resources (Provisions, etc.)",
        "Amount": float(max(other, 0.0)),
        "% of Total": float(max(other, 0.0) / max(afr, 1.0) * 100.0),
    })
    return pd.DataFrame(rows)
