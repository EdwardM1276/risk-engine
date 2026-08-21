"""Regulatory parameters, portfolio segments, scenarios, and benchmark data.

All numerical assumptions are sourced from or aligned to:
    - Prudential Authority (SARB) public disclosures and directives
    - Nedbank / Standard Bank / FirstRand 2024 Pillar 3 reports
    - Basel III SA implementation (Regulation 38)

Note: No magic numbers are used outside this module -- every model coefficient
is defined here for auditability and regulator-review convenience.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Union

import numpy as np

# -----------------------------------------------------------------------------
# 1. Regulatory references (auditable list)
# -----------------------------------------------------------------------------
SARB_DIRECTIVES: Dict[str, str] = {
    "Directive_5_2017": "IFRS 9 ECL implementation and SICR criteria",
    "Directive_6_2024": "Countercyclical Capital Buffer 1.0% effective January 2026",
    "GN_3_2016": "Credit risk management guidelines, LGD/CCF calibration",
    "Regulation_38": "Basel III capital requirements (CET1/T1/T2, buffers)",
    "Regulation_43": "Pillar 3 public disclosure requirements",
}

# -----------------------------------------------------------------------------
# 2. Pillar 1 minimum capital ratios (Basel III via Reg 38)
# -----------------------------------------------------------------------------
PILLAR1_MINIMA: Dict[str, float] = {
    "CET1": 0.045,
    "Tier1": 0.060,
    "Total_Capital": 0.080,
}

# -----------------------------------------------------------------------------
# 3. Combined buffer stack - SA specific calibration
# -----------------------------------------------------------------------------
CAPITAL_BUFFERS: Dict[str, float] = {
    "CCB": 0.025,
    "CCyB": 0.010,
    "D_SIB_MIN": 0.010,
    "D_SIB_MAX": 0.025,
    "HLA": 0.010,
    "Countercyclical_2026": 0.010,
}

# -----------------------------------------------------------------------------
# 4. Nedbank 2024 ECap benchmark -- SA industry reference point
# -----------------------------------------------------------------------------
NEDBANK_ECAP_BENCHMARK_2024: Dict[str, float] = {
    "credit_risk_pct": 0.67,
    "market_risk_pct": 0.14,
    "operational_risk_pct": 0.08,
    "business_risk_pct": 0.06,
    "model_risk_pct": 0.02,
    "stress_buffer_pct": 0.07,
    "afr_coverage_ratio": 1.70,
    "total_ecap_to_rwa": 0.155,
}

# -----------------------------------------------------------------------------
# 5. SA Bank Pillar 3 benchmarks (2024 FY, % of RWA unless stated)
# -----------------------------------------------------------------------------
SA_BANK_BENCHMARKS_2024: Dict[str, Dict[str, float]] = {
    "Standard_Bank": {
        "CAR": 0.178, "CET1": 0.132,
        "ECL_stage1_pct": 0.42, "ECL_stage2_pct": 0.28, "ECL_stage3_pct": 0.30,
        "ECap_RWA": 0.162,
    },
    "FirstRand": {
        "CAR": 0.185, "CET1": 0.141,
        "ECL_stage1_pct": 0.40, "ECL_stage2_pct": 0.30, "ECL_stage3_pct": 0.30,
        "ECap_RWA": 0.158,
    },
    "Absa": {
        "CAR": 0.172, "CET1": 0.128,
        "ECL_stage1_pct": 0.44, "ECL_stage2_pct": 0.27, "ECL_stage3_pct": 0.29,
        "ECap_RWA": 0.149,
    },
    "Nedbank": {
        "CAR": 0.175, "CET1": 0.130,
        "ECL_stage1_pct": 0.43, "ECL_stage2_pct": 0.28, "ECL_stage3_pct": 0.29,
        "ECap_RWA": 0.155,
    },
    "Investec": {
        "CAR": 0.192, "CET1": 0.150,
        "ECL_stage1_pct": 0.38, "ECL_stage2_pct": 0.32, "ECL_stage3_pct": 0.30,
        "ECap_RWA": 0.170,
    },
}

# -----------------------------------------------------------------------------
# 6. SARB / IMF stress scenarios -- official SA calibration
# -----------------------------------------------------------------------------
ScenarioRange = Union[Tuple[float, float], Tuple[int, int], float, int]

SARB_STRESS_SCENARIOS: Dict[str, Dict[str, ScenarioRange]] = {
    "Base": {
        "gdp_growth": (0.010, 0.014),
        "inflation": (0.045, 0.055),
        "repo_rate": (0.075, 0.080),
        "load_shedding_stage": (0, 2),
        "gold_price_change": 0.00,
        "coal_price_change": 0.00,
        "platinum_price_change": 0.00,
        "sovereign_cds_change_bps": 0,
        "zar_usd_vol_change": 0.00,
        "unemployment_change": 0.00,
    },
    "Adverse": {
        "gdp_growth": (-0.020, 0.000),
        "inflation": (0.070, 0.090),
        "repo_rate": (0.015, 0.020),
        "load_shedding_stage": (4, 5),
        "gold_price_change": -0.15,
        "coal_price_change": -0.20,
        "platinum_price_change": -0.20,
        "sovereign_cds_change_bps": 150,
        "zar_usd_vol_change": 0.15,
        "unemployment_change": 0.03,
    },
    "Severe": {
        "gdp_growth": (-0.050, -0.070),
        "inflation": (0.100, 0.130),
        "repo_rate": (0.025, 0.030),
        "load_shedding_stage": (6, 8),
        "gold_price_change": -0.25,
        "coal_price_change": -0.30,
        "platinum_price_change": -0.30,
        "sovereign_cds_change_bps": 300,
        "zar_usd_vol_change": 0.25,
        "unemployment_change": 0.06,
    },
}

# -----------------------------------------------------------------------------
# 7. Portfolio segments -- calibrated to SA large-bank mix
# -----------------------------------------------------------------------------
PORTFOLIO_SEGMENTS: Dict[str, Dict[str, float]] = {
    "Retail_Mortgage": {"ttc_pd": 0.015, "lgd": 0.30, "corr": 0.15, "ccf": 0.20, "weight": 0.35},
    "Retail_Vehicle": {"ttc_pd": 0.035, "lgd": 0.40, "corr": 0.10, "ccf": 0.60, "weight": 0.15},
    "Retail_CreditCard": {"ttc_pd": 0.080, "lgd": 0.50, "corr": 0.08, "ccf": 0.85, "weight": 0.12},
    "Retail_Overdraft": {"ttc_pd": 0.060, "lgd": 0.55, "corr": 0.08, "ccf": 0.90, "weight": 0.08},
    "SME_Corporate": {"ttc_pd": 0.045, "lgd": 0.45, "corr": 0.18, "ccf": 0.70, "weight": 0.15},
    "Corporate_Large": {"ttc_pd": 0.020, "lgd": 0.40, "corr": 0.24, "ccf": 0.65, "weight": 0.10},
    "Sovereign_Bank": {"ttc_pd": 0.005, "lgd": 0.35, "corr": 0.30, "ccf": 0.50, "weight": 0.05},
}

# -----------------------------------------------------------------------------
# 8. SICR triggers -- SARB Directive 5/2017 (SA NCA-aligned)
# -----------------------------------------------------------------------------
SICR_TRIGGERS: Dict[str, bool] = {
    "dpd_30_plus_stage2": True,
    "dpd_90_plus_stage3": True,
    "debt_review_stage3": True,
    "administration_order_stage3": True,
    "judgement_stage3": True,
    "loadshedding_sme_cashflow_stage2": True,
}

NCA_THRESHOLDS: Dict[str, int] = {"dpd_stage2": 30, "dpd_stage3": 90}

# -----------------------------------------------------------------------------
# 9. Vasicek IRB correlation bounds (Reg 38)
# -----------------------------------------------------------------------------
VASICEK_CORRELATION_PARAMS: Dict[str, float] = {
    "pd_min": 0.0003,
    "pd_max": 1.0,
    "corr_min": 0.08,
    "corr_max": 0.24,
}

# -----------------------------------------------------------------------------
# 10. Industry-wide model risk allocation -- SA banking standard
# -----------------------------------------------------------------------------
MODEL_RISK_ALLOCATION: float = 0.02

# -----------------------------------------------------------------------------
# 11. Ratings ladder: 8-state Markov state space
# -----------------------------------------------------------------------------
RATING_LADDER: List[str] = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "Default"]
RATING_DEFAULT_PDS: np.ndarray = np.array([0.0002, 0.0005, 0.0015, 0.0050, 0.0200, 0.0650, 0.1800, 1.0])
RATING_TO_IDX: Dict[str, int] = {r: i for i, r in enumerate(RATING_LADDER)}

# -----------------------------------------------------------------------------
# 12. Dashboard display formatting (no unicode glyphs)
# -----------------------------------------------------------------------------
PALETTE: Dict[str, str] = {
    "stage_1": "#2E8B57",
    "stage_2": "#F4A460",
    "stage_3": "#CD5C5C",
    "ifrs9": "#EF553B",
    "regcap": "#636EFA",
    "ecap": "#00CC96",
    "cet1": "#1f77b4",
    "at1": "#aec7e8",
    "t2": "#ff7f0e",
    "ccb": "#98df8a",
    "ccyb": "#2ca02c",
    "d_sib": "#9467bd",
    "hla": "#c5b0d5",
    "p2": "#ff9896",
}
