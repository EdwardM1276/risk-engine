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
    "D_SIB_MIN": 0.005,
    "D_SIB_MAX": 0.025,
    "Countercyclical_2026": 0.010,
}

# D-SIB (HLA) buffer tiers per SARB Regulation 38 / Basel D-SIB framework.
# Systemic-importance buckets map to CET1 add-ons of 0.5% - 2.5% of RWA.
D_SIB_BUFFER_TIERS: Dict[int, float] = {
    1: 0.005,
    2: 0.010,
    3: 0.015,
    4: 0.020,
    5: 0.025,
}
DEFAULT_D_SIB_BUCKET: int = 3  # large SA D-SIB average add-on ~1.5%

# Basel III leverage ratio (SARB Regulation 38: 4% minimum for banks;
# D-SIB leverage buffer set at 50% of the risk-weighted D-SIB buffer).
LEVERAGE_REQUIREMENTS: Dict[str, float] = {
    "minimum": 0.040,
    "d_sib_buffer_scaling": 0.50,
}

# Basel III output floor: floored RWA = max(modelled RWA, floor_pct x standardised RWA).
# Fully phased-in floor is 72.5% (2030); SA phase-in per Regulation 38 amendments.
OUTPUT_FLOOR: Dict[str, object] = {
    "floor_pct": 0.725,
    "phase_in": {
        2023: 0.50, 2024: 0.55, 2025: 0.60, 2026: 0.65,
        2027: 0.70, 2028: 0.725, 2029: 0.725, 2030: 0.725,
    },
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
# 5b. Master SA bank benchmark table (2024 FY Pillar 3 / annual reports)
#     Metrics are fractions unless stated. Capitec is a retail-only outlier
#     and is excluded from the D-SIB average.
# -----------------------------------------------------------------------------
SA_DSIB_BENCHMARKS_2024: Dict[str, Dict[str, float]] = {
    "Standard_Bank": {
        "rwa_density": 0.528, "cet1": 0.120, "tier1": 0.140, "total_capital": 0.156,
        "leverage": 0.074, "credit_rwa_share": 0.71, "oprisk_rwa_share": 0.14,
        "market_rwa_share": 0.049, "lcr": 1.263, "nsfr": 1.164,
        "total_exposure_bn": 1690.0, "is_dsib": 1.0,
    },
    "FirstRand": {
        "rwa_density": 0.455, "cet1": 0.117, "tier1": 0.136, "total_capital": 0.156,
        "leverage": 0.062, "credit_rwa_share": 0.75, "oprisk_rwa_share": 0.15,
        "market_rwa_share": 0.033, "lcr": 1.282, "nsfr": 1.178,
        "total_exposure_bn": 1600.0, "is_dsib": 1.0,
    },
    "Absa": {
        "rwa_density": 0.409, "cet1": 0.104, "tier1": 0.127, "total_capital": 0.151,
        "leverage": 0.052, "credit_rwa_share": 0.75, "oprisk_rwa_share": 0.15,
        "market_rwa_share": 0.036, "lcr": 1.282, "nsfr": 1.157,
        "total_exposure_bn": 1935.0, "is_dsib": 1.0,
    },
    "Nedbank": {
        "rwa_density": 0.565, "cet1": 0.128, "tier1": 0.141, "total_capital": 0.158,
        "leverage": 0.080, "credit_rwa_share": 0.72, "oprisk_rwa_share": 0.13,
        "market_rwa_share": 0.059, "lcr": 1.392, "nsfr": 1.173,
        "total_exposure_bn": 1073.0, "is_dsib": 1.0,
    },
    "Investec": {
        "rwa_density": 0.401, "cet1": 0.113, "tier1": 0.134, "total_capital": 0.156,
        "leverage": 0.054, "credit_rwa_share": 0.72, "oprisk_rwa_share": 0.14,
        "market_rwa_share": 0.044, "lcr": 1.323, "nsfr": 1.060,
        "total_exposure_bn": 639.0, "is_dsib": 1.0,
    },
    "Capitec": {
        "rwa_density": 0.530, "cet1": 0.342, "tier1": 0.342, "total_capital": 0.350,
        "leverage": 0.181, "credit_rwa_share": 0.67, "oprisk_rwa_share": 0.12,
        "market_rwa_share": 0.0005, "lcr": 23.18, "nsfr": 2.207,
        "total_exposure_bn": 230.0, "is_dsib": 0.0,
    },
}

# Averaged D-SIB benchmark profile (excludes Capitec for capital/liquidity
# ratios; RWA density averaged across all six banks per benchmark table).
BANK_BENCHMARK_PROFILES: Dict[str, Dict[str, float]] = {
    "D_SIB_AVERAGE": {
        "rwa_density": 0.48,
        "cet1": 0.116,
        "tier1": 0.136,
        "total_capital": 0.155,
        "leverage": 0.064,
        "credit_rwa_share": 0.72,
        "oprisk_rwa_share": 0.14,
        "market_rwa_share": 0.04,
        "other_rwa_share": 0.10,
        "ecl_ead_target": 0.015,
        "lcr": 1.308,
        "nsfr": 1.146,
        "total_exposure_bn": 1500.0,
        "d_sib_bucket": 3,
    },
}
for _bank, _b in SA_DSIB_BENCHMARKS_2024.items():
    BANK_BENCHMARK_PROFILES[_bank] = {
        "rwa_density": _b["rwa_density"],
        "cet1": _b["cet1"],
        "tier1": _b["tier1"],
        "total_capital": _b["total_capital"],
        "leverage": _b["leverage"],
        "credit_rwa_share": _b["credit_rwa_share"],
        "oprisk_rwa_share": _b["oprisk_rwa_share"],
        "market_rwa_share": _b["market_rwa_share"],
        "other_rwa_share": max(
            0.0,
            1.0 - _b["credit_rwa_share"] - _b["oprisk_rwa_share"] - _b["market_rwa_share"],
        ),
        "ecl_ead_target": 0.015,
        "lcr": _b["lcr"],
        "nsfr": _b["nsfr"],
        "total_exposure_bn": _b["total_exposure_bn"],
        "d_sib_bucket": 3 if _b["is_dsib"] else 0,
    }

DEFAULT_BANK_PROFILE: str = "D_SIB_AVERAGE"

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
        "gdp_growth": (-0.070, -0.050),
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
# TTC PDs / LGDs calibrated to the averaged D-SIB benchmark profile
# (portfolio RWA density ~48%, ECL/EAD ~1.5%). Standardised risk weights
# (`sa_rw`) follow the Basel III final standardised approach.
PORTFOLIO_SEGMENTS: Dict[str, Dict[str, float]] = {
    "Retail_Mortgage": {"ttc_pd": 0.008, "lgd": 0.20, "corr": 0.15, "ccf": 0.20, "weight": 0.35, "sa_rw": 0.30},
    "Retail_Vehicle": {"ttc_pd": 0.015, "lgd": 0.35, "corr": 0.10, "ccf": 0.60, "weight": 0.10, "sa_rw": 0.75},
    "Retail_CreditCard": {"ttc_pd": 0.040, "lgd": 0.80, "corr": 0.04, "ccf": 0.85, "weight": 0.08, "sa_rw": 0.75},
    "Retail_Overdraft": {"ttc_pd": 0.050, "lgd": 0.75, "corr": 0.05, "ccf": 0.90, "weight": 0.05, "sa_rw": 0.75},
    "SME_Corporate": {"ttc_pd": 0.025, "lgd": 0.45, "corr": 0.11, "ccf": 0.70, "weight": 0.15, "sa_rw": 0.85},
    "Corporate_Large": {"ttc_pd": 0.015, "lgd": 0.40, "corr": 0.18, "ccf": 0.65, "weight": 0.20, "sa_rw": 0.80},
    "Sovereign_Bank": {"ttc_pd": 0.002, "lgd": 0.10, "corr": 0.23, "ccf": 0.50, "weight": 0.07, "sa_rw": 0.05},
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
