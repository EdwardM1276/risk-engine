"""ECap allocation: Nedbank 2024 benchmark split, business/model/stress-risk add-ons."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from config.params import MODEL_RISK_ALLOCATION, NEDBANK_ECAP_BENCHMARK_2024


def allocate_nedbank_ecap_benchmark(
    total_rwa: float,
    ifrs9_ecl_total: float,
    credit_ecap_from_simulation: Optional[float] = None,
    market_ecap_from_simulation: Optional[float] = None,
    operational_ecap_from_simulation: Optional[float] = None,
    macro_conditions: Optional[Dict[str, float]] = None,
) -> Dict:
    """Return total ECap split by 6 risk types per Nedbank 2024 benchmark weights."""
    w = dict(NEDBANK_ECAP_BENCHMARK_2024)
    pct_rwa = w["total_ecap_to_rwa"]
    if macro_conditions:
        ls = float(macro_conditions.get("load_shedding_stage", 0)) / 8.0
        gdp = float(macro_conditions.get("gdp_yoy", 0.012))
        scale = 1.0 + 0.7 * ls + 0.8 * max(0.0, 0.012 - gdp) * 10.0
        pct_rwa = pct_rwa * float(np.clip(scale, 0.9, 1.8))

    total_ecap = total_rwa * pct_rwa
    credit = total_ecap * w["credit_risk_pct"]
    market = total_ecap * w["market_risk_pct"]
    op = total_ecap * w["operational_risk_pct"]
    business = total_ecap * w["business_risk_pct"]
    model = total_ecap * w["model_risk_pct"]
    stress = total_ecap * w["stress_buffer_pct"]

    if credit_ecap_from_simulation is not None:
        credit = float(credit_ecap_from_simulation)
        residual = max(0.0, total_ecap - credit)
        rw = np.array([w["market_risk_pct"], w["operational_risk_pct"],
                       w["business_risk_pct"], w["model_risk_pct"], w["stress_buffer_pct"]], dtype=float)
        rw = rw / max(rw.sum(), 1e-9)
        market, op, business, model, stress = (residual * rw).tolist()
        total_ecap = credit + market + op + business + model + stress

    if market_ecap_from_simulation is not None:
        delta = market_ecap_from_simulation - market
        market = float(market_ecap_from_simulation)
        credit = max(credit - delta * 0.6, 0.0)
        stress = max(stress - delta * 0.4, 0.0)

    if operational_ecap_from_simulation is not None:
        delta = operational_ecap_from_simulation - op
        op = float(operational_ecap_from_simulation)
        credit = max(credit - delta * 0.6, 0.0)
        business = max(business - delta * 0.4, 0.0)

    model_floor = total_ecap * MODEL_RISK_ALLOCATION
    if model < model_floor:
        shortfall = model_floor - model
        model = model_floor
        credit = max(credit - shortfall * 0.6, 0.0)
        stress = max(stress - shortfall * 0.4, 0.0)

    total = credit + market + op + business + model + stress
    return {
        "total_ecap": float(total),
        "components": {
            "Credit Risk ECap": float(credit),
            "Market Risk ECap": float(market),
            "Operational Risk ECap": float(op),
            "Business Risk ECap": float(business),
            "Model Risk ECap": float(model),
            "Stress Buffer ECap": float(stress),
        },
        "weights": {k: float(v) for k, v in [
            ("Credit Risk %", w["credit_risk_pct"]),
            ("Market Risk %", w["market_risk_pct"]),
            ("Operational Risk %", w["operational_risk_pct"]),
            ("Business Risk %", w["business_risk_pct"]),
            ("Model Risk %", w["model_risk_pct"]),
            ("Stress Buffer %", w["stress_buffer_pct"]),
        ]},
        "total_ecap_pct_rwa": float(total / max(total_rwa, 1.0)),
        "ecl_vs_ecap_ratio": float(ifrs9_ecl_total / max(total, 1.0)),
    }


def compute_business_risk_ecap(
    total_assets: float,
    net_interest_margin: float = 0.032,
    revenue_volatility: float = 0.08,
    alpha: float = 0.999,
) -> float:
    """Business risk ECap via revenue volatility approach (SA large-bank calibration)."""
    from scipy.stats import norm
    expected_rev = total_assets * net_interest_margin
    z = norm.ppf(alpha)
    return float(expected_rev * revenue_volatility * z)
