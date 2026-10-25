"""Economic capital aggregation with benchmark fallbacks for unsimulated risks."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from config.params import MODEL_RISK_ALLOCATION, NEDBANK_ECAP_BENCHMARK_2024
from engine.money import post_sum


def allocate_nedbank_ecap_benchmark(
    total_rwa: float,
    ifrs9_ecl_total: float,
    credit_ecap_from_simulation: Optional[float] = None,
    market_ecap_from_simulation: Optional[float] = None,
    operational_ecap_from_simulation: Optional[float] = None,
    macro_conditions: Optional[Dict[str, float]] = None,
) -> Dict:
    """Aggregate supplied risk measures and disclose benchmark fallbacks.

    Published benchmark percentages are used only for components without a
    supplied risk estimate. Simulated credit, market, and operational values
    are therefore not forced into a benchmark-sized total.
    """
    w = dict(NEDBANK_ECAP_BENCHMARK_2024)
    pct_rwa = w["total_ecap_to_rwa"]
    if macro_conditions:
        ls = float(macro_conditions.get("load_shedding_stage", 0)) / 8.0
        gdp = float(macro_conditions.get("gdp_yoy", 0.012))
        scale = 1.0 + 0.7 * ls + 0.8 * max(0.0, 0.012 - gdp) * 10.0
        pct_rwa = pct_rwa * float(np.clip(scale, 0.9, 1.8))

    benchmark_total = total_rwa * pct_rwa
    credit = benchmark_total * w["credit_risk_pct"]
    market = benchmark_total * w["market_risk_pct"]
    op = benchmark_total * w["operational_risk_pct"]
    business = benchmark_total * w["business_risk_pct"]
    model = benchmark_total * w["model_risk_pct"]
    other = benchmark_total * w["other_risk_pct"]
    stress = benchmark_total * w["stress_buffer_pct"]
    supplied = any(value is not None for value in (
        credit_ecap_from_simulation,
        market_ecap_from_simulation,
        operational_ecap_from_simulation,
    ))

    if credit_ecap_from_simulation is not None:
        credit = max(float(credit_ecap_from_simulation), 0.0)

    if market_ecap_from_simulation is not None:
        market = max(float(market_ecap_from_simulation), 0.0)

    if operational_ecap_from_simulation is not None:
        op = max(float(operational_ecap_from_simulation), 0.0)

    risk_mix = {
        "Credit Risk ECap": float(w["credit_risk_pct"]),
        "Market Risk ECap": float(w["market_risk_pct"]),
        "Operational Risk ECap": float(w["operational_risk_pct"]),
        "Business Risk ECap": float(w["business_risk_pct"]),
        "Model Risk ECap": float(w["model_risk_pct"]),
        "Other Risk ECap": float(w["other_risk_pct"]),
    }
    risk_mix_total = float(sum(risk_mix.values()))
    if not np.isclose(risk_mix_total, 1.0, atol=1e-10):
        raise ValueError(f"ECap minimum risk mix must sum to 100%; got {risk_mix_total:.6f}")
    total = float(post_sum([credit, market, op, business, model, other, stress]))
    return {
        "total_ecap": float(total),
        "benchmark_total_ecap": float(benchmark_total),
        "allocation_method": (
            "Hybrid economic-capital indicator (simulation + benchmark fallbacks + stress overlay)"
            if supplied else "Hybrid benchmark-derived economic-capital indicator + stress overlay"
        ),
        "components": {
            "Credit Risk ECap": float(credit),
            "Market Risk ECap": float(market),
            "Operational Risk ECap": float(op),
            "Business Risk ECap": float(business),
            "Model Risk ECap": float(model),
            "Other Risk ECap (benchmark residual)": float(other),
            "Stress Buffer ECap": float(stress),
        },
        "aggregation_policy": "Hybrid economic-capital indicator: supplied credit, market and operational measures replace benchmark fallbacks; business, model and other components are benchmark-derived; stress is a deterministic overlay. No cross-risk diversification is applied.",
        "minimum_risk_mix_weights": risk_mix,
        "minimum_risk_mix_weight_sum": risk_mix_total,
        "stress_buffer_pct_of_benchmark_total": float(w["stress_buffer_pct"]),
        "weights": risk_mix,
        "model_risk_allocation_floor_configured": float(MODEL_RISK_ALLOCATION),
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
