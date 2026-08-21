"""Scenario engine: SARB-standard stress scenarios + idiosyncratic custom shocks.

Implements the three-tier SARB / IMF framework used by the Prudential Authority:
    Base / Adverse / Severe, with severity-interpolation and idiosyncratic add-ons.
All shock labels and display formatting are plain-text only with no emoji glyphs.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from config.params import SARB_STRESS_SCENARIOS


# -----------------------------------------------------------------------------
# Public scenario retrieval
# -----------------------------------------------------------------------------

def get_scenario_parameters(
    scenario_name: str,
    severity_multiplier: float = 1.0,
    seed: int = 2024,
) -> Dict[str, object]:
    """Retrieve SARB-standard scenario parameters.

    Severity multiplier allows smooth interpolation between scenario bounds.
    Use <1.0 for milder shocks, >1.0 for amplified tail stress.
    """
    scenario_name = scenario_name.title()
    if scenario_name not in SARB_STRESS_SCENARIOS:
        raise ValueError(
            f"Unknown scenario {scenario_name}. "
            f"Choose from {list(SARB_STRESS_SCENARIOS.keys())}"
        )

    base = SARB_STRESS_SCENARIOS[scenario_name]
    rng = np.random.default_rng(seed)

    params: Dict[str, object] = {}
    for key, value in base.items():
        if isinstance(value, tuple) and len(value) == 2:
            low, high = value
            if "growth" in key or "change" in key:
                v = float(rng.uniform(low, high) * severity_multiplier)
            else:
                mid = (low + high) / 2.0
                span = (high - low) / 2.0
                v = mid + span * severity_multiplier * float(rng.uniform(-1.0, 1.0))
                if "stage" in key:
                    v = int(np.clip(v, 0, 8))
            params[key] = v
        else:
            if isinstance(value, (int, float)):
                params[key] = float(value) * severity_multiplier
            else:
                params[key] = value

    return params


def scenario_to_macro_conditions(scenario_params: Dict[str, object]) -> Dict[str, float]:
    """Translate scenario parameters to the macro_conditions dict used by models."""

    def _to_float(v, default: float) -> float:
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, tuple) and len(v) == 2:
            return (float(v[0]) + float(v[1])) / 2.0
        return default

    def _to_int(v, default: int) -> int:
        if isinstance(v, (int, float)):
            return int(v)
        if isinstance(v, tuple) and len(v) == 2:
            return int((float(v[0]) + float(v[1])) / 2.0)
        return default

    repo_adj = _to_float(scenario_params.get("repo_rate", 0.0), 0.0)
    gdp = _to_float(scenario_params.get("gdp_growth", 0.012), 0.012)
    cpi = _to_float(scenario_params.get("inflation", 0.05), 0.05)
    unemp_delta = _to_float(scenario_params.get("unemployment_change", 0.0), 0.0)
    ls = _to_int(scenario_params.get("load_shedding_stage", 2), 2)

    return {
        "repo_rate": 0.0775 + repo_adj if abs(repo_adj) < 0.5 else repo_adj,
        "gdp_yoy": gdp,
        "cpi_yoy": 0.05 + (cpi - 0.05),
        "unemployment_rate": 0.32 + unemp_delta,
        "load_shedding_stage": int(np.clip(ls, 0, 8)),
    }


def scenario_to_market_data(scenario_params: Dict[str, object]) -> Dict[str, float]:
    """Extract market-data shocks from scenario parameters."""

    def _to_float(v, default: float) -> float:
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, tuple) and len(v) == 2:
            return (float(v[0]) + float(v[1])) / 2.0
        return default

    ls = _to_float(scenario_params.get("load_shedding_stage", 2), 2.0)
    gdp = _to_float(scenario_params.get("gdp_growth", 0.0), 0.0)

    return {
        "gold_price_change": _to_float(scenario_params.get("gold_price_change", 0.0), 0.0),
        "coal_price_change": _to_float(scenario_params.get("coal_price_change", 0.0), 0.0),
        "platinum_price_change": _to_float(scenario_params.get("platinum_price_change", 0.0), 0.0),
        "sovereign_cds_change_bps": float(
            scenario_params.get("sovereign_cds_change_bps", 0)
        ),
        "jse_property_change": -0.5 * ls / 8.0 - 0.3 * abs(gdp),
        "zar_usd_vol_change": _to_float(scenario_params.get("zar_usd_vol_change", 0.0), 0.0),
    }


# -----------------------------------------------------------------------------
# Idiosyncratic custom shocks (beyond standard SARB 3-scenario framework)
# -----------------------------------------------------------------------------

def create_idiosyncratic_scenario(
    sovereign_downgrade: bool = False,
    commodity_collapse: bool = False,
    cyber_incident: bool = False,
    housing_crash: bool = False,
    smes_failure_wave: bool = False,
    seed: int = 2024,
) -> Dict[str, object]:
    """Compose idiosyncratic shock layers over an Adverse base scenario.

    Label naming convention: use plain descriptive text, no glyphs.
    """
    base = dict(SARB_STRESS_SCENARIOS["Adverse"])
    rng = np.random.default_rng(seed)

    if sovereign_downgrade:
        base["sovereign_cds_change_bps"] = max(
            int(base.get("sovereign_cds_change_bps", 0)), 250
        )
        base["zar_usd_vol_change"] = max(
            float(base.get("zar_usd_vol_change", 0.0)), 0.25
        )
        base["inflation"] = (max(base["inflation"][0], 0.09), max(base["inflation"][1], 0.12))
        base["repo_rate"] = (max(base["repo_rate"][0], 0.02), max(base["repo_rate"][1], 0.035))

    if commodity_collapse:
        base["gold_price_change"] = min(float(base.get("gold_price_change", 0.0)), -0.30)
        base["platinum_price_change"] = min(
            float(base.get("platinum_price_change", 0.0)), -0.40
        )
        base["coal_price_change"] = min(float(base.get("coal_price_change", 0.0)), -0.40)
        base["gdp_growth"] = (
            min(base["gdp_growth"][0], -0.04),
            min(base["gdp_growth"][1], -0.02),
        )
        base["unemployment_change"] = max(
            float(base.get("unemployment_change", 0.0)), 0.05
        )

    if cyber_incident:
        base["operational_shock_ecap_mult"] = 2.5
        base["load_shedding_stage"] = (
            max(base["load_shedding_stage"][0], 6),
            max(base["load_shedding_stage"][1], 8),
        )

    if housing_crash:
        base["property_price_shock_pct"] = -0.30
        base["load_shedding_stage"] = (
            max(base["load_shedding_stage"][0], 5),
            max(base["load_shedding_stage"][1], 7),
        )

    if smes_failure_wave:
        base["sme_pd_multiplier"] = 2.5
        base["unemployment_change"] = max(
            float(base.get("unemployment_change", 0.0)), 0.06
        )
        base["load_shedding_stage"] = (
            max(base["load_shedding_stage"][0], 5),
            max(base["load_shedding_stage"][1], 7),
        )

    for k, v in base.items():
        if isinstance(v, tuple) and len(v) == 2:
            base[k] = (
                v[0] + float(rng.uniform(-0.005, 0.005)),
                v[1] + float(rng.uniform(-0.005, 0.005)),
            )

    return base


# -----------------------------------------------------------------------------
# Dashboard display helper (plain-text, percentage / bps / stage formatting)
# -----------------------------------------------------------------------------

def compute_scenario_shock_summary(scenario_params: Dict[str, object]) -> pd.DataFrame:
    """Summarize all shocks applied in a scenario for clean dashboard display."""
    rows: list[Dict[str, str]] = []
    labels: Dict[str, str] = {
        "gdp_growth": "GDP Growth",
        "inflation": "Inflation (CPI)",
        "repo_rate": "Repo Rate (change)",
        "load_shedding_stage": "Load-Shedding Stage",
        "gold_price_change": "Gold Price (change)",
        "coal_price_change": "Coal Price (change)",
        "platinum_price_change": "Platinum Price (change)",
        "sovereign_cds_change_bps": "Sovereign CDS (bps change)",
        "zar_usd_vol_change": "ZAR/USD Volatility (change)",
        "unemployment_change": "Unemployment (pp change)",
    }

    for key, label in labels.items():
        if key not in scenario_params:
            continue
        v = scenario_params[key]
        is_bps = "bps" in label
        is_stage = "Stage" in label

        if isinstance(v, tuple) and len(v) == 2:
            v0, v1 = float(v[0]), float(v[1])
            if is_stage:
                display = f"Stage {int(v0)}-{int(v1)}"
            elif is_bps:
                display = f"{int(v0)} to {int(v1)} bps"
            else:
                display = f"{v0 * 100:+.1f}% to {v1 * 100:+.1f}%"
            rows.append({"Shock Driver": label, "Range": display})
        else:
            fv = float(v) if not isinstance(v, (int, float)) else float(v)
            if is_bps:
                display = f"{fv:+.0f} bps"
            elif is_stage:
                display = f"Stage {int(fv)}"
            else:
                display = f"{fv * 100:+.2f}%"
            rows.append({"Shock Driver": label, "Range": display})

    return pd.DataFrame(rows)
