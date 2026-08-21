"""IFRS 9 PD model: 8-state Markov transition matrices, TTC to PIT conversion.

The core model is a discrete-time Markov chain over the 8-grade internal
rating scale (AAA to Default). Transition probabilities are scaled by a
single systematic credit factor which itself is a linear combination of SA
macro drivers (GDP gap, unemployment gap, loadshedding stage).
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import norm

from config.params import (
    PORTFOLIO_SEGMENTS,
    RATING_DEFAULT_PDS,
    RATING_TO_IDX,
    SICR_TRIGGERS,
)

N_STATES: int = 8


def build_markov_transition_matrix(macro_conditions: Dict[str, float]) -> np.ndarray:
    """Return an (8 x 8) row-stochastic transition matrix for a given macro state."""
    gdp_gap = float(macro_conditions.get("gdp_yoy", 0.0) - 0.015)
    unemp_gap = float(macro_conditions.get("unemployment_rate", 0.32) - 0.32)
    ls_stage = float(macro_conditions.get("load_shedding_stage", 2))
    ls_impact = (ls_stage / 8.0) * 0.35

    stress_factor = np.clip(-gdp_gap * 4.0 + unemp_gap * 2.5 + ls_impact, -1.5, 2.5)

    base = np.array([
        [92.0, 5.5, 1.5, 0.6, 0.25, 0.10, 0.04, 0.01],
        [4.0, 88.0, 5.5, 1.5, 0.6, 0.25, 0.10, 0.05],
        [0.8, 5.0, 85.0, 6.5, 1.8, 0.6, 0.20, 0.10],
        [0.2, 1.2, 6.5, 81.0, 7.5, 2.5, 0.8, 0.30],
        [0.05, 0.4, 2.0, 7.5, 76.0, 9.5, 3.0, 1.55],
        [0.02, 0.15, 0.8, 3.0, 9.0, 68.0, 12.0, 7.03],
        [0.01, 0.05, 0.3, 1.0, 3.0, 10.0, 56.0, 29.64],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0],
    ]) / 100.0

    bias = np.clip(0.005 * stress_factor, -0.02, 0.03)
    adj = np.zeros_like(base)
    for i in range(N_STATES - 1):
        for j in range(N_STATES):
            if j == N_STATES - 1:
                adj[i, j] = np.clip(base[i, j] * (1 + bias * 2.0), 0.0, 1.0)
            elif j > i:
                adj[i, j] = np.clip(base[i, j] * (1 + bias), 0.0, 1.0)
            elif j < i:
                adj[i, j] = np.clip(base[i, j] * max(0.0, (1 - bias * 0.8)), 0.0, 1.0)
            else:
                adj[i, j] = base[i, j]

    for i in range(N_STATES):
        s = adj[i, :].sum()
        if s > 0:
            adj[i, :] = adj[i, :] / s
    adj[-1, :] = 0.0
    adj[-1, -1] = 1.0
    return adj


def convert_ttc_to_pit(
    portfolio_df: pd.DataFrame,
    macro_conditions: Dict[str, float],
    forecast_horizon_months: int = 12,
) -> pd.DataFrame:
    """Project every account from TTC PDs to PIT using stress-conditioned Markov chain.

    Output columns added:
        - pit_pd_12m, lifetime_pd, ttc_pd, credit_index, pd_multiplier
    """
    result = portfolio_df.copy()
    trans = build_markov_transition_matrix(macro_conditions)

    gdp_yoy = float(macro_conditions.get("gdp_yoy", 0.0))
    unemp = float(macro_conditions.get("unemployment_rate", 0.32))
    ls_stage = float(macro_conditions.get("load_shedding_stage", 2))
    cpi_yoy = float(macro_conditions.get("cpi_yoy", 0.05))
    repo = float(macro_conditions.get("repo_rate", 0.0775))

    n_12m = max(1, int(np.ceil(forecast_horizon_months / 12)))
    n_life = 5

    seg_arr = result["segment"].values
    ls_vuln_arr = result["loadshedding_vulnerability_score"].values.astype(float) / 5.0
    mob_arr = result["months_on_book"].values.astype(float)
    rating_arr = result["internal_rating"].values
    ttc_pd_arr = result["base_segment_ttc_pd"].values.astype(float)
    dpd_arr = result["dpd"].values.astype(int)
    debt_arr = result["debt_review_flag"].values.astype(bool)
    judge_arr = result["judgement_flag"].values.astype(bool)

    pit_12m = np.zeros(len(result), dtype=float)
    life_pd = np.zeros(len(result), dtype=float)
    ci = np.zeros(len(result), dtype=float)

    for idx in range(len(result)):
        start_idx = RATING_TO_IDX.get(rating_arr[idx], 3)
        seg = seg_arr[idx]
        ls_vuln = ls_vuln_arr[idx]

        mult = 1.0
        if "Retail" in seg:
            u_e = 1.8 if "Mortgage" in seg else (2.5 if "CreditCard" in seg else 2.2)
            mult *= 1.0 + (unemp - 0.32) * u_e
            mult *= 1.0 + (repo - 0.0775) * (3.0 if "Mortgage" in seg else 2.0)
            mult *= 1.0 + (cpi_yoy - 0.05) * 1.2
        elif "SME" in seg:
            mult *= 1.0 + (ls_stage / 8.0) * ls_vuln * 2.8
            mult *= 1.0 + (gdp_yoy - 0.015) * (-5.0)
            mult *= 1.0 + (unemp - 0.32) * 1.5
        elif "Corporate" in seg:
            mult *= 1.0 + (ls_stage / 8.0) * ls_vuln * 1.2
            mult *= 1.0 + (gdp_yoy - 0.015) * (-3.5)
        else:
            mult *= 1.0 + (ls_stage / 8.0) * 0.3

        seasoning = 1.0 + 0.6 * np.exp(-mob_arr[idx] / 24.0)
        mult = np.clip(mult * seasoning, 0.25, 8.0)

        dist = np.zeros(N_STATES, dtype=float)
        dist[start_idx] = 1.0
        default_12m = 0.0
        default_life = 0.0
        for p in range(n_life):
            dist = dist @ trans
            if p < n_12m:
                default_12m = dist[-1]
            default_life = dist[-1]

        ttc = ttc_pd_arr[idx]
        pd_ratio = default_12m / max(ttc, 1e-6)
        pit = np.clip(ttc * mult * (0.4 + 0.6 * pd_ratio), 1e-4, 0.999)
        life = np.clip(pit * (1 + 0.6 * (n_life - n_12m)), pit, 0.999)

        dpd, debt, judge = dpd_arr[idx], debt_arr[idx], judge_arr[idx]
        if dpd >= 90 or judge or debt:
            pit = np.clip(pit * 2.5, pit, 0.999)
            life = np.clip(life * 1.6, life, 0.999)
        elif dpd >= 30:
            pit = np.clip(pit * 1.4, pit, 0.999)

        ci[idx] = np.clip(
            norm.ppf(np.clip(pit, 1e-6, 1 - 1e-6)) - norm.ppf(np.clip(ttc, 1e-6, 1 - 1e-6)),
            -3.0, 3.0,
        )
        pit_12m[idx] = pit
        life_pd[idx] = life

    result["pit_pd_12m"] = pit_12m
    result["lifetime_pd"] = life_pd
    result["ttc_pd"] = ttc_pd_arr
    result["credit_index"] = ci
    result["pd_multiplier"] = pit_12m / np.clip(ttc_pd_arr, 1e-6, None)
    return result
