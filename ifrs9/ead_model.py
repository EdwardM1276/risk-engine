"""IFRS 9 EAD model: elevated SA CCFs reflecting SA retail overdraft culture.

Basel standard CCFs are raised by 8-10 percentage points to reflect SA
borrower behaviour (statistically observed vs EU/US benchmarks). Additional
stress drawdown applied during loadshedding / unemployment spikes -- the
"working capital defence drawdown" effect observed in SARB stress tests.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def calculate_elevated_ccf_ead(
    portfolio_df: pd.DataFrame,
    macro_conditions: Dict[str, float],
) -> pd.DataFrame:
    """Return portfolio with `ead`, `ccf_applied`, `current_utilisation`, etc."""
    result = portfolio_df.copy()
    n = len(result)

    gdp_yoy = float(macro_conditions.get("gdp_yoy", 0.0))
    unemp = float(macro_conditions.get("unemployment_rate", 0.32))
    ls_stage = float(macro_conditions.get("load_shedding_stage", 2))
    repo = float(macro_conditions.get("repo_rate", 0.0775))

    seg = result["segment"].values
    base_ccf = result["base_segment_ccf"].values.astype(float)
    principal = result["principal_outstanding"].values.astype(float)
    undrawn = result["undrawn_limit"].values.astype(float)
    dpd = result["dpd"].values.astype(int)
    ls_vuln = result["loadshedding_vulnerability_score"].values.astype(float) / 5.0
    mob = result["months_on_book"].values.astype(float)

    ead = np.zeros(n, dtype=float)
    ccf = np.zeros(n, dtype=float)
    util = np.zeros(n, dtype=float)
    sdd = np.zeros(n, dtype=float)

    for i in range(n):
        sf = 0.0
        sf += (unemp - 0.32) * (1.2 if seg[i].startswith("Retail") else 0.8)
        sf += (ls_stage / 8.0) * ls_vuln[i] * (0.30 if "SME" in seg[i] else 0.10)
        sf += max(0.0, 0.015 - gdp_yoy) * 1.5
        sf += (repo - 0.0775) * (-0.6)
        sf = float(np.clip(sf, -0.15, 0.50))

        if seg[i] == "Retail_Mortgage":
            c = float(np.clip(base_ccf[i] + 0.02 + sf * 0.3, 0.10, 0.45))
        elif seg[i] == "Retail_Vehicle":
            c = float(np.clip(base_ccf[i] + 0.02 + sf * 0.4, 0.10, 0.80))
        elif seg[i] == "Retail_CreditCard":
            c = float(np.clip(base_ccf[i] + 0.10 + sf * 0.8, 0.40, 0.98))
        elif seg[i] == "Retail_Overdraft":
            c = float(np.clip(base_ccf[i] + 0.10 + sf * 1.0, 0.40, 0.98))
        elif seg[i] == "SME_Corporate":
            c = float(np.clip(base_ccf[i] + 0.08 + sf * 0.9, 0.30, 0.95))
        elif seg[i] == "Corporate_Large":
            c = float(np.clip(base_ccf[i] + 0.05 + sf * 0.7, 0.25, 0.92))
        else:
            c = float(np.clip(base_ccf[i] + 0.03 + sf * 0.4, 0.20, 0.80))

        if dpd[i] >= 60:
            c = float(np.clip(c * 1.12, c, 0.99))
        elif dpd[i] >= 30:
            c = float(np.clip(c * 1.06, c, 0.99))
        if mob[i] < 12:
            c = float(np.clip(c * 1.08, c, 0.99))

        ccf[i] = c
        ead[i] = principal[i] + undrawn[i] * c
        total_line = max(principal[i] + undrawn[i], 1.0)
        util[i] = principal[i] / total_line
        sdd[i] = (c - base_ccf[i]) / max(base_ccf[i], 0.01)

    result["ead"] = ead
    result["ccf_applied"] = ccf
    result["current_utilisation"] = util
    result["stress_drawdown_vs_base_pct"] = sdd
    result["undrawn_component_ead"] = undrawn * ccf
    return result
