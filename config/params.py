from typing import Dict
PILLAR1_MINIMA = {'CET1': 0.045, 'Tier1': 0.060, 'Total_Capital': 0.080}
CAPITAL_BUFFERS = {'CCB': 0.025, 'CCyB': 0.010, 'D_SIB': 0.015}

# Updated SICR Triggers (Fix 3)
SICR_TRIGGERS = {
    "relative_increase_investment": 2.0,
    "relative_increase_sub_investment": 1.5,
    "absolute_increase_bps": 150,
    "dPD_threshold": 0.02
}

# Nedbank 2024 Pillar 3 Segment Weights
PORTFOLIO_SEGMENTS = {
    "Retail_Mortgage":    {"weight": 0.35, "pd": 0.008, "lgd": 0.20, "corr": 0.15},
    "Corporate_Large":    {"weight": 0.20, "pd": 0.015, "lgd": 0.40, "corr": 0.20},
    "SME_Corporate":      {"weight": 0.15, "pd": 0.025, "lgd": 0.45, "corr": 0.25},
    "Retail_Vehicle":     {"weight": 0.10, "pd": 0.015, "lgd": 0.35, "corr": 0.18},
    "Retail_CreditCard":  {"weight": 0.08, "pd": 0.040, "lgd": 0.80, "corr": 0.20},
    "Retail_Overdraft":   {"weight": 0.05, "pd": 0.050, "lgd": 0.75, "corr": 0.22},
    "Sovereign_Bank":     {"weight": 0.07, "pd": 0.002, "lgd": 0.10, "corr": 0.10},
}