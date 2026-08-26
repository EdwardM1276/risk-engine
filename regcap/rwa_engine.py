import numpy as np
from scipy.stats import norm
import pandas as pd

def compute_vasicek_rwa(pd, lgd, ead, maturity=2.5):
    # Basel III Paragraph 272: Asset Correlation (R)
    pd = np.clip(pd, 0.0003, 0.999)
    r = 0.12 * (1 - np.exp(-50 * pd)) / (1 - np.exp(-50)) + 0.24 * (1 - (1 - np.exp(-50 * pd)) / (1 - np.exp(-50)))
    
    # Basel III Paragraph 273: Maturity Adjustment (MA)
    b = (0.11852 - 0.05478 * np.log(pd)) ** 2
    ma = (1 + (maturity - 2.5) * b) / (1 - 1.5 * b)
    
    # ASRF Capital Charge (K)
    z = (norm.ppf(pd) + np.sqrt(r) * norm.ppf(0.999)) / np.sqrt(1 - r)
    k = (lgd * norm.cdf(z) - pd * lgd) * ma
    
    # RWA = K * 12.5 * EAD
    return ead * k * 12.5

def compute_total_rwa(portfolio_df, macro_scenarios=None):
    ead = portfolio_df['principal'] + portfolio_df.get('undrawn', 0) * 0.75
    rwa_vals = compute_vasicek_rwa(portfolio_df['pd'].values, portfolio_df['lgd'].values, ead.values)
    return {"total_rwa": float(np.sum(rwa_vals)), "methodology": {"credit": "Basel III IRB ASRF"}}