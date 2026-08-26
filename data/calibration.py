import pandas as pd
import numpy as np
from scipy.optimize import minimize_scalar
from data.acquisition import scale_portfolio_to_target
from regcap.rwa_engine import compute_vasicek_rwa
from config.params import PORTFOLIO_SEGMENTS

def calibrate_synthetic_book(target_metrics: dict, n_accounts: int = 5000) -> pd.DataFrame:
    # 1. Generate distribution using Nedbank 2024 weights
    weights = target_metrics.get("segment_weights") or {
        "Retail_Mortgage": 0.35, "Corporate_Large": 0.20, "SME_Corporate": 0.15,
        "Retail_Vehicle": 0.10, "Retail_CreditCard": 0.08, "Retail_Overdraft": 0.05,
        "Sovereign_Bank": 0.07
    }
    segments = list(weights.keys())
    probs = list(weights.values())
    
    data = []
    for _ in range(n_accounts):
        seg = np.random.choice(segments, p=probs)
        cfg = PORTFOLIO_SEGMENTS[seg]
        data.append({
            'segment': seg,
            'pd': cfg['pd'],
            'lgd': cfg['lgd'],
            'principal': 100000,
            'undrawn': 20000
        })
    df = pd.DataFrame(data)
    
    # 2. Scale Exposure to target (e.g., R1,100bn)
    df = scale_portfolio_to_target(df, target_metrics['total_exposure_bn'])

    # 3. Solve for LGD scalar to hit 59% RWA density
    target_density = target_metrics.get("rwa_density_pct", 59) / 100
    
    def objective(lgd_scale):
        temp_lgd = np.clip(df['lgd'] * lgd_scale, 0.05, 0.90)
        ead = df['principal'] + df['undrawn'] * 0.75
        rwas = compute_vasicek_rwa(df['pd'].values, temp_lgd.values, ead.values)
        current_density = rwas.sum() / ead.sum()
        return (current_density - target_density)**2

    res = minimize_scalar(objective, bounds=(0.1, 2.0), method='bounded')
    df['lgd'] = np.clip(df['lgd'] * res.x, 0.05, 0.90)
    
    return df

def compute_rwa_density(df):
    ead = df['principal'] + df['undrawn'] * 0.75
    rwa = compute_vasicek_rwa(df['pd'].values, df['lgd'].values, ead.values)
    return rwa.sum() / ead.sum()