import numpy as np
import pandas as pd
from regcap.rwa_engine import compute_vasicek_rwa
from config.params import PORTFOLIO_SEGMENTS

def generate_raw_portfolio(n_accounts: int, segment_weights: dict = None) -> pd.DataFrame:
    if segment_weights is None:
        # Nedbank's lower RWA density is driven by a high share of secured lending (Mortgages)
        segment_weights = {
            'Retail_Mortgage': 0.65,  # Aggressive shift to low-RWA mortgages
            'SME_Corporate': 0.10,
            'Corporate_Large': 0.10,
            'Retail_Vehicle': 0.05,
            'Retail_CreditCard': 0.02,
            'Retail_Overdraft': 0.02,
            'Sovereign_Bank': 0.06
        }

    segments = list(segment_weights.keys())
    probs = list(segment_weights.values())

    data = []
    for _ in range(n_accounts):
        seg = np.random.choice(segments, p=probs)
        config = PORTFOLIO_SEGMENTS[seg]

        data.append({
            'account_id': f'ACC_{_}',
            'segment': seg,
            # Use a very tight Beta distribution to keep PDs in a healthy range
            'pd': np.random.beta(0.1, 200) * config['pd'],
            'lgd': config['lgd'] * 0.8, # Assume better collateral recovery for calibration
            'principal': 1000000,
            'undrawn': 250000,
            'correlation': config['corr']
        })
    return pd.DataFrame(data)

def compute_rwa_density(df: pd.DataFrame) -> float:
    eads = df['principal'] + df['undrawn'] * 0.75
    rwas = compute_vasicek_rwa(df['pd'].values, df['lgd'].values, eads.values)
    return float(rwas.sum() / eads.sum())