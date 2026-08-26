import pandas as pd
import numpy as np

def scale_portfolio_to_target(portfolio: pd.DataFrame, target_exposure_bn: float) -> pd.DataFrame:
    # Drawn Exposure = Principal + (Undrawn * CCF 0.75)
    current_ead = (portfolio['principal'] + portfolio['undrawn'] * 0.75).sum()
    target_ead = target_exposure_bn * 1e9
    factor = target_ead / current_ead
    
    portfolio['principal'] *= factor
    portfolio['undrawn'] *= factor
    if 'collateral_value' in portfolio.columns:
        portfolio['collateral_value'] *= factor
    return portfolio