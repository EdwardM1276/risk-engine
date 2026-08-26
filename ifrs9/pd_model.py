import numpy as np

COEFFICIENTS = {
    "gdp_growth": -4.0,
    "unemployment": 2.5,
    "repo_rate": 0.8,
    "inflation": 0.6,
    "loadshedding": 0.25
}

def convert_ttc_to_pit(df, macro_params):
    # Simplified logit shift for demonstration
    logit_pd = np.log(df['pd'] / (1 - df['pd']))
    shift = (macro_params.get('gdp_yoy', 0) * COEFFICIENTS['gdp_growth'] +
             macro_params.get('unemployment_rate', 0) * COEFFICIENTS['unemployment'])
    df['pit_pd'] = 1 / (1 + np.exp(-(logit_pd + shift)))
    return df