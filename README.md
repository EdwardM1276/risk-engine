Purpose
A South African-specific credit risk volatility engine for simulating macroeconomic stress impacts on banking portfolios. It calculates IFRS 9 ECL, Basel III regulatory capital, and Monte Carlo-based Economic Capital.

Key Technologies
Python, Streamlit, Pandas, NumPy, SciPy, Plotly, ReportLab, World Bank/FRED APIs.

Top-Level Structure
ecap/: Monte Carlo copula simulation engine.
ifrs9/: ECL staging and impairment logic.
regcap/: Basel III RWA and capital stack calculations.
data/: Acquisition, normalization, and synthetic book calibration.
dashboard/: Streamlit UI for scenario analysis and visualization.
run.py: Primary orchestrator and CLI entrypoint.
config/: Centralized regulatory parameters and model coefficients.
tests/: Sanity checks against D-SIB Pillar 3 benchmarks.
Key Concepts
AFR: Available Financial Resources; total capital for ECap coverage.
Coverage Ratio: Ratio of AFR to total Economic Capital required.
EAD: Exposure at Default; principal plus CCF-adjusted undrawn amounts.
ECL Stages: IFRS 9 classification (Stage 1, 2, or 3) based on SICR.
SICR: Significant Increase in Credit Risk; trigger for Stage 2 migration.
Vasicek ASRF: Model used for IRB-style credit risk RWA calculation.
T-Copula: Dependency model capturing tail risk in correlated defaults.
D-SIB: Domestic Systemically Important Bank; specific SA regulatory class.
Loadshedding Vulnerability: SA-specific risk factor for SME and Retail segments.
RWA Density: Ratio of Risk-Weighted Assets to total Exposure.
Model Risk Reserve: Capital add-on (e.g., 2-5%) for model uncertainty.
Erosion Path: Stress test showing capital depletion across shock intensities.
CCF: Credit Conversion Factor; percentage of undrawn lines expected to default.
PIT PD: Point-in-Time Probability of Default; macro-adjusted risk metric.
Sovereign CDS: Macro variable representing South African sovereign risk.

https://risk-engine-gqdwpubmpfdm6qyv44yybz.streamlit.app/
