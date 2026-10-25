# South African Credit Risk Volatility Engine

An educational research prototype for synthetic credit portfolios. It connects IFRS 9-style staging and ECL, Basel-inspired RWA and capital sensitivities, Monte Carlo losses, benchmark-derived economic capital, and an illustrative AFR coverage measure. It is not a bank model, regulatory return, IFRS 9 implementation, solvency assessment, or independently validated model.

The default dashboard mode is **Synthetic demonstration**. The optional **Public context + synthetic accounts** mode combines public aggregate context with generated borrower-like rows; it does not create observed South African loan-level data. Governed institutional data can be supplied through the portfolio CSV contract documented in [DATA_DICTIONARY.md](DATA_DICTIONARY.md), subject to the institution’s own authorisation and controls.

The dashboard opens on a plain-language **Learn the Model** tab with short lessons, examples drawn from the active run, and quick understanding checks. **Model Uncertainty** explains the estimate ranges and the meaning and limits of the 99.9% level. The bank comparison is shown as a table with public source references and notes about non-comparable assumptions. The separate UCI observed-data CLI uses an `.xls` workbook; `xlrd` is included in `requirements.txt` for that reader.

## Set up and run

Use Python 3.11 and create an isolated environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m compileall -q config data ecap engine ifrs9 nca regcap scenarios run.py dashboard
python run.py pipeline --scenario Base --n-accounts 100 --n-mc-sims 300 --seed 2024
python -m streamlit run streamlit_app.py
```

The `requirements.txt` declares supported dependency ranges. The repository tests use `unittest`; `pytest` is optional. CI is configured for Python 3.11. See [VALIDATION_REPORT.md](VALIDATION_REPORT.md) for the interpreter and dependency versions available during local verification.

## Scenarios and reproducibility

The scenario engine applies configured Base, Adverse and Severe paths, plus optional idiosyncratic overlays. Supply `as_of_date` when calling `run_engine_end_to_end` for historical runs. The run package records the as-of date, model/reference versions, seed, run ID, configuration digest, and a JSON snapshot under `outputs/runs/`. `reproduce_run(run_id)` replays the stored request and checks its configuration digest. Synthetic generation is seeded and date-anchored.

The current replay snapshot stores run configuration, not immutable raw snapshots for every acquired public provider response. The synthetic example includes preserved input snapshots and hashes in [the sample run package](sample_runs/sample_2026-10-25/). Public-source replay can therefore still depend on upstream availability and changes. See [VALIDATION_REPORT.md](VALIDATION_REPORT.md) for recorded checks and their limits.

## Method and limitations

- ECL is an **IFRS 9-style proxy** using simplified SICR, lifetime PD and undiscounted `EAD × PD × LGD`; it does not implement full contractual cash flows, EIR discounting or probability-weighted economic scenarios.
- Credit RWA uses a Vasicek/ASRF inspired calculation and a dated output-floor sensitivity. Standardised credit weights are proxies, while market, operational and other RWA use benchmark-scaled values; the result is not official regulatory RWA.
- Economic capital is a **hybrid indicator**: simulated credit/market/operational estimates replace selected benchmark components, remaining components use benchmark assumptions, and a deterministic stress overlay is added without cross-risk diversification. Credit 99.9% VaR from the interactive sample size is exploratory.
- AFR is a proxy built from benchmark-constructed capital resources and a fixed 30% ECL addition. It is not actual bank AFR.
- Calibration to configured public benchmark aggregates is benchmark alignment, not independent validation.

Read [MODEL_SPECIFICATION.md](MODEL_SPECIFICATION.md), [PARAMETER_REGISTRY.md](PARAMETER_REGISTRY.md), [DATA_DICTIONARY.md](DATA_DICTIONARY.md), [MODEL_RISK_CODEBASE_REVIEW.md](MODEL_RISK_CODEBASE_REVIEW.md), and [VALIDATION_REPORT.md](VALIDATION_REPORT.md) before interpreting outputs.

## Repository map

`data/` acquires and generates data; `scenarios/` creates scenario transformations; `ifrs9/` calculates ECL-style results; `regcap/` calculates RWA/capital proxies; `ecap/` simulates losses and computes the hybrid indicator; `run.py` orchestrates and persists run metadata; `dashboard/` presents results; `tests/` contains software and numerical contract tests.

## Deployment

The dashboard is intended for local research use. Do not upload confidential borrower data to a public host. Streamlit deployment, access control, secrets handling, encryption, retention, monitoring, change approval, and incident response have not been established as a governed production service. Container instructions in [DOCKER.md](DOCKER.md) are convenience tooling, not a production deployment approval.
