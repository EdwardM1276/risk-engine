# Validation report

## Scope and execution environment

This report records checks performed in the detached worktree based on upstream `main` commit `4b89985833c00dc5e9b98b2f6773c9e68ccf77c9`. The comprehensive prompt references `MODEL_RISK_CODEBASE_REVIEW.md`, but that file is absent from this checkout. The remediation tracker therefore maps the review areas in the supplied prompt to inspected code and tests; it cannot establish a line-by-line comparison with the missing document.

The available local interpreter was Python 3.9.13 with NumPy 1.26.4, pandas 2.2.2, SciPy 1.13.1 and Streamlit 1.32.2. The declared dependencies require Python 3.11 and newer package versions. The project suite and smoke run were executed with the available interpreter; CI is configured for Python 3.11 but no remote CI run is claimed.

## Test and runtime results

- `python -m unittest discover -s tests -v`: **60 tests passed**.
- `python -m compileall -q config data ecap engine ifrs9 nca regcap scenarios run.py dashboard`: passed (recorded after final changes below).
- The deterministic Base, Adverse and Severe run package is in [`sample_runs/sample_2026-10-25`](sample_runs/sample_2026-10-25). It uses 100 synthetic accounts, requested EAD ZAR 100 billion, seed 2024, 300 Student-t simulations (`df=6`), and as-of date 2026-10-20.
- Streamlit headless startup and `/_stcore/health` were checked on the available older Streamlit version. This does not constitute browser-level UI testing or verification against the declared dependency set.
- Public-data acquisition was not exercised against live providers in this validation; acquisition semantics are covered by unit tests. No credentials or institutional loan data were used.

## Scenario results

| Scenario | ECL (ZAR bn) | RWA (ZAR bn) | RWA density | CET1 ratio | Hybrid ECap indicator (ZAR bn) | Illustrative AFR coverage | 99.9% credit tail draws |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base | 1.5000 | 47.9973 | 48.00% | 11.60% | 18.3419 | 0.4301 | 1 of 300 |
| Adverse | 1.8949 | 54.2304 | 53.48% | 10.27% | 20.9603 | 0.3821 | 1 of 300 |
| Severe | 2.7746 | 60.8590 | 59.66% | 9.15% | 31.8559 | 0.2597 | 1 of 300 |

All EAD totals were approximately ZAR 100bn. The floor was 65% on the sample as-of date; it did not bind in these cases. The capital and coverage outputs are benchmark/prototype calculations, not a bank solvency assessment. The coverage numerator is explicitly a benchmark-constructed AFR proxy including 30% of ECL.

Compared with the earlier upstream smoke run under the same stated portfolio size, seed, draw count and date, Base RWA/CET1 were effectively unchanged (about ZAR 48.0bn / 11.60%); Base ECap increased from about ZAR 18.0843bn to ZAR 18.3419bn and proxy coverage moved from 0.4363 to 0.4301. Adverse ECL/RWA/CET1 were about ZAR 2.0551bn / 53.9148bn / 10.33% before and ZAR 1.8949bn / 54.2304bn / 10.27% after. Severe ECL/RWA/CET1 were about ZAR 2.8798bn / 60.2820bn / 9.24% before and ZAR 2.7746bn / 60.8590bn / 9.15% after. ECap/coverage also changed. These movements reflect the now-connected custom market/operational shocks, separated hybrid residual and stress overlay, and calibration/staging changes. Values are a small deterministic synthetic sample and should not be treated as a clean empirical model comparison.

## Calculation and calibration checks

- A numerical ASRF reference test independently evaluates the Vasicek conditional PD equation at PD 1%, LGD 40%, asset correlation 15%, and 99.9% confidence, and checks that unexpected loss is `(conditional PD - PD) × LGD` to 10 decimal places.
- Effective-dated output floor tests cover 0% through 2025-06-30, 60% from 2025-07-01, 65% from 2026-01-01, 70% from 2027-01-01 and 72.5% from 2028-01-01.
- Sample calibration reached ECL/EAD 1.499999% against a 1.5% target (residual −8.87e-9); credit RWA density 34.5581% against a 34.56% target (residual −1.91e-5). Both targets were marked attainable and converged within factor bounds [0.25, 4.0]. The RWA calibration adjusts a prototype ordinary-LGD scalar, not a validated downturn parameter.
- Tests verify that an unreachable ECL target is flagged, PD calibration restages when macro conditions are supplied, synthetic time series are date-aligned/as-of bounded, and custom sovereign, commodity, cyber, housing and SME failure-wave shocks reach their configured downstream calculations.
- Correlation matrix repair is reported with before/after minimum eigenvalue and Frobenius adjustment. No repair was needed for Base or Adverse in this sample. Severe required a material repair with Frobenius adjustment 0.416737; this is a model input concern requiring review, not evidence that the repaired dependence structure is valid.

## Tail quality and uncertainty

For 300 simulations, the empirical 99.9% tail corresponds to one order statistic (expected tail count 0.3); one seed/replicate was used. Each scenario therefore emits a tail adequacy warning. Nested prefix stability is not called convergence; no formal tail confidence interval is reported. Reported heuristic ranges are explicitly illustrative, and the credit tail interval is withheld.

## Limitations and unresolved validation

- Synthetic data supports software and known-generating-process checks only. The package preserves synthetic inputs for repeatable demonstration; replay of all live public provider responses is not implemented.
- Public aggregate information is context only and is not account-level observed data. This run did not fetch live public data.
- No institution-authorised default, recovery, migration, collateral, contractual cash-flow or reconciled capital-resource history was available. Out-of-time performance, empirical calibration, loss forecasts and actual AFR cannot be validated here.
- The RWA, capital stack, output floor application and ECap aggregation remain stated proxies. The model is not a full IFRS 9 implementation, Basel/SARB regulatory calculator, regulatory return, bank solvency assessment or production model.
- Independent model validation and browser-level dashboard checks remain future work.
