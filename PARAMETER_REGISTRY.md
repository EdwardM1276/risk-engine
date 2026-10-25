# Parameter registry

**Registry status:** research-prototype inventory. Values below are the values or source contracts used by the current code, not approved institution-specific model parameters. Review dates are the repository date (2026-10-25) unless a source date is shown.

| Parameter / group | Value or range | Unit | Type and lineage | Effective date / status |
|---|---:|---|---|---|
| `PILLAR1_MINIMA` | CET1 4.5%; Tier 1 6.0%; total 8.0% | ratio of RWA | Regulatory minima configured in `config/params.py`; verify against the applicable amended Regulations and institution-specific directions before use | Prototype reference; not a complete requirement matrix |
| `CAPITAL_BUFFERS.CCB` | 2.5% | ratio of RWA | Regulatory framework assumption in `config/params.py` | Buffer proxy; capital quality not fully modelled |
| `CAPITAL_BUFFERS.CCyB` | 1.0% | ratio of RWA | Configured from South African CCyB policy reference | Effective 2026-01-01 in the local specification; verify applicable rate for institution/date |
| `D_SIB_BUFFER_TIERS` | 0.5%–2.5% | ratio of RWA | Bucket schedule in `config/params.py`; benchmark/prototype mapping | Actual bank bucket requires official institution-specific evidence |
| `OUTPUT_FLOOR.phase_in` | 60%, 65%, 70%, 72.5% | fraction | PA Circular C4/2025, section 1.1.1; [official circular](https://www.resbank.co.za/content/dam/sarb/publications/prudential-authority/pa-deposit-takers/banks-circulars/2025/C4-2025-Basel%20III%20post-crisis%20reforms%20reporting.pdf) | 2025-07-01, 2026-01-01, 2027-01-01, 2028-01-01; historical lookup is date-effective |
| `PORTFOLIO_SEGMENTS.*.ttc_pd` | Segment values in `config/params.py` | annual probability | Synthetic portfolio assumption; no observed South African calibration | Prototype assumption |
| `PORTFOLIO_SEGMENTS.*.lgd` | Segment values in `config/params.py` | loss fraction | Synthetic ordinary LGD assumption | Prototype assumption |
| `PORTFOLIO_SEGMENTS.*.corr` | Segment values in `config/params.py` | correlation | Copula dependence assumption; not estimated from local default history | Prototype assumption |
| `PORTFOLIO_SEGMENTS.*.sa_rw` | Segment values in `config/params.py` | risk weight | Standardised credit RWA proxy; classification/mitigation not fully implemented | Not official standardised RWA |
| `SICR_TRIGGERS`, `NCA_THRESHOLDS` | See `config/params.py` | ratio, basis points, days, flags | Rule-of-thumb staging configuration; not a complete approved accounting policy | Prototype assumption |
| `NEDBANK_ECAP_BENCHMARK_2024` | Components in `config/params.py`; minimum-mix weights sum to 100%; 7% stress overlay | fraction of RWA-derived benchmark total | Public benchmark-derived allocation plus explicitly labelled 3% residual; no controlled institution component schedule | Benchmark year 2024; hybrid method |
| `MODEL_RISK_ALLOCATION` | 2% | fraction | Prototype minimum model-risk component allocation | Not a regulatory requirement |
| AFR ECL uplift | 30% of modelled ECL | fraction | Fixed prototype proxy in `run.py` | Not actual eligible AFR |
| Credit ECap | `VaR(99.9%) − simulated mean` | ZAR | Exploratory Monte Carlo estimate using PIT PD and ordinary LGD | One seed by default; formal uncertainty withheld |
| Market ECap | ES 97.5% | ZAR | Configured factor volatility/correlation proxy in `ecap/copula_mc.py` | Not validated or horizon-harmonised |
| Operational ECap | Pareto/event simulation | ZAR | Configured event-frequency and severity assumptions in `ecap/copula_mc.py` | Not calibrated to bank loss history |
| Macro/scenario paths | Base, Adverse, Severe and idiosyncratic shocks | rates, percentage changes, index units as labelled in code | Configured deterministic scenario assumptions in `scenarios/stress_engine.py`; fetched series are context/provenance unless directly used | User-selected as-of date; not a fitted forecast |

## Governance fields

For future governed parameters, record: stable key, description, numeric value/range, unit, source URL/document section, source publication date, retrieval date, jurisdiction, effective interval, type (`regulatory`, `empirical`, `benchmark`, `prototype assumption`), owner/reviewer, review status, sensitivity evidence, and change history. Current code does not yet provide a complete owner/approval workflow for every parameter.
