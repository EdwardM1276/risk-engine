# Data dictionary and governed import contract

**Schema version:** 1.0 (proposed portable CSV contract). The current synthetic generator produces a richer frame; names below are canonical fields or equivalent model fields. All monetary amounts are ZAR unless a source explicitly declares another currency. Dates are ISO-8601. A governed import must preserve source values, source units, transformation lineage, and an as-of/value date.

## Core exposure fields

| Field | Type / unit | Required | Validation / lineage |
|---|---|---:|---|
| `account_id` | string, anonymised token | Yes | Unique within dataset; reversible identity mapping stays with the institution and is never imported |
| `as_of_date` | date | Yes | Observation date; no future observations relative to run date |
| `currency` | ISO 4217 code | Yes | Convert only with dated FX source and retain original amount/currency |
| `segment` | controlled string | Yes | Segment/product mapping and version required |
| `principal_outstanding` | decimal ZAR | Yes | Finite, non-negative; balance at as-of date |
| `undrawn_limit` | decimal ZAR | Yes | Finite, non-negative; committed undrawn amount at as-of date |
| `ccf` | fraction [0,1] | Required for off-balance exposures | Empirical or approved model value with source/version; never infer silently |
| `ead` | decimal ZAR | Derived or supplied | Reconcile to drawn balance + converted undrawn amount; record calculation method |
| `ttc_pd` | annual probability [0,1] | Required for IRB-style analysis | Source model/version, horizon, calibration date, rating grade required |
| `pit_pd_12m` | 12-month probability [0,1] | Required for ECL/simulation | Scenario, model version and macro as-of date required |
| `lifetime_pd` | cumulative probability [0,1] | Required for Stage 2/3 ECL | Contractual horizon/survival curve and method required; scalar proxy must be labelled |
| `internal_rating` | string | Required when available | Rating scale/version and effective date required; keep separate from PD |
| `lgd` | fraction [0,1] | Required | Ordinary LGD basis, recovery horizon, discount rate and collateral treatment required |
| `downturn_lgd` | fraction [0,1] | Required for regulatory calculation | Downturn method, downturn period and approval evidence required |
| `collateral_value` | decimal ZAR | Optional | Valuation date, valuation type, currency, lien/priority and haircut lineage required |
| `loan_to_value` | fraction | Optional | Specify numerator/denominator and valuation date |
| `dpd` | integer days | Required | Non-negative; measurement date and cure rules required |
| `stage` | integer 1, 2 or 3 | Required if supplied | Institution staging result/version/date; do not overwrite with prototype staging silently |
| `default_date` | date or null | Required for performance panel | Default definition and event source required |
| `recovery_amount` | decimal ZAR | Required for recovery panel | Recovery date, currency, costs, collateral source and discounting required |
| `restructuring_flag` | boolean | Optional | Effective date, reason and distressed/non-distressed classification required |
| `cash_flow_date`, `cash_flow_amount` | date, decimal ZAR | Required for full ECL implementation | Contractual/expected classification, currency and EIR discount basis required |

## Lineage and integrity checks

Each dataset should carry a separate manifest with dataset ID, schema version, source system, extraction timestamp, population definition, observation window, as-of date, source currency, de-identification method, approval reference, file hash, row count, and quality results. Validate required columns, unique account keys, finite/ranged values, duplicate rows, missingness, date ordering, EAD reconciliation, currency reconciliation, and portfolio total exposure. Outliers are flagged with thresholds and reviewed; they are not silently clipped.

Raw observed fields must remain separate from generated or transformed fields. Public aggregate context must not be expanded into pseudo-account records and labelled observed. The current FDIC/public path is a context/proxy mode; generated borrower attributes remain synthetic. Keep all identity/token translation outside this research prototype under the institution's governance controls.
