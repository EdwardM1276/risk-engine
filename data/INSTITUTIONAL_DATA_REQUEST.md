# Institutional Account-Level Data Request

## Purpose

Provide an authorized, anonymized historical extract for IFRS 9 PD, LGD, EAD, staging, stress testing, and validation. The extract must be supplied through an approved bank, bureau, consortium, or secure data-room channel. Do not place identifiable customer data in this repository.

## Required grain

One row per `account_id` and `observation_date` snapshot. Preserve repeated observations for the same account. Include at least 36 months of history and enough forward performance to observe 12-month and lifetime default outcomes.

## Required fields

| Field | Type | Definition |
|---|---|---|
| `account_id` | string | Irreversible pseudonym; stable across periods |
| `observation_date` | date | Snapshot date |
| `segment` | string | Product or risk segment |
| `institution_size` | string | Bank classification |
| `province` | string | Coarsened geography, if approved |
| `principal_outstanding` | decimal | Drawn balance in reporting currency |
| `undrawn_limit` | decimal | Undrawn committed limit |
| `collateral_value` | decimal | Approved collateral value and valuation date if available |
| `loan_to_value` | decimal | LTV at observation date |
| `months_on_book` | integer | Account seasoning |
| `tenure_years` | decimal | Contractual or observed tenure |
| `dpd` | integer | Days past due at observation date |
| `internal_rating` | string | Rating grade at observation date |
| `debt_review_flag` | boolean | Debt review indicator |
| `judgement_flag` | boolean | Legal judgement indicator |
| `administration_order` | boolean | Administration-order indicator |
| `ifrs9_stage` | integer | Bank-assigned stage, if available |
| `default_flag` | boolean | Default status at observation date |
| `default_date` | date | First default date, nullable |
| `cure_date` | date | First approved cure date, nullable |
| `write_off_date` | date | Write-off date, nullable |
| `recovery_cashflow` | decimal | Recovery cash flow in reporting currency |
| `recovery_date` | date | Recovery cash-flow date |
| `credit_limit` | decimal | Total contractual limit |
| `utilisation` | decimal | Drawn balance divided by credit limit |
| `interest_rate` | decimal | Contractual effective rate or approved proxy |
| `cashflow_amount` | decimal | Contractual cash-flow amount, if available |
| `cashflow_date` | date | Contractual cash-flow date |

## Supporting reference tables

- Segment-to-product mapping and reporting-currency definition.
- Rating-scale definitions and rating migration history.
- Default definition, cure definition, restructuring treatment, and write-off policy.
- Collateral type, valuation, haircut, realization cost, and valuation-date definitions.
- IFRS 9 stage policy, SICR thresholds, scenario weights, and model version history.
- Data dictionary, missing-value codes, extraction timestamp, and source-system lineage.

## Privacy and governance requirements

- No names, identity numbers, addresses, account numbers, phone numbers, or free-text notes.
- Use irreversible pseudonymization with a bank-controlled re-identification key that is never shared.
- Obtain model-risk, legal, privacy, and data-owner approval before transfer.
- Transfer through an approved encrypted channel and retain the checksum of the received file.
- Record population scope, exclusions, currencies, extraction SQL/version, and historical restatements.
- Supply a small independently verified golden subset for validation, not just aggregate totals.

## Acceptance checks

The engine should reject the extract if:

- Required columns are absent.
- Account-period keys are duplicated unexpectedly.
- Dates are invalid or outside the approved study window.
- Balances, limits, or recoveries have unexplained sign or currency errors.
- Default dates precede account observation dates without an approved definition.
- Default, DPD, stage, rating, collateral, and recovery fields are entirely missing.
- Aggregate balances do not reconcile to the bank's control totals.

This request is an acquisition contract, not a synthetic-data substitute. Documentation or public disclosures may define concepts and validation ranges, but must not be used to manufacture account observations.
