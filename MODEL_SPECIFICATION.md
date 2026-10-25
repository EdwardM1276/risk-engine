# Model specification

**Version:** research-prototype specification, 2026-10-25  
**Scope:** executable contracts in this repository; parameter/source detail is in [PARAMETER_REGISTRY.md](PARAMETER_REGISTRY.md).  
**Use restriction:** educational and exploratory analysis only. No empirical South African account-level validation or independent model validation has been completed.

## Purpose and flow

The pipeline is: scenario definition → synthetic/public/institutional input acquisition → PIT PD, LGD and EAD transformations → staging/ECL → credit and benchmark-scaled non-credit RWA → illustrative capital stack → correlated loss simulation → hybrid ECap indicator → AFR proxy coverage. Public mode provides public macro/financial context and, where configured, aggregate US bank information; generated account rows remain synthetic.

Amounts use nominal ZAR unless otherwise stated. PD, LGD, CCF, rates and correlations are decimal fractions. EAD, ECL, RWA, capital and ECap are ZAR. Scenario `as_of_date` is the effective date for historical transformations and regulatory schedule lookups.

## Credit inputs and definitions

- **TTC PD** is the synthetic/account input through-the-cycle one-year PD (`ttc_pd`). It feeds the IRB-inspired RWA formula by default.
- **PIT PD** (`pit_pd_12m`) is calculated from TTC PD and macro factors using the logistic shift in `ifrs9/pd_model.py`; it is a one-year conditional proxy and drives default thresholds and 12-month ECL.
- **Lifetime PD** (`lifetime_pd`) is a simplified cumulative proxy generated from PIT PD and a configured horizon. It is not a survival curve or term structure.
- **Internal rating** is a synthetic category for segmentation and SICR heuristics. It is not a calibrated rating scale.
- **EAD** is drawn principal plus undrawn commitment multiplied by the stress-adjusted CCF: `EAD = principal_outstanding + undrawn_limit × ccf_applied`.
- **LGD** is a scenario-adjusted ordinary loss severity informed by segment, collateral and macro assumptions. **Downturn LGD** is a separate stress proxy. ECL currently consumes ordinary LGD; IRB RWA currently consumes ordinary LGD after a benchmark calibration scalar. These are prototype conventions, not regulatory LGD parameter approval.

## Staging and ECL

Stage 3 is assigned by configured credit-impaired/backstop indicators. Stage 2 is assigned for configured SICR signals, including relative/absolute PD deterioration, delinquency and selected qualitative signals. Stage 1 is the remainder. Trigger values are heuristics, not a complete IFRS 9 policy.

The executable simplified calculation is:

```text
12m ECL       = EAD × PIT_PD_12m × ordinary_LGD
lifetime ECL  = EAD × lifetime_PD × ordinary_LGD
stage ECL     = lifetime ECL for Stage 2 or 3; otherwise 12m ECL
```

The code does not discount expected cash shortfalls with effective interest rates and does not probability-weight multiple forecast scenarios. It is therefore not a complete IFRS 9 ECL calculation.

## Credit RWA and floor proxy

For PD `p`, LGD `L`, asset correlation `R`, confidence `α`, maturity `M`, and `Φ` the standard normal CDF:

```text
K = L × [Φ((Φ⁻¹(p) + √R × Φ⁻¹(α)) / √(1-R)) - p] × MA
credit RWA = EAD × K × 12.5
```

`α=99.9%`; expected loss is excluded from `K`; maturity adjustment is applied to corporate/SME classes only and maturity is clipped to 1–5 years. Current RWA uses TTC PD, ordinary LGD multiplied by the reported calibration factor, and exposure-class correlations. This is an ASRF-inspired prototype, not a complete IRB implementation.

The standardised credit base is `EAD × configured segment proxy risk weight`. Market, operational and other RWA are benchmark-scaled rather than independently calculated under their full standardised approaches. The output floor applies `max(modelled RWA, floor percentage × incomplete standardised RWA proxy)`. South African effective dates are 60% from 2025-07-01, 65% from 2026-01-01, 70% from 2027-01-01 and 72.5% from 2028-01-01, based on PA Circular C4/2025, section 1.1.1. Pre-implementation dates return 0%. The result is a sensitivity proxy, not regulatory RWA or a compliance status.

## Capital requirements and resources

`regcap/capital_stack.py` applies prototype hard minima, CCB, CCyB, D-SIB and leverage assumptions. It does not implement an institution-specific capital-quality matrix, supervisory add-ons, directions, or a reconciled resource schedule. Reported capital-resource values are benchmark-constructed when no controlled schedule is supplied. A capital ratio from this mode is not an assessment of a real bank.

## Economic capital

Credit losses are generated from a Gaussian or Student-t copula. The t construction uses a shared chi-square scale for the multivariate latent vector. Loss per simulated account is default indicator × EAD × ordinary LGD × stochastic severity multiplier. The 99.9% VaR less simulated mean is displayed as a credit ECap proxy. At small counts the tail has too few observations; nested prefix stability from one seed is not independent replication and cannot establish tail accuracy. No formal tail interval is reported.

Market ECap is an ES 97.5% proxy from a configured factor simulation. Operational ECap uses configured event/severity assumptions. These measures do not share a validated common horizon or calibration. The aggregate is explicitly a **hybrid economic-capital indicator**: simulated credit/market/operational measures replace those benchmark components; business, model and other risk components remain benchmark-derived; a deterministic stress buffer is overlaid; no diversification benefit is applied. The minimum risk composition sums to 100%, including an explicit 3% benchmark residual assumption. The 7% stress buffer is separate from that mix.

## AFR and coverage

The illustrative numerator is benchmark-constructed total capital resources plus `30% × modelled ECL`. The 30% is a prototype proxy, not an eligibility rule. The coverage ratio is `illustrative AFR / hybrid ECap indicator`. It must not be described as actual AFR or real-bank solvency coverage. Coverage erosion applies deterministic scenario multipliers to AFR and ECap; these are stress overlays, not a joint simulation.

## Calibration, data, and reproducibility

Calibration aligns synthetic aggregates with configured benchmark targets. The ECL routine reapplies the PD scalar, restages when macro conditions are supplied, and recomputes dependent ECL fields. Calibration results expose target, initial/achieved values, factor bounds, attainability and residual; a discrete stage transition can make an aggregate target unreachable. The same target is not independent validation. Synthetic dates are seeded and anchored to normalized month-ends no later than the as-of date. The run snapshot records request/configuration identity; the committed sample package additionally preserves its exact synthetic input frames and hashes. Current replay re-executes the request; the stored run snapshot does not preserve every immutable public provider response, so public replay is not guaranteed if upstream data change. Input schema and expected units are listed in [DATA_DICTIONARY.md](DATA_DICTIONARY.md).

## Limitations and future validation

Future work requiring governed evidence includes empirical default/recovery histories, rating migration and macro sensitivity calibration, reconciled capital schedules, bank-approved model outputs, out-of-time validation, independent review, and a complete regulatory calculation specification. No production readiness, regulatory compliance, empirical calibration or independent validation is claimed.
