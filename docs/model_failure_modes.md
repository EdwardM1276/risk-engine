# Model Failure-Modes Register — SA Credit Risk Volatility Engine

Each entry records a failure mode, trigger condition, detection control, and current status. This register is living documentation; new model features must add or update an entry before merge.

## FM-01 Tail under-convergence (Credit MC)
- Component: `ecap/copula_mc.py`
- Failure: 99.9% VaR/ES estimated from too few tail observations.
- Trigger: `n_sims` below approximately 100,000 at 99.9%.
- Detection: `tail_estimate_warning` and convergence flags; tests assert the warning at small sample sizes.
- Status: MITIGATED.

## FM-02 Output floor misapplication
- Component: `regcap/rwa_engine.py`
- Failure: reported RWA uses modelled value while the floor binds, or vice versa.
- Trigger: modelled RWA below floor percentage times standardized RWA.
- Detection: `output_floor_binding` and binding/non-binding unit tests.
- Status: MITIGATED.

## FM-03 Double application of severity
- Component: `scenarios/stress_engine.py` and `run.py`
- Failure: severity is applied to an already stressed parameter and then applied again through shock composition.
- Trigger: Adverse or Severe scenario combined with an overlapping idiosyncratic shock.
- Detection: severity-sweep monotonicity test; total ECL must be non-decreasing.
- Status: MITIGATED by regression coverage.

## FM-04 Hard-coded time-varying inputs
- Component: `config/params.py`
- Failure: rates and buffers compiled as constants destroy replay and governance.
- Trigger: a time-varying scalar is added directly to model code.
- Detection: review and `config/reference_data.py` effective-dated records.
- Status: MITIGATED for governed reference-rate access.

## FM-05 T-copula degrees-of-freedom instability
- Component: `ecap/copula_mc.py`
- Failure: degrees of freedom at or below 2 produces unstable tail variance.
- Trigger: `copula_type='t'` with invalid `t_df`.
- Detection: input validation raises `ValueError`.
- Status: CLOSED.

## FM-06 Silent benchmark fallback
- Component: `config/reference_data.py`
- Failure: missing rates silently fall back to stale values.
- Trigger: lookup before first effective record or after cessation.
- Detection: `RateUnavailable`; no stale fallback path exists.
- Status: CLOSED.

## FM-07 Principal-at-default recomputation
- Component: `nca/in_duplum.py`
- Failure: cap base is recomputed from current principal instead of frozen default-event principal.
- Trigger: a default episode is constructed from current balance.
- Detection: frozen `DefaultEpisode` and explicit principal-at-default field.
- Status: CLOSED.

## FM-08 Rounding policy inconsistency
- Component: posted monetary outputs
- Failure: per-item rounding diverges from round-at-posting and results are not reproducible to cents.
- Trigger: float arithmetic crosses a posted-output boundary without `engine.money` helpers.
- Detection: `round_post` and `post_sum` tests.
- Status: MITIGATED at ECL, ECap, summary, and NCA boundaries.

## FM-09 Back-valued restaging
- Component: `ifrs9/staging.py`
- Failure: late-recorded accounts are restaged retroactively from run time.
- Trigger: staging logic uses operational timestamp rather than account `value_date`.
- Detection: value-date contract and temporal regression test.
- Status: MITIGATED by preserving account state at `value_date`.

## FM-10 Run non-reproducibility
- Component: `run.py` and `engine/reproducibility.py`
- Failure: identical inputs produce materially different outputs without a recorded configuration identity.
- Trigger: missing seed, as-of date, or governed version in the snapshot.
- Detection: config digest, stored snapshot, run ID sequence, and reproduction round-trip.
- Status: MITIGATED.

## Monetary policy

Internal simulation arrays may remain float64 for performance. Posted totals use Decimal conversion through text and half-even rounding once at the posting boundary. This policy is not a claim that the model is otherwise financially or legally production-ready.
