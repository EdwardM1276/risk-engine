# Remediation tracker

This tracker maps the supplied model-risk review to the current upstream `main` codebase (base commit `4b89985833c00dc5e9b98b2f6773c9e68ccf77c9`) and the changes made in this worktree. It does not replace the historical review. **Completed** means implemented and covered by tests where stated; **partial** means the prototype has a control or disclosure but important work remains; **deferred** means it requires unavailable governed data or a broader methodology.

| Review area | Status | Implementation and evidence | Remaining limitation |
|---|---|---|---|
| Scenario mapping, repo level/shock, idiosyncratic propagation | Partial | `scenarios/stress_engine.py`, `ifrs9/pd_model.py`, `ecap/copula_mc.py`, `tests/test_remediation_contracts.py`; five custom-shock downstream tests | Assumptions remain configured rather than empirically fitted; gold/composite market channels remain stylized |
| TTC/PIT PD, ordinary/downturn LGD, EAD and ASRF contracts | Partial | `ifrs9/`, `regcap/rwa_engine.py`; ASRF numerical reference test and bounded PIT transform tests in `tests/test_risk_sanity.py` | RWA uses ordinary LGD with a calibration scalar; no institution-approved downturn LGD, full exposure classification, or independent regulatory reference implementation |
| Standardised RWA and output floor | Partial | RWA labelled sensitivity proxy; date-effective SA floor implementation in `regcap/rwa_engine.py`; boundary tests in `tests/test_remediation_contracts.py` | Standardised non-credit components remain benchmark-scaled. No official RWA or compliance claim is supported |
| Capital requirements / capital quality | Partial | Configured D-SIB bucket schedule and ratio tests | Buffer composition and institution-specific Pillar 2/add-ons, eligibility, deductions and supervisory directions are not completely modelled |
| ECap mix, hybrid method, stress overlay | Partial | `ecap/allocation.py` makes hybrid method explicit; minimum mix reconciles to 100% with named residual; stress overlay separate; test added | Benchmark residual is assumed; component horizons/confidence standards are not harmonised; no joint cross-risk simulation/diversification |
| Copula construction and matrix validity | Partial | Shared t scale; eigenvalue-based correlation repair with eigenvalue and adjustment diagnostics; tested positive definiteness | Materially repaired matrices still require model review; dependence assumptions are synthetic |
| Extreme-tail evidence and uncertainty labels | Partial | Prefix stability is no longer called convergence; low-tail warning; credit interval withheld; heuristic ranges named; tests added | Only one seed is run by default; repeated independent replications, quantile intervals, importance sampling and validated tail accuracy remain future work |
| AFR / coverage | Partial | UI and returned coverage object label benchmark-constructed capital plus 30% ECL as proxy | No actual eligible AFR or reconciled capital resource schedule |
| Synthetic/public data semantics and quality | Partial | Synthetic is default; public option is named “Public context + synthetic accounts”; acquisition tests preserve public-vs-synthetic distinction | Complete immutable provider snapshots, outlier diagnostics and broad source-lineage reporting remain incomplete |
| Calibration transparency | Partial | `data/calibration.py`; bounded bisection solves, target attainability, initial/achieved values, residual, bounds, iteration counts, and stage recomputation after PD calibration; `tests/test_remediation_contracts.py` and `tests/test_risk_sanity.py` | Calibration still aligns to configured benchmark aggregates and is not independent validation; the LGD factor is a prototype RWA scalar |
| Snapshot and replay | Partial | `run.py`, `engine/reproducibility.py`, `tests/test_reproducibility.py`; synthetic deterministic replay test passes | Run configuration is persisted, but raw public provider frames are not all preserved; public replay can fetch changed upstream data |
| Model specification, parameter registry, data dictionary, README | Completed for prototype scope | Added `MODEL_SPECIFICATION.md`, `PARAMETER_REGISTRY.md`, `DATA_DICTIONARY.md`; README updated | Governance owner/approval workflow and complete machine-readable parameter metadata remain future work |
| Software verification | Partial | Baseline and post-change unit suites; compile checks; deterministic scenario smoke runs | Streamlit browser exercise, CI workflow evidence, mutation/property tests and comprehensive shock tests remain incomplete |
| Institutional, out-of-time and independent validation | Deferred | Future data interface documented; synthetic known-truth checks are described as software checks only | Requires authorised account/default/recovery/migration/capital data and independent reviewer; not claimed complete |

The supplied prompt names `MODEL_RISK_CODEBASE_REVIEW.md` as a source document, but that file is not present in the fetched upstream `main` checkout. This tracker therefore records the prompt's specified review areas against the inspected source and does not claim a line-by-line reconciliation to the missing review history.

## Decisions made

1. The standardised RWA and floor stay as sensitivity proxies because the current engine does not compute complete standardised market, operational and credit-mitigation requirements.
2. Tail prefix stability is descriptive only. One-seed empirical 99.9% VaR does not receive a formal confidence interval or a green convergence claim.
3. ECap is labelled a hybrid indicator. The benchmark minimum risk mix has an explicit 3% residual assumption; the stress buffer is kept separate as an overlay.
4. The 30% ECL addition remains only as a visibly labelled AFR proxy because no actual AFR resource schedule is supplied.
5. Historical output-floor logic uses effective dates from PA Circular C4/2025 rather than year-only/current-system-date logic.
