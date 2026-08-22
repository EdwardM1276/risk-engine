# South African Credit Risk Volatility Engine
## Comprehensive Project Report

**Report date:** 22 August 2026  
**Project type:** Python risk analytics prototype and Streamlit decision-support dashboard  
**Repository:** `EdwardM1276/risk-engine`  
**Deployment target:** Streamlit Community Cloud

---

## Executive Summary

The South African Credit Risk Volatility Engine is an end-to-end analytical prototype that connects macroeconomic stress scenarios to credit loss, capital requirements, economic capital, and available financial resources. Its intended purpose is to show how a South African banking portfolio could be evaluated across IFRS 9 expected credit loss (ECL), Basel-style risk-weighted assets and regulatory capital, Monte Carlo economic capital, and coverage under increasingly severe stress.

The project achieved several meaningful outcomes. It established a top-level orchestration layer, generated a synthetic South African loan book, implemented a complete model pipeline, exposed the results through a Streamlit dashboard, added scenario and idiosyncratic shock controls, created a GitHub repository, and initiated a public Streamlit deployment. It also resolved a data-contract defect between a dataclass and dictionary-style consumers, simplified fragile dashboard visualisations, and added defensive rendering around individual tabs.

The project is useful as a learning tool, architecture demonstrator, model prototyping environment, and conversation aid for risk teams. It is not, in its current form, suitable for direct use in regulatory capital calculation, financial reporting, impairment provisioning, limit setting, or an audited bank production environment. The main reasons are synthetic rather than institution-specific data, simplified calibration, incomplete validation and testing, assumptions that are illustrative rather than empirically evidenced, and several methodological limitations in the simulation layer. Its value therefore lies in demonstrating an integrated workflow and identifying where a production capability would need stronger data, governance, validation, and operational controls.

---

## 1. Project Aims

The project was designed around five connected aims:

1. **Integrate major banking risk concepts.** Bring together IFRS 9 ECL, Basel III-style capital, economic capital, stress testing, and coverage analysis in one reproducible workflow.
2. **Represent South African risk drivers.** Include macroeconomic variables, loadshedding, commodity prices, sovereign credit conditions, rand volatility, unemployment, South African provinces, and local banking benchmarks.
3. **Make scenario transmission visible.** Show how a macro or idiosyncratic shock can move through exposure, default risk, loss-given-default, ECL, RWA, capital, economic capital, and coverage.
4. **Create a usable analytical interface.** Provide an interactive dashboard through which a user can alter scenario severity, portfolio size, account count, Monte Carlo simulations, institution classification, copula family, and idiosyncratic shocks.
5. **Create a repeatable project entry point.** Allow the full engine and dashboard to be launched through a top-level command rather than requiring users to understand each module separately.

The project is therefore best understood as an integrated prototype, rather than as a single narrowly defined model.

---

## 2. Scope and Architecture

The main architecture is a sequential pipeline coordinated by `run.py`:

```text
Scenario configuration
        |
        v
Synthetic data acquisition
        |
        v
IFRS 9 PD, LGD, EAD, staging and ECL
        |
        v
RWA calculation and regulatory capital stack
        |
        v
Credit, market and operational Monte Carlo capital
        |
        v
Economic capital allocation
        |
        v
AFR coverage and stress erosion
        |
        v
Benchmarks, uncertainty bands and dashboard
```

### Main components

- **`config/params.py`** centralises scenario ranges, portfolio segment assumptions, regulatory ratios, capital buffers, benchmark values, SICR triggers, and display colours.
- **`data/acquisition.py`** creates macroeconomic, loadshedding, market, time-series, and loan portfolio data. It returns a `RawDataset` container with dictionary-compatible access.
- **`ifrs9/`** contains PD conversion, LGD adjustment, EAD calculation, staging, and ECL application.
- **`regcap/`** contains RWA and regulatory capital calculations.
- **`ecap/`** contains credit copula simulation, market risk simulation, operational risk simulation, allocation, and coverage analysis.
- **`scenarios/`** converts named scenarios into macro and market conditions and applies idiosyncratic overlays.
- **`run.py`** executes the full pipeline and exposes pipeline and dashboard command modes.
- **`dashboard/app.py`** presents engine controls, headline metrics, scenario summaries, tables, and simplified charts.
- **`streamlit_app.py`** is the deployment entry point for Streamlit Cloud.
- **`.streamlit/config.toml`** provides deployment and theme settings.
- **`tests/test_orchestrator.py`** provides a basic regression check for the command-line interface.

---

## 3. What Was Achieved

### 3.1 End-to-end orchestration

A top-level `run_engine_end_to_end()` function now executes the full workflow in a defined sequence. The function returns a structured result dictionary containing run metadata, scenario data, raw data, IFRS 9 outputs, regulatory capital results, Monte Carlo outputs, economic capital, coverage analysis, benchmarks, and uncertainty bands.

This is a substantial improvement over a collection of disconnected scripts. It gives the project one controlling abstraction and makes the dashboard a consumer of a stable result contract.

The command layer also supports explicit modes, including a pipeline mode for batch execution and a dashboard mode for launching Streamlit. This creates a clearer operational path for both development and demonstration.

### 3.2 South African portfolio and scenario representation

The synthetic portfolio is split across seven segments:

- Retail mortgage
- Retail vehicle finance
- Retail credit card
- Retail overdraft
- SME and corporate lending
- Large corporate lending
- Sovereign and bank exposures

The data generator includes account-level fields such as principal outstanding, undrawn limits, collateral, loan-to-value, days past due, debt review, judgement, administration order, province, loadshedding vulnerability, internal rating, probability of default assumptions, LGD assumptions, correlation, and credit conversion factor.

The scenario configuration includes Base, Adverse, and Severe cases, with shocks to GDP, inflation, repo rates, loadshedding, commodities, sovereign CDS, rand volatility, and unemployment. Additional idiosyncratic switches represent sovereign downgrade, commodity collapse, cyber incident, housing crash, and SME failure wave.

This gives the prototype a locally meaningful narrative rather than presenting a generic global credit model.

### 3.3 IFRS 9 modelling flow

The IFRS 9 component follows a recognizable sequence:

1. Convert through-the-cycle PD assumptions into point-in-time PDs.
2. Adjust LGD for macro and market conditions.
3. Calculate EAD using drawn and undrawn exposure with CCF assumptions.
4. Assign IFRS 9 stages.
5. Apply 12-month ECL for performing and SICR exposures and lifetime ECL for credit-impaired exposures.

Staging includes 30-day and 90-day delinquency backstops, debt review, judgement, administration orders, rating-based triggers, quantitative PD deterioration tests, and an SME loadshedding cash-flow indicator.

The staging output records both stage and reason, which is valuable for explainability and audit discussion even though the calibration is not sufficient for a production impairment model.

### 3.4 Regulatory capital and RWA representation

The project includes a risk-weighted asset calculation and a capital stack with CET1, Tier 1, total capital, conservation buffer, countercyclical buffer, D-SIB, HLA, and Pillar 2 components. It also calculates capital ratios, available capital, buffer utilisation, leverage, surplus, and payout restriction indicators.

This allows the user to see the difference between credit loss and capital adequacy. That distinction is important: an increase in ECL is not identical to a breach of a capital ratio, and a portfolio can experience material losses while remaining above minimum regulatory thresholds.

### 3.5 Economic capital and coverage analysis

The economic capital layer uses:

- A Gaussian or t-copula credit default simulation
- A correlated market-risk simulation with an expected-shortfall-style measure
- An operational risk scenario simulation using frequency and Pareto severity
- A benchmark allocation across risk types
- Available Financial Resources compared with total economic capital
- A stress erosion path under increasing shock intensity

The coverage output includes a main ratio, a surplus or shortfall, a decomposition of coverage drivers, and an erosion path. This makes the model more decision-oriented than a simple loss calculator.

### 3.6 Dashboard and presentation layer

The Streamlit dashboard contains seven conceptual tabs:

1. Shock propagation
2. IFRS 9 and staging
3. Regulatory capital and RWA
4. Economic capital and Monte Carlo
5. Coverage and erosion
6. South African bank benchmarks
7. Uncertainty and model risk

The original dashboard used several complex Plotly visualisations. Those visuals created rendering instability across tabs. They were replaced with simpler Streamlit bar, line, histogram, and table components, with per-section exception handling and fallback warnings. This is a practical improvement for reliability and presentation.

### 3.7 Deployment work

The project was:

- Committed to Git
- Pushed to the public repository `EdwardM1276/risk-engine`
- Configured with `streamlit_app.py` as the Streamlit Cloud entry point
- Given a `.streamlit/config.toml` configuration
- Deployed to Streamlit Community Cloud

The first deployment exposed a dependency problem. Streamlit Cloud attempted to use Python 3.14 with older pinned packages, and the build failed while trying to build Pillow. The dependency specification was then modernised and pushed as commit `70c0481`.

The deployment process itself was useful because it tested the repository as an external consumer would see it, rather than relying only on a local virtual environment.

---

## 4. How the Project Was Achieved

### 4.1 Centralised parameters

A deliberate design choice was to place model coefficients and regulatory assumptions in `config/params.py`. This improves traceability and reduces the risk of scattered magic numbers. It also makes scenario and calibration review easier because a reviewer has one main location to inspect.

The limitation is that centralisation improves organisation but does not prove that the values are correct, current, or appropriate for a particular institution. Parameter governance still requires source documentation, ownership, approval, versioning, and validation evidence.

### 4.2 Deterministic execution

Most generators and simulations accept explicit seeds. This makes local testing and demonstrations reproducible. The orchestrator also records run metadata, including account count, simulation count, copula type, duration, scenario, and idiosyncratic shock hash.

Reproducibility is an important engineering foundation. It does not eliminate model uncertainty, but it makes differences between runs explainable.

### 4.3 Defensive data contracts

A key runtime issue occurred because `acquire_all_data()` returned a `RawDataset` dataclass while downstream code accessed it with dictionary syntax. The solution was to add mapping-compatible methods such as `__getitem__`, `get`, `keys`, `items`, `values`, and `__iter__`.

This fixed the immediate TypeError while preserving the typed container. It was a good local repair because it addressed compatibility at the data boundary without forcing a broad rewrite of the engine.

### 4.4 Simplification of the visual layer

The dashboard originally relied heavily on Plotly subplots, pie charts, waterfall charts, sunbursts, dual axes, polar charts, and annotated traces. These are visually expressive but increase the number of assumptions about data types, column names, empty frames, and browser rendering.

The revised dashboard returns simple DataFrames from helper functions and uses built-in Streamlit charts where possible. Each tab isolates rendering failures with `try`/`except` blocks so one incompatible table or chart does not take down the full page.

This is not a substitute for fixing every data-quality issue. It is a sensible reliability choice for an exploratory application.

### 4.5 Basic regression testing

The project includes tests confirming that the orchestrator parser exists, exposes pipeline mode, and accepts dashboard help. The pipeline command was also executed successfully during development.

The testing approach remains minimal. The tests verify the command surface, but they do not validate numerical correctness, scenario monotonicity, stage allocation, capital calculations, Monte Carlo calibration, or the complete dashboard interaction path.

---

## 5. Significance of the Project

The project is significant in four ways.

### 5.1 It demonstrates risk integration

Many risk prototypes calculate ECL, RWA, or economic capital in isolation. This project demonstrates the more important organisational question: how do these measures interact when a scenario changes? The propagation view encourages users to think about loss, capital, and coverage as connected outcomes.

### 5.2 It makes local risk drivers concrete

The inclusion of loadshedding, commodity exposure, sovereign CDS, rand volatility, provincial concentration, and local bank benchmarks makes the project more relevant to a South African context than a generic portfolio simulator.

### 5.3 It exposes model and data dependencies

Because the engine moves through several layers, it makes dependencies visible. For example, scenario severity affects PD, staging, ECL, RWA, simulations, economic capital, and coverage. This helps explain why a risk number is not merely an output but the result of a chain of assumptions.

### 5.4 It provides a foundation for future industrialisation

The modular structure creates a reasonable starting point for replacing synthetic data with governed data, replacing illustrative parameters with validated calibration, and adding proper model risk controls. The project is therefore more valuable as a scaffold than as a final decision engine.

---

## 6. Honest Industry Usefulness Evaluation

### 6.1 Where it is useful today

The project has real value in the following settings:

- **Training and education:** It can demonstrate IFRS 9 staging, ECL, regulatory capital, stress testing, Monte Carlo simulation, and economic capital in one application.
- **Prototype design:** It can help a team test how a future risk platform might be divided into acquisition, modelling, orchestration, reporting, and governance layers.
- **Workshop facilitation:** Risk, finance, treasury, and senior management can use it to discuss scenario transmission and coverage without exposing confidential bank data.
- **Requirements discovery:** The dashboard can help identify which controls, outputs, and drill-downs users would expect before a production build.
- **Model comparison experiments:** Researchers can compare scenario assumptions, portfolio mixes, and simulation settings under controlled synthetic conditions.
- **Demonstration and portfolio evidence:** It shows practical software integration, domain modelling, data structures, dashboard development, testing, and deployment.

### 6.2 Where it is not yet suitable

It should not currently be used as the source of:

- Regulatory capital submissions
- IFRS 9 financial statement provisions
- Board-approved risk appetite limits
- Credit pricing or underwriting decisions
- Customer-level adverse decisions
- Capital planning or ICAAP conclusions without independent validation
- Formal stress-test submissions
- External investor or regulator reporting

The central reason is not that the architecture is wrong. The problem is evidential strength. A production banking model must demonstrate that its data, definitions, calibration, implementation, monitoring, limitations, and governance are fit for a specific institution and purpose.

### 6.3 Potential industry value after development

With substantial additional work, the project could evolve into:

- A scenario analysis workbench for portfolio managers
- A challenger model for existing ECL or capital models
- A capital planning and stress-testing prototype
- A model-risk benchmarking tool
- A controlled management information dashboard
- A research platform for South African macro-financial transmission

The highest-value path is likely a governed internal challenger or scenario workbench, not immediate replacement of established bank models.

### 6.4 Overall assessment

**Current maturity:** early-to-mid prototype.  
**Educational value:** high.  
**Architecture demonstration value:** high.  
**Decision-support value with explicit caveats:** moderate.  
**Production banking readiness:** low.  
**Regulatory readiness:** very low without major validation and governance work.

That assessment is not a criticism of the project’s ambition. It is the realistic distinction between a coherent prototype and a model that can support audited financial or regulatory decisions.

---

## 7. Technical and Methodological Limitations

### 7.1 Synthetic data dominates the outputs

The portfolio, macro series, market series, and loadshedding history are generated synthetically. The generator is designed to look plausible, but plausible is not the same as representative of an actual bank’s portfolio, observed default history, recovery behaviour, product definitions, or risk concentrations.

Results should therefore be read as scenario illustrations, not forecasts of a real institution.

### 7.2 Calibration is illustrative

The project references South African regulatory and bank benchmark concepts, but the numerical values are not a substitute for official regulatory interpretation, internal bank calibration, audited Pillar 3 disclosures, or approved accounting methodology. A production implementation would need documented sources, effective dates, mapping logic, and review of every material assumption.

### 7.3 Monte Carlo stability is limited

The dashboard allows relatively small simulation counts. Tail estimates at 99.9% confidence are inherently unstable with small samples. The operational-risk implementation uses a low scenario count in its default configuration, making the extreme quantile heavily dependent on a small number of observations.

The output should include confidence intervals, convergence checks, effective sample information, and warnings when the simulation count is inadequate for the requested confidence level.

### 7.4 Copula implementation requires review

The Gaussian and t-copula paths do not appear to use exactly the same threshold construction. In particular, the t-copula branch transforms the latent variable through a normal CDF and compares it directly with the PD, while the Gaussian path uses a normal inverse-PD threshold. That difference may be intentional, but it should be explicitly justified and tested because it can change default probabilities and tail behaviour.

A model validation review should examine the factor distribution, dependence matrix, positive-definiteness treatment, marginal calibration, default threshold construction, and sensitivity to degrees of freedom.

### 7.5 Regulatory concepts are simplified

The project uses labels such as Basel III, Regulation 38, IFRS 9, FRTB-style, AMA-style, and SARB-aligned. These labels communicate the intended framework, but the implementation is a simplified analytical approximation. It does not constitute a complete implementation of all relevant regulatory rules, reporting templates, eligibility criteria, transitional provisions, capital deductions, exposure classes, or supervisory expectations.

### 7.6 Uncertainty bands are partly heuristic

Some uncertainty intervals are specified as fixed percentage ranges rather than being estimated from a full calibration, parameter posterior, resampling procedure, or scenario distribution. They are useful for communicating uncertainty but should not be described as statistically complete confidence intervals without further methodological work.

### 7.7 Testing is narrow

The current test verifies the CLI surface. It does not provide comprehensive model validation. Important gaps include:

- Unit tests for each model component
- Golden datasets with expected numerical outputs
- Property tests for non-negative EAD, ECL, RWA, and capital
- Scenario monotonicity tests
- Reproducibility tests across seeds
- Monte Carlo convergence tests
- Data-quality and schema tests
- Dashboard tab interaction tests
- Deployment smoke tests
- Performance tests at production portfolio sizes

### 7.8 Dashboard resilience can hide defects

The use of broad exception handling prevents a single visual from crashing the dashboard, which improves user experience. However, broad catches can also hide genuine data or model defects. Production code should log the exception, expose a diagnostic identifier, and distinguish between an expected empty state and an unexpected implementation error.

### 7.9 Persistence is not production-grade

The audit summary is written to a CSV file as a best-effort operation. This is adequate for a local prototype but not for controlled audit trails. A production system would need immutable run records, user identity, parameter versioning, source-data snapshots, approvals, access control, retention, and monitoring.

### 7.10 Streamlit Cloud is suitable for demonstration, not necessarily bank deployment

Streamlit Cloud provides a convenient public demonstration environment. It is not automatically appropriate for confidential bank data, regulated workloads, enterprise authentication, controlled secrets, high availability, or internal model governance. Any industrial deployment would require a security and operating-model review.

---

## 8. Challenges Faced and Resolutions

### Challenge 1: No clear project-level launch layer

**Problem:** The project had multiple modules and a dashboard, but no single obvious command for the whole system.  
**Resolution:** A top-level CLI and orchestrator were established in `run.py`, with pipeline and dashboard modes.

### Challenge 2: Dataclass and dictionary contract mismatch

**Problem:** The data acquisition function returned `RawDataset`, while the orchestrator used dictionary indexing, causing a `TypeError`.  
**Resolution:** `RawDataset` gained mapping-compatible access methods, preserving the dataclass while supporting existing consumers.

### Challenge 3: Fragile dashboard charts

**Problem:** Complex Plotly visuals crashed on certain tabs when data shapes or types did not match chart expectations.  
**Resolution:** Chart helpers were changed to return simpler DataFrames, built-in Streamlit charts were used, and tab sections gained graceful fallback handling.

### Challenge 4: Presentation quality and emoji cleanup

**Problem:** The dashboard contained presentation noise and emoji characters that were inconsistent with the desired professional style.  
**Resolution:** Dashboard text and styling were cleaned up, with a more restrained executive layout.

### Challenge 5: Streamlit dependency deployment failure

**Problem:** Streamlit Cloud selected Python 3.14 and failed while building an older Pillow dependency pulled by the original package pins.  
**Resolution:** The dependency specification was modernised to compatible version ranges, committed, and pushed. A runtime pin was also added during the troubleshooting process.

### Challenge 6: Browser and shell environment differences

**Problem:** Local PowerShell command execution was complicated by a Python-backed terminal session and Windows path escaping.  
**Resolution:** Repository and deployment actions were performed through the active Python process and browser tools, with the repository and deployment state independently verified.

---

## 9. Recommended Development Roadmap

### Phase 1: Make the prototype technically dependable

- Add component-level unit tests.
- Add a small golden dataset with expected outputs.
- Add schema validation at every module boundary.
- Replace broad exception catches with structured logging.
- Add explicit empty-data and invalid-parameter handling.
- Add scenario monotonicity and reproducibility tests.
- Record model version and parameter version in every run.

### Phase 2: Improve modelling credibility

- Replace synthetic macro and portfolio data with governed historical datasets.
- Calibrate PD, LGD, CCF, cure, prepayment, and recovery assumptions to observed experience.
- Validate staging triggers against actual account histories.
- Rework copula threshold and marginal calibration consistently.
- Increase tail-simulation counts and add convergence diagnostics.
- Separate expected loss, unexpected loss, stress loss, and capital definitions clearly.
- Validate benchmark values against current official disclosures and effective regulatory dates.

### Phase 3: Add risk governance

- Produce model documentation and a formal limitations register.
- Add independent validation and challenger testing.
- Define change-control, approval, and parameter ownership processes.
- Add data lineage and source-data quality reporting.
- Implement controlled audit storage rather than CSV-only persistence.
- Add access control and protect confidential configuration and portfolio data.
- Establish monitoring for drift, overrides, failures, and output anomalies.

### Phase 4: Industrialise the user experience

- Separate the computational engine from the dashboard service.
- Add asynchronous job execution for large portfolios and simulations.
- Add downloadable run packs with parameters, results, charts, and validation checks.
- Provide scenario comparison and run-to-run diffing.
- Add role-specific views for risk, finance, treasury, and executives.
- Consider an internal deployment platform with enterprise identity and logging.

---

## 10. Conclusion

The South African Credit Risk Volatility Engine successfully demonstrates an integrated risk analytics workflow from scenario definition through ECL, RWA, regulatory capital, economic capital, and coverage. The strongest achievement is not any individual formula; it is the creation of a coherent, runnable, inspectable system with a clear orchestration layer and a usable dashboard.

The project has meaningful educational, prototyping, and stakeholder-communication value. It can help users understand how South African macroeconomic shocks may be transmitted through a banking portfolio and how different risk measures relate to each other.

Its current limitations are equally important. The outputs are generated from synthetic data and illustrative assumptions; the tests are narrow; the tail simulations require stronger stability evidence; regulatory and accounting concepts are simplified; and the audit and deployment controls are not yet suitable for a production bank environment.

The honest conclusion is that the project is a strong foundation and a credible prototype, but not yet an industry-grade risk decision engine. Its greatest potential value is as a transparent scenario workbench or challenger-model foundation that can be progressively strengthened with real data, validated calibration, formal model governance, robust testing, and controlled deployment.

---

## 11. Review-Driven Improvements Implemented

Following the independent critical review, the project was reworked in several high-priority areas:

- The credit simulation now uses consistent one-factor loadings in both Gaussian and t-copula modes.
- The t-copula now applies a shared chi-square scale to the latent vector and compares it with t-distribution PD thresholds. It no longer applies a normal CDF to t-distributed values.
- Invalid copula names, non-positive simulation counts, and unstable t degrees of freedom now raise explicit validation errors.
- Credit simulation results expose the number of empirical observations in the 99.9% tail and warn when that tail is too small to support a stable estimate.
- RWA output now states that credit risk is IRB-style Vasicek ASRF, while market and operational measures are stress-sensitive approximations rather than full regulatory implementations.
- Economic capital allocation now aggregates supplied simulated credit, market, and operational components directly. Published benchmark percentages are retained as fallbacks for components that are not simulated, and the allocation method is disclosed in the result.
- Regression tests now cover reproducibility, non-negative simulated losses, invalid simulation parameters, tail-sample warnings, and RWA methodology metadata.

These changes improve conceptual transparency and reduce material implementation risk, but they do not remove the need for real-data calibration, independent validation, stronger tail methods, and formal model governance.

---

## Appendix A: Repository Deliverables

- `run.py`: end-to-end orchestrator and CLI
- `dashboard/app.py`: Streamlit dashboard
- `streamlit_app.py`: Streamlit Cloud entry point
- `.streamlit/config.toml`: Streamlit configuration
- `requirements.txt`: Python dependencies
- `runtime.txt`: Python runtime declaration
- `config/params.py`: central assumptions and benchmarks
- `data/acquisition.py`: synthetic data acquisition and portfolio generation
- `ifrs9/`: ECL and staging modules
- `regcap/`: RWA and regulatory capital modules
- `ecap/`: economic capital and coverage modules
- `scenarios/`: stress scenario modules
- `tests/test_orchestrator.py`: CLI regression tests

## Appendix B: Evidence of Completion

- Local pipeline command executed successfully during development.
- CLI regression tests passed during development.
- Dashboard launched locally through the project command layer.
- GitHub repository created and populated.
- Streamlit Cloud app created from `main` and `streamlit_app.py`.
- Deployment dependency issue identified from live logs and addressed in commit `70c0481`.
