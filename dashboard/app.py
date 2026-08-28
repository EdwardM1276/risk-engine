import streamlit as st
import pandas as pd
import plotly.express as px
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run import run_engine_end_to_end
from data.acquisition import fetch_uci_credit_card_defaults, summarize_uci_default_benchmark
from config.params import (
    SARB_DIRECTIVES, NEDBANK_ECAP_BENCHMARK_2024, PORTFOLIO_SEGMENTS,
    BANK_BENCHMARK_PROFILES, DEFAULT_BANK_PROFILE, D_SIB_BUFFER_TIERS,
)

st.set_page_config(
    page_title="SA Credit Risk Volatility Engine",
    page_icon="bank",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLOR_SCHEME = {
    "Stage 1": "#2E8B57",
    "Stage 2": "#F4A460",
    "Stage 3": "#CD5C5C",
    "IFRS 9 ECL": "#EF553B",
    "Basel III RegCap": "#636EFA",
    "Economic Capital": "#00CC96",
    "CET1": "#1f77b4",
    "AT1": "#aec7e8",
    "T2": "#ff7f0e",
    "CCB": "#98df8a",
    "CCyB": "#2ca02c",
    "D-SIB": "#9467bd",
    "HLA": "#c5b0d5",
    "Pillar 2": "#ff9896",
}

FORMAT_ZAR = lambda x: f"R{x:,.0f}" if x < 1e9 else f"R{x/1e9:,.2f}bn"
FORMAT_PCT = lambda x: f"{x*100:.2f}%"


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


@st.cache_data(ttl=86400, show_spinner=False)
def load_public_credit_benchmark():
    """Cache the observed UCI benchmark for the current dashboard session."""
    return summarize_uci_default_benchmark(fetch_uci_credit_card_defaults())


def plot_stage_distribution(ifrs9):
    stage_summary = pd.DataFrame({
        "Stage": ["Stage 1", "Stage 2", "Stage 3"],
        "Accounts": [
            _safe_float(ifrs9.get("stage_distribution", {}).get(1, 0)),
            _safe_float(ifrs9.get("stage_distribution", {}).get(2, 0)),
            _safe_float(ifrs9.get("stage_distribution", {}).get(3, 0)),
        ],
        "ECL_bn": [
            _safe_float(ifrs9.get("ecl_by_stage", {}).get(1, 0)) / 1e9,
            _safe_float(ifrs9.get("ecl_by_stage", {}).get(2, 0)) / 1e9,
            _safe_float(ifrs9.get("ecl_by_stage", {}).get(3, 0)) / 1e9,
        ],
    })
    return stage_summary


def plot_capital_waterfall(regcap):
    pct_map = regcap.get("capital_stack", {}).get("percentages", {})
    rows = [
        ("CET1", _safe_float(pct_map.get("CET1 Min (Pillar 1)", 0.0), 0.0)),
        ("Tier 1 Add-on", _safe_float(pct_map.get("Tier1 Min (Pillar 1)", 0.0), 0.0) - _safe_float(pct_map.get("CET1 Min (Pillar 1)", 0.0), 0.0)),
        ("Total Capital Add-on", _safe_float(pct_map.get("Total Cap Min (Pillar 1)", 0.0), 0.0) - _safe_float(pct_map.get("Tier1 Min (Pillar 1)", 0.0), 0.0)),
        ("CCB", _safe_float(pct_map.get("CCB (Capital Conservation)", 0.0), 0.0)),
        ("CCyB", _safe_float(pct_map.get("CCyB (Countercyclical 2026)", 0.0), 0.0)),
        ("D-SIB", _safe_float(pct_map.get("D-SIB Buffer", 0.0), 0.0)),
        ("Pillar 2", _safe_float(pct_map.get("Pillar 2 Add-on", 0.0), 0.0)),
    ]
    return pd.DataFrame(rows, columns=["Component", "Share_of_RWA"]).assign(Share_of_RWA=lambda d: d["Share_of_RWA"] * 100)


def plot_ecap_allocation(ecap):
    components = ecap.get("components", {})
    df = pd.DataFrame({
        "Component": list(components.keys()),
        "Value": [float(v) for v in components.values()],
    })
    return df.sort_values("Value", ascending=False)


def plot_loss_distribution(credit_mc, ecl_total):
    losses = pd.Series(credit_mc.get("simulated_losses", []), dtype=float) / 1e9
    loss_df = pd.DataFrame({"Portfolio Loss (R bn)": losses})
    return loss_df


def plot_propagation_path(result, result_severe=None):
    rows = []
    for name, r in [("Base", result)] + ([("Severe", result_severe)] if result_severe else []):
        try:
            rows.extend([
                {"Scenario": name, "Step": "EAD", "Value": _safe_float(r.get("ifrs9", {}).get("ead_total", 0.0)) / 1e9},
                {"Scenario": name, "Step": "IFRS 9 ECL", "Value": _safe_float(r.get("ifrs9", {}).get("ecl_total", 0.0)) / 1e9},
                {"Scenario": name, "Step": "RWA", "Value": _safe_float(r.get("regcap", {}).get("total_rwa", 0.0)) / 1e9},
                {"Scenario": name, "Step": "ECap", "Value": _safe_float(r.get("economic_capital", {}).get("total_ecap", 0.0)) / 1e9},
                {"Scenario": name, "Step": "AFR", "Value": _safe_float(r.get("coverage", {}).get("main", {}).get("AFR", 0.0)) / 1e9},
            ])
        except Exception:
            continue
    return pd.DataFrame(rows)


def plot_coverage_erosion(coverage):
    erosion_path = coverage.get("erosion_path")
    if erosion_path is None or erosion_path.empty:
        return pd.DataFrame({"Shock Intensity (x)": [1.0], "Coverage Ratio": [1.0], "ECap Expansion %": [0.0], "AFR Erosion %": [0.0]})
    df = erosion_path.copy()
    for col in ["Shock Intensity (x)", "Coverage Ratio", "ECap Expansion %", "AFR Erosion %"]:
        if col not in df.columns:
            df[col] = 0.0
    return df[["Shock Intensity (x)", "Coverage Ratio", "ECap Expansion %", "AFR Erosion %"]]


def plot_uncertainty_bands(uncertainty):
    rows = []
    for metric, band in uncertainty.items():
        if "CAR" in metric or "Coverage" in metric:
            continue
        rows.append({
            "Metric": metric,
            "Central (R bn)": _safe_float(band.get("central", 0.0)) / 1e9,
            "Lower 95% (R bn)": _safe_float(band.get("lower_95", 0.0)) / 1e9,
            "Upper 95% (R bn)": _safe_float(band.get("upper_95", 0.0)) / 1e9,
            "Model Risk (R bn)": _safe_float(band.get("model_risk", 0.0)) / 1e9,
        })
    return pd.DataFrame(rows)


def plot_benchmark_radar(bench_df, regcap_result):
    if bench_df is None or bench_df.empty:
        return pd.DataFrame({"Metric": ["Total CAR", "CET1 Ratio"], "Value": [0.0, 0.0]})
    return bench_df.copy()


def main():
    st.markdown(
        """
        <style>
        :root {
            --page-bg: #f3f6fb;
            --panel-bg: rgba(255,255,255,0.88);
            --panel-border: #dfe7f5;
            --ink: #10233b;
            --muted: #475569;
            --primary: #1d4ed8;
            --primary-soft: #dbeafe;
            --success: #0f766e;
            --shadow: 0 10px 24px rgba(15, 23, 42, 0.08);
        }

        .stApp {
            background: linear-gradient(180deg, #f8fafc 0%, #eef4ff 100%);
        }
        .block-container {
            padding-top: 0.8rem !important;
            padding-bottom: 2rem !important;
            max-width: 1500px !important;
        }
        h1 {
            color: var(--ink);
            font-size: 2.2rem !important;
            margin-bottom: 0.15rem !important;
            letter-spacing: -0.04em;
        }
        h2 {
            color: #1f3b66;
            border-bottom: 1px solid #dbe3f0;
            padding-bottom: 0.35rem;
            margin-top: 1.2rem !important;
            margin-bottom: 0.8rem !important;
            font-size: 1.15rem !important;
            letter-spacing: 0.01em;
        }
        .stTabs [role="tablist"] {
            gap: 0.45rem;
            border-bottom: 1px solid #dbe3f0;
            margin-bottom: 0.8rem;
        }
        .stTabs [role="tab"] {
            border-radius: 0.75rem 0.75rem 0 0;
            background: #edf2fb;
            color: var(--muted);
            padding: 0.55rem 0.9rem;
            font-weight: 600;
            border: 1px solid transparent;
        }
        .stTabs [role="tab"][aria-selected="true"] {
            background: linear-gradient(180deg, var(--primary) 0%, #1e40af 100%);
            color: white;
            border-color: rgba(29,78,216,0.25);
            box-shadow: 0 6px 18px rgba(29,78,216,0.18);
        }
        div[data-testid="stMetric"] {
            background: rgba(255,255,255,0.9);
            border: 1px solid var(--panel-border);
            border-left: 4px solid var(--primary);
            border-radius: 0.8rem;
            padding: 0.7rem 0.9rem 0.5rem 0.9rem;
            box-shadow: var(--shadow);
            min-height: 110px;
        }
        div[data-testid="stMetric"] > label {
            color: var(--muted);
            font-weight: 600;
            font-size: 0.78rem !important;
        }
        div[data-testid="stMetric"] > div {
            font-size: 1.25rem !important;
            line-height: 1.2;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
            border-right: 1px solid rgba(148,163,184,0.18);
        }
        [data-testid="stSidebar"] .stMarkdown,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] .stCheckbox label,
        [data-testid="stSidebar"] .stCheckbox label p,
        [data-testid="stSidebar"] .stCheckbox label span,
        [data-testid="stSidebar"] .block-container {
            color: #ffffff !important;
        }
        [data-testid="stSidebar"] .stSelectbox > div,
        [data-testid="stSidebar"] .stNumberInput > div,
        [data-testid="stSidebar"] .stSlider > div,
        [data-testid="stSidebar"] .stCheckbox {
            background: rgba(15, 23, 42, 0.42);
            border-radius: 0.55rem;
        }
        .stDataFrame, .stTable {
            border-radius: 0.75rem;
            overflow: hidden;
            box-shadow: var(--shadow);
        }
        .stAlert {
            border-radius: 0.8rem;
        }
        .presentation-header {
            background: linear-gradient(135deg, rgba(29,78,216,0.12), rgba(59,130,246,0.04));
            border: 1px solid rgba(29,78,216,0.1);
            border-radius: 0.9rem;
            padding: 0.8rem 1rem;
            margin-bottom: 0.8rem;
        }
        </style>
        """, unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="presentation-header">
            <div style="font-size: 0.78rem; letter-spacing: 0.08em; text-transform: uppercase; color: #2563eb; font-weight: 700; margin-bottom: 0.2rem;">Executive risk dashboard</div>
            <div style="font-size: 2.1rem; font-weight: 700; color: #10233b; line-height: 1.1;">South African Credit Risk Volatility Engine</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("IFRS 9 ECL, Basel III RegCap, and Economic Capital aligned with SARB Directives 5/2017, 6/2024, and Regulation 38/43")

    with st.sidebar:
        st.header("Engine Controls")
        data_source = st.selectbox("Data Source", ["synthetic", "public"], index=1,
                       help="Public mode uses live World Bank, SARB, FRED, and FDIC data")
        scenario = st.selectbox("Scenario (SARB/IMF Standard)", ["Base", "Adverse", "Severe"], index=0)
        severity = st.slider("Severity Multiplier", 0.5, 2.5, 1.0, 0.1,
                              help="Interpolate between standard scenario bounds")
        institution_size = st.selectbox("Institution Classification",
                                         ["Large_D-SIB", "Large", "Medium", "Small_Mutual"], index=0)
        profile_names = list(BANK_BENCHMARK_PROFILES.keys())
        bank_profile = st.selectbox("Benchmark / Calibration Profile", profile_names,
                                     index=profile_names.index(DEFAULT_BANK_PROFILE),
                                     help="Calibration targets (RWA density, ECL/EAD, capital ratios)")
        profile = BANK_BENCHMARK_PROFILES[bank_profile]
        d_sib_bucket = st.selectbox("D-SIB Bucket (HLA tier)", [0, 1, 2, 3, 4, 5],
                                     index=int(profile.get("d_sib_bucket", 3)),
                                     format_func=lambda b: f"Bucket {b} ({D_SIB_BUFFER_TIERS.get(b, 0.0)*100:.1f}% CET1)" if b else "Bucket 0 (not a D-SIB)")
        total_exposure = st.number_input("Total Portfolio Exposure (R bn)", 100, 5000,
                                          int(profile["total_exposure_bn"]), 50) * 1e9
        n_accounts = st.slider("Portfolio Records", 500, 5000, 2000, 500)
        n_mc_sims = st.slider("Monte Carlo Simulations", 500, 5000, 1500, 250,
                               help="Correlated default copula simulations")
        copula_type = st.selectbox("Copula Family", ["t", "Gaussian"], index=0,
                                    help="t-copula captures tail dependence (SA systemic clustering)")

        st.subheader("Idiosyncratic Shocks")
        sov_shock = st.checkbox("Sovereign Downgrade", value=False, key="shock_sovereign")
        comm_shock = st.checkbox("Commodity Collapse", value=False, key="shock_commodity")
        cyber_shock = st.checkbox("Cyber / Operational Catastrophe", value=False, key="shock_cyber")
        housing_shock = st.checkbox("Housing Market Crash", value=False, key="shock_housing")
        sme_shock = st.checkbox("SME Failure Wave", value=False, key="shock_sme")

        seed = st.number_input("Random Seed", 0, 9999, 2024, 1)
        st.divider()
        with st.expander("Regulatory References"):
            for k, v in SARB_DIRECTIVES.items():
                st.markdown(f"- **{k}**: {v}")

    idio_shocks = {
        "sovereign_downgrade": sov_shock,
        "commodity_collapse": comm_shock,
        "cyber_incident": cyber_shock,
        "housing_crash": housing_shock,
        "smes_failure_wave": sme_shock,
    }
    any_idio = any(idio_shocks.values())
    run_label = f"{scenario}" + (" + Idiosyncratic" if any_idio else "") + f" (Severity {severity}x)"

    with st.spinner(f"Running Risk Engine: {run_label} - {n_accounts:,} accounts, {n_mc_sims:,} sims..."):
        result = run_engine_end_to_end(
            scenario=scenario,
            total_exposure=total_exposure,
            n_accounts=n_accounts,
            seed=seed,
            institution_size=institution_size,
            severity_multiplier=severity,
            idiosyncratic_shocks=idio_shocks if any_idio else None,
            n_mc_sims=n_mc_sims,
            copula_type=copula_type,
            data_source=data_source,
            bank_profile=bank_profile,
            d_sib_bucket=d_sib_bucket,
        )

    ifrs9 = result["ifrs9"]
    regcap = result["regcap"]
    ecap = result["economic_capital"]
    coverage = result["coverage"]["main"]
    ratios = regcap["capital_ratios"]
    bench_df = result["benchmark_comparison"]
    output_floor = result["output_floor"]
    rwa_density = result["rwa_density"]
    ecl_ead = ifrs9["ecl_ead_ratio"]

    st.success(f"Engine completed in {result['run_metadata']['duration_seconds']:.2f}s - Scenario: **{run_label}**")
    quality = result["raw_data"]["data_quality"]
    if quality.get("status") != "PASSED":
        st.warning(
            f"Data quality status: {quality.get('status', 'REVIEW')}. "
            "This run is research-use only; inspect the evidence panel before interpreting results."
        )
    with st.expander("Evidence and Data Provenance"):
        st.json({
            "data_source": quality.get("data_source"),
            "validation_ready": quality.get("validation_ready", False),
            "synthetic_fallback_used": quality.get("synthetic_fallback_used", False),
            "gap_flags": quality.get("gap_flags", []),
            "placeholder_fields": quality.get("placeholder_fields", []),
            "source_notes": quality.get("source_notes", {}),
        })

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        ead_target = result["run_metadata"]["target_total_exposure"]
        st.metric("Total Exposure (EAD)", FORMAT_ZAR(ifrs9["ead_total"]),
                   delta=f"Target: {FORMAT_ZAR(ead_target)}")
    with col2:
        st.metric("IFRS 9 ECL", FORMAT_ZAR(ifrs9["ecl_total"]),
                   delta=f"{ecl_ead*100:.2f}% EAD (bench {profile['ecl_ead_target']*100:.1f}%)",
                   delta_color="inverse")
    with col3:
        st.metric("Total RWA", FORMAT_ZAR(regcap["total_rwa"]),
                   delta=f"{rwa_density*100:.1f}% density (bench {profile['rwa_density']*100:.0f}%)")
    with col4:
        st.metric("CET1 Ratio", FORMAT_PCT(ratios["CET1 Ratio"]),
                   delta=f"Bench: {profile['cet1']*100:.1f}%",
                   delta_color="normal" if ratios["CET1 Ratio"] >= 0.11 else "inverse")
    with col5:
        st.metric("Total ECap Required", FORMAT_ZAR(ecap["total_ecap"]),
                   delta=f"{ecap['total_ecap_pct_rwa']*100:.1f}% of RWA")
    with col6:
        st.metric("AFR Coverage Ratio", f"{coverage['Coverage Ratio']*100:.1f}%",
                   delta=coverage["Coverage Rating"].split(" - ")[0],
                   delta_color="inverse" if coverage["Coverage Ratio"] < 1.2 else "normal")

    st.divider()

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Shock Propagation", "IFRS 9 & Staging", "RegCap & RWA",
        "Economic Capital & MC", "Coverage & Erosion",
        "Benchmarks vs SA Banks", "Uncertainty & Model Risk"
    ])

    with tab1:
        st.subheader("Scenario Shock Drivers")
        try:
            st.dataframe(result["scenario"]["shock_summary"], use_container_width=True, hide_index=True)
        except Exception:
            st.warning("Scenario detail was unavailable, but the engine still completed successfully.")

        st.subheader("Propagation Through Capital Stack")
        result_severe = None
        if scenario != "Severe":
            with st.spinner("Running severe comparison..."):
                try:
                    result_severe = run_engine_end_to_end(
                        scenario="Severe", total_exposure=total_exposure, n_accounts=n_accounts,
                        seed=seed, institution_size=institution_size, severity_multiplier=severity,
                        idiosyncratic_shocks=idio_shocks if any_idio else None,
                        n_mc_sims=max(500, n_mc_sims // 2), copula_type=copula_type,
                        data_source=data_source,
                        bank_profile=bank_profile, d_sib_bucket=d_sib_bucket,
                    )
                except Exception:
                    result_severe = None
        try:
            propagation = plot_propagation_path(result, result_severe)
            if propagation.empty:
                st.info("No propagation data available for this scenario.")
            else:
                st.line_chart(propagation.pivot_table(index="Step", columns="Scenario", values="Value", aggfunc="last"))
        except Exception:
            st.warning("The propagation chart was simplified because the underlying dataset was not ready for plotting.")

    with tab2:
        st.subheader("IFRS 9 Stage Distribution & ECL Allocation")
        try:
            stage_df = plot_stage_distribution(ifrs9)
            st.bar_chart(stage_df.set_index("Stage")["Accounts"])
            st.bar_chart(stage_df.set_index("Stage")["ECL_bn"])
        except Exception:
            st.warning("Stage graphics were not available; the summary metrics remain visible.")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Weighted Avg PIT PD (12m)", FORMAT_PCT(ifrs9["avg_pit_pd"]))
        with c2:
            st.metric("Weighted Avg LGD", FORMAT_PCT(ifrs9["avg_lgd"]))
        with c3:
            st.metric("Lifetime / 12m ECL Ratio",
                       f"{ifrs9['ecl_lifetime_total'] / max(ifrs9['ecl_12m_total'], 1):.2f}x")

        st.subheader("Portfolio-Level ECL Detail (by Segment)")
        try:
            portfolio = pd.DataFrame(ifrs9["portfolio"])
            df_seg = portfolio.groupby("segment", as_index=False).agg(
                n_accounts=("account_id", "count"),
                EAD=("ead", "sum"),
                ECL=("ecl", "sum"),
                avg_PD=("pit_pd_12m", "mean"),
                avg_LGD=("lgd", "mean"),
            )
            df_seg["ECL Rate %"] = (df_seg["ECL"] / df_seg["EAD"] * 100).fillna(0.0)
            df_seg["EAD"] = df_seg["EAD"].map(lambda x: f"R{x/1e6:,.1f}m")
            df_seg["ECL"] = df_seg["ECL"].map(lambda x: f"R{x/1e6:,.1f}m")
            df_seg["avg_PD"] = df_seg["avg_PD"].map(lambda x: f"{x*100:.2f}%")
            df_seg["avg_LGD"] = df_seg["avg_LGD"].map(lambda x: f"{x*100:.1f}%")
            df_seg["ECL Rate %"] = df_seg["ECL Rate %"].map(lambda x: f"{x:.2f}%")
            st.dataframe(df_seg.rename(columns={"segment": "Product Segment"}), use_container_width=True, hide_index=True)
        except Exception:
            st.warning("Segment detail could not be rendered; the portfolio summary is still available above.")

    with tab3:
        st.subheader("Capital Stack")
        st.caption(
            "Prototype methodology: IRB-style Vasicek credit RWA, with FRTB-style "
            "market and SA-OR-style operational stress approximations."
        )
        try:
            capital_df = plot_capital_waterfall(regcap)
            st.bar_chart(capital_df.set_index("Component")["Share_of_RWA"])
        except Exception:
            st.warning("Capital stack chart could not be rendered.")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("CET1 Ratio", FORMAT_PCT(ratios["CET1 Ratio"]),
                       delta=f"Surplus: {FORMAT_ZAR(max(0, ratios['CET1 Surplus (vs CET1+Buffers)']))}",
                       delta_color="inverse" if ratios["CET1 Surplus (vs CET1+Buffers)"] < 0 else "normal")
        with c2:
            st.metric("Tier 1 Ratio", FORMAT_PCT(ratios["Tier1 Ratio"]),
                       delta=f"Bench: {profile['tier1']*100:.1f}%")
        with c3:
            st.metric("Total CAR", FORMAT_PCT(ratios["Total CAR"]),
                       delta=f"Bench: {profile['total_capital']*100:.1f}%")
        with c4:
            st.metric("Combined Buffer Usage", f"{ratios['Combined Buffer Utilisation %']*100:.1f}%",
                       delta=ratios["Conservation Level"].split(" - ")[0],
                       delta_color="inverse" if ratios["Combined Buffer Utilisation %"] > 1.0 else "normal")

        c1, c2, c3, c4 = st.columns(4)
        d_sib_info = regcap["capital_stack"]["d_sib"]
        with c1:
            st.metric("Leverage Ratio", FORMAT_PCT(ratios["Leverage Ratio"]),
                       delta=f"Req: {ratios['Leverage Requirement']*100:.2f}% | Bench: {profile['leverage']*100:.1f}%",
                       delta_color="inverse" if ratios["Leverage Surplus"] < 0 else "normal")
        with c2:
            st.metric("D-SIB Bucket", f"Bucket {d_sib_info['bucket']}",
                       delta=f"HLA buffer: {d_sib_info['buffer_rate']*100:.1f}% CET1")
        with c3:
            st.metric("Payout Restriction", FORMAT_PCT(ratios["Payout Distribution Restriction %"]),
                       delta="MDA trigger" if ratios["Payout Distribution Restriction %"] >= 1.0 else "Flexible")
        with c4:
            st.metric("Output Floor",
                       "Binding" if output_floor["floor_applied"] else "Not binding",
                       delta=f"Floor ({output_floor['floor_pct']*100:.1f}% of SA): {FORMAT_ZAR(output_floor['floor_rwa'])}",
                       delta_color="inverse" if output_floor["floor_applied"] else "normal")

        st.subheader("Output Floor Mechanics")
        st.caption("Final RWA = max(modelled IRB RWA, floor % x standardised RWA) - Basel III output floor")
        floor_df = pd.DataFrame([
            {"Measure": "Modelled (IRB) RWA", "R bn": output_floor["modelled_rwa"] / 1e9},
            {"Measure": "Standardised RWA", "R bn": output_floor["standardised_rwa"] / 1e9},
            {"Measure": f"Floor ({output_floor['floor_pct']*100:.1f}% of SA)", "R bn": output_floor["floor_rwa"] / 1e9},
            {"Measure": "Final RWA", "R bn": output_floor["final_rwa"] / 1e9},
        ])
        st.bar_chart(floor_df.set_index("Measure"))
        st.caption(
            f"Floor headroom: {FORMAT_ZAR(output_floor['floor_headroom'])} "
            f"({'floor is binding' if output_floor['floor_applied'] else 'modelled RWA above floor'})"
        )

        st.subheader("RWA Breakdown")
        try:
            rwa_bd = result["rwa_breakdown"]
            rwa_df = pd.DataFrame({"Risk Type": list(rwa_bd.keys()), "RWA (R bn)": [float(x) / 1e9 for x in rwa_bd.values()]})
            st.bar_chart(rwa_df.set_index("Risk Type"))
        except Exception:
            st.warning("RWA breakdown could not be plotted.")

    with tab4:
        st.subheader("ECap Allocation")
        try:
            ecap_df = plot_ecap_allocation(ecap)
            st.bar_chart(ecap_df.set_index("Component")["Value"] / 1e9)
        except Exception:
            st.warning("Economic capital breakdown chart could not be rendered.")

        st.subheader("Credit Portfolio Loss Distribution")
        try:
            loss_df = plot_loss_distribution(result["monte_carlo"]["credit"], ifrs9["ecl_total"])
            st.plotly_chart(px.histogram(loss_df, x="Portfolio Loss (R bn)", nbins=60),
                             use_container_width=True)
            credit_tail = result["monte_carlo"]["credit"]
            if credit_tail.get("tail_estimate_warning", False):
                st.warning(
                    "The 99.9% credit tail contains fewer than 100 simulated "
                    "observations; treat ECap as directional, not a stable estimate."
                )
            convergence = credit_tail.get("convergence", {})
            if convergence and not convergence.get("converged", False):
                st.warning(
                    "The 99.9% tail has not converged at the configured tolerance. "
                    "Treat the estimate as directional, not decision-grade."
                )
            elif convergence:
                st.success("The cumulative tail estimate passed the configured convergence tolerance.")
        except Exception:
            st.warning("Loss distribution could not be rendered; key summary metrics remain visible.")

        c1, c2, c3 = st.columns(3)
        credit_mc = result["monte_carlo"]["credit"]
        with c1:
            st.metric("Credit VaR (99.9%)", FORMAT_ZAR(credit_mc["VaR"][0.999]))
        with c2:
            st.metric("Credit ES (99.9%)", FORMAT_ZAR(credit_mc["Expected_Shortfall"][0.999]))
        with c3:
            st.metric("Credit ECap (UL 99.9%)", FORMAT_ZAR(credit_mc["credit_ecap_999"]),
                       delta=f"{credit_mc['credit_ecap_999_pct_ead']*100:.2f}% EAD")

        st.markdown(f"**ECL vs ECap Ratio:** {ecap['ecl_vs_ecap_ratio']:.2f}x")
        st.caption(f"ECap method: {ecap.get('allocation_method', 'Not disclosed')}")

    with tab5:
        st.subheader("AFR vs ECap Coverage")
        cov_main = result["coverage"]["main"]
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Coverage Ratio", f"{cov_main['Coverage Ratio']*100:.1f}%",
                       delta=f"vs Nedbank 2024: {cov_main['Coverage vs Benchmark %']:.1f}%")
        with c2:
            st.metric("Available Financial Resources (AFR)", FORMAT_ZAR(cov_main["AFR"]))
        with c3:
            st.metric("Total ECap Required", FORMAT_ZAR(cov_main["Total ECap Required"]))
        with c4:
            st.metric("Surplus / (Shortfall)", FORMAT_ZAR(cov_main["Surplus / (Shortfall)"]),
                       delta_color="inverse" if cov_main["Surplus / (Shortfall)"] < 0 else "normal")

        try:
            coverage_df = plot_coverage_erosion(result["coverage"])
            st.line_chart(coverage_df.set_index("Shock Intensity (x)"))
        except Exception:
            st.warning("Coverage erosion plot was simplified because the input was not available.")

        st.subheader("Coverage Decomposition")
        try:
            st.dataframe(result["coverage"]["decomposition"], use_container_width=True, hide_index=True)
        except Exception:
            st.warning("Coverage decomposition table could not be rendered.")

    with tab6:
        st.subheader("Bank Benchmark Comparison")
        st.caption(
            f"Calibration profile: **{bank_profile.replace('_', ' ')}** "
            f"(RWA density {profile['rwa_density']*100:.0f}%, CET1 {profile['cet1']*100:.1f}%, "
            f"Tier 1 {profile['tier1']*100:.1f}%, Total {profile['total_capital']*100:.1f}%, "
            f"leverage {profile['leverage']*100:.1f}%, ECL/EAD {profile['ecl_ead_target']*100:.1f}%)"
        )
        try:
            st.dataframe(bench_df, use_container_width=True, hide_index=True)
        except Exception:
            st.warning("Benchmark table could not be rendered.")
        try:
            radar_df = plot_benchmark_radar(bench_df, regcap)
            st.bar_chart(radar_df.set_index("Metric") if "Metric" in radar_df.columns else radar_df)
        except Exception:
            st.warning("Benchmark summary chart was simplified due to plotting incompatibility.")

        st.subheader("Observed Public Credit Benchmark")
        try:
            observed_benchmark = load_public_credit_benchmark()
            st.metric("UCI Observed Default Rate", f"{observed_benchmark['overall_default_rate'] * 100:.2f}%")
            st.dataframe(pd.DataFrame(observed_benchmark["default_rate_by_dpd"]), use_container_width=True, hide_index=True)
            st.caption("UCI Taiwan credit-card observations are a PD challenger benchmark, not bank-specific IFRS 9 data.")
        except Exception as exc:
            st.warning(f"Observed public benchmark unavailable: {exc}")

    with tab7:
        st.subheader("Model Uncertainty")
        try:
            u_df = plot_uncertainty_bands(result["uncertainty_bands"])
            st.bar_chart(u_df.set_index("Metric"))
            st.dataframe(u_df, use_container_width=True, hide_index=True)
        except Exception:
            st.warning("Uncertainty chart could not be rendered, but the model risk summary remains available.")

        st.subheader("Model Risk Allocation")
        model_risk = NEDBANK_ECAP_BENCHMARK_2024["model_risk_pct"] * ecap["total_ecap"]
        st.info(f"**Model Risk Reserve:** R{model_risk/1e6:,.1f}m ({NEDBANK_ECAP_BENCHMARK_2024['model_risk_pct']*100:.1f}% of total ECap) - explicitly allocated per SA best practice.")

        with st.expander("Audit Trail: Assumptions & Parameterisation"):
            st.json({
                "Scenario": result["scenario"]["parameters"],
                "Portfolio Segments": {k: {kk: (f"{vv*100:.2f}%" if "pd" in kk or "lgd" in kk or "corr" in kk or "ccf" in kk else vv)
                                              for kk, vv in v.items()} for k, v in PORTFOLIO_SEGMENTS.items()},
                "Engine Metadata": {
                    "n_accounts": result["run_metadata"]["n_accounts"],
                    "n_mc_sims": result["run_metadata"]["n_mc_sims"],
                    "copula_type": result["run_metadata"]["copula_type"],
                    "duration_seconds": round(result["run_metadata"]["duration_seconds"], 3),
                    "run_start": str(result["run_metadata"]["run_start"]),
                },
                "Calibration (audit trail)": {
                    k: (round(v, 6) if isinstance(v, float) else v)
                    for k, v in result["calibration"].items() if k != "targets"
                },
            })


if __name__ == "__main__":
    main()
