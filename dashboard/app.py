import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run import run_engine_end_to_end
from config.params import (
    SARB_DIRECTIVES, PILLAR1_MINIMA, CAPITAL_BUFFERS, SA_BANK_BENCHMARKS_2024,
    NEDBANK_ECAP_BENCHMARK_2024, PORTFOLIO_SEGMENTS,
)
from scenarios.stress_engine import (
    get_scenario_parameters, scenario_to_macro_conditions, scenario_to_market_data,
    create_idiosyncratic_scenario, compute_scenario_shock_summary,
)

st.set_page_config(
    page_title="SA Credit Risk Volatility Engine",
    page_icon="🏦",
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


def plot_stage_distribution(ifrs9):
    s1 = ifrs9["stage_distribution"].get(1, 0)
    s2 = ifrs9["stage_distribution"].get(2, 0)
    s3 = ifrs9["stage_distribution"].get(3, 0)
    total = max(s1 + s2 + s3, 1)

    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "domain"}, {"type": "xy"}]])
    fig.add_trace(go.Pie(
        labels=["Stage 1 (Performing)", "Stage 2 (SICR)", "Stage 3 (Credit Impaired)"],
        values=[s1, s2, s3],
        marker=dict(colors=[COLOR_SCHEME["Stage 1"], COLOR_SCHEME["Stage 2"], COLOR_SCHEME["Stage 3"]]),
        textinfo="label+percent",
        hole=0.4,
    ), row=1, col=1)

    s1e = ifrs9["ecl_by_stage"].get(1, 0)
    s2e = ifrs9["ecl_by_stage"].get(2, 0)
    s3e = ifrs9["ecl_by_stage"].get(3, 0)
    fig.add_trace(go.Bar(
        x=["Stage 1", "Stage 2", "Stage 3"],
        y=[s1e / 1e9, s2e / 1e9, s3e / 1e9],
        marker=dict(color=[COLOR_SCHEME["Stage 1"], COLOR_SCHEME["Stage 2"], COLOR_SCHEME["Stage 3"]]),
        text=[f"R{s1e/1e9:,.2f}bn", f"R{s2e/1e9:,.2f}bn", f"R{s3e/1e9:,.2f}bn"],
        textposition="outside",
    ), row=1, col=2)

    fig.update_layout(title_text="IFRS 9: Accounts vs ECL Distribution by Stage",
                      template="plotly_white", height=420, showlegend=False)
    fig.update_yaxes(title_text="ECL (R bn)", row=1, col=2)
    return fig


def plot_capital_waterfall(regcap):
    pcts = regcap["capital_stack"]["percentages"]
    items = [
        ("CET1 Min", pcts["CET1 Min (Pillar 1)"], COLOR_SCHEME["CET1"]),
        ("T1 Add", pcts["Tier1 Min (Pillar 1)"] - pcts["CET1 Min (Pillar 1)"], COLOR_SCHEME["AT1"]),
        ("T2 Add", pcts["Total Cap Min (Pillar 1)"] - pcts["Tier1 Min (Pillar 1)"], COLOR_SCHEME["T2"]),
        ("CCB", pcts["CCB (Capital Conservation)"], COLOR_SCHEME["CCB"]),
        ("CCyB 2026", pcts["CCyB (Countercyclical 2026)"], COLOR_SCHEME["CCyB"]),
        ("D-SIB", pcts["D-SIB Buffer"], COLOR_SCHEME["D-SIB"]),
        ("HLA", pcts["HLA Buffer"], COLOR_SCHEME["HLA"]),
        ("Pillar 2", pcts["Pillar 2 Add-on"], COLOR_SCHEME["Pillar 2"]),
    ]
    labels = [x[0] for x in items]
    values = [x[1] * 100 for x in items]
    colors = [x[2] for x in items]

    fig = go.Figure(go.Waterfall(
        name="Capital Stack",
        orientation="v",
        measure=["absolute", "relative", "relative", "relative", "relative", "relative", "relative", "relative"],
        x=labels,
        y=values,
        text=[f"{v:.2f}pp" for v in values],
        textposition="outside",
        decreasing=dict(marker=dict(color="#6baed6")),
        increasing=dict(marker=dict(color=colors)),
        totals=dict(marker=dict(color="#c7e9c0")),
        connector=dict(line=dict(color="rgb(63, 63, 63)")),
    ))
    fig.add_hline(y=sum(values), line_dash="dash", line_color="red",
                  annotation_text=f"Total Required: {sum(values):.2f}% RWA")
    fig.update_layout(title_text="Capital Stack Waterfall: Pillar 1 → Buffers → Pillar 2 (SARB Reg 38)",
                      template="plotly_white", height=480, yaxis_title="% of RWA")
    return fig


def plot_ecap_allocation(ecap):
    comp = ecap["components"]
    labels = list(comp.keys())
    values = list(comp.values())
    total = sum(values)
    pcts = [v / max(total, 1) * 100 for v in values]
    text = [f"{l}: R{v/1e9:,.2f}bn ({p:.1f}%)" for l, v, p in zip(labels, values, pcts)]

    fig = go.Figure(data=[go.Sunburst(
        labels=["Total ECap"] + labels,
        parents=[""] + ["Total ECap"] * len(labels),
        values=[total] + values,
        branchvalues="total",
        marker=dict(colors=["#f0f0f0", "#2e8b57", "#4682b4", "#ff7f0e", "#9467bd", "#d62728", "#ffbb78"]),
        hovertext=["Total"] + text,
    )])
    fig.update_layout(
        title_text=f"Economic Capital Allocation (Nedbank 2024 Benchmark) - Total: R{total/1e9:,.2f}bn",
        template="plotly_white", height=500,
    )
    return fig


def plot_loss_distribution(credit_mc, ecl_total):
    losses = credit_mc["simulated_losses"] / 1e9
    el = credit_mc["expected_loss"] / 1e9
    var_999 = credit_mc["VaR"][0.999] / 1e9
    es_999 = credit_mc["Expected_Shortfall"][0.999] / 1e9

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=losses, nbinsx=100, name="Portfolio Loss",
                               marker=dict(color="#636EFA", opacity=0.75),
                               histnorm="probability density"))
    fig.add_vline(x=el, line_dash="dot", line_color="green",
                  annotation_text=f"Expected Loss: R{el:,.2f}bn", annotation_position="top left")
    fig.add_vline(x=var_999, line_dash="dash", line_color="orange",
                  annotation_text=f"VaR 99.9%: R{var_999:,.2f}bn", annotation_position="top right")
    fig.add_vline(x=es_999, line_dash="solid", line_color="red",
                  annotation_text=f"ES 99.9%: R{es_999:,.2f}bn")

    fig.update_layout(
        title_text=f"Simulated Credit Loss Distribution ({credit_mc['n_sims']:,} sims, {credit_mc['copula_type']}-copula)",
        xaxis_title="Portfolio Loss (R bn)", yaxis_title="Density",
        template="plotly_white", height=450, showlegend=False,
    )
    return fig


def plot_propagation_path(result, result_severe=None):
    scenarios = [("Base", result)]
    if result_severe is not None:
        scenarios.append(("Severe", result_severe))

    rows = []
    for name, r in scenarios:
        ead = r["ifrs9"]["ead_total"]
        ecl = r["ifrs9"]["ecl_total"]
        cr = r["regcap"]["total_rwa"]
        ec = r["economic_capital"]["total_ecap"]
        afr = r["coverage"]["main"]["AFR"]

        rows.extend([
            {"Scenario": name, "Step": "1. EAD Exposure", "Value": ead / 1e9},
            {"Scenario": name, "Step": "2. IFRS 9 ECL", "Value": ecl / 1e9},
            {"Scenario": name, "Step": "3. Total RWA", "Value": cr / 1e9},
            {"Scenario": name, "Step": "4. Total ECap Req", "Value": ec / 1e9},
            {"Scenario": name, "Step": "5. AFR Available", "Value": afr / 1e9},
        ])

    df = pd.DataFrame(rows)
    fig = px.line(df, x="Step", y="Value", color="Scenario", markers=True,
                  color_discrete_map={"Base": "#636EFA", "Severe": "#EF553B"})
    fig.update_traces(line=dict(width=3))
    fig.update_layout(
        title_text="Shock Propagation: EAD → ECL → RWA → ECap → AFR Coverage",
        yaxis_title="Amount (R bn)", template="plotly_white", height=450,
    )
    for _, row in df.iterrows():
        fig.add_annotation(x=row["Step"], y=row["Value"], text=f"R{row['Value']:,.1f}bn",
                           showarrow=False, yshift=10)
    return fig


def plot_coverage_erosion(coverage):
    df = coverage["erosion_path"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Shock Intensity (x)"], y=df["Coverage Ratio"],
                             mode="lines+markers", line=dict(color="#00CC96", width=3),
                             name="Coverage Ratio"))
    fig.add_hline(y=1.0, line_dash="dash", line_color="red",
                  annotation_text="Minimum 100% Coverage", annotation_position="top left")
    fig.add_hline(y=NEDBANK_ECAP_BENCHMARK_2024["afr_coverage_ratio"],
                  line_dash="dot", line_color="blue",
                  annotation_text="Nedbank 2024 Benchmark: 170%", annotation_position="bottom right")
    fig.add_trace(go.Scatter(x=df["Shock Intensity (x)"], y=df["ECap Expansion %"],
                             mode="lines", line=dict(color="#EF553B", dash="dash"),
                             name="ECap Expansion % (RHS)", yaxis="y2"))
    fig.add_trace(go.Scatter(x=df["Shock Intensity (x)"], y=df["AFR Erosion %"],
                             mode="lines", line=dict(color="#636EFA", dash="dot"),
                             name="AFR Erosion % (RHS)", yaxis="y2"))

    fig.update_layout(
        title_text="Coverage Ratio Erosion Under Increasing Stress Intensity",
        template="plotly_white", height=460,
        yaxis_title="AFR / ECap Coverage Ratio",
        yaxis2=dict(title="% Change", overlaying="y", side="right"),
        xaxis_title="Shock Intensity Multiplier",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def plot_uncertainty_bands(uncertainty):
    items = []
    for metric, band in uncertainty.items():
        if "CAR" in metric or "Coverage" in metric:
            continue
        c = band["central"]
        lo = band["lower_95"]
        hi = band["upper_95"]
        mr = band["model_risk"]
        items.append({
            "Metric": metric,
            "Central (R bn)": c / 1e9,
            "Lower 95%": lo / 1e9,
            "Upper 95%": hi / 1e9,
            "Model Risk (R bn)": mr / 1e9,
            "95% Interval Width (R bn)": (hi - lo) / 1e9,
        })

    df = pd.DataFrame(items)
    fig = go.Figure()
    for _, r in df.iterrows():
        fig.add_trace(go.Scatter(
            x=[r["Lower 95%"], r["Central (R bn)"], r["Upper 95%"]],
            y=[r["Metric"], r["Metric"], r["Metric"]],
            mode="lines+markers",
            line=dict(color=COLOR_SCHEME["Economic Capital"], width=4),
            marker=dict(size=[8, 12, 8], color=["#888", "#000", "#888"]),
            showlegend=False,
            name=r["Metric"],
            hovertemplate=f"{r['Metric']}: 95% [{r['Lower 95%']:,.2f}, {r['Upper 95%']:,.2f}] bn<br>Central: R{r['Central (R bn)']:,.2f}bn",
        ))
    fig.update_layout(
        title_text="Model Uncertainty: 95% Confidence Intervals + Model Risk Allocation",
        xaxis_title="Amount (R bn)", template="plotly_white", height=400,
    )
    return fig, df


def plot_benchmark_radar(bench_df, regcap_result):
    banks = list(SA_BANK_BENCHMARKS_2024.keys())
    categories = ["Total CAR", "CET1 Ratio", "Stage 1%", "Stage 2%", "Stage 3%", "ECap/RWA"]

    engine_ratios = regcap_result["capital_ratios"]
    engine_row = [engine_ratios["Total CAR"]] * len(categories)

    fig = go.Figure()

    for bank in banks:
        b = SA_BANK_BENCHMARKS_2024[bank]
        vals = [b["CAR"], b["CET1"], b["ECL_stage1_pct"], b["ECL_stage2_pct"], b["ECL_stage3_pct"], b["ECap_RWA"]]
        fig.add_trace(go.Scatterpolar(r=vals, theta=categories, fill="toself", name=bank, opacity=0.4))

    stage_ecl_pct = bench_df.iloc[0]
    ecl_s1 = float(stage_ecl_pct["Stage 1 ECL %"].replace("%", "")) / 100
    ecl_s2 = float(stage_ecl_pct["Stage 2 ECL %"].replace("%", "")) / 100
    ecl_s3 = float(stage_ecl_pct["Stage 3 ECL %"].replace("%", "")) / 100
    ecap_pct = NEDBANK_ECAP_BENCHMARK_2024["total_ecap_to_rwa"]
    engine_vals = [engine_ratios["Total CAR"], engine_ratios["CET1 Ratio"], ecl_s1, ecl_s2, ecl_s3, ecap_pct]
    fig.add_trace(go.Scatterpolar(r=engine_vals, theta=categories, fill="toself",
                                   name="Engine Output", line=dict(color="#EF553B", width=3)))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 0.35])),
        title_text="SA Bank Benchmark Comparison: Engine vs Top 5 D-SIBs (2024 Pillar 3)",
        template="plotly_white", height=520,
    )
    return fig


def main():
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1rem; padding-bottom: 2rem; max-width: 95% !important;}
        h1 {color: #1a365d;}
        h2 {color: #2c5282; border-bottom: 2px solid #cbd5e0; padding-bottom: 0.3rem;}
        .metric-container {background: #f7fafc; padding: 1rem; border-radius: 0.6rem; border-left: 4px solid #3182ce;}
        </style>
        """, unsafe_allow_html=True,
    )

    st.title("🏦 South African Credit Risk Volatility Engine")
    st.caption("IFRS 9 ECL • Basel III RegCap • Economic Capital — aligned with SARB Directives 5/2017, 6/2024 & Regulation 38/43")

    with st.sidebar:
        st.header("⚙️ Engine Controls")
        scenario = st.selectbox("Scenario (SARB/IMF Standard)", ["Base", "Adverse", "Severe"], index=0)
        severity = st.slider("Severity Multiplier", 0.5, 2.5, 1.0, 0.1,
                              help="Interpolate between standard scenario bounds")
        institution_size = st.selectbox("Institution Classification",
                                         ["Large_D-SIB", "Large", "Medium", "Small_Mutual"], index=0)
        total_exposure = st.number_input("Total Portfolio Exposure (R bn)", 100, 5000, 500, 50) * 1e9
        n_accounts = st.slider("Synthetic Accounts", 500, 5000, 2000, 500)
        n_mc_sims = st.slider("Monte Carlo Simulations", 500, 5000, 1500, 250,
                               help="Correlated default copula simulations")
        copula_type = st.selectbox("Copula Family", ["t", "Gaussian"], index=0,
                                    help="t-copula captures tail dependence (SA systemic clustering)")

        st.subheader("Idiosyncratic Shocks")
        sov_shock = st.checkbox("Sovereign Downgrade", value=False)
        comm_shock = st.checkbox("Commodity Collapse", value=False)
        cyber_shock = st.checkbox("Cyber / Operational Catastrophe", value=False)
        housing_shock = st.checkbox("Housing Market Crash", value=False)
        sme_shock = st.checkbox("SME Failure Wave", value=False)

        seed = st.number_input("Random Seed", 0, 9999, 2024, 1)
        st.divider()
        with st.expander("📋 Regulatory References"):
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

    with st.spinner(f"🔄 Running Risk Engine: {run_label} — {n_accounts:,} accounts, {n_mc_sims:,} sims..."):
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
        )

    ifrs9 = result["ifrs9"]
    regcap = result["regcap"]
    ecap = result["economic_capital"]
    coverage = result["coverage"]["main"]
    ratios = regcap["capital_ratios"]
    bench_df = result["benchmark_comparison"]

    st.success(f"✅ Engine completed in {result['run_metadata']['duration_seconds']:.2f}s — Scenario: **{run_label}**")

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric("Total Exposure (EAD)", FORMAT_ZAR(ifrs9["ead_total"]))
    with col2:
        st.metric("IFRS 9 ECL", FORMAT_ZAR(ifrs9["ecl_total"]),
                   delta=f"{ifrs9['ecl_total']/max(ifrs9['ead_total'],1)*100:.2f}% EAD", delta_color="inverse")
    with col3:
        st.metric("Total RWA", FORMAT_ZAR(regcap["total_rwa"]),
                   delta=f"{regcap['total_rwa']/max(ifrs9['ead_total'],1)*100:.1f}% EAD density")
    with col4:
        st.metric("Total Capital Adequacy Ratio", FORMAT_PCT(ratios["Total CAR"]),
                   delta="Pass" if ratios["Total CAR"] > 0.115 else "Warning",
                   delta_color="normal" if ratios["Total CAR"] > 0.115 else "inverse")
    with col5:
        st.metric("Total ECap Required", FORMAT_ZAR(ecap["total_ecap"]),
                   delta=f"{ecap['total_ecap_pct_rwa']*100:.1f}% of RWA")
    with col6:
        st.metric("AFR Coverage Ratio", f"{coverage['Coverage Ratio']*100:.1f}%",
                   delta=coverage["Coverage Rating"].split(" - ")[0],
                   delta_color="inverse" if coverage["Coverage Ratio"] < 1.2 else "normal")

    st.divider()

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Shock Propagation", "🧾 IFRS 9 & Staging", "🏛️ RegCap & RWA",
        "📈 Economic Capital & MC", "🛡️ Coverage & Erosion",
        "🎯 Benchmarks vs SA Banks", "⚠️ Uncertainty & Model Risk"
    ])

    with tab1:
        st.subheader("1. Scenario Shock Drivers")
        st.dataframe(result["scenario"]["shock_summary"], use_container_width=True, hide_index=True)

        st.subheader("2. Propagation Through Capital Stack")
        result_severe = None
        if scenario != "Severe":
            with st.spinner("Running Severe for propagation comparison..."):
                try:
                    result_severe = run_engine_end_to_end(
                        scenario="Severe", total_exposure=total_exposure, n_accounts=n_accounts,
                        seed=seed, institution_size=institution_size, severity_multiplier=severity,
                        idiosyncratic_shocks=idio_shocks if any_idio else None,
                        n_mc_sims=max(500, n_mc_sims//2), copula_type=copula_type,
                    )
                except Exception:
                    result_severe = None
        st.plotly_chart(plot_propagation_path(result, result_severe), use_container_width=True)

    with tab2:
        st.subheader("IFRS 9 Stage Distribution & ECL Allocation")
        st.plotly_chart(plot_stage_distribution(ifrs9), use_container_width=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Weighted Avg PIT PD (12m)", FORMAT_PCT(ifrs9["avg_pit_pd"]))
        with c2:
            st.metric("Weighted Avg LGD", FORMAT_PCT(ifrs9["avg_lgd"]))
        with c3:
            st.metric("Lifetime / 12m ECL Ratio",
                       f"{ifrs9['ecl_lifetime_total'] / max(ifrs9['ecl_12m_total'], 1):.2f}x")

        st.subheader("Portfolio-Level ECL Detail (by Segment)")
        df_seg = ifrs9["portfolio"].groupby("segment").agg(
            n_accounts=("account_id", "count"),
            EAD=("ead", "sum"),
            ECL=("ecl", "sum"),
            avg_PD=("pit_pd_12m", "mean"),
            avg_LGD=("lgd", "mean"),
            Stage3_ECL=("ecl", lambda x: x[ifrs9["portfolio"].loc[x.index, "ifrs9_stage"] == 3].sum()),
        ).reset_index()
        df_seg["ECL Rate %"] = df_seg["ECL"] / df_seg["EAD"] * 100
        for c in ["EAD", "ECL", "Stage3_ECL"]:
            df_seg[c] = df_seg[c].apply(lambda x: f"R{x/1e6:,.1f}m")
        df_seg["avg_PD"] = df_seg["avg_PD"].apply(lambda x: f"{x*100:.2f}%")
        df_seg["avg_LGD"] = df_seg["avg_LGD"].apply(lambda x: f"{x*100:.1f}%")
        df_seg["ECL Rate %"] = df_seg["ECL Rate %"].apply(lambda x: f"{x:.2f}%")
        st.dataframe(df_seg.rename(columns={"segment": "Product Segment"}),
                     use_container_width=True, hide_index=True)

    with tab3:
        st.subheader("Capital Stack Waterfall")
        st.plotly_chart(plot_capital_waterfall(regcap), use_container_width=True)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("CET1 Ratio", FORMAT_PCT(ratios["CET1 Ratio"]),
                       delta=f"Surplus: {FORMAT_ZAR(max(0, ratios['CET1 Surplus (vs CET1+Buffers)']))}",
                       delta_color="inverse" if ratios["CET1 Surplus (vs CET1+Buffers)"] < 0 else "normal")
        with c2:
            st.metric("Combined Buffer Usage", f"{ratios['Combined Buffer Utilisation %']*100:.1f}%",
                       delta=ratios["Conservation Level"].split(" - ")[0],
                       delta_color="inverse" if ratios["Combined Buffer Utilisation %"] > 1.0 else "normal")
        with c3:
            st.metric("Leverage Ratio", FORMAT_PCT(ratios["Leverage Ratio"]))
        with c4:
            st.metric("Payout Restriction", FORMAT_PCT(ratios["Payout Distribution Restriction %"]),
                       delta="MDA trigger" if ratios["Payout Distribution Restriction %"] >= 1.0 else "Flexible")

        st.subheader("RWA Breakdown")
        rwa_bd = result["rwa_breakdown"]
        fig_rwa = px.pie(names=list(rwa_bd.keys()), values=list(rwa_bd.values()),
                         title="Total RWA Composition by Risk Type",
                         color_discrete_sequence=["#636EFA", "#00CC96", "#EF553B"], hole=0.45)
        fig_rwa.update_traces(textinfo="label+percent+value", texttemplate="%{label}<br>%{percent}<br>R%{value:,.0f}")
        st.plotly_chart(fig_rwa, use_container_width=True)

    with tab4:
        st.subheader("ECap Allocation (Nedbank 2024 Benchmark Framework)")
        st.plotly_chart(plot_ecap_allocation(ecap), use_container_width=True)

        st.subheader("Credit Portfolio Loss Distribution (Simulated)")
        st.plotly_chart(plot_loss_distribution(result["monte_carlo"]["credit"], ifrs9["ecl_total"]),
                         use_container_width=True)

        c1, c2, c3 = st.columns(3)
        credit_mc = result["monte_carlo"]["credit"]
        with c1:
            st.metric("Credit VaR (99.9%)", FORMAT_ZAR(credit_mc["VaR"][0.999]))
        with c2:
            st.metric("Credit ES (99.9%)", FORMAT_ZAR(credit_mc["Expected_Shortfall"][0.999]))
        with c3:
            st.metric("Credit ECap (UL 99.9%)", FORMAT_ZAR(credit_mc["credit_ecap_999"]),
                       delta=f"{credit_mc['credit_ecap_999_pct_ead']*100:.2f}% EAD")

        st.markdown(f"**ECL vs ECap Ratio:** {ecap['ecl_vs_ecap_ratio']:.2f}x (expect 1.0x in severe stress)")

    with tab5:
        st.subheader("AFR vs ECap Coverage Ratio")
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

        st.plotly_chart(plot_coverage_erosion(result["coverage"]), use_container_width=True)

        st.subheader("Coverage Decomposition: AFR Sources vs ECap Uses")
        st.dataframe(result["coverage"]["decomposition"], use_container_width=True, hide_index=True)

    with tab6:
        st.subheader("SA Bank Benchmark Comparison (Pillar 3, 2024)")
        st.dataframe(bench_df, use_container_width=True, hide_index=True)
        st.plotly_chart(plot_benchmark_radar(bench_df, regcap), use_container_width=True)

    with tab7:
        st.subheader("Model Uncertainty & 95% Confidence Intervals")
        u_fig, u_df = plot_uncertainty_bands(result["uncertainty_bands"])
        st.plotly_chart(u_fig, use_container_width=True)
        st.dataframe(u_df, use_container_width=True, hide_index=True)

        st.subheader("Model Risk Allocation (SA Banking Industry Standard: 2% of ECap)")
        model_risk = NEDBANK_ECAP_BENCHMARK_2024["model_risk_pct"] * ecap["total_ecap"]
        st.info(f"**Model Risk Reserve:** R{model_risk/1e6:,.1f}m ({NEDBANK_ECAP_BENCHMARK_2024['model_risk_pct']*100:.1f}% of total ECap) — explicitly allocated per SA best practice.")

        with st.expander("📜 Audit Trail: Assumptions & Parameterisation"):
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
                }
            })


if __name__ == "__main__":
    main()
