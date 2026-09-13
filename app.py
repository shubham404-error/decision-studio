

"""Paisaan by CapitalSense — a public, session-only Streamlit MVP."""

from __future__ import annotations

from datetime import date
from io import BytesIO
import os

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from core.contracts import CalculationRequest, CalculationResult
from core.engines import (
    calculate_capital_gains,
    calculate_dcf,
    calculate_goal_plan,
    calculate_portfolio_risk,
    calculate_returns,
    calculate_trade_plan,
)
from services.ai import PERSONAL_CALL_LIMIT, explain_result, extract_image_records, gemini_available
from services.documents import MAX_IMAGES, parse_holdings_csv, prepare_image
from services.market_data import fetch_price_history


st.set_page_config(page_title="Paisaan by CapitalSense", page_icon="₹", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
    @import url('https://api.fontshare.com/v2/css?f[]=clash-display@400,500,600,700&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif !important;
    }
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Clash Display', sans-serif !important;
    }

    .block-container {max-width: 1200px; padding-top: 2.2rem; padding-bottom: 3rem;}
    [data-testid="stMetric"] {background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); padding: .8rem; border-radius: .8rem;}
    .cs-kicker {color: #a0aec0; text-transform: uppercase; letter-spacing: .08em; font-size: .75rem; font-weight: 700;}
    .cs-hero {padding: 1.5rem 0 .5rem 0;}
    .cs-card {border: 1px solid rgba(255,255,255,0.1); border-radius: 14px; padding: 1rem 1.1rem; background: rgba(255,255,255,0.02); min-height: 130px;}
    .cs-note {border-left: 4px solid #4263eb; padding: .75rem 1rem; background: rgba(66, 99, 235, 0.1); border-radius: 0 .6rem .6rem 0;}
    </style>
    """,
    unsafe_allow_html=True,

    )
    
    
def main_page():
    def money(value: float | None) -> str:
        return "—" if value is None or not np.isfinite(value) else f"₹{value:,.0f}"
    
    
    def percent(value: float | None, digits: int = 1) -> str:
        return "—" if value is None or not np.isfinite(value) else f"{value * 100:.{digits}f}%"
    
    
    def init_state() -> None:
        defaults = {"ai_calls": 0, "holdings": [], "image_drafts": [], "portfolio_confirmed": False, "paisaan_mode": True}
        for key, value in defaults.items():
            st.session_state.setdefault(key, value)
    
    
    def gemini_key() -> str | None:
        try:
            return st.secrets.get("GEMINI_API_KEY")
        except Exception:
            return os.getenv("GEMINI_API_KEY")
    
    
    def render_flow(title: str, when: str, needs: str, limitation: str) -> None:
        st.markdown(f'<div class="cs-kicker">Guided decision workspace</div><h1>{title}</h1>', unsafe_allow_html=True)
        if st.session_state.paisaan_mode:
            st.caption("Pehchaan → Soch → Plan. Understand the money story first; then let the numbers do the talking, mah brother.")
        cols = st.columns(3)
        for column, heading, body in zip(cols, ["When to use this", "What you need", "What it will not tell you"], [when, needs, limitation]):
            with column:
                st.markdown(f'<div class="cs-card"><b>{heading}</b><br><br>{body}</div>', unsafe_allow_html=True)
        st.write("")
    
    
    def render_result(result: CalculationResult, primary: list[tuple[str, str, float | None, str]]) -> None:
        metrics = st.columns(len(primary))
        for column, label, value, kind in zip(metrics, [item[0] for item in primary], [item[2] for item in primary], [item[3] for item in primary]):
            rendered = money(value) if kind == "money" else percent(value) if kind == "percent" else "—" if value is None else f"{value:,.2f}"
            column.metric(label, rendered)
        st.markdown("#### Understand this result")
        explain, changes, next_step = st.columns(3)
        with explain:
            st.markdown("**What it means**")
            st.write(primary[0][1])
        with changes:
            st.markdown("**What could change it**")
            st.write(result.warnings[0] if result.warnings else "Your inputs and the assumptions shown below drive this result.")
        with next_step:
            st.markdown("**Safe next check**")
            st.write("Change one assumption at a time and compare the scenario before acting.")
        with st.expander("Assumptions, data quality, and limitations"):
            for note in result.assumptions:
                st.write(f"• {note}")
            for note in result.data_quality:
                st.caption(note)
            for warning in result.warnings[1:]:
                st.warning(warning)
    
    
    def ai_explainer(result: CalculationResult, confirmed_context: dict | None = None) -> None:
        st.markdown("#### Ask the AI explainer")
        st.caption("Gemini explains confirmed deterministic results. It cannot change the calculation, execute a trade, or guarantee an outcome.")
        suggested = st.selectbox(
            "Start with a question",
            [
                "What am I missing in this analysis?",
                "What assumption has the biggest effect?",
                "Where is the risk concentrated?",
                "What should I verify before relying on this result?",
            ],
            key=f"ai_suggestion_{result.input_fingerprint}",
        )
        question = st.text_input("Or ask in your own words", value=suggested, key=f"ai_question_{result.input_fingerprint}")
        key = gemini_key()
        if not key:
            st.info("AI explanation is unavailable until a Gemini key is configured in Streamlit secrets. The calculation remains fully usable.")
            return
        if st.session_state.ai_calls >= PERSONAL_CALL_LIMIT:
            st.info("This session has reached its AI explanation limit. You can still change inputs and rerun every deterministic calculation.")
            return
        if st.button("Explain my confirmed result", key=f"ai_run_{result.input_fingerprint}"):
            with st.spinner("Preparing a bounded explanation..."):
                try:
                    answer = explain_result(key, os.getenv("GEMINI_MODEL", "gemini-3.5-flash"), question, result.as_context(), confirmed_context)
                    st.session_state.ai_calls += 1
                    st.markdown(answer)
                except Exception as exc:
                    st.warning(f"AI explanation is unavailable right now. Your deterministic result is unchanged. ({exc})")
    
    
    @st.cache_data(ttl=900, max_entries=16, show_spinner=False)
    def cached_prices(symbols: tuple[str, ...]) -> tuple[pd.DataFrame, str, str | None]:
        return fetch_price_history(list(symbols))
    
    
    def home_page() -> None:
        st.markdown('<div class="cs-hero"><div class="cs-kicker">CapitalSense presents</div><h1>Paisaan.</h1><p>First, pehchaan your money. Then make the next move with clarity.</p></div>', unsafe_allow_html=True)
        if st.session_state.paisaan_mode:
            st.markdown('<div class="cs-note"><b>Money follows, mah brother.</b><br>Here, that means: know what you own, know your risks, and make the decision before chasing the outcome.</div>', unsafe_allow_html=True)
        st.info("Educational decision support only. No brokerage connection, no financial advice, and no permanent storage of uploaded data.")
        choices = [
            ("Portfolio Health Check", "Review returns, concentration, drawdown, risk, and diversification."),
            ("Goals & Retirement Planner", "Test a SIP, step-up, inflation, FIRE, and NPS planning scenario."),
            ("Value a Stock", "Explore DCF and WACC assumptions through a valuation range."),
            ("Sell / Tax-Aware Return", "Estimate capital gains and what remains after assumed tax."),
            ("Trade Plan", "Size a risk-defined trade and explore option payoff assumptions."),
        ]
        for index in range(0, len(choices), 2):
            columns = st.columns(2)
            for column, (name, description) in zip(columns, choices[index:index + 2]):
                with column:
                    st.markdown(f'<div class="cs-card"><b>{name}</b><br><br>{description}<br><br><span style="color:#5c6b7c">Choose it from the sidebar to begin.</span></div>', unsafe_allow_html=True)
        st.markdown("### How the app handles your data")
        st.write("CSV and image uploads stay in this browser session. Image extraction is optional and needs your explicit consent before a selected image is sent to Gemini. Review every extracted field before analysis.")
    
    
    def portfolio_page() -> None:
        render_flow(
            "Portfolio Health Check",
            "Use this before adding money, selling a holding, or deciding whether your portfolio is carrying more risk than you intended.",
            "A holdings CSV, a few holdings entered manually, or sample data. Market-history risk metrics need recognised NSE symbols.",
            "It cannot predict future returns or tell you what to buy. It describes the portfolio you confirm.",
        )
        with st.expander("1. Add holdings", expanded=True):
            sample_col, csv_col = st.columns(2)
            with sample_col:
                if st.button("Load sample portfolio"):
                    st.session_state.holdings = [
                        {"symbol": "RELIANCE", "quantity": 20.0, "average_cost": 2400.0, "current_price": 2900.0, "value": 58000.0},
                        {"symbol": "HDFCBANK", "quantity": 40.0, "average_cost": 1500.0, "current_price": 1700.0, "value": 68000.0},
                        {"symbol": "INFY", "quantity": 30.0, "average_cost": 1350.0, "current_price": 1600.0, "value": 48000.0},
                    ]
                    st.session_state.portfolio_confirmed = False
            with csv_col:
                uploaded_csv = st.file_uploader("Import holdings CSV (max 5 MB)", type=["csv"], key="holdings_csv")
                if uploaded_csv and st.button("Read CSV into this session"):
                    parsed = parse_holdings_csv(uploaded_csv.getvalue())
                    for warning in parsed.warnings:
                        st.warning(warning)
                    if parsed.records:
                        st.session_state.holdings = parsed.records
                        st.session_state.portfolio_confirmed = False
                        st.success(f"Loaded {len(parsed.records)} holding rows. Review them below.")
    
            st.markdown("**Or edit the holdings directly**")
            table = pd.DataFrame(st.session_state.holdings or [{"symbol": "", "quantity": 0.0, "average_cost": 0.0, "current_price": 0.0, "value": 0.0}])
            edited = st.data_editor(table, num_rows="dynamic", use_container_width=True, key="holdings_editor")
            if st.button("Confirm holdings for analysis"):
                valid = edited.copy()
                valid["symbol"] = valid["symbol"].astype(str).str.strip().str.upper()
                valid = valid[valid["symbol"].ne("") & valid["symbol"].ne("NAN")]
                st.session_state.holdings = valid.to_dict("records")
                st.session_state.portfolio_confirmed = bool(st.session_state.holdings)
                st.success("Pehchaan unlocked: your holdings are confirmed for this session. Nothing was saved outside it.")
    
        with st.expander("Optional: analyse a statement image", expanded=False):
            st.warning("Do not upload passwords, PINs, Aadhaar, PAN, or documents you do not have permission to share. This public app does not save uploads, but image extraction sends the selected image to Gemini after consent.")
            images = st.file_uploader("Upload up to five JPG, PNG, or WebP images (10 MB each)", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True, key="statement_images")
            consent = st.checkbox("I understand that the selected image will be sent to the configured Google Gemini service for extraction. I will review the extracted draft before analysis.", key="image_consent")
            if images and len(images) > MAX_IMAGES:
                st.error(f"Choose no more than {MAX_IMAGES} images in one session.")
            elif images:
                st.caption(f"{len(images)} image(s) selected. Start with one clear holdings or transaction screenshot.")
                selected = st.selectbox("Image to extract", range(len(images)), format_func=lambda index: images[index].name)
                if consent and st.button("Extract a reviewable draft from selected image"):
                    key = gemini_key()
                    if not key:
                        st.info("Image extraction needs a Gemini key in Streamlit secrets. You can still add holdings manually or import CSV.")
                    elif st.session_state.ai_calls >= PERSONAL_CALL_LIMIT:
                        st.info("This session has reached its bounded AI limit. Use CSV or manual entry instead.")
                    else:
                        image, error = prepare_image(images[selected].getvalue(), images[selected].type)
                        if error:
                            st.error(error)
                        else:
                            with st.spinner("Extracting a draft for your review..."):
                                try:
                                    draft = extract_image_records(key, os.getenv("GEMINI_MODEL", "gemini-3.5-flash"), image)
                                    st.session_state.ai_calls += 1
                                    st.session_state.image_drafts = draft.records
                                    st.session_state.image_draft_warnings = draft.warnings
                                except Exception as exc:
                                    st.warning(f"Image extraction is unavailable right now. ({exc})")
            if st.session_state.image_drafts:
                st.markdown("**2. Review extracted draft — it is not analysed until you confirm it**")
                for warning in st.session_state.get("image_draft_warnings", []):
                    st.warning(warning)
                extracted = st.data_editor(pd.DataFrame(st.session_state.image_drafts), num_rows="dynamic", use_container_width=True, key="image_draft_editor")
                if st.button("Use reviewed image draft as holdings"):
                    st.session_state.holdings = extracted.to_dict("records")
                    st.session_state.portfolio_confirmed = True
                    st.success("Reviewed draft confirmed for this session.")
    
        if not st.session_state.portfolio_confirmed:
            st.info("Confirm holdings above to unlock analysis. This pause is intentional: it prevents an uploaded draft from being mistaken for verified data.")
            return
        holdings = st.session_state.holdings
        holdings_frame = pd.DataFrame(holdings)
        if "value" not in holdings_frame:
            holdings_frame["value"] = 0.0
        holdings_frame["value"] = pd.to_numeric(holdings_frame["value"], errors="coerce").fillna(0.0)
        if holdings_frame["value"].sum() <= 0 and {"quantity", "current_price"}.issubset(holdings_frame.columns):
            holdings_frame["value"] = pd.to_numeric(holdings_frame["quantity"], errors="coerce").fillna(0) * pd.to_numeric(holdings_frame["current_price"], errors="coerce").fillna(0)
        st.markdown("### 3. Review concentration before calculating risk")
        total_value = float(holdings_frame["value"].sum())
        if total_value > 0:
            allocation = holdings_frame.assign(weight=holdings_frame["value"] / total_value).sort_values("weight", ascending=False)
            left, right = st.columns([1, 1])
            left.plotly_chart(px.pie(allocation, names="symbol", values="value", title="Confirmed allocation"), use_container_width=True)
            right.dataframe(allocation[["symbol", "value", "weight"]], use_container_width=True, hide_index=True, column_config={"value": st.column_config.NumberColumn("Value", format="₹%d"), "weight": st.column_config.NumberColumn("Weight", format="%.1%%")})
        else:
            st.warning("Add market values or quantity/current price to assess concentration. You can still use manual return data below.")
    
        with st.expander("4. Risk inputs", expanded=True):
            use_market = st.checkbox("Fetch one-year public price history for recognised NSE symbols", value=True)
            manual_return = st.number_input("If market data is unavailable, enter an assumed annual return for the return summary (%)", value=12.0, step=0.5) / 100
            initial_value = st.number_input("Total amount invested / starting value", min_value=0.0, value=max(total_value * 0.85, 0.0), step=1000.0)
            start_date = st.date_input("Approximate first investment date", value=date(date.today().year - 3, 1, 1))
            confidence = st.selectbox("VaR confidence", [0.90, 0.95, 0.99], index=1, format_func=lambda value: f"{value:.0%}")
        prices = pd.DataFrame()
        source, as_of = "Manual inputs", None
        symbols = tuple(symbol for symbol in holdings_frame.get("symbol", pd.Series(dtype=str)).dropna().astype(str).tolist() if symbol)
        if use_market and symbols:
            with st.spinner("Fetching public price history only for confirmed symbols..."):
                prices, source, as_of = cached_prices(symbols)
        if not prices.empty:
            returns_frame = prices.pct_change().dropna(how="all")
            common = [column for column in returns_frame.columns if column.replace(".NS", "") in symbols]
            daily_returns = returns_frame[common].mean(axis=1).dropna().to_numpy() if common else np.array([])
            st.caption(f"Data source: {source}. As of: {as_of or 'unavailable'}. Data can be delayed or incomplete.")
        else:
            daily_returns = np.repeat((1 + manual_return) ** (1 / 252) - 1, 30)
            st.caption(f"Data source: {source}. Showing an assumed-return illustration, not historical risk.")
        risk_request = CalculationRequest("portfolio_risk", {"returns": daily_returns.tolist(), "confidence": confidence, "holdings": holdings_frame.to_dict("records")})
        risk = calculate_portfolio_risk(risk_request)
        return_request = CalculationRequest("returns", {"cashflows": [{"date": start_date, "amount": -initial_value}], "ending_value": total_value, "period_returns": daily_returns.tolist()}, as_of_date=date.today())
        returns = calculate_returns(return_request)
        render_result(risk, [("Annual volatility", "The range of normal portfolio movement implied by its return history.", risk.outputs.get("annualized_volatility"), "percent"), ("Max drawdown", "The largest peak-to-trough decline in the observed period.", risk.outputs.get("max_drawdown"), "percent"), ("Historical daily VaR", "A one-day loss threshold from historical returns at the selected confidence.", risk.outputs.get("historical_var_daily"), "percent"), ("Top holding", "The largest confirmed allocation; high concentration can dominate outcomes.", risk.outputs.get("top_holding_weight"), "percent")])
        st.markdown("### Returns summary")
        summary_cols = st.columns(4)
        summary_cols[0].metric("Invested", money(returns.outputs.get("invested_amount")))
        summary_cols[1].metric("Current value", money(returns.outputs.get("ending_value")))
        summary_cols[2].metric("XIRR", percent(returns.outputs.get("xirr")))
        summary_cols[3].metric("CAGR", percent(returns.outputs.get("cagr")))
        if risk.tables_and_chart_series.get("cumulative_growth"):
            chart = pd.DataFrame({"Growth of ₹1": risk.tables_and_chart_series["cumulative_growth"], "Drawdown": risk.tables_and_chart_series["drawdowns"]})
            st.line_chart(chart)
        ai_explainer(risk, {"holdings": holdings_frame[["symbol", "value"]].to_dict("records"), "market_source": source})
    
    
    def goals_page() -> None:
        render_flow("Goals & Retirement Planner", "Use this to check whether a planned monthly investment is broadly aligned with a future spending goal.", "Your SIP, step-up, horizon, expected return, inflation, and present monthly expenses.", "It cannot guarantee retirement readiness, product returns, or statutory NPS/PPF benefits.")
        c1, c2, c3 = st.columns(3)
        with c1:
            monthly_sip = st.number_input("Monthly investment", min_value=0.0, value=15000.0, step=1000.0)
            years = st.slider("Years to goal", 1, 40, 20)
        with c2:
            annual_return = st.slider("Expected annual return", 3.0, 18.0, 11.0, 0.5) / 100
            step_up = st.slider("Annual SIP step-up", 0.0, 25.0, 10.0, 1.0) / 100
        with c3:
            inflation = st.slider("Expected inflation", 2.0, 12.0, 6.0, 0.5) / 100
            expenses = st.number_input("Current monthly expenses", min_value=0.0, value=60000.0, step=5000.0)
            nps_share = st.slider("Illustrative NPS share of planned contributions", 0, 100, 20) / 100
        result = calculate_goal_plan(CalculationRequest("goal_planner", {"monthly_sip": monthly_sip, "annual_step_up": step_up * 100, "years": years, "annual_return": annual_return, "inflation": inflation, "monthly_expenses": expenses, "nps_share": nps_share}))
        render_result(result, [("Projected corpus", "Illustrated corpus after planned monthly investing and annual step-ups.", result.outputs["projected_corpus"], "money"), ("Indicative FIRE corpus", "A 25× future annual-spending illustration, not a retirement guarantee.", result.outputs["indicative_fire_corpus"], "money"), ("Goal gap", "The difference between the projection and the illustrative corpus target.", result.outputs["goal_gap"], "money")])
        st.line_chart(pd.DataFrame(result.tables_and_chart_series["yearly_projection"]).set_index("year")[["corpus"]])
        ai_explainer(result)
    
    
    def valuation_page() -> None:
        render_flow("Value a Stock", "Use this to test whether your valuation story survives conservative assumptions.", "A normalised FCFF estimate, growth view, capital structure, beta, and share count.", "It cannot produce a certain fair price or replace company research.")
        st.markdown("### Base case inputs")
        c1, c2, c3 = st.columns(3)
        with c1:
            fcff = st.number_input("Current annual FCFF", min_value=0.0, value=1000.0, step=100.0)
            growth = st.slider("Five-year FCFF growth", -10.0, 35.0, 12.0, 0.5) / 100
            terminal_growth = st.slider("Terminal growth", 1.0, 8.0, 5.0, 0.25) / 100
        with c2:
            beta = st.slider("Equity beta", 0.2, 3.0, 1.0, 0.05)
            risk_free = st.slider("Risk-free rate", 4.0, 10.0, 7.0, 0.25) / 100
            premium = st.slider("Equity risk premium", 3.0, 10.0, 6.0, 0.25) / 100
        with c3:
            debt = st.number_input("Net debt (cash negative)", value=0.0, step=100.0)
            shares = st.number_input("Shares outstanding", min_value=1.0, value=100.0, step=1.0)
            equity_weight = st.slider("Equity weight in capital", 10, 95, 75) / 100
        base_inputs = {"fcff": fcff, "growth": growth, "terminal_growth": terminal_growth, "beta": beta, "risk_free": risk_free, "equity_premium": premium, "cost_debt": 0.09, "tax_rate": 0.25, "equity_weight": equity_weight, "net_debt": debt, "shares_outstanding": shares, "horizon_years": 5}
        scenarios = {"Bear": {**base_inputs, "growth": growth - 0.04, "terminal_growth": max(0.01, terminal_growth - 0.01)}, "Base": base_inputs, "Bull": {**base_inputs, "growth": growth + 0.04, "terminal_growth": terminal_growth + 0.005}}
        results = {name: calculate_dcf(CalculationRequest("dcf_valuation", inputs, scenario_name=name)) for name, inputs in scenarios.items()}
        comparison = pd.DataFrame([{"Scenario": name, "Value/share": result.outputs["value_per_share"], "WACC": result.outputs["wacc"], "Enterprise value": result.outputs["enterprise_value"]} for name, result in results.items()])
        st.dataframe(comparison, hide_index=True, use_container_width=True, column_config={"Value/share": st.column_config.NumberColumn(format="₹%.2f"), "WACC": st.column_config.NumberColumn(format="%.2%%"), "Enterprise value": st.column_config.NumberColumn(format="₹%.0f")})
        base = results["Base"]
        render_result(base, [("Base value / share", "A present-value estimate under the base case, not a target price.", base.outputs["value_per_share"], "money"), ("WACC", "The discount rate that has a large effect on the present value.", base.outputs["wacc"], "percent"), ("Equity value", "Enterprise value after the user-entered net debt adjustment.", base.outputs["equity_value"], "money")])
        st.line_chart(pd.DataFrame(base.tables_and_chart_series["forecast"]).set_index("year")[["fcff", "present_value"]])
        ai_explainer(base)
    
    
    def tax_page() -> None:
        render_flow("Sell / Tax-Aware Return", "Use this before estimating what you may keep after selling an investment.", "Purchase value, expected sale value, costs, holding period, and a tax-rate assumption you can verify.", "It is not a tax return or a complete view of surcharge, cess, set-off, grandfathering, or special rules.")
        c1, c2, c3 = st.columns(3)
        with c1:
            purchase = st.number_input("Purchase value", min_value=0.0, value=500000.0, step=10000.0)
            sale = st.number_input("Expected sale value", min_value=0.0, value=700000.0, step=10000.0)
        with c2:
            costs = st.number_input("Brokerage and transaction costs", min_value=0.0, value=1000.0, step=500.0)
            months = st.number_input("Holding period (months)", min_value=0, value=18, step=1)
        with c3:
            asset = st.selectbox("Asset type", ["listed_equity", "other_asset"], format_func=lambda item: "Listed equity / equity mutual fund" if item == "listed_equity" else "Other asset — verify your applicable rule")
            tax_year = st.selectbox("Assumption version", ["FY 2026-27 estimate", "FY 2025-26 estimate"])
        with st.expander("Advanced tax assumptions — confirm these before relying on the result"):
            short_rate = st.slider("Short-term tax rate", 0.0, 40.0, 20.0, 0.5) / 100
            long_rate = st.slider("Long-term tax rate", 0.0, 40.0, 12.5, 0.5) / 100
            exemption = st.number_input("Listed-equity LTCG exemption", min_value=0.0, value=125000.0, step=25000.0)
        result = calculate_capital_gains(CalculationRequest("capital_gains", {"purchase_value": purchase, "sale_value": sale, "transaction_costs": costs, "holding_months": months, "asset_type": asset, "short_rate": short_rate, "long_rate": long_rate, "long_exemption": exemption}, tax_year=tax_year))
        render_result(result, [("Estimated tax", "Indicative tax based on your selected rate and asset assumptions.", result.outputs["estimated_tax"], "money"), ("Net sale proceeds", "Expected sale value after entered costs and illustrative tax.", result.outputs["net_sale_proceeds"], "money"), ("After-tax return", "Return after entered costs and estimated tax.", result.outputs["after_tax_return"], "percent")])
        ai_explainer(result)
    
    
    def trade_page() -> None:
        render_flow("Trade Plan", "Use this to cap downside before placing a simulated or real trade elsewhere.", "Capital available, risk limit, entry, stop, target, and optional option assumptions.", "It cannot provide live margin, broker execution, option-chain data, or a trade recommendation.")
        st.markdown("### Risk-defined position")
        c1, c2, c3 = st.columns(3)
        with c1:
            capital = st.number_input("Capital available", min_value=0.0, value=200000.0, step=5000.0)
            risk_percent = st.slider("Maximum capital at risk", 0.1, 5.0, 1.0, 0.1)
        with c2:
            entry = st.number_input("Planned entry", min_value=0.0, value=1000.0, step=1.0)
            stop = st.number_input("Stop-loss", min_value=0.0, value=950.0, step=1.0)
            target = st.number_input("Target", min_value=0.0, value=1100.0, step=1.0)
        with c3:
            leverage = st.slider("Illustrative leverage", 1.0, 10.0, 1.0, 0.5)
            charges = st.slider("Round-trip charge rate", 0.0, 0.5, 0.10, 0.01) / 100
        with st.expander("Optional: option payoff and theoretical value"):
            option_type = st.selectbox("Option type", ["call", "put"])
            spot = st.number_input("Current underlying price", min_value=0.0, value=entry, step=1.0)
            strike = st.number_input("Strike", min_value=0.0, value=entry, step=1.0)
            premium = st.number_input("Premium paid", min_value=0.0, value=30.0, step=1.0)
            volatility = st.slider("Implied-volatility assumption", 5.0, 100.0, 25.0, 1.0) / 100
            expiry = st.slider("Days to expiry", 1, 365, 30)
        result = calculate_trade_plan(CalculationRequest("trade_plan", {"capital": capital, "risk_percent": risk_percent, "entry": entry, "stop": stop, "target": target, "leverage": leverage, "charges_rate": charges, "option_spot": spot, "option_strike": strike, "option_premium": premium, "option_volatility": volatility, "option_expiry_days": expiry, "option_type": option_type}))
        render_result(result, [("Risk budget", "Your stated maximum rupee loss before charges and slippage.", result.outputs["risk_budget"], "money"), ("Position quantity", "The lower of the risk-budget and capital-cap quantity.", result.outputs["position_quantity"], "number"), ("Risk / reward", "Target distance divided by stop-loss distance.", result.outputs["risk_reward_ratio"], "number"), ("Margin illustration", "Exposure divided by your chosen illustrative leverage, not broker margin.", result.outputs["margin_estimate"], "money")])
        if result.tables_and_chart_series["option_payoff"]:
            payoff = pd.DataFrame(result.tables_and_chart_series["option_payoff"])
            st.plotly_chart(px.line(payoff, x="underlying", y="pnl_per_unit", title="Illustrative option payoff at expiry"), use_container_width=True)
        st.caption(f"Theoretical Black-Scholes option value: {money(result.outputs.get('black_scholes_value'))}")
        ai_explainer(result)
    
    
    def tools_page() -> None:
        render_flow("Tools", "Use this directory when you already know the calculation you need.", "A clear decision context and inputs you can verify.", "It does not replace the guided workspaces, which give more useful cues and checks.")
        st.dataframe(pd.DataFrame([
            {"Tool": "Portfolio risk", "Where it lives": "Portfolio Health Check", "Why": "Risk, drawdown, return, and concentration need one confirmed holdings view."},
            {"Tool": "XIRR / CAGR / TWRR", "Where it lives": "Portfolio Health Check", "Why": "Return measures are meaningful only beside their cashflows and time period."},
            {"Tool": "SIP / FIRE / NPS planning", "Where it lives": "Goals & Retirement Planner", "Why": "One scenario beats several disconnected savings calculators."},
            {"Tool": "DCF / WACC", "Where it lives": "Value a Stock", "Why": "Scenario sensitivity is more honest than a single fair value."},
            {"Tool": "Capital gains", "Where it lives": "Sell / Tax-Aware Return", "Why": "After-tax proceeds depend on assumptions users can inspect."},
            {"Tool": "Position size / options", "Where it lives": "Trade Plan", "Why": "Risk budget, charges, payoff, and margin belong together."},
        ]), hide_index=True, use_container_width=True)
    
    
    def main() -> None:
        init_state()
        with st.sidebar:
            st.markdown("## ₹ Paisaan")
            st.caption("by CapitalSense · money clarity, mah brother")
            st.session_state.paisaan_mode = st.toggle("Paisaan mode", value=st.session_state.paisaan_mode, help="Adds light meme-inspired cues. Turn it off for a more classic research-terminal tone.")
            page = st.radio("Choose a decision", ["Home", "Portfolio Health Check", "Goals & Retirement", "Value a Stock", "Sell / Tax-Aware Return", "Trade Plan", "Tools"], label_visibility="collapsed")
            st.divider()
            st.caption("Session-only uploads · optional AI explanation · no brokerage connection")
            st.caption(f"AI explanations left this session: {max(0, PERSONAL_CALL_LIMIT - st.session_state.ai_calls)}")
        pages = {
            "Home": home_page,
            "Portfolio Health Check": portfolio_page,
            "Goals & Retirement": goals_page,
            "Value a Stock": valuation_page,
            "Sell / Tax-Aware Return": tax_page,
            "Trade Plan": trade_page,
            "Tools": tools_page,
        }
        pages[page]()
        st.divider()
        st.caption("Paisaan by CapitalSense is educational decision support. Pehchaan first; verify tax rules, prices, instrument data, and suitability independently before acting.")
    
    
    if __name__ == "__main__":
        main()
    

def guide_page():
    st.markdown("""
    <style>
    .terminal-card {
        border: 1px solid rgba(255,255,255,0.08); 
        border-radius: 12px; 
        padding: 1.5rem; 
        background: #14161f; 
        height: 100%;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .terminal-card:hover {
        border-color: rgba(255,255,255,0.2);
    }
    .terminal-icon {
        font-size: 2rem; 
        margin-bottom: 0.8rem;
    }
    .terminal-title {
        font-family: 'Clash Display', sans-serif;
        font-size: 1.3rem; 
        font-weight: 600; 
        color: #ffffff; 
        margin-bottom: 0.5rem;
    }
    .terminal-subtitle {
        font-size: 0.85rem; 
        font-style: italic; 
        color: #8c9baf; 
        margin-bottom: 1rem;
    }
    .terminal-text {
        font-size: 0.9rem; 
        color: #cbd5e1; 
        line-height: 1.5;
    }
    .highlight {
        color: #4facfe;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)

    # Hero Section
    st.markdown("""
    <div style="text-align: center; padding: 3rem 1rem 4rem 1rem; background: radial-gradient(circle at top, rgba(20,24,36,0.8) 0%, rgba(15,17,26,0) 70%); border-radius: 12px; margin-bottom: 2rem;">
        <h1 style="color: #4facfe; font-size: 3.5rem; font-weight: 800; margin-bottom: 0.5rem; font-family: 'Clash Display', sans-serif; letter-spacing: -1px;">Decision Studio</h1>
        <p style="color: #8c9baf; font-size: 1.1rem; max-width: 650px; margin: 0 auto;">Institutional-grade mathematical models, simplified for retail investors. Measure risk, project returns, and evaluate complex financial decisions with precision.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.image("gif.gif")
    st.write("")
    
    st.markdown("<h2 style='font-family: \"Clash Display\", sans-serif; margin-bottom: 1.5rem;'>Core Calculators</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="terminal-card">
            <div class="terminal-icon">📈</div>
            <div class="terminal-title">Portfolio Health Check</div>
            <div class="terminal-subtitle">Answers: "Is my portfolio too concentrated or risky?"</div>
            <div class="terminal-text">Upload a CSV of your holdings to instantly review returns, concentration, drawdown risk, and diversification metrics.</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="terminal-card">
            <div class="terminal-icon">💰</div>
            <div class="terminal-title">DCF Valuation</div>
            <div class="terminal-subtitle">Answers: "What is the intrinsic value of this asset?"</div>
            <div class="terminal-text">Calculate the intrinsic value of an asset based on projected future cash flows and your <span class="highlight">required rate of return</span>.</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown("""
        <div class="terminal-card">
            <div class="terminal-icon">📉</div>
            <div class="terminal-title">Trade Plan</div>
            <div class="terminal-subtitle">Answers: "How much should I risk on this trade?"</div>
            <div class="terminal-text">Structures a complete entry/exit strategy including position sizing based on your <span class="highlight">risk tolerance</span> and stop-loss levels.</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.write("")
    
    st.markdown("<h2 style='font-family: \"Clash Display\", sans-serif; margin-bottom: 1.5rem;'>Planning & Tax Models</h2>", unsafe_allow_html=True)
    
    col4, col5 = st.columns(2)
    
    with col4:
        st.markdown("""
        <div class="terminal-card">
            <div class="terminal-icon">🎯</div>
            <div class="terminal-title">Goal Planning</div>
            <div class="terminal-subtitle">Answers: "How much SIP do I need for my target?"</div>
            <div class="terminal-text">Determine the required monthly SIP or lumpsum investment needed for a specific financial target. Factors in inflation and <span class="highlight">step-up</span> contributions.</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col5:
        st.markdown("""
        <div class="terminal-card">
            <div class="terminal-icon">🏛️</div>
            <div class="terminal-title">Tax-Aware Return</div>
            <div class="terminal-subtitle">Answers: "What do I actually keep after taxes?"</div>
            <div class="terminal-text">Project tax implications to optimize <span class="highlight">short-term vs. long-term</span> capital gains brackets before executing a sell decision.</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.write("")
    st.write("")
    
    st.markdown("<h2 style='font-family: \"Clash Display\", sans-serif; margin-bottom: 1.5rem;'>Inputs & Best Practices</h2>", unsafe_allow_html=True)
    
    col6, col7 = st.columns(2)
    
    with col6:
        st.markdown("""
        <div class="terminal-card">
            <div class="terminal-icon">🔒</div>
            <div class="terminal-title">Privacy First Data</div>
            <div class="terminal-subtitle">Answers: "Where does my uploaded data go?"</div>
            <div class="terminal-text">
                <b>Session-Only:</b> Your uploaded CSVs remain entirely in your browser. We do not store your financial data.<br><br>
                <b>AI Parser:</b> Securely upload images, and Gemini AI will extract the financial records.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col7:
        st.markdown("""
        <div class="terminal-card">
            <div class="terminal-icon">🚦</div>
            <div class="terminal-title">Garbage In, Garbage Out</div>
            <div class="terminal-subtitle">Answers: "How reliable are these results?"</div>
            <div class="terminal-text">
                <b>Assumptions:</b> Always expand the limitations section under results. The math is deterministic; bad inputs yield bad outputs.<br><br>
                <b>AI Explainer:</b> Use the built-in AI Explainer to understand exactly what is driving the result.
            </div>
        </div>
        """, unsafe_allow_html=True)

pages = {
    "Start": [
        st.Page(guide_page, title="User Guide", icon="📖", default=True)
    ],
    "Calculators": [
        st.Page(main_page, title="Decision Studio", icon="🧮")
    ]
}

pg = st.navigation(pages, position="sidebar")
pg.run()
