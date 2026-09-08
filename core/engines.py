"""Pure financial calculation engines. No Streamlit, network, or file-system calls."""

from __future__ import annotations

from datetime import date
from math import exp, log, pi, sqrt
from statistics import NormalDist
from typing import Iterable

import numpy as np
import pandas as pd

from .contracts import CalculationRequest, CalculationResult


NORMAL = NormalDist()
TRADING_DAYS = 252


def _finite(value: float | int | None, default: float = 0.0) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def _result(request: CalculationRequest, outputs: dict, **kwargs) -> CalculationResult:
    return CalculationResult(
        calculator_id=request.calculator_id,
        outputs=outputs,
        input_fingerprint=request.input_fingerprint,
        **kwargs,
    )


def _xnpv(rate: float, cashflows: list[tuple[date, float]]) -> float:
    start = cashflows[0][0]
    return sum(amount / (1 + rate) ** ((when - start).days / 365.25) for when, amount in cashflows)


def xirr(cashflows: Iterable[tuple[date, float]]) -> float | None:
    """Bisection-based XIRR; returns None where an economically valid rate is absent."""
    flows = sorted((when, _finite(amount)) for when, amount in cashflows)
    if len(flows) < 2 or not any(amount < 0 for _, amount in flows) or not any(amount > 0 for _, amount in flows):
        return None
    low, high = -0.9999, 10.0
    f_low, f_high = _xnpv(low, flows), _xnpv(high, flows)
    while f_low * f_high > 0 and high < 1_000_000:
        high *= 2
        f_high = _xnpv(high, flows)
    if f_low * f_high > 0:
        return None
    for _ in range(160):
        mid = (low + high) / 2
        f_mid = _xnpv(mid, flows)
        if abs(f_mid) < 1e-7:
            return mid
        if f_low * f_mid <= 0:
            high, f_high = mid, f_mid
        else:
            low, f_low = mid, f_mid
    return (low + high) / 2


def calculate_returns(request: CalculationRequest) -> CalculationResult:
    """Calculate money-weighted and time-weighted outcomes from user-confirmed inputs."""
    raw_flows = request.inputs.get("cashflows", [])
    cashflows: list[tuple[date, float]] = []
    for item in raw_flows:
        when = item["date"] if isinstance(item["date"], date) else date.fromisoformat(str(item["date"]))
        cashflows.append((when, _finite(item["amount"])))
    ending_value = _finite(request.inputs.get("ending_value"))
    as_of = request.as_of_date or date.today()
    if ending_value:
        cashflows.append((as_of, ending_value))

    xirr_value = xirr(cashflows)
    invested = -sum(amount for _, amount in cashflows if amount < 0)
    gain = ending_value - invested
    start_dates = [when for when, amount in cashflows if amount < 0]
    years = max((as_of - min(start_dates)).days / 365.25, 0) if start_dates else 0
    cagr = ((ending_value / invested) ** (1 / years) - 1) if invested > 0 and ending_value > 0 and years > 0 else None

    period_returns = np.asarray(request.inputs.get("period_returns", []), dtype=float)
    period_returns = period_returns[np.isfinite(period_returns)]
    twrr = float(np.prod(1 + period_returns) - 1) if period_returns.size else None
    warnings = []
    if xirr_value is None:
        warnings.append("XIRR needs at least one contribution and one positive ending value or withdrawal.")
    if not period_returns.size:
        warnings.append("TWRR is unavailable until period-by-period return data is supplied.")
    return _result(
        request,
        {
            "invested_amount": invested,
            "ending_value": ending_value,
            "gain": gain,
            "xirr": xirr_value,
            "cagr": cagr,
            "twrr": twrr,
            "investment_years": years,
        },
        assumptions=["Cashflows are dated and use investor sign convention: contributions negative, withdrawals/value positive."],
        warnings=warnings,
        data_quality=["Uses only user-confirmed values; no market price is inferred."],
    )


def calculate_portfolio_risk(request: CalculationRequest) -> CalculationResult:
    """Compute portfolio risk statistics from daily portfolio and optional benchmark returns."""
    returns = np.asarray(request.inputs.get("returns", []), dtype=float)
    returns = returns[np.isfinite(returns)]
    confidence = min(max(_finite(request.inputs.get("confidence"), 0.95), 0.8), 0.999)
    risk_free = _finite(request.inputs.get("risk_free_rate"), 0.065)
    holdings = request.inputs.get("holdings", [])
    warnings: list[str] = []
    if returns.size < 20:
        warnings.append("Risk estimates are unstable with fewer than 20 daily observations.")
    if returns.size == 0:
        return _result(request, {}, warnings=["Daily return history is required for portfolio risk analysis."], data_quality=["No calculation performed."])

    mean_daily = float(np.mean(returns))
    vol_daily = float(np.std(returns, ddof=1)) if returns.size > 1 else 0.0
    annual_return = (1 + mean_daily) ** TRADING_DAYS - 1
    annual_vol = vol_daily * sqrt(TRADING_DAYS)
    downside = returns[returns < 0]
    downside_dev = float(np.std(downside, ddof=1) * sqrt(TRADING_DAYS)) if downside.size > 1 else 0.0
    sharpe = (annual_return - risk_free) / annual_vol if annual_vol else None
    sortino = (annual_return - risk_free) / downside_dev if downside_dev else None
    cumulative = np.cumprod(1 + returns)
    drawdowns = cumulative / np.maximum.accumulate(cumulative) - 1
    historical_var = -float(np.quantile(returns, 1 - confidence))
    historical_cvar = -float(np.mean(returns[returns <= np.quantile(returns, 1 - confidence)]))
    z = NORMAL.inv_cdf(1 - confidence)
    parametric_var = max(0.0, -(mean_daily + z * vol_daily))

    benchmark = np.asarray(request.inputs.get("benchmark_returns", []), dtype=float)
    benchmark = benchmark[np.isfinite(benchmark)]
    beta = None
    if benchmark.size == returns.size and benchmark.size > 1 and np.var(benchmark, ddof=1) > 0:
        beta = float(np.cov(returns, benchmark, ddof=1)[0, 1] / np.var(benchmark, ddof=1))
    elif request.inputs.get("benchmark_returns"):
        warnings.append("Benchmark return history must align with portfolio return history to calculate beta.")

    total_value = sum(max(0.0, _finite(row.get("value"))) for row in holdings)
    weights = [max(0.0, _finite(row.get("value"))) / total_value for row in holdings] if total_value else []
    top_weight = max(weights) if weights else None
    hhi = sum(weight**2 for weight in weights) if weights else None
    concentration = "Unavailable"
    if top_weight is not None:
        concentration = "High" if top_weight >= 0.35 or (hhi or 0) >= 0.25 else "Moderate" if top_weight >= 0.2 else "Diversified"

    return _result(
        request,
        {
            "annualized_return": annual_return,
            "annualized_volatility": annual_vol,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "beta": beta,
            "max_drawdown": float(np.min(drawdowns)),
            "historical_var_daily": historical_var,
            "historical_cvar_daily": historical_cvar,
            "parametric_var_daily": parametric_var,
            "confidence": confidence,
            "top_holding_weight": top_weight,
            "concentration_hhi": hhi,
            "concentration_assessment": concentration,
        },
        tables_and_chart_series={"cumulative_growth": cumulative.tolist(), "drawdowns": drawdowns.tolist()},
        assumptions=[f"Annualisation uses {TRADING_DAYS} trading days.", "Parametric VaR assumes normally distributed daily returns."],
        warnings=warnings,
        data_quality=[f"Calculated from {returns.size} daily portfolio return observations."],
    )


def calculate_goal_plan(request: CalculationRequest) -> CalculationResult:
    monthly_sip = max(0.0, _finite(request.inputs.get("monthly_sip")))
    step_up = max(0.0, _finite(request.inputs.get("annual_step_up"))) / 100
    years = max(1, int(_finite(request.inputs.get("years"), 15)))
    annual_return = _finite(request.inputs.get("annual_return"), 0.11)
    inflation = _finite(request.inputs.get("inflation"), 0.06)
    current_expenses = max(0.0, _finite(request.inputs.get("monthly_expenses")))
    nps_share = min(max(_finite(request.inputs.get("nps_share"), 0.0), 0.0), 1.0)
    monthly_rate = annual_return / 12
    corpus = 0.0
    rows = []
    for year in range(1, years + 1):
        contribution = monthly_sip * (1 + step_up) ** (year - 1)
        for _ in range(12):
            corpus = corpus * (1 + monthly_rate) + contribution
        rows.append({"year": year, "annual_contribution": contribution * 12, "corpus": corpus})
    future_monthly_expenses = current_expenses * (1 + inflation) ** years
    fire_corpus = future_monthly_expenses * 12 * 25
    nps_corpus = corpus * nps_share
    return _result(
        request,
        {
            "projected_corpus": corpus,
            "future_monthly_expenses": future_monthly_expenses,
            "indicative_fire_corpus": fire_corpus,
            "goal_gap": max(0.0, fire_corpus - corpus),
            "nps_allocated_corpus": nps_corpus,
            "non_nps_corpus": corpus - nps_corpus,
        },
        tables_and_chart_series={"yearly_projection": rows},
        assumptions=["Contributions are made at each month-end.", "FIRE corpus uses a 4% withdrawal-rate illustration; it is not a guarantee.", "NPS allocation is a planning split, not a statutory benefit calculation."],
        warnings=["Returns and inflation are assumptions. Test a lower-return and higher-inflation scenario before relying on the result."],
        data_quality=["No product recommendation or live scheme return is used."],
    )


def calculate_dcf(request: CalculationRequest) -> CalculationResult:
    fcff = max(0.0, _finite(request.inputs.get("fcff")))
    growth = _finite(request.inputs.get("growth"), 0.12)
    terminal_growth = _finite(request.inputs.get("terminal_growth"), 0.05)
    beta = max(0.0, _finite(request.inputs.get("beta"), 1.0))
    risk_free = _finite(request.inputs.get("risk_free"), 0.07)
    equity_premium = _finite(request.inputs.get("equity_premium"), 0.06)
    cost_debt = _finite(request.inputs.get("cost_debt"), 0.09)
    tax_rate = min(max(_finite(request.inputs.get("tax_rate"), 0.25), 0.0), 0.5)
    equity_weight = min(max(_finite(request.inputs.get("equity_weight"), 0.75), 0.01), 0.99)
    net_debt = _finite(request.inputs.get("net_debt"))
    shares = max(1.0, _finite(request.inputs.get("shares_outstanding"), 1.0))
    horizon = min(max(int(_finite(request.inputs.get("horizon_years"), 5)), 3), 10)
    cost_equity = risk_free + beta * equity_premium
    wacc = equity_weight * cost_equity + (1 - equity_weight) * cost_debt * (1 - tax_rate)
    rows, pv_sum, current = [], 0.0, fcff
    for year in range(1, horizon + 1):
        current *= 1 + growth
        discounted = current / (1 + wacc) ** year
        pv_sum += discounted
        rows.append({"year": year, "fcff": current, "present_value": discounted})
    warnings = []
    if wacc <= terminal_growth:
        warnings.append("WACC must be greater than terminal growth; terminal value has been withheld.")
        terminal_value = None
        enterprise = pv_sum
    else:
        terminal_value = current * (1 + terminal_growth) / (wacc - terminal_growth)
        enterprise = pv_sum + terminal_value / (1 + wacc) ** horizon
    equity_value = enterprise - net_debt
    return _result(
        request,
        {
            "cost_of_equity": cost_equity,
            "wacc": wacc,
            "terminal_value": terminal_value,
            "enterprise_value": enterprise,
            "equity_value": equity_value,
            "value_per_share": equity_value / shares,
        },
        tables_and_chart_series={"forecast": rows},
        assumptions=["Uses FCFF and Gordon-growth terminal value.", "Cost of equity uses CAPM inputs supplied by the user."],
        warnings=warnings + ["DCF output is highly sensitive to FCFF, WACC, and terminal growth."],
        data_quality=["All operating and capital-structure inputs are user supplied."],
    )


def calculate_capital_gains(request: CalculationRequest) -> CalculationResult:
    buy = max(0.0, _finite(request.inputs.get("purchase_value")))
    sell = max(0.0, _finite(request.inputs.get("sale_value")))
    costs = max(0.0, _finite(request.inputs.get("transaction_costs")))
    holding_months = max(0, int(_finite(request.inputs.get("holding_months"))))
    asset_type = str(request.inputs.get("asset_type", "listed_equity"))
    short_rate = max(0.0, _finite(request.inputs.get("short_rate"), 0.20))
    long_rate = max(0.0, _finite(request.inputs.get("long_rate"), 0.125))
    exemption = max(0.0, _finite(request.inputs.get("long_exemption"), 125000))
    long_term = holding_months >= 12 if asset_type == "listed_equity" else holding_months >= 24
    gain = sell - buy - costs
    taxable_gain = max(0.0, gain - exemption) if long_term and asset_type == "listed_equity" else max(0.0, gain)
    rate = long_rate if long_term else short_rate
    tax = taxable_gain * rate
    return _result(
        request,
        {
            "capital_gain": gain,
            "taxable_gain": taxable_gain,
            "estimated_tax": tax,
            "net_sale_proceeds": sell - costs - tax,
            "after_tax_return": (sell - costs - tax) / buy - 1 if buy else None,
            "long_term": long_term,
            "tax_rate_used": rate,
        },
        assumptions=[f"Tax-year assumptions: {request.tax_year or 'FY 2026-27 estimate'}.", "Tax rates and exemption are editable planning assumptions."],
        warnings=["This is an indicative capital-gains estimate, not a tax return or tax advice. Surcharge, cess, set-off, grandfathering, and special cases are excluded."],
        data_quality=["Uses user-selected asset type, holding period, and rate assumptions."],
    )


def _norm_cdf(value: float) -> float:
    return NORMAL.cdf(value)


def calculate_trade_plan(request: CalculationRequest) -> CalculationResult:
    capital = max(0.0, _finite(request.inputs.get("capital")))
    risk_pct = min(max(_finite(request.inputs.get("risk_percent"), 1.0) / 100, 0.0), 0.1)
    entry = max(0.0, _finite(request.inputs.get("entry")))
    stop = max(0.0, _finite(request.inputs.get("stop")))
    target = max(0.0, _finite(request.inputs.get("target")))
    leverage = max(1.0, _finite(request.inputs.get("leverage"), 1.0))
    charges_rate = max(0.0, _finite(request.inputs.get("charges_rate"), 0.001))
    per_unit_risk = abs(entry - stop)
    risk_budget = capital * risk_pct
    risk_quantity = int(risk_budget // per_unit_risk) if per_unit_risk else 0
    capital_quantity = int((capital * leverage) // entry) if entry else 0
    quantity = min(risk_quantity, capital_quantity)
    gross_exposure = quantity * entry
    charges = gross_exposure * charges_rate * 2
    reward_per_unit = abs(target - entry)
    risk_reward = reward_per_unit / per_unit_risk if per_unit_risk else None

    spot = max(0.0, _finite(request.inputs.get("option_spot")))
    strike = max(0.0, _finite(request.inputs.get("option_strike")))
    premium = max(0.0, _finite(request.inputs.get("option_premium")))
    volatility = max(0.0001, _finite(request.inputs.get("option_volatility"), 0.25))
    expiry_years = max(1 / 365, _finite(request.inputs.get("option_expiry_days"), 30) / 365)
    option_type = str(request.inputs.get("option_type", "call"))
    black_scholes = None
    if spot and strike:
        d1 = (log(spot / strike) + (0.065 + volatility**2 / 2) * expiry_years) / (volatility * sqrt(expiry_years))
        d2 = d1 - volatility * sqrt(expiry_years)
        call = spot * _norm_cdf(d1) - strike * exp(-0.065 * expiry_years) * _norm_cdf(d2)
        put = call - spot + strike * exp(-0.065 * expiry_years)
        black_scholes = call if option_type == "call" else put
    payoff = []
    if strike and premium:
        for underlying in np.linspace(strike * 0.75, strike * 1.25, 41):
            intrinsic = max(0.0, underlying - strike) if option_type == "call" else max(0.0, strike - underlying)
            payoff.append({"underlying": float(underlying), "pnl_per_unit": float(intrinsic - premium)})
    return _result(
        request,
        {
            "risk_budget": risk_budget,
            "position_quantity": quantity,
            "gross_exposure": gross_exposure,
            "estimated_round_trip_charges": charges,
            "risk_reward_ratio": risk_reward,
            "margin_estimate": gross_exposure / leverage,
            "black_scholes_value": black_scholes,
        },
        tables_and_chart_series={"option_payoff": payoff},
        assumptions=["Margin is an illustrative exposure/leverage estimate, not broker SPAN margin.", "Option pricing uses a European Black-Scholes illustration with a fixed 6.5% risk-free rate."],
        warnings=["Position size is capped by the stated risk budget and available leveraged capital.", "Option models and broker charges can differ materially from live execution."],
        data_quality=["No live broker margin, option chain, or execution data is used."],
    )
