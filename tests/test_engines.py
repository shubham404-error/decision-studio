from datetime import date

import pytest

from core.contracts import CalculationRequest
from core.engines import (
    calculate_capital_gains,
    calculate_dcf,
    calculate_goal_plan,
    calculate_portfolio_risk,
    calculate_returns,
    calculate_trade_plan,
)


def test_returns_calculates_xirr_for_one_contribution_and_ending_value():
    request = CalculationRequest(
        "returns",
        {"cashflows": [{"date": date(2024, 1, 1), "amount": -100_000}], "ending_value": 121_000},
        as_of_date=date(2026, 1, 1),
    )
    result = calculate_returns(request)
    assert result.outputs["xirr"] == pytest.approx(0.10, abs=0.005)
    assert result.outputs["gain"] == 21_000


def test_portfolio_risk_identifies_concentration_and_loss_tail():
    request = CalculationRequest(
        "portfolio_risk",
        {
            "returns": [0.01, -0.02, 0.01, -0.03, 0.005] * 8,
            "confidence": 0.95,
            "holdings": [{"symbol": "ONE", "value": 80_000}, {"symbol": "TWO", "value": 20_000}],
        },
    )
    result = calculate_portfolio_risk(request)
    assert result.outputs["concentration_assessment"] == "High"
    assert result.outputs["historical_var_daily"] > 0
    assert result.outputs["max_drawdown"] < 0


def test_goal_plan_projects_a_positive_corpus_and_gap():
    result = calculate_goal_plan(CalculationRequest("goal", {"monthly_sip": 10_000, "annual_step_up": 10, "years": 20, "annual_return": 0.11, "inflation": 0.06, "monthly_expenses": 50_000, "nps_share": 0.2}))
    assert result.outputs["projected_corpus"] > 0
    assert result.outputs["future_monthly_expenses"] > 50_000
    assert len(result.tables_and_chart_series["yearly_projection"]) == 20


def test_dcf_with_valid_discount_rate_returns_value_per_share():
    result = calculate_dcf(CalculationRequest("dcf", {"fcff": 1000, "growth": 0.1, "terminal_growth": 0.05, "beta": 1, "risk_free": 0.07, "equity_premium": 0.06, "cost_debt": 0.09, "tax_rate": 0.25, "equity_weight": 0.75, "net_debt": 0, "shares_outstanding": 100, "horizon_years": 5}))
    assert result.outputs["wacc"] > 0
    assert result.outputs["terminal_value"] > 0
    assert result.outputs["value_per_share"] > 0


def test_capital_gains_uses_equity_long_term_exemption():
    result = calculate_capital_gains(CalculationRequest("tax", {"purchase_value": 500_000, "sale_value": 800_000, "transaction_costs": 0, "holding_months": 18, "asset_type": "listed_equity", "short_rate": 0.2, "long_rate": 0.125, "long_exemption": 125_000}, tax_year="FY 2026-27 estimate"))
    assert result.outputs["long_term"] is True
    assert result.outputs["taxable_gain"] == 175_000
    assert result.outputs["estimated_tax"] == pytest.approx(21_875)


def test_trade_plan_never_exceeds_risk_budget_quantity():
    result = calculate_trade_plan(CalculationRequest("trade", {"capital": 100_000, "risk_percent": 1, "entry": 1000, "stop": 950, "target": 1100, "leverage": 1, "charges_rate": 0.001, "option_spot": 1000, "option_strike": 1000, "option_premium": 30, "option_volatility": 0.25, "option_expiry_days": 30, "option_type": "call"}))
    assert result.outputs["position_quantity"] == 20
    assert result.outputs["black_scholes_value"] > 0
    assert len(result.tables_and_chart_series["option_payoff"]) == 41
