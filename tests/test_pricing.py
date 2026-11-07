from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from var_es.portfolio import CURVE_TENORS, FACTORS, MarketState, Portfolio, mark_to_market
from var_es.pricing import FixedRateBond, gk_greeks, gk_price


@pytest.fixture
def market():
    return MarketState(
        xiu=38.0,
        usdcad=1.36,
        cad_rate=0.036,
        usd_rate=0.042,
        y_0_25=0.038,
        y_1=0.035,
        y_2=0.033,
        y_5=0.032,
        y_10=0.036,
        fx_vol=0.09,
    )


@pytest.mark.parametrize("kind", ["call", "put"])
def test_gk_greeks_match_finite_differences(kind):
    spot, strike, t, vol, cad, usd = 1.36, 1.35, 0.25, 0.09, 0.036, 0.042
    greek = gk_greeks(spot, strike, t, vol, cad, usd, kind)

    def value(s=spot, maturity=t, sigma=vol):
        return gk_price(s, strike, maturity, sigma, cad, usd, kind)

    h = 1e-4
    assert greek.delta == pytest.approx((value(spot + h) - value(spot - h)) / (2 * h), rel=1e-6)
    assert greek.gamma == pytest.approx((value(spot + h) + value(spot - h) - 2 * value()) / h**2, rel=1e-5)
    assert greek.vega == pytest.approx((value(sigma=vol + h) - value(sigma=vol - h)) / (2 * h), rel=1e-6)
    theta_fd = (value(maturity=t - h) - value(maturity=t + h)) / (2 * h)
    assert greek.theta == pytest.approx(theta_fd, rel=1e-6)


def test_gk_put_call_parity_and_expiry():
    spots = np.array([1.20, 1.35, 1.50])
    strike, t, vol, cad, usd = 1.35, 0.25, 0.09, 0.036, 0.042
    call = gk_price(spots, strike, t, vol, cad, usd, "call")
    put = gk_price(spots, strike, t, vol, cad, usd, "put")
    np.testing.assert_allclose(call - put, spots * np.exp(-usd * t) - strike * np.exp(-cad * t))
    np.testing.assert_allclose(gk_price(spots, strike, 0.0, vol, cad, usd, "call"), [0.0, 0.0, 0.15])


def test_bond_at_par_and_parallel_risk(market):
    rates = market.zero_rates
    bond = FixedRateBond.at_par(2_000_000, 10.0, CURVE_TENORS, rates)
    risk = bond.parallel_risk(CURVE_TENORS, rates)
    up = bond.value(CURVE_TENORS, rates + 0.0001)
    down = bond.value(CURVE_TENORS, rates - 0.0001)
    assert risk.pv == pytest.approx(bond.face)
    assert risk.dv01 == pytest.approx((down - up) / 2, rel=1e-6)
    assert risk.convexity == pytest.approx((up + down - 2 * risk.pv) / (risk.pv * 0.0001**2), rel=1e-6)
    assert down > risk.pv > up


def test_bond_uses_term_specific_discounts():
    tenors = (0.5, 1.0)
    rates = np.array([0.01, 0.06])
    bond = FixedRateBond(100.0, 1.0, 0.05)
    expected = 2.5 * np.exp(-0.01 * 0.5) + 102.5 * np.exp(-0.06)
    assert bond.value(tenors, rates) == pytest.approx(expected)


def test_portfolio_vector_revaluation_preserves_the_held_vintage(market):
    book = Portfolio.from_market(market)
    assert book.value(market) == pytest.approx(10_000_000)
    shock = np.array([0.03, -0.003, 0.001, 0.02, 0.01])
    changed = replace(
        market,
        xiu=market.xiu * np.exp(shock[0]),
        y_5=market.y_5 + shock[1],
        y_10=market.y_10 + shock[2],
        usdcad=market.usdcad * np.exp(shock[3]),
        fx_vol=market.fx_vol + shock[4],
    )
    elapsed = 3 / 365
    pnl = book.pnl(market, np.stack([np.zeros_like(shock), shock]), elapsed)
    assert pnl[0] == pytest.approx(book.pnl_between(market, market, elapsed))
    assert pnl[1] == pytest.approx(book.pnl_between(market, changed, elapsed))
    assert book.option_strike == market.usdcad
    assert book.restrike(changed).option_strike == changed.usdcad
    assert book.marks(changed)["fx_straddle"] != book.restrike(changed).marks(changed)["fx_straddle"]
    with pytest.raises(ValueError, match="five market factors"):
        book.value(market, shocks=np.zeros(4))


def test_portfolio_factor_gradient_and_hessian_match_finite_differences(market):
    book = Portfolio.from_market(market)
    gradient, hessian = book.sensitivities(market)
    h = 2e-5
    steps = h * np.eye(len(FACTORS))
    zero_pnl = book.pnl(market, np.zeros(len(FACTORS)))
    up, down = book.pnl(market, steps), book.pnl(market, -steps)
    np.testing.assert_allclose(gradient, (up - down) / (2 * h), rtol=2e-6, atol=1e-4)
    np.testing.assert_allclose(np.diag(hessian), (up + down - 2 * zero_pnl) / h**2, rtol=2e-4)
    for i, j in [(1, 2), (3, 4)]:
        x, y = steps[i], steps[j]
        mixed_fd = (book.pnl(market, x + y) - book.pnl(market, x - y)
                    - book.pnl(market, -x + y) + book.pnl(market, -x - y)) / (4 * h**2)
        assert hessian[i, j] == pytest.approx(mixed_fd, rel=3e-4, abs=2)
    assert hessian[3, 3] < 0
    np.testing.assert_array_equal(hessian, hessian.T)


def test_daily_marks_use_fixed_sizing_and_reset_only_option_vintage(market):
    changed = replace(market, xiu=market.xiu * 1.02, usdcad=market.usdcad * 1.02)
    frame = pd.DataFrame([vars(market), vars(changed)], index=pd.date_range("2025-01-02", periods=2))
    book = Portfolio.from_market(market)
    marks = mark_to_market(frame, book)
    assert marks.index.equals(frame.index)
    assert marks.iloc[1].to_dict() == pytest.approx(book.restrike(changed).marks(changed))
    assert marks.iloc[1]["equity"] == pytest.approx(5_100_000)
