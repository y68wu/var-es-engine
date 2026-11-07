import numpy as np
import pandas as pd
import pytest

from var_es.market import CURVE_COLUMNS, FACTOR_COLUMNS, align_market, ewma_volatility, factor_changes


def test_ewma_warmup_then_causal_update():
    returns = np.array([0.01, -0.02, 0.04, -0.30])
    spot = pd.Series(np.exp(np.r_[0.0, returns.cumsum()]))
    result = ewma_volatility(spot, decay=0.94, warmup=2)
    initial_variance = (0.01**2 + 0.02**2) / 2

    assert result.iloc[:2].isna().all()
    assert result.iloc[2] == pytest.approx(np.sqrt(252 * initial_variance))
    assert result.iloc[3] == pytest.approx(np.sqrt(252 * (0.94 * initial_variance + 0.06 * 0.04**2)))
    pd.testing.assert_series_equal(result.iloc[:-1], ewma_volatility(spot.iloc[:-1], warmup=2))


def test_join_is_causal_for_carry_rates_and_does_not_fill_fx_or_curve():
    dates = pd.bdate_range("2024-01-02", periods=55)
    xiu = pd.Series(100 * np.exp(np.arange(len(dates)) / 1_000), index=dates)
    fx = pd.Series(1.35 * np.exp(np.arange(len(dates)) / 2_000), index=dates)
    curve = pd.DataFrame(0.03, index=dates, columns=list(CURVE_COLUMNS.values()))
    curve["y_5"] += np.arange(len(dates)) / 100_000
    corra = pd.Series(0.02, index=dates)
    corra.iloc[-1] = 0.04
    dgs3mo = pd.Series(0.05, index=dates[:-7])
    missing_fx, missing_curve = dates[34], dates[35]
    market = align_market(xiu, fx.drop(missing_fx), curve.drop(missing_curve), corra, dgs3mo)

    assert missing_fx not in market.index
    assert missing_curve not in market.index
    assert dates[-1] not in market.index  # USD rate is more than seven calendar days old.
    assert market["usd_rate"].iloc[-1] == 0.05
    full_rates = align_market(xiu, fx, curve, corra, corra)
    assert full_rates["cad_rate"].iloc[-1] == 0.02  # Same-date 0.04 is not yet available.

    changes = factor_changes(full_rates)
    assert tuple(changes.columns) == FACTOR_COLUMNS
    assert changes.index.equals(full_rates.index[1:])
    assert changes.iloc[-1].to_numpy() == pytest.approx([0.001, 0.00001, 0.0, 0.0005, 0.0])
