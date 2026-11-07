"""External diagnostics work on dated data without downloading market feeds."""

import numpy as np
import pandas as pd
import pytest

from var_es.context import fx_validation, vix_regimes


def test_fx_comparison_aligns_common_dates_before_calculating_returns():
    dates = pd.date_range("2026-09-01", periods=6)
    boc = pd.Series([1.40, 1.41, 1.39, 1.37, 1.43, 1.45], index=dates)
    yahoo = boc * 1.002
    result = fx_validation(yahoo.drop(dates[1]), boc.drop(dates[3]))
    assert result["overlap_days"] == 4
    assert result["return_observations"] == 3
    assert result["return_correlation"] == pytest.approx(1)
    assert result["mean_abs_return_gap_bp"] == pytest.approx(0)
    assert result["median_abs_spot_gap_bp"] == pytest.approx(20)
    assert result["p95_abs_spot_gap_bp"] == pytest.approx(20)
    assert result["lag_return_observations"] == 1


def test_fx_lag_diagnostic_compares_identical_interior_sample():
    dates = pd.date_range("2026-09-01", periods=7)
    boc = pd.Series(np.exp([0.0, .01, -.03, .02, -.02, .05, .01]), index=dates)
    yahoo = boc.shift(1, fill_value=1.10)
    result = fx_validation(yahoo, boc)
    correlations = {row["yahoo_offset_sessions"]: row["correlation"]
                    for row in result["lag_correlations"]}
    assert result["lag_return_observations"] == 4
    assert correlations[1] == pytest.approx(1)
    assert correlations[0] < 1
    assert correlations[-1] < 1


def test_vix_strictly_precedes_date_and_excludes_stale_values():
    dates = pd.to_datetime(["2026-01-02", "2026-01-03", "2026-01-09", "2026-01-10"])
    vix = pd.Series([12, 20], index=pd.to_datetime(["2026-01-01", "2026-01-02"]))
    forecasts = pd.DataFrame({
        "date": dates, "method": "test", "var": [10, 20, 30, 40],
        "exception": [True, False, True, True],
    })
    result = vix_regimes(forecasts, vix)
    assert result["matched_days"] == 3
    assert result["unmatched_days"] == 1
    low, elevated = result["groups"]
    assert low["days"] == 1
    assert elevated["days"] == 2
    assert low["models"][0]["exception_rate"] == 1
    assert elevated["models"][0]["exception_rate"] == .5
    assert elevated["models"][0]["mean_var_99"] == 25


def test_vix_keeps_absent_regime_distinct_from_zero_observed_exceptions():
    dates = pd.to_datetime(["2026-01-02", "2026-01-03"])
    vix = pd.Series([12], index=pd.to_datetime(["2026-01-01"]))
    forecasts = pd.DataFrame({
        "date": dates, "method": "test", "var": [10, 20], "exception": [False, False],
    })
    low, elevated = vix_regimes(forecasts, vix)["groups"]
    assert low["models"][0]["exception_rate"] == 0
    assert elevated["models"][0]["exception_rate"] is None
