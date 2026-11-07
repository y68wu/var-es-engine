import numpy as np
import pandas as pd
import pytest

from var_es.backtest import coverage_tests, rolling_backtest, traffic_light
from var_es.risk import RiskEstimate


@pytest.mark.parametrize("exception", [False, True])
def test_coverage_likelihoods_are_finite_at_probability_boundaries(exception):
    result = coverage_tests([exception] * 250)
    realized_probability = 0.01 if exception else 0.99
    assert result.kupiec_lr == pytest.approx(-2 * 250 * np.log(realized_probability))
    assert result.conditional_lr == pytest.approx(-2 * 249 * np.log(realized_probability))
    assert result.independence_lr == 0.0
    assert result.independence_p == 1.0


def test_christoffersen_detects_clusters_without_changing_kupiec():
    scattered = np.zeros(200, dtype=bool)
    scattered[5::20] = True
    clustered = np.zeros(200, dtype=bool)
    clustered[70:80] = True
    first = coverage_tests(scattered, p=0.05)
    second = coverage_tests(clustered, p=0.05)
    assert first.kupiec_lr == second.kupiec_lr == 0
    assert (second.n00, second.n01, second.n10, second.n11) == (188, 1, 1, 9)
    assert first.independence_p > 0.05
    assert second.independence_p < 0.001
    assert second.conditional_p < 0.001


@pytest.mark.parametrize("count, zone", [(0, "green"), (4, "green"), (5, "yellow"),
                                          (9, "yellow"), (10, "red"), (250, "red")])
def test_basel_traffic_light_uses_only_the_final_250_hits(count, zone):
    hits = [True] * 50 + [False] * (250 - count) + [True] * count
    light = traffic_light(hits)
    assert (light.zone, light.exceptions, light.observations) == (zone, count, 250)


def test_rolling_forecast_excludes_realized_shock_and_requires_strict_exceedance():
    dates = pd.date_range("2024-01-01", periods=5)
    shocks = pd.DataFrame({"factor": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=dates)
    fitted_windows = {}

    def forecast(date, window):
        fitted_windows[date] = window[:, 0].tolist()
        return {"model": RiskEstimate(var=4.0, es=4.5)}

    def realized(date, shock):
        assert shock.shape == (1,)
        return -shock[0]

    results = rolling_backtest(shocks, forecast, realized, window=2, days=2)
    assert fitted_windows == {dates[3]: [2.0, 3.0], dates[4]: [3.0, 4.0]}
    assert results.index.names == ["date", "method"]
    assert results["exception"].tolist() == [False, True]
    assert results["loss"].tolist() == [4.0, 5.0]
