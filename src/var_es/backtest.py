"""Out-of-sample rolling forecasts and 99% VaR exception diagnostics."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.special import xlog1py, xlogy
from scipy.stats import chi2

from .risk import FloatArray, RiskEstimate


@dataclass(frozen=True)
class CoverageResult:
    observations: int
    exceptions: int
    exception_rate: float
    kupiec_lr: float
    kupiec_p: float
    independence_lr: float
    independence_p: float
    conditional_lr: float
    conditional_p: float
    n00: int
    n01: int
    n10: int
    n11: int


@dataclass(frozen=True)
class TrafficLight:
    zone: str
    exceptions: int
    observations: int


def _hit_array(exceptions: ArrayLike) -> NDArray[np.int64]:
    hits = np.asarray(exceptions)
    if hits.ndim != 1 or not np.isin(hits, (False, True)).all():
        raise ValueError("Exceptions must be a one-dimensional sequence of booleans or 0/1")
    return hits.astype(np.int64)


def _bernoulli_log_likelihood(failures: int, hits: int, p: float) -> float:
    return float(xlog1py(failures, -p) + xlogy(hits, p))


def _fitted_log_likelihood(failures: int, hits: int) -> float:
    n = failures + hits
    if n == 0:
        return 0.0
    return _bernoulli_log_likelihood(failures, hits, hits / n)


def coverage_tests(exceptions: ArrayLike, p: float = 0.01) -> CoverageResult:
    """Run Kupiec POF and Christoffersen independence and joint tests.

    Kupiec uses all hits. The two Markov likelihood tests condition on the
    first hit and use the remaining sequence. Report asymptotic chi-squared
    p-values (1, 1 and 2 degrees of freedom), which have limited reliability
    for sparse exceptions or unobserved transition rows.
    """
    hits = _hit_array(exceptions)
    if not (0 < p < 1):
        raise ValueError("The nominal exception probability must lie between zero and one")
    n = len(hits)
    if n < 2:
        raise ValueError("At least two observations are required")
    count = int(hits.sum())
    no_hit = n - count
    kupiec = max(
        0.0,
        2 * (_fitted_log_likelihood(no_hit, count) - _bernoulli_log_likelihood(no_hit, count, p)),
    )

    transitions = 2 * hits[:-1] + hits[1:]
    n00, n01, n10, n11 = map(int, np.bincount(transitions, minlength=4))
    markov = _fitted_log_likelihood(n00, n01) + _fitted_log_likelihood(n10, n11)
    independent = _fitted_log_likelihood(n00 + n10, n01 + n11)
    nominal = _bernoulli_log_likelihood(n00 + n10, n01 + n11, p)
    independence = max(0.0, 2 * (markov - independent))
    conditional = max(0.0, 2 * (markov - nominal))

    return CoverageResult(
        n, count, count / n,
        kupiec, float(chi2.sf(kupiec, 1)),
        independence, float(chi2.sf(independence, 1)),
        conditional, float(chi2.sf(conditional, 2)),
        n00, n01, n10, n11,
    )


def traffic_light(exceptions: ArrayLike) -> TrafficLight:
    """Apply the Basel 99% VaR traffic light to the last 250 observations."""
    hits = _hit_array(exceptions)
    if len(hits) < 250:
        raise ValueError("Basel traffic light requires 250 exception observations")
    count = int(hits[-250:].sum())
    if count <= 4:
        zone = "green"
    elif count <= 9:
        zone = "yellow"
    else:
        zone = "red"
    return TrafficLight(zone, count, 250)


def rolling_backtest(
    shocks: pd.DataFrame,
    forecast: Callable[[pd.Timestamp, FloatArray], Mapping[str, RiskEstimate]],
    realized_pnl: Callable[[pd.Timestamp, FloatArray], float],
    *,
    window: int = 500,
    days: int = 750,
) -> pd.DataFrame:
    """Score the final days against forecasts fitted to strictly earlier shocks.

    The two callbacks receive the realization date. The forecast callback must
    use positions and market levels known before that date. The realized P&L
    callback receives that day's one-dimensional factor shock vector.
    """
    if window < 2 or days < 1 or len(shocks) < window + days:
        raise ValueError("Insufficient shocks for the requested training and backtest periods")
    if not isinstance(shocks.index, pd.DatetimeIndex):
        raise ValueError("The shock index must contain realization dates")
    if not shocks.index.is_unique or not shocks.index.is_monotonic_increasing:
        raise ValueError("Shock dates must be unique and in chronological order")

    values = shocks.to_numpy(dtype=float)
    if values.shape[1] < 1 or not np.isfinite(values).all():
        raise ValueError("Shocks must contain one or more finite factors")
    rows = []
    methods = None
    for i in range(len(values) - days, len(values)):
        date = shocks.index[i]
        estimates = forecast(date, values[i - window : i])
        if methods is None:
            methods = set(estimates)
        if not methods or set(estimates) != methods:
            raise ValueError("Forecasts must return the same nonempty set of methods each day")
        pnl = float(realized_pnl(date, values[i]))
        for method, estimate in estimates.items():
            if not np.isfinite((pnl, estimate.var, estimate.es)).all():
                raise ValueError("P&L and forecasts must be finite")
            rows.append((date, method, pnl, -pnl, estimate.var, estimate.es, -pnl > estimate.var))

    return pd.DataFrame.from_records(
        rows, columns=["date", "method", "pnl", "loss", "var", "es", "exception"]
    ).set_index(["date", "method"])
