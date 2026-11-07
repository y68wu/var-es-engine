"""Portfolio marks and full or Taylor revaluation under five market factors."""

from dataclasses import dataclass, fields, replace

import numpy as np
import pandas as pd

from .market import FACTOR_COLUMNS as FACTORS
from .pricing import FixedRateBond, gk_greeks, gk_price


CURVE_TENORS = (0.25, 1.0, 2.0, 5.0, 10.0)
MARK_COMPONENTS = ("equity", "goc_5y", "goc_10y", "fx_straddle", "cash")
MIN_VOL = 0.0001


@dataclass(frozen=True)
class MarketState:
    """One CAD market close, with yields and annualized FX volatility in decimals."""

    xiu: float
    usdcad: float
    cad_rate: float
    usd_rate: float
    y_0_25: float
    y_1: float
    y_2: float
    y_5: float
    y_10: float
    fx_vol: float

    @classmethod
    def from_row(cls, row):
        return cls(**{field.name: float(row[field.name]) for field in fields(cls)})

    @property
    def zero_rates(self):
        return np.array([self.y_0_25, self.y_1, self.y_2, self.y_5, self.y_10])


@dataclass(frozen=True)
class Portfolio:
    """Fixed holdings with explicit strike and remaining tenor at the risk date."""

    equity_units: float
    goc_5y: FixedRateBond
    goc_10y: FixedRateBond
    option_notional_usd: float
    option_strike: float
    option_maturity: float
    cash_cad: float

    @classmethod
    def from_market(cls, state: MarketState, *, capital_cad=10_000_000, option_notional_usd=15_000_000):
        """Size XIU/5Y/10Y at 50%/30%/20% and retain the straddle premium as cash."""
        rates = state.zero_rates
        book = cls(
            equity_units=0.50 * capital_cad / state.xiu,
            goc_5y=FixedRateBond.at_par(0.30 * capital_cad, 5.0, CURVE_TENORS, rates),
            goc_10y=FixedRateBond.at_par(0.20 * capital_cad, 10.0, CURVE_TENORS, rates),
            option_notional_usd=option_notional_usd,
            option_strike=state.usdcad,
            option_maturity=0.25,
            cash_cad=0.0,
        )
        return replace(book, cash_cad=-book.marks(state)["fx_straddle"])

    def restrike(self, state: MarketState):
        """Create a fresh spot-ATM 3M vintage for a dated risk snapshot."""
        return replace(self, option_strike=state.usdcad, option_maturity=0.25)

    def _remaining_option_time(self, elapsed):
        if not 0 <= elapsed <= self.option_maturity:
            raise ValueError("elapsed must be between the risk date and option expiry")
        return self.option_maturity - elapsed

    def _components(self, state, shocks, elapsed):
        maturity = self._remaining_option_time(elapsed)
        equity = state.xiu * np.exp(shocks[..., 0])
        fx = state.usdcad * np.exp(shocks[..., 3])
        vol = np.maximum(state.fx_vol + shocks[..., 4], MIN_VOL)
        rates = state.zero_rates + np.pad(
            shocks[..., 1:3], [(0, 0)] * (shocks.ndim - 1) + [(3, 0)]
        )
        # Only the 5Y and 10Y knots move under the five-factor model.
        call = gk_price(fx, self.option_strike, maturity, vol, state.cad_rate, state.usd_rate, "call")
        put = gk_price(fx, self.option_strike, maturity, vol, state.cad_rate, state.usd_rate, "put")
        return (
            self.equity_units * equity,
            self.goc_5y.value(CURVE_TENORS, rates, elapsed),
            self.goc_10y.value(CURVE_TENORS, rates, elapsed),
            -self.option_notional_usd * (call + put),
            self.cash_cad,
        )

    def marks(self, state: MarketState, elapsed=0.0):
        """Return CAD component marks and total for this vintage."""
        result = dict(zip(MARK_COMPONENTS, self._components(state, np.zeros(len(FACTORS)), elapsed)))
        result["total"] = sum(result.values())
        return result

    def value(self, state: MarketState, shocks=None, elapsed=0.0):
        """Return CAD market value for shocks with shape (..., 5)."""
        shocks = np.zeros(len(FACTORS)) if shocks is None else np.asarray(shocks, dtype=float)
        if shocks.ndim < 1 or shocks.shape[-1] != len(FACTORS):
            raise ValueError("last dimension of shocks must contain the five market factors")
        if not np.isfinite(shocks).all():
            raise ValueError("shocks must be finite")
        return sum(self._components(state, shocks, elapsed))

    def pnl(self, state: MarketState, shocks, elapsed=0.0):
        """CAD scenario P&L against the original state with strike and holdings fixed."""
        return self.value(state, shocks, elapsed) - self.value(state)

    def pnl_between(self, previous: MarketState, current: MarketState, elapsed=0.0):
        """Full-state revaluation for the held vintage, including carry-rate moves."""
        return self.value(current, elapsed=elapsed) - self.value(previous)

    def sensitivities(self, state: MarketState, elapsed=0.0):
        """Analytical gradient and Hessian in factor units at zero shock."""
        maturity = self._remaining_option_time(elapsed)
        gradient = np.zeros(len(FACTORS))
        hessian = np.zeros((len(FACTORS), len(FACTORS)))
        gradient[0] = hessian[0, 0] = self.equity_units * state.xiu
        for bond in (self.goc_5y, self.goc_10y):
            times, weights, discounted = bond.discounted_cashflows(CURVE_TENORS, state.zero_rates, elapsed)
            loadings = times[:, None] * weights[:, -2:]
            gradient[1:3] -= discounted @ loadings
            hessian[1:3, 1:3] += loadings.T @ (discounted[:, None] * loadings)

        vol = max(state.fx_vol, MIN_VOL)
        vol_active = float(state.fx_vol > MIN_VOL)
        for kind in ("call", "put"):
            greek = gk_greeks(state.usdcad, self.option_strike, maturity, vol, state.cad_rate, state.usd_rate, kind)
            size = -self.option_notional_usd
            log_delta = greek.delta * state.usdcad
            gradient[3] += size * log_delta
            gradient[4] += size * greek.vega * vol_active
            hessian[3, 3] += size * (greek.gamma * state.usdcad**2 + log_delta)
            hessian[3, 4] += size * greek.vanna * state.usdcad * vol_active
            hessian[4, 4] += size * greek.volga * vol_active
        hessian[4, 3] = hessian[3, 4]
        return gradient, hessian


def mark_to_market(market: pd.DataFrame, portfolio: Portfolio) -> pd.DataFrame:
    """Mark dated ATM risk snapshots; adjacent reset marks do not represent held P&L."""
    marks = []
    for row in market.itertuples(index=False):
        state = MarketState.from_row(row._asdict())
        marks.append(portfolio.restrike(state).marks(state))
    return pd.DataFrame(marks, index=market.index, columns=[*MARK_COMPONENTS, "total"])
