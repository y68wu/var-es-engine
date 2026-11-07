"""European FX option and fixed-rate bond valuation in CAD."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.special import ndtr


OptionKind = Literal["call", "put"]


def _option_sign(kind: OptionKind) -> int:
    if kind not in ("call", "put"):
        raise ValueError("kind must be 'call' or 'put'")
    return 1 if kind == "call" else -1


def _gk_terms(spot, strike, maturity, vol, cad_rate, usd_rate):
    if not np.isfinite(strike) or strike <= 0:
        raise ValueError("strike must be positive and finite")
    if not np.isfinite(maturity) or maturity < 0:
        raise ValueError("maturity must be nonnegative and finite")
    spot, vol, cad_rate, usd_rate = np.broadcast_arrays(spot, vol, cad_rate, usd_rate)
    if not all(np.isfinite(x).all() for x in (spot, vol, cad_rate, usd_rate)):
        raise ValueError("spot, volatility and rates must be finite")
    if np.any(spot <= 0) or np.any(vol <= 0):
        raise ValueError("spot and volatility must be positive")
    if maturity == 0:
        return spot, vol, cad_rate, usd_rate, None, None
    sigma_t = vol * np.sqrt(maturity)
    d1 = (np.log(spot / strike) + (cad_rate - usd_rate) * maturity) / sigma_t + 0.5 * sigma_t
    return spot, vol, cad_rate, usd_rate, d1, d1 - sigma_t


def gk_price(spot, strike, maturity, vol, cad_rate, usd_rate, kind: OptionKind):
    """Price a European option in CAD per USD of notional, vectorized over market inputs.

    Spot and strike are CAD per USD. Rates are continuously compounded decimals,
    volatility is annualized, and maturity is in years.
    """
    sign = _option_sign(kind)
    spot, _, cad_rate, usd_rate, d1, d2 = _gk_terms(
        spot, strike, maturity, vol, cad_rate, usd_rate
    )
    if maturity == 0:
        return np.maximum(sign * (spot - strike), 0.0)
    discounted_spot = spot * np.exp(-usd_rate * maturity)
    discounted_strike = strike * np.exp(-cad_rate * maturity)
    return sign * (discounted_spot * ndtr(sign * d1) - discounted_strike * ndtr(sign * d2))


@dataclass(frozen=True)
class OptionGreeks:
    delta: float
    gamma: float
    vega: float
    theta: float
    vanna: float
    volga: float


def gk_greeks(spot, strike, maturity, vol, cad_rate, usd_rate, kind: OptionKind) -> OptionGreeks:
    """Return spot Greeks, vega per unit volatility, and calendar theta per year."""
    sign = _option_sign(kind)
    if maturity == 0:
        raise ValueError("Greeks require positive time to expiry")
    spot, vol, cad_rate, usd_rate, d1, d2 = _gk_terms(
        spot, strike, maturity, vol, cad_rate, usd_rate
    )
    sqrt_t = np.sqrt(maturity)
    df_usd = np.exp(-usd_rate * maturity)
    df_cad = np.exp(-cad_rate * maturity)
    density = np.exp(-0.5 * d1**2) / np.sqrt(2 * np.pi)
    delta = sign * df_usd * ndtr(sign * d1)
    gamma = df_usd * density / (spot * vol * sqrt_t)
    vega = spot * df_usd * density * sqrt_t
    theta = (
        -spot * df_usd * density * vol / (2 * sqrt_t)
        + usd_rate * spot * delta
        - sign * cad_rate * strike * df_cad * ndtr(sign * d2)
    )
    vanna = -df_usd * density * d2 / vol
    volga = vega * d1 * d2 / vol
    return OptionGreeks(delta, gamma, vega, theta, vanna, volga)


def interpolation_weights(times, tenors):
    """Linear interpolation of zero rates, with flat endpoint extrapolation."""
    return np.column_stack([np.interp(times, tenors, knot) for knot in np.eye(len(tenors))])


@dataclass(frozen=True)
class BondRisk:
    pv: float
    dv01: float
    duration: float
    convexity: float


@dataclass(frozen=True)
class FixedRateBond:
    """Synthetic bond with semiannual coupons and any initial short stub."""

    face: float
    maturity: float
    coupon: float

    def cashflows(self):
        if not np.isfinite(self.maturity) or self.maturity <= 0:
            raise ValueError("bond maturity must be positive and finite")
        times = np.sort(np.arange(self.maturity, 0, -0.5))
        coupons = self.face * self.coupon * np.diff(times, prepend=0.0)
        coupons[-1] += self.face
        return times, coupons

    @classmethod
    def at_par(cls, face, maturity, tenors, rates):
        times, _ = cls(face, maturity, 0.0).cashflows()
        discounts = np.exp(-(interpolation_weights(times, tenors) @ rates) * times)
        accruals = np.diff(times, prepend=0.0)
        coupon = (1 - discounts[-1]) / (discounts @ accruals)
        return cls(face, maturity, coupon)

    def discounted_cashflows(self, tenors, rates, elapsed=0.0):
        times, cashflows = self.cashflows()
        # Cash paid during the horizon stays in the P&L with zero reinvestment.
        remaining = np.maximum(times - elapsed, 0.0)
        weights = interpolation_weights(remaining, tenors)
        discount = np.exp(-(np.asarray(rates) @ weights.T) * remaining)
        return remaining, weights, cashflows * discount

    def value(self, tenors, rates, elapsed=0.0):
        """CAD value across one or more zero curves, including any paid coupons."""
        _, _, discounted = self.discounted_cashflows(tenors, rates, elapsed)
        return discounted.sum(axis=-1)

    def parallel_risk(self, tenors, rates) -> BondRisk:
        """Positive DV01 for a long bond and PV-normalized parallel convexity."""
        times, _, discounted = self.discounted_cashflows(tenors, rates)
        pv = discounted.sum()
        dollar_duration = discounted @ times
        return BondRisk(pv, dollar_duration * 1e-4, dollar_duration / pv, discounted @ times**2 / pv)
