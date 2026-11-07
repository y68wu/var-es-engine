"""One-day VaR and expected shortfall of a CAD P&L distribution."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import ndtri

FloatArray = NDArray[np.float64]
METHODS = ("historical", "delta_normal", "delta_gamma", "mc_normal", "mc_t")

# Equal probability midpoints for rearranging a nonmonotone CF quantile curve.
_CF_GRID_SIZE = 32_768
_CF_NORMAL_GRID = ndtri((np.arange(_CF_GRID_SIZE) + 0.5) / _CF_GRID_SIZE)


@dataclass(frozen=True)
class RiskEstimate:
    """Upper-tail measures of loss, with positive values denoting risk."""

    var: float
    es: float


@dataclass(frozen=True)
class MCDraws:
    """Standardized innovations to reuse across valuations."""

    normal: FloatArray
    student_t: FloatArray


def make_mc_draws(
    paths: int, dimensions: int, df: float = 6.0, seed: int = 0
) -> MCDraws:
    """Draw standard normal and unit-covariance multivariate t innovations.

    The two families share Gaussian draws for comparison. Each Student-t path
    uses a single chi-squared draw for all factors, preserving joint tails.
    """
    if paths < 2 or dimensions < 1 or not np.isfinite(df) or df <= 2:
        raise ValueError("Require at least two paths, one factor and finite df > 2")
    rng = np.random.default_rng(seed)
    normal = rng.standard_normal((paths, dimensions))
    scale = np.sqrt((df - 2) / rng.chisquare(df, size=paths))
    return MCDraws(normal, normal * scale[:, None])


def _empirical_risk(
    losses: FloatArray, var_level: float, es_level: float
) -> RiskEstimate:
    ordered = np.sort(losses)
    n = ordered.size
    var = ordered[int(np.ceil(var_level * n)) - 1]

    # Integrate the empirical quantile over [es_level, 1]. The first
    # included order statistic can have only a fraction of its usual mass.
    tail_start = es_level * n
    boundary = int(tail_start)
    fraction = boundary + 1 - tail_start
    es = (fraction * ordered[boundary] + ordered[boundary + 1 :].sum()) / (
        n - tail_start
    )
    return RiskEstimate(float(var), float(es))


def _normal_tail_mean(level: float) -> float:
    z = ndtri(level)
    return float(np.exp(-0.5 * z**2 - 0.5 * np.log(2 * np.pi)) / (1 - level))


def _covariance_factor(covariance: FloatArray) -> FloatArray:
    try:
        return np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError:
        # A zero-variance or collinear factor gives a semidefinite covariance.
        # The eigenfactor preserves null directions without adding artificial risk.
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        tolerance = np.finfo(float).eps * len(eigenvalues) * eigenvalues[-1]
        return eigenvectors * np.sqrt(np.where(eigenvalues > tolerance, eigenvalues, 0))


def _delta_gamma_cumulants(
    mean: FloatArray,
    factor: FloatArray,
    gradient: FloatArray,
    hessian: FloatArray,
    base_pnl: float,
) -> tuple[float, float, float, float]:
    """Cumulants of loss = constant + linear'Z + Z'quadratic Z."""
    quadratic = -0.5 * (factor.T @ hessian @ factor)
    linear = -factor.T @ (gradient + hessian @ mean)
    constant = -base_pnl - gradient @ mean - 0.5 * (mean @ hessian @ mean)
    quadratic_sq = quadratic @ quadratic
    quadratic_linear = quadratic @ linear
    return (
        float(constant + np.trace(quadratic)),
        float(linear @ linear + 2 * np.trace(quadratic_sq)),
        float(6 * linear @ quadratic_linear + 8 * np.trace(quadratic_sq @ quadratic)),
        float(48 * (quadratic_linear @ quadratic_linear + np.sum(quadratic_sq**2))),
    )


def _cornish_fisher_risk(
    cumulants: tuple[float, float, float, float], var_level: float, es_level: float
) -> RiskEstimate:
    mean, variance, third, fourth = cumulants
    if variance == 0:
        return RiskEstimate(mean, mean)
    std = np.sqrt(variance)
    skew = third / std**3
    excess = fourth / variance**2

    # Q(p) = mean + std * polynomial(Phi^-1(p)).
    cubic = excess / 24 - skew**2 / 18
    quadratic = skew / 6
    linear = 1 - excess / 8 + 5 * skew**2 / 36
    constant = -quadratic
    increasing = (
        cubic > 0 and linear >= quadratic**2 / (3 * cubic)
    ) or (cubic == quadratic == 0 and linear >= 0)
    if not increasing:
        # Increasing rearrangement approximated on equal probability midpoints.
        # Both VaR and ES then come from the same valid quantile curve.
        losses = mean + std * np.polyval(
            (cubic, quadratic, linear, constant), _CF_NORMAL_GRID
        )
        return _empirical_risk(losses, var_level, es_level)

    z_var = ndtri(var_level)
    z_es = ndtri(es_level)
    var = mean + std * np.polyval((cubic, quadratic, linear, constant), z_var)
    # Conditional normal moments E[Z], E[Z^2], E[Z^3] given Z > z_es.
    tail_mean = _normal_tail_mean(es_level)
    es = mean + std * (
        cubic * (z_es**2 + 2) * tail_mean
        + quadratic * (1 + z_es * tail_mean)
        + linear * tail_mean
        + constant
    )
    return RiskEstimate(float(var), float(es))


def estimate_risk(
    shocks: ArrayLike,
    full_revalue: Callable[[FloatArray], FloatArray],
    gradient: ArrayLike,
    hessian: ArrayLike,
    *,
    paths: int = 10_000,
    df: float = 6.0,
    seed: int = 0,
    var_level: float = 0.99,
    es_level: float = 0.975,
    base_pnl: float = 0.0,
    draws: MCDraws | None = None,
) -> dict[str, RiskEstimate]:
    """Estimate four models of one-day risk, including two MC innovations.

    Each row is a joint observed factor shock. Derivatives are of P&L in the
    same units and factor order. The vectorized repricer returns one CAD P&L
    per row, with profits positive. VaR and ES are reported in CAD losses.

    Delta-normal, delta-gamma and both MC variants share the sample shock
    mean and covariance. Delta-gamma applies Cornish–Fisher to the first four
    loss cumulants. MC uses correlated normal or covariance-matched Student-t
    innovations followed by full repricing. Pass pre-generated draws to reuse
    the same innovations across forecasts, overriding paths, df and seed.

    The base P&L shifts both Taylor models. Set it to the repricer's P&L at
    zero shock if that model includes deterministic carry.
    """
    shocks = np.asarray(shocks, dtype=float)
    gradient = np.asarray(gradient, dtype=float)
    hessian = np.asarray(hessian, dtype=float)
    if shocks.ndim != 2 or shocks.shape[0] < 2 or shocks.shape[1] < 1:
        raise ValueError("Shocks must contain at least two observations of one or more factors")
    observations, dimensions = shocks.shape
    if gradient.shape != (dimensions,) or hessian.shape != (dimensions, dimensions):
        raise ValueError("Gradient and Hessian must match the factor dimension")
    if not all(np.isfinite(array).all() for array in (shocks, gradient, hessian, base_pnl)):
        raise ValueError("Shocks, derivatives and base P&L must be finite")
    if not np.allclose(hessian, hessian.T, atol=1e-10, rtol=1e-10):
        raise ValueError("The Hessian must be symmetric")
    if not (0 < var_level < 1 and 0 < es_level < 1):
        raise ValueError("Confidence levels must lie strictly between zero and one")

    # Center relative to an observation so a constant factor stays exactly
    # constant after floating-point averaging.
    relative = shocks - shocks[0]
    relative_mean = relative.mean(axis=0)
    mean = shocks[0] + relative_mean
    centered = relative - relative_mean
    covariance = centered.T @ centered / (observations - 1)
    factor = _covariance_factor(covariance)

    if draws is None:
        draws = make_mc_draws(paths, dimensions, df, seed)
    if (
        draws.normal.ndim != 2
        or draws.normal.shape != draws.student_t.shape
        or draws.normal.shape[0] < 2
        or draws.normal.shape[1] != dimensions
        or not np.isfinite(draws.normal).all()
        or not np.isfinite(draws.student_t).all()
    ):
        raise ValueError("Monte Carlo draws must be finite and match the factor dimension")

    def full_risk(scenarios: FloatArray) -> RiskEstimate:
        pnl = np.asarray(full_revalue(scenarios), dtype=float)
        if pnl.shape != (len(scenarios),) or not np.isfinite(pnl).all():
            raise ValueError("The repricer must return one finite P&L per scenario")
        return _empirical_risk(-pnl, var_level, es_level)

    delta_mean = -base_pnl - gradient @ mean
    delta_std = np.linalg.norm(factor.T @ gradient)
    delta_normal = RiskEstimate(
        float(delta_mean + delta_std * ndtri(var_level)),
        float(delta_mean + delta_std * _normal_tail_mean(es_level)),
    )
    cumulants = _delta_gamma_cumulants(mean, factor, gradient, hessian, base_pnl)
    return {
        "historical": full_risk(shocks),
        "delta_normal": delta_normal,
        "delta_gamma": _cornish_fisher_risk(cumulants, var_level, es_level),
        "mc_normal": full_risk(mean + draws.normal @ factor.T),
        "mc_t": full_risk(mean + draws.student_t @ factor.T),
    }
