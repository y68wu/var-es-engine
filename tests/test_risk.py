import numpy as np
import pytest
from scipy.stats import ncx2, norm

from var_es.risk import (
    METHODS,
    _cornish_fisher_risk,
    _delta_gamma_cumulants,
    estimate_risk,
    make_mc_draws,
)


def test_empirical_expected_shortfall_uses_fractional_tail_mass():
    shocks = -np.arange(1.0, 6.0)[:, None]
    estimates = estimate_risk(
        shocks,
        lambda scenarios: scenarios[:, 0],
        [1.0],
        [[0.0]],
        var_level=0.8,
        es_level=0.7,
    )
    assert tuple(estimates) == METHODS
    assert estimates["historical"].var == 4.0
    assert estimates["historical"].es == pytest.approx((5.0 + 0.5 * 4.0) / 1.5)


def test_linear_delta_gamma_and_delta_normal_match_closed_form_with_carry():
    shocks = np.array([[-0.03], [0.005], [0.01], [0.02]])
    gradient = 10_000.0
    carry = -3.0
    estimates = estimate_risk(
        shocks,
        lambda scenarios: carry + gradient * scenarios[:, 0],
        [gradient],
        [[0.0]],
        base_pnl=carry,
    )
    mean_loss = -carry - gradient * shocks.mean()
    std_loss = gradient * shocks.std(ddof=1)
    var = mean_loss + std_loss * norm.ppf(0.99)
    es = mean_loss + std_loss * norm.pdf(norm.ppf(0.975)) / 0.025
    for method in ("delta_normal", "delta_gamma"):
        assert estimates[method].var == pytest.approx(var)
        assert estimates[method].es == pytest.approx(es)


def test_quadratic_cumulants_with_nonzero_shock_mean_match_noncentral_chi_square():
    mean = 0.4
    sigma = 0.3
    gradient = -2.0
    hessian = -4.0
    carry = 0.5
    cumulants = _delta_gamma_cumulants(
        np.array([mean]),
        np.array([[sigma]]),
        np.array([gradient]),
        np.array([[hessian]]),
        carry,
    )
    # Loss = -0.5 + 2X + 2X^2 = -1 + 2(X+0.5)^2.
    scale = 2 * sigma**2
    nc = ((mean + 0.5) / sigma) ** 2
    expected_mean, variance, skew, excess = ncx2.stats(1, nc, moments="mvsk")
    assert cumulants == pytest.approx(
        (-1 + scale * expected_mean, scale**2 * variance,
         scale**3 * variance**1.5 * skew, scale**4 * variance**2 * excess)
    )


def test_nonmonotone_cornish_fisher_is_rearranged_for_upper_tail_measures():
    # Moments of chi-square(1), whose CF polynomial has a nonmonotone region.
    cumulants = (1.0, 2.0, 8.0, 48.0)
    lower = _cornish_fisher_risk(cumulants, 0.10, 0.10)
    middle = _cornish_fisher_risk(cumulants, 0.20, 0.20)
    upper = _cornish_fisher_risk(cumulants, 0.99, 0.99)
    assert lower.var <= middle.var <= upper.var
    assert lower.es <= middle.es <= upper.es
    assert upper.es >= upper.var


def test_constant_shocks_and_singular_covariance_preserve_deterministic_loss():
    gradient = np.array([1.0, -1.0])
    for factor in (np.repeat(0.1, 60), np.linspace(-1.0, 1.0, 60)):
        shocks = np.column_stack([factor, factor])
        estimates = estimate_risk(
            shocks,
            lambda scenarios: 5.0 + scenarios @ gradient,
            gradient,
            np.zeros((2, 2)),
            base_pnl=5.0,
        )
        for estimate in estimates.values():
            assert estimate.var == pytest.approx(-5.0)
            assert estimate.es == pytest.approx(-5.0)


def test_student_t_uses_covariance_matching_and_common_radial_scale():
    draws = make_mc_draws(100_000, 3, df=6, seed=17)
    np.testing.assert_array_equal(draws.student_t, make_mc_draws(100_000, 3, 6, 17).student_t)
    np.testing.assert_allclose(np.cov(draws.student_t, rowvar=False), np.eye(3), atol=0.03)
    scale = draws.student_t / draws.normal
    np.testing.assert_allclose(scale[:, 0], scale[:, 2])
    assert np.quantile(draws.student_t[:, 0], 0.99) > np.quantile(draws.normal[:, 0], 0.99)


def test_short_gamma_risk_is_missed_by_delta_normal_at_zero_delta():
    shocks = np.linspace(-0.04, 0.04, 500)[:, None]
    estimates = estimate_risk(
        shocks,
        lambda scenarios: -1_000_000 * scenarios[:, 0] ** 2,
        [0.0],
        [[-2_000_000]],
    )
    assert estimates["delta_normal"].var == 0.0
    for method in ("historical", "delta_gamma", "mc_normal", "mc_t"):
        assert estimates[method].var > 0.0
        assert estimates[method].es > 0.0
