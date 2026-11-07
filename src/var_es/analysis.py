"""Run the empirical case study and produce tables and charts."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from .backtest import coverage_tests, rolling_backtest, traffic_light
from .market import factor_changes, load_market
from .portfolio import CURVE_TENORS, MIN_VOL, MarketState, Portfolio, mark_to_market
from .pricing import gk_greeks
from .risk import estimate_risk, make_mc_draws

WINDOW = 500
BACKTEST_DAYS = 750
PATHS = 10_000
SEED = 20261107
DF = 6.0
VAR_LEVEL = 0.99
ES_LEVEL = 0.975
METHOD_LABELS = {
    "historical": "Historical",
    "delta_normal": "Delta-normal",
    "delta_gamma": "Delta-gamma",
    "mc_normal": "MC Gaussian",
    "mc_t": "MC Student-t",
}
HISTORICAL_PERIODS = (
    ("Global financial crisis", "2008-09-01", "2008-11-30"),
    ("COVID-19 selloff", "2020-03-01", "2020-03-31"),
    ("2022 rate hikes", "2022-01-01", "2022-12-31"),
)


def stress_tests(market: pd.DataFrame, book: Portfolio) -> pd.DataFrame:
    """Replay historical paths and apply standalone and combined shocks."""
    state = MarketState.from_row(market.iloc[-1])
    base_marks = book.marks(state)
    results = []

    for name, start, end in HISTORICAL_PERIODS:
        prior = market.loc[market.index < start]
        period = market.loc[start:end]
        if prior.empty or period.empty:
            raise ValueError(f"Market data does not cover {name}")

        # Re-anchor relative spot moves and absolute yield/volatility moves
        # onto the same report-date positions for every episode.
        changes = period - prior.iloc[-1]
        replay = changes + market.iloc[-1]
        for column in ("xiu", "usdcad"):
            replay[column] = period[column] / prior.iloc[-1][column] * market.iloc[-1][column]
        replay["fx_vol"] = replay["fx_vol"].clip(lower=MIN_VOL)
        path = pd.DataFrame(
            [book.marks(MarketState.from_row(row)) for _, row in replay.iterrows()],
            index=replay.index,
        ).subtract(base_marks)
        worst_date = path["total"].idxmin()
        for measure, date in (("period end", path.index[-1]), ("worst close", worst_date)):
            results.append(
                {"scenario": name, "measure": measure, "anchor": prior.index[-1],
                 "scenario_date": date, **path.loc[date].to_dict()}
            )

    rate_fields = ("cad_rate", "y_0_25", "y_1", "y_2", "y_5", "y_10")
    rate_up = {field: getattr(state, field) + 0.02 for field in rate_fields}
    # A 10% decline in the USD value of one CAD implies USD/CAD divided by 0.90.
    cad_down = {"usdcad": state.usdcad / 0.90}
    vol_up = {"fx_vol": state.fx_vol * 2}
    hypothetical = (
        ("CAD curve +200 bp", rate_up),
        ("CAD depreciation 10%", cad_down),
        ("FX volatility doubles", vol_up),
        ("Combined hypothetical", {**rate_up, **cad_down, **vol_up}),
    )
    for name, changes in hypothetical:
        marks = book.marks(replace(state, **changes))
        results.append({
            "scenario": name, "measure": "instantaneous", "anchor": market.index[-1],
            "scenario_date": pd.NaT,
            **{leg: marks[leg] - base_marks[leg] for leg in marks},
        })
    return pd.DataFrame(results)


def approximation_curve(book: Portfolio, state: MarketState) -> pd.DataFrame:
    """Compare local approximations with exact pricing across FX-only shocks."""
    moves = np.linspace(-0.15, 0.15, 301)
    shocks = np.zeros((len(moves), 5))
    shocks[:, 3] = np.log1p(moves)
    gradient, hessian = book.sensitivities(state)
    linear = shocks @ gradient
    quadratic = linear + 0.5 * np.einsum("ij,jk,ik->i", shocks, hessian, shocks)
    exact = book.pnl(state, shocks)
    return pd.DataFrame({
        "fx_spot_move": moves,
        "full_revaluation": exact,
        "delta_pnl": linear,
        "delta_gamma_pnl": quadratic,
        "delta_error": linear - exact,
        "delta_gamma_error": quadratic - exact,
    })


def position_sensitivities(book: Portfolio, state: MarketState) -> pd.DataFrame:
    """Report position-sized option Greeks and parallel and key-rate bond risk."""
    rows = []
    for kind in ("call", "put"):
        greeks = gk_greeks(
            state.usdcad, book.option_strike, book.option_maturity,
            state.fx_vol, state.cad_rate, state.usd_rate, kind,
        )
        for metric, scale, units in (
            ("delta", 1, "CAD per unit of USD/CAD spot"),
            ("gamma", 1, "CAD per squared unit of USD/CAD spot"),
            ("vega", 0.01, "CAD per volatility point"),
            ("theta", 1 / 365, "CAD per calendar day"),
        ):
            rows.append((f"short FX {kind}", metric,
                         -book.option_notional_usd * getattr(greeks, metric) * scale, units))
    for label, bond in (("GoC 5Y", book.goc_5y), ("GoC 10Y", book.goc_10y)):
        risk = bond.parallel_risk(CURVE_TENORS, state.zero_rates)
        times, weights, discounted = bond.discounted_cashflows(CURVE_TENORS, state.zero_rates)
        key_rates = (discounted * times) @ weights * 1e-4
        rows.extend([
            (label, "parallel DV01", risk.dv01, "CAD per bp"),
            (label, "5Y key-rate DV01", key_rates[-2], "CAD per bp"),
            (label, "10Y key-rate DV01", key_rates[-1], "CAD per bp"),
            (label, "parallel convexity", risk.convexity, "years squared"),
        ])
    return pd.DataFrame(rows, columns=["position", "measure", "value", "units"])


def study(market: pd.DataFrame) -> dict:
    """Estimate current risk, run a strict out-of-sample backtest and stress the book."""
    changes = factor_changes(market)
    if len(changes) < WINDOW + BACKTEST_DAYS:
        raise ValueError(f"Study requires at least {WINDOW + BACKTEST_DAYS} aligned daily changes")
    states = {date: MarketState.from_row(row) for date, row in market.iterrows()}
    previous = {date: states[prev] for prev, date in zip(market.index[:-1], market.index[1:])}
    latest = states[market.index[-1]]
    reference = Portfolio.from_market(latest)
    draws = make_mc_draws(PATHS, changes.shape[1], df=DF, seed=SEED)

    def forecast(book, state, window):
        gradient, hessian = book.sensitivities(state)
        return estimate_risk(
            window, lambda shocks: book.pnl(state, shocks), gradient, hessian,
            paths=PATHS, df=DF, var_level=VAR_LEVEL, es_level=ES_LEVEL, draws=draws,
        )

    latest_window = changes.iloc[-WINDOW:].to_numpy()
    current = forecast(reference, latest, latest_window)
    comparison = []
    for label, book in (
        ("with short straddle", reference),
        ("without options", Portfolio.from_market(latest, option_notional_usd=0)),
    ):
        gradient, hessian = book.sensitivities(latest)
        linear = estimate_risk(
            latest_window, lambda shocks: shocks @ gradient, gradient, hessian, draws=draws
        )["mc_normal"]
        full = current["mc_normal"] if book is reference else forecast(
            book, latest, latest_window
        )["mc_normal"]
        comparison.append({
            "portfolio": label, "linear_var_99": linear.var, "full_var_99": full.var,
            "linear_es_97_5": linear.es, "full_es_97_5": full.es,
        })
    # Use only information from the first forecast date to size the backtest book.
    first_realization = changes.index[-BACKTEST_DAYS]
    backtest_book = Portfolio.from_market(previous[first_realization])

    def historical_forecast(date, window):
        state = previous[date]
        return forecast(backtest_book.restrike(state), state, window)

    def realized(date, shock):
        state = previous[date]
        return float(backtest_book.restrike(state).pnl(state, shock))

    backtest = rolling_backtest(
        changes, historical_forecast, realized, window=WINDOW, days=BACKTEST_DAYS
    )
    diagnostics = []
    for method in METHOD_LABELS:
        hits = backtest.xs(method, level="method")["exception"].to_numpy()
        coverage = coverage_tests(hits, p=1 - VAR_LEVEL)
        traffic = traffic_light(hits)
        diagnostics.append({
            "method": method, **asdict(coverage),
            "traffic_last_250": traffic.zone, "exceptions_last_250": traffic.exceptions,
        })
    snapshot = pd.DataFrame.from_dict(
        {method: {"var_99": risk.var, "es_97_5": risk.es} for method, risk in current.items()},
        orient="index",
    ).rename_axis("method")
    daily_marks = mark_to_market(market.loc[first_realization:], backtest_book)
    prior_dates = market.index[-BACKTEST_DAYS - 1:-1]
    daily_marks["held_1d_pnl_with_carry"] = [
        backtest_book.restrike(states[prior_date]).pnl_between(
            states[prior_date], states[date], elapsed=(date - prior_date).days / 365
        )
        for prior_date, date in zip(prior_dates, daily_marks.index)
    ]
    return {
        "snapshot": snapshot,
        "backtest": backtest,
        "diagnostics": pd.DataFrame(diagnostics).set_index("method"),
        "stress": stress_tests(market, reference),
        "curve": approximation_curve(reference, latest),
        "marks": pd.Series(reference.marks(latest), name="cad"),
        "daily_marks": daily_marks,
        "comparison": pd.DataFrame(comparison).set_index("portfolio"),
        "sensitivities": position_sensitivities(reference, latest),
    }


def write_results(results: dict, output: Path, figures: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    local = output / "local"
    local.mkdir(exist_ok=True)
    for name in ("snapshot", "diagnostics", "marks", "comparison"):
        results[name].to_csv(output / f"{name}.csv", float_format="%.8g")
    for name in ("stress", "sensitivities"):
        results[name].to_csv(output / f"{name}.csv", index=False, float_format="%.8g")
    for name in ("backtest", "curve", "daily_marks"):
        results[name].to_csv(local / f"{name}.csv", float_format="%.10g")
    plot_results(results["curve"], results["diagnostics"], figures)


def plot_results(curve: pd.DataFrame, diagnostics: pd.DataFrame, figures: Path) -> None:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, PercentFormatter

    ink, muted, grid = "#14263a", "#4b637a", "#dbe5ec"
    teal, red, blue = "#007c78", "#b33a32", "#4864ae"
    with plt.rc_context({
        "font.family": "DejaVu Sans", "font.size": 10, "text.color": ink,
        "axes.labelcolor": muted, "xtick.color": muted, "ytick.color": muted,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.spines.left": False, "axes.spines.bottom": False, "axes.axisbelow": True,
        "xtick.major.size": 0, "ytick.major.size": 0,
    }):
        fig, (left, right) = plt.subplots(1, 2, figsize=(11.5, 4.1),
                                         gridspec_kw={"width_ratios": (1.2, 1)})
        fig.subplots_adjust(left=0.085, right=0.975, bottom=0.19, top=0.74, wspace=0.51)
        fig.text(0.085, 0.93, "Short FX gamma changes the loss profile", fontsize=19, weight="bold")
        fig.text(0.085, 0.855, "CAD 10M reference NAV  /  USD/CAD short 3M straddle", color=muted)
        left.set_title("FX-only shock · P&L approximation error", loc="left", weight="bold", pad=12)
        left.plot(curve["fx_spot_move"], curve["delta_error"] / 1e6,
                  color=red, linewidth=2.3, label="Delta only")
        left.plot(curve["fx_spot_move"], curve["delta_gamma_error"] / 1e6,
                  color=teal, linewidth=2.3, label="Delta-gamma")
        left.axhline(0, color=muted, linewidth=0.8)
        left.grid(axis="y", color=grid)
        left.set_xlabel("USD/CAD spot shock")
        left.set_ylabel("Approx. P&L minus full revaluation (CAD m)")
        left.xaxis.set_major_formatter(PercentFormatter(1))
        left.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.1f}"))
        left.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.34), ncol=2)

        counts = diagnostics.loc[list(METHOD_LABELS), "exceptions"]
        positions = np.arange(len(counts))
        right.set_title("99% VaR exceptions · 750 observations", loc="left", weight="bold", pad=12)
        right.barh(positions, counts, height=0.53, color=[blue, red, teal, blue, blue])
        right.set_yticks(positions, METHOD_LABELS.values())
        right.invert_yaxis()
        right.set_xlim(0, max(9, counts.max() * 1.23))
        right.xaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=5))
        right.grid(axis="x", color=grid)
        right.axvline(7.5, color=muted, linewidth=1, linestyle="--")
        for i, count in enumerate(counts):
            right.text(count + 0.22, i, str(count), va="center", weight="bold")
        right.set_xlabel("Dashed line marks 7.5 expected exceptions")
        fig.savefig(figures / "risk-dashboard.png", dpi=170, facecolor="white")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market", type=Path, default=Path("data/market.csv"))
    parser.add_argument("--output", type=Path, default=Path("reports"))
    parser.add_argument("--figures", type=Path, default=Path("figures"))
    args = parser.parse_args()
    market = load_market(args.market)
    results = study(market)
    write_results(results, args.output, args.figures)
    backtest_dates = results["backtest"].index.get_level_values("date")
    print(f"Market as of {market.index[-1]:%Y-%m-%d}")
    print(f"Backtest {backtest_dates.min():%Y-%m-%d} to {backtest_dates.max():%Y-%m-%d}")
    print(results["snapshot"].round(0).to_string())
    print(results["diagnostics"][["exceptions", "exceptions_last_250", "traffic_last_250"]].to_string())


if __name__ == "__main__":
    main()
