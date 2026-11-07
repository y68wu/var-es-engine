"""Compare independent public feeds and break out backtest results by prior VIX."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .market import DEFAULT_MARKET_PATH, FRED_URL, ONE_DAY, load_market

FX_URL = "https://www.bankofcanada.ca/valet/observations/FXUSDCAD/json"
FX_SERIES_START = pd.Timestamp("2017-01-01")
VIX_THRESHOLD = 20
VIX_MAX_AGE = pd.Timedelta(days=7)


def _iso_date(date: pd.Timestamp) -> str:
    return date.date().isoformat()


def _clean_series(series: pd.Series) -> pd.Series:
    clean = series.dropna().sort_index().astype(float)
    if not isinstance(clean.index, pd.DatetimeIndex) or clean.index.hasnans or not clean.index.is_unique:
        raise ValueError("An external series requires unique observation dates")
    if not np.isfinite(clean).all() or (clean <= 0).any():
        raise ValueError("External quotes and index closes must be finite and positive")
    return clean


def fx_validation(yahoo: pd.Series, boc: pd.Series) -> dict:
    """Compare price levels and log returns on the same intersection of dates."""
    paired = pd.concat(
        [_clean_series(yahoo).rename("yahoo"), _clean_series(boc).rename("boc")],
        axis=1,
        join="inner",
    ).sort_index()
    if len(paired) < 3:
        raise ValueError("FX feed comparison needs at least three common observations")

    log_returns = np.log(paired).diff().iloc[1:]
    if log_returns.std().eq(0).any():
        raise ValueError("FX return correlation requires variation in both feeds")
    # Compare all offsets on the same interior dates. A lag is descriptive of
    # the vendor labels and is never used to move a forecast input.
    lagged = pd.DataFrame({
        "boc": log_returns["boc"],
        "previous": log_returns["yahoo"].shift(1),
        "same": log_returns["yahoo"],
        "next": log_returns["yahoo"].shift(-1),
    }).dropna()
    lag_correlations = []
    for offset, column in ((-1, "previous"), (0, "same"), (1, "next")):
        correlation = lagged["boc"].corr(lagged[column]) if len(lagged) > 1 else None
        lag_correlations.append({
            "yahoo_offset_sessions": offset,
            "correlation": float(correlation) if correlation is not None and np.isfinite(correlation) else None,
        })
    abs_return_gap_bp = (log_returns["yahoo"] - log_returns["boc"]).abs() * 10_000
    abs_spot_gap_bp = (paired["yahoo"] / paired["boc"] - 1).abs() * 10_000
    return {
        "start_date": _iso_date(paired.index[0]),
        "end_date": _iso_date(paired.index[-1]),
        "overlap_days": len(paired),
        "return_observations": len(log_returns),
        "return_correlation": float(log_returns["yahoo"].corr(log_returns["boc"])),
        "mean_abs_return_gap_bp": float(abs_return_gap_bp.mean()),
        "median_abs_spot_gap_bp": float(abs_spot_gap_bp.median()),
        "p95_abs_spot_gap_bp": float(abs_spot_gap_bp.quantile(0.95)),
        "lag_return_observations": len(lagged),
        "lag_correlations": lag_correlations,
    }


def vix_regimes(backtest: pd.DataFrame, vix: pd.Series) -> dict:
    """Assign a strictly earlier VIX close, then summarize exception rates."""
    days = pd.DatetimeIndex(backtest["date"].unique()).sort_values()
    methods = backtest["method"].unique()
    if (
        days.empty
        or days.hasnans
        or backtest["method"].isna().any()
        or backtest.duplicated(["date", "method"]).any()
        or len(backtest) != len(days) * len(methods)
    ):
        raise ValueError("The backtest requires one forecast per method and realization date")
    if not np.isfinite(backtest["var"]).all() or not backtest["exception"].isin([True, False]).all():
        raise ValueError("The backtest requires finite VaR values and Boolean exception flags")

    prior_vix = _clean_series(vix).reindex(
        days - ONE_DAY, method="pad", tolerance=VIX_MAX_AGE - ONE_DAY
    )
    below = prior_vix < VIX_THRESHOLD
    elevated = prior_vix >= VIX_THRESHOLD
    groups = []
    for regime_id, label, mask in (
        ("below_20", "VIX below 20", below),
        ("at_least_20", "VIX 20 or higher", elevated),
    ):
        regime_dates = days[mask.to_numpy()]
        results = backtest[backtest["date"].isin(regime_dates)]
        grouped = results.groupby("method").agg(
            exceptions=("exception", "sum"), mean_var_99=("var", "mean")
        ).reindex(methods)
        count = len(regime_dates)
        groups.append({
            "id": regime_id,
            "label": label,
            "days": count,
            "models": [{
                "method": method,
                "exceptions": int(row["exceptions"]) if count else 0,
                "exception_rate": float(row["exceptions"] / count) if count else None,
                "mean_var_99": float(row["mean_var_99"]) if count else None,
            } for method, row in grouped.iterrows()],
        })

    return {
        "threshold": VIX_THRESHOLD,
        "max_age_days": VIX_MAX_AGE.days,
        "start_date": _iso_date(days[0]),
        "end_date": _iso_date(days[-1]),
        "matched_days": int(prior_vix.notna().sum()),
        "unmatched_days": int(prior_vix.isna().sum()),
        "groups": groups,
    }


def download_context(market: pd.DataFrame, backtest: pd.DataFrame) -> dict:
    """Download two public reference feeds and publish derived diagnostics."""
    market_first = max(market.index.min(), FX_SERIES_START)
    market_last = market.index.max()
    backtest_first = backtest["date"].min()
    backtest_last = backtest["date"].max()
    with requests.Session() as session:
        fx_response = session.get(FX_URL, params={
            "start_date": _iso_date(market_first),
            "end_date": _iso_date(market_last),
        }, timeout=45)
        fx_response.raise_for_status()
        vix_response = session.get(FRED_URL, params={
            "id": "VIXCLS", "cosd": _iso_date(backtest_first - VIX_MAX_AGE),
            "coed": _iso_date(backtest_last - ONE_DAY),
        }, timeout=45)
        vix_response.raise_for_status()

    observations = fx_response.json()["observations"]
    boc = pd.Series(
        [pd.to_numeric(row.get("FXUSDCAD", {}).get("v"), errors="coerce") for row in observations],
        index=pd.to_datetime([row["d"] for row in observations]),
        dtype=float,
    )
    vix_table = pd.read_csv(BytesIO(vix_response.content), na_values=["."])
    vix = pd.Series(
        vix_table["VIXCLS"].to_numpy(dtype=float), index=pd.to_datetime(vix_table.iloc[:, 0])
    )
    return {
        "retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sources": [
            {
                "id": "FXUSDCAD", "provider": "Bank of Canada Valet",
                "label": "USD/CAD indicative daily average",
                "url": "https://www.bankofcanada.ca/rates/exchange/daily-exchange-rates/",
                "download_url": fx_response.url,
                "response_sha256": hashlib.sha256(fx_response.content).hexdigest(),
            },
            {
                "id": "VIXCLS", "provider": "Cboe via FRED",
                "label": "Cboe Volatility Index daily close",
                "url": "https://fred.stlouisfed.org/series/VIXCLS",
                "download_url": vix_response.url,
                "response_sha256": hashlib.sha256(vix_response.content).hexdigest(),
            },
        ],
        "fx_validation": fx_validation(market["usdcad"], boc),
        "vix_regimes": vix_regimes(backtest, vix),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market", type=Path, default=DEFAULT_MARKET_PATH)
    parser.add_argument("--backtest", type=Path, default=Path("reports/local/backtest.csv"))
    parser.add_argument("--output", type=Path, default=Path("reports/context.json"))
    args = parser.parse_args()

    market = load_market(args.market)
    backtest = pd.read_csv(args.backtest, parse_dates=["date"], dtype={"exception": "boolean"})
    result = download_context(market, backtest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    fx = result["fx_validation"]
    regimes = result["vix_regimes"]
    print(f"Saved FX check over {fx['overlap_days']:,} observations and "
          f"VIX diagnostics over {regimes['matched_days']:,} backtest days to {args.output}")


if __name__ == "__main__":
    main()
