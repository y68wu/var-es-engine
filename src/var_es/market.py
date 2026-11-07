"""Acquire, align and validate the public daily market inputs."""

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

SAMPLE_START = "2007-01-01"
SAMPLE_END = "2026-10-21"
DEFAULT_MARKET_PATH = Path("data/market.csv")
CURVE_COLUMNS = {
    "ZC025YR": "y_0_25",
    "ZC100YR": "y_1",
    "ZC200YR": "y_2",
    "ZC500YR": "y_5",
    "ZC1000YR": "y_10",
}
MARKET_COLUMNS = ("xiu", "usdcad", "cad_rate", "usd_rate", *CURVE_COLUMNS.values(), "fx_vol")
FACTOR_COLUMNS = ("equity_log_return", "y_5_change", "y_10_change", "fx_log_return", "fx_vol_change")
CURVE_URL = "https://www.bankofcanada.ca/stats/results/csv"
CORRA_URL = "https://www.bankofcanada.ca/valet/observations/AVG.INTWO/json"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
RATE_MAX_AGE = pd.Timedelta(days=7)
ONE_DAY = pd.Timedelta(days=1)


def ewma_volatility(
    spot: pd.Series, decay: float = 0.94, warmup: int = 30, sessions_per_year: int = 252
) -> pd.Series:
    """Annualize EWMA squared log returns, seeded with the first warmup returns."""
    if not 0 < decay < 1 or warmup < 1 or sessions_per_year < 1:
        raise ValueError("Require 0 < decay < 1, positive warmup and positive annualization")
    if not np.isfinite(spot).all() or (spot <= 0).any():
        raise ValueError("FX spot must contain finite positive observations")
    squared_returns = np.log(spot).diff().pow(2).dropna()
    if len(squared_returns) < warmup:
        return pd.Series(np.nan, index=spot.index, name="fx_vol")

    # Insert the initial variance at its observation date, then apply the recursive EWMA.
    variance_input = squared_returns.iloc[warmup - 1 :].copy()
    variance_input.iloc[0] = squared_returns.iloc[:warmup].mean()
    variance = variance_input.ewm(alpha=1 - decay, adjust=False).mean()
    return np.sqrt(sessions_per_year * variance).reindex(spot.index).rename("fx_vol")


def align_market(
    xiu: pd.Series,
    usdcad: pd.Series,
    curve: pd.DataFrame,
    corra: pd.Series,
    dgs3mo: pd.Series,
) -> pd.DataFrame:
    """Use joint spot/curve dates and latest strictly prior published-rate proxies.

    The two rate series must already be decimals, as must the BoC zero yields.
    """
    market = pd.concat(
        [xiu.rename("xiu"), usdcad.rename("usdcad"), curve[list(CURVE_COLUMNS.values())]],
        join="inner",
        axis=1,
    ).sort_index().dropna()

    for column, rates in (("cad_rate", corra), ("usd_rate", dgs3mo)):
        # Strict precedence on observation dates, including when markets use different calendars.
        prior = rates.dropna().sort_index()
        matched = prior.reindex(
            market.index - ONE_DAY, method="pad", tolerance=RATE_MAX_AGE - ONE_DAY
        )
        market[column] = matched.to_numpy()

    market = market.dropna()
    market["fx_vol"] = ewma_volatility(market["usdcad"])
    market = market.loc[:, list(MARKET_COLUMNS)].dropna().rename_axis("date")
    return _validate_market(market)


def _validate_market(market: pd.DataFrame) -> pd.DataFrame:
    missing = set(MARKET_COLUMNS).difference(market.columns)
    if missing:
        raise ValueError(f"Missing market columns: {', '.join(sorted(missing))}")
    if (
        market.empty
        or not isinstance(market.index, pd.DatetimeIndex)
        or market.index.hasnans
        or market.index.has_duplicates
        or not market.index.is_monotonic_increasing
    ):
        raise ValueError("Market data requires a nonempty, sorted, unique daily date index")
    selected = market.loc[:, list(MARKET_COLUMNS)]
    if not np.isfinite(selected).all().all():
        raise ValueError("Market data requires finite prices, rates, and volatility")
    if (selected[["xiu", "usdcad", "fx_vol"]] <= 0).any().any():
        raise ValueError("Market data requires positive XIU prices, FX spot and FX volatility")
    return selected


def load_market(path: str | Path | None = None) -> pd.DataFrame:
    """Load a previously downloaded market snapshot without a network request."""
    market_path = Path(path) if path is not None else DEFAULT_MARKET_PATH
    if not market_path.is_file():
        raise FileNotFoundError(f"{market_path} is missing. Run python -m var_es.market to download it.")
    market = pd.read_csv(market_path, parse_dates=["date"], index_col="date", float_precision="round_trip")
    return _validate_market(market)


def factor_changes(market: pd.DataFrame) -> pd.DataFrame:
    """Produce five simultaneous shocks indexed by the ending observation date."""
    market = _validate_market(market)
    return pd.DataFrame(
        {
            "equity_log_return": np.log(market["xiu"]).diff(),
            "y_5_change": market["y_5"].diff(),
            "y_10_change": market["y_10"].diff(),
            "fx_log_return": np.log(market["usdcad"]).diff(),
            "fx_vol_change": market["fx_vol"].diff(),
        },
        index=market.index,
        columns=list(FACTOR_COLUMNS),
    ).iloc[1:]


def _get(session: requests.Session, url: str, params: dict[str, str]) -> requests.Response:
    response = session.get(url, params=params, timeout=45)
    response.raise_for_status()
    return response


def _yahoo_series(
    yf, ticker: str, column: str, start: pd.Timestamp, end: pd.Timestamp
) -> pd.Series:
    # yfinance uses an exclusive end date. Strip the local timezone without changing trading dates.
    history = yf.Ticker(ticker).history(
        start=start.date().isoformat(),
        end=(end + ONE_DAY).date().isoformat(),
        auto_adjust=False,
        actions=False,
        timeout=30,
    )
    if history.empty or column not in history or history[column].dropna().empty:
        raise RuntimeError(f"Yahoo Finance returned no {column} for {ticker}. Try the download again.")
    index = history.index.tz_localize(None).normalize()
    return pd.Series(history[column].to_numpy(dtype=float), index=index)


def download_market(
    path: str | Path = DEFAULT_MARKET_PATH,
    start: str = SAMPLE_START,
    end: str = SAMPLE_END,
    metadata_path: str | Path | None = None,
) -> dict:
    """Retrieve a dated snapshot, save local inputs and record sample provenance."""
    import yfinance as yf

    start_date, end_date = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    if end_date <= start_date:
        raise ValueError("Download end date must be later than the start date")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(output.parent / "cache" / "yfinance"))
    first, last = start_date.date().isoformat(), end_date.date().isoformat()
    prior_week = (start_date - RATE_MAX_AGE).date().isoformat()

    xiu = _yahoo_series(yf, "XIU.TO", "Adj Close", start_date, end_date)
    usdcad = _yahoo_series(yf, "USDCAD=X", "Close", start_date, end_date)
    with requests.Session() as session:
        curve_csv = _get(
            session, CURVE_URL,
            {
                "lookupPage": "lookup_yield_curve.php", "startRange": "1986-01-01",
                "searchRange": "", "dFrom": first, "dTo": last,
            },
        )
        corra_json = _get(session, CORRA_URL, {"start_date": prior_week, "end_date": last})
        dgs3mo_csv = _get(session, FRED_URL, {"id": "DGS3MO", "cosd": prior_week, "coed": last})

    curve = pd.read_csv(
        BytesIO(curve_csv.content), skipinitialspace=True, na_values=["na"], parse_dates=["Date"],
        index_col="Date", usecols=["Date", *CURVE_COLUMNS],
    ).rename(columns=CURVE_COLUMNS)
    corra_rows = corra_json.json()["observations"]
    corra = pd.Series(
        [pd.to_numeric(row.get("AVG.INTWO", {}).get("v"), errors="coerce") for row in corra_rows],
        index=pd.to_datetime([row["d"] for row in corra_rows]),
        dtype=float,
    ) / 100
    fred = pd.read_csv(BytesIO(dgs3mo_csv.content), na_values=["."])
    dgs3mo = pd.Series(
        fred["DGS3MO"].to_numpy(dtype=float), index=pd.to_datetime(fred.iloc[:, 0])
    ) / 100

    market = align_market(xiu, usdcad, curve, corra, dgs3mo)
    csv_bytes = market.to_csv(float_format="%.12g", lineterminator="\n").encode("utf-8")
    output.write_bytes(csv_bytes)
    metadata = {
        "retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "requested_start": first,
        "requested_end_inclusive": last,
        "sample_first_date": market.index[0].date().isoformat(),
        "sample_last_date": market.index[-1].date().isoformat(),
        "sample_observations": len(market),
        "market_csv_sha256": hashlib.sha256(csv_bytes).hexdigest(),
        "software": {"pandas": pd.__version__, "yfinance": yf.__version__},
        "sources": {
            "xiu": "Yahoo Finance XIU.TO adjusted close via yfinance",
            "usdcad": "Yahoo Finance USDCAD=X close via yfinance",
            "goc_zero_coupon_curve": curve_csv.url,
            "corra": corra_json.url,
            "usd_3m_treasury": dgs3mo_csv.url,
        },
        "alignment": "Joint XIU, USD/CAD and GoC curve dates; CORRA and DGS3MO taken from strictly prior observations up to seven calendar days old.",
        "fx_volatility": {"decay": 0.94, "warmup_returns": 30, "sessions_per_year": 252},
    }
    provenance = Path(metadata_path) if metadata_path else output.with_name(f"{output.stem}_metadata.json")
    provenance.parent.mkdir(parents=True, exist_ok=True)
    provenance.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=SAMPLE_START)
    parser.add_argument("--end", default=SAMPLE_END, help="Inclusive requested end date")
    parser.add_argument("--output", type=Path, default=DEFAULT_MARKET_PATH)
    parser.add_argument("--metadata", type=Path, help="Override the output provenance JSON path")
    args = parser.parse_args()
    metadata = download_market(args.output, args.start, args.end, args.metadata)
    print(f"Saved {metadata['sample_observations']:,} sessions through {metadata['sample_last_date']} to {args.output}")


if __name__ == "__main__":
    main()
