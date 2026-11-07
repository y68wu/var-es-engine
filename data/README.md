# Market inputs

The reported run requested 2007-01-01 through 2026-10-21, inclusive. The aligned panel contains 4,888 observations from 2007-02-13 through 2026-10-21 after the volatility warmup. The [sample metadata](market_metadata.json) records the download URLs, versions and the SHA-256 of the local panel.

From the repository root, after installing the package, run:

```bash
python -m var_es.market --start 2007-01-01 --end 2026-10-21 --output data/market.csv
```

The downloader writes the input panel and its metadata. The market CSV and client cache stay local under `data/`. Yahoo Finance price history is subject to its terms of use. Source corrections and later ETF distributions can revise historical adjusted closes, so a later download can produce a different hash. [yfinance project documentation](https://github.com/ranaroussi/yfinance)

| Input | Provider and units |
| --- | --- |
| XIU | [Yahoo Finance](https://finance.yahoo.com/quote/XIU.TO/history/), CAD adjusted close for `XIU.TO`. Adjustments reflect splits and distributions, so the series acts as a total-return proxy. |
| USD/CAD | [Yahoo Finance](https://finance.yahoo.com/quote/USDCAD%3DX/history/), `USDCAD=X` daily close in CAD per USD. |
| GoC zero curve | [Bank of Canada](https://www.bankofcanada.ca/rates/interest-rates/bond-yield-curves/), official zero-coupon curve at 0.25, 1, 2, 5 and 10 years in decimal continuously compounded yield. [Curve methodology](https://www.bis.org/publications/paper-25-zero-coupon-yield-curves-technical-documentation.pdf#page=22) |
| CAD carry proxy | [Bank of Canada Valet](https://www.bankofcanada.ca/rates/interest-rates/corra/), CORRA series `AVG.INTWO` converted from percent to decimal. |
| USD carry proxy | [Federal Reserve Board via FRED](https://fred.stlouisfed.org/series/DGS3MO), 3-month U.S. Treasury constant-maturity yield `DGS3MO` converted from percent to decimal. |
| FX volatility proxy | EWMA of squared USD/CAD log returns on the aligned dates, seeded with the mean of the first 30 squared returns. The daily variance uses 0.94 persistence and is annualized with 252 sessions. |

The panel uses dates with observed XIU, FX and GoC curve data. Each carry rate uses its latest strictly earlier observation, up to seven calendar days old. Close times differ across feeds, and successive joint observations can span holidays. The Bank of Canada publishes the zero-coupon curves with a lag of about two weeks, so this is a retrospective exercise using curves by observation date. CORRA and the U.S. Treasury bill yield approximate the flat 3-month pricing inputs.

The five factors are XIU and USD/CAD log returns and decimal changes in the GoC 5Y and 10Y yields and annualized FX volatility. Each change is indexed by the ending date.

## Independent market context

After running `cad-var-es`, use `python -m var_es.context` to generate [the contextual diagnostics](../reports/context.json). The command downloads the two series below and saves derived statistics, source URLs and response hashes. The original feed responses stay in memory during processing.

| Series | How it is used |
| --- | --- |
| Bank of Canada Valet `FXUSDCAD` | Indicative CAD per USD daily average, compared with Yahoo `USDCAD=X` close over dates present in both feeds and the main panel. [Daily exchange rates](https://www.bankofcanada.ca/rates/exchange/daily-exchange-rates/), [publication methodology](https://www.bankofcanada.ca/rates/exchange/background-information-on-foreign-exchange-rates/) |
| Cboe `VIXCLS` via FRED | Daily closing index level. Assign the latest observation dated strictly before each realization, within seven calendar days, to the below-20 or 20-and-above group. [Cboe Volatility Index via FRED](https://fred.stlouisfed.org/series/VIXCLS) |

The FX check compares spot levels and log returns across common observation dates. The median absolute spot gap is about 18 bp across 2,432 dates. Same-date return correlation is about 0.37, and it rises to about 0.75 when the Yahoo return is labeled one shared session later. The offset check uses an identical interior sample for each offset and illustrates sensitivity to quote timing and date labels. The risk model retains its original feed alignment.

The VIX split assigns all 750 realization dates. Delta-normal breached 6 of 623 times (0.96%) below 20 and 6 of 127 times (4.72%) at 20 or above. The threshold is illustrative. VIX is a U.S. equity volatility indicator used to describe outcomes after the backtest. The split leaves the pricing inputs, model estimation windows and forecasts as originally generated.
