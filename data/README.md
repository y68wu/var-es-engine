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
