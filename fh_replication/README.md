# fh_replication — Fassas & Hourvouliades (2019) VIX Futures Curve Geometry

Replication of **Fassas & Hourvouliades (2019), "VIX Futures as a Market
Timing Indicator"** (JIFMIM 59, 21–36), extended from the paper's window to
the full CME VIX-futures history.

The paper's linear cross-sectional "curve geometry" model (their Eq. 1) is fit
**once per trading day** across all listed VIX futures contracts:

```
price(i,t) = α₀(t) + β₀(t) · TtM(i,t) + ε(i,t)
```

where `TtM` is time-to-maturity in years. The daily slope `β₀(t)` summarises
the term-structure shape — `β > 0` contango, `β < 0` backwardation — and the
intercept `α₀(t)` proxies short-end spot volatility. Days need at least 3
listed contracts to be fit.

This module has a dual role:

1. **Standalone replication** producing a daily results panel and three figures.
2. **Shared signal library**: `compute_vix_term_slope()` is *the* source of the
   VIX term-structure slope signal imported by all five modules of
   `experiment2 - Return Regression` and by `experiment3 - Nasdaq`
   (as `from fh_replication.fh_replication import compute_vix_term_slope`,
   with the repo root on `sys.path`).

## Files

| File | Role |
|---|---|
| `fh_replication.py` | The entire module: loader, daily cross-sectional OLS, slope wrapper, three plotting functions, `run_replication()` orchestrator, `__main__` entry |
| `output/` | Generated artifacts (see below) |

## Key functions (pipeline order)

| Function | What it does |
|---|---|
| `load_vix_futures_term_structure()` | Reads the two `VolatilityIndexFuture_*` parquets, filters `curve_group == "VX"`, computes `ttm_years` from last trade dates, drops expired rows → long DataFrame `[date, security, price, ttm_years]` |
| `fit_daily_cross_section(vx_df, min_contracts=3)` | The core model: one OLS per day → date-indexed panel with `alpha, beta, se_alpha, se_beta, t_alpha, t_beta, r2, r2_adj, n_contracts, residual_std` |
| `compute_vix_term_slope(vx_df, min_contracts=3)` | Thin public wrapper returning only the daily `beta` series (renamed `term_slope`) — the downstream API |
| `plot_main_results` / `plot_sample_fits` / `plot_summary_stats` | The three figures |
| `run_replication(data_dir, output_dir)` | Load → fit → print summary stats → write CSV → render figures |

## Outputs (`output/`)

| File | Content |
|---|---|
| `fh_daily_results.csv` | Daily panel of all cross-sectional fit statistics (git-ignored; regenerated on run) |
| `fh_main_results.png` | 3-panel time series: β ± 2·SE, daily R², t(β) with ±1.96/±2.576 significance bands; crisis periods shaded |
| `fh_sample_fits.png` | 2×2 price-vs-TtM scatters with fitted line on four representative dates (2008 crisis backwardation, 2014 calm contango, 2020-03 COVID, 2022 hiking cycle) |
| `fh_summary_stats.png` | Histograms + KDE of the daily β and R² distributions |

Console output includes β/R²/t(β) summary statistics and the
contango/backwardation frequency split.

## How to run

```bash
python fh_replication/fh_replication.py
```

No CLI arguments; paths are anchored to the file location, so any working
directory works. Reads `data/VolatilityIndexFuture_security_meta.parquet` and
`data/VolatilityIndexFuture_historical.parquet`.

Dependencies: `numpy`, `pandas`, `matplotlib`, `scipy`, `statsmodels`,
`pyarrow` (repo-root `requirements.txt`).
