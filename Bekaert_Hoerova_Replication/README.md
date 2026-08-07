# Bekaert_Hoerova_Replication — Bekaert & Hoerova (2014) HAR-RV-VIX Baseline

Replication of **Bekaert & Hoerova (2014), "The VIX, the Variance Premium and
Stock Market Volatility"** (ECB WP 1675) — specifically **Model 8
(HAR-RV-VIX)**, the paper's best RV forecasting specification — plus an
extension comparing the VIX against the **SPX 1-month variance swap fair
strike (VS)** as the risk-neutral variance measure.

Two deliberate deviations from the paper:

1. **RV proxy** — daily squared front-month ES futures returns `(r·100)²`
   instead of 5-minute intraday RV (see `experiment1 - Intraday Variant/` for
   the intraday version).
2. **Risk-neutral leg** — both `VIX²/12` and `VS²/12` are estimated, to test
   whether the theoretically exact variance-swap strike improves on the VIX.

This directory doubles as the repo's **shared library**: `data_prep.py` and
`har_model.py` are imported by `experiment1 - HAR VRP Computation`,
`experiment1 - Intraday Variant`, and (indirectly)
`experiment2 - Predictive Return Regression and Timing Backtests`.

## Model

```
RV22_fwd(t+1..t+22) = c + α·X²(t)/12 + β^m·RV22(t) + β^w·RV5(t) + β^d·RV1(t) + ε
```

where `X` is VIX or VS; all variance quantities are in monthly %² units
(`RV1 = daily·22`, `RV5 = 5d mean·22`, `RV22 = 22d sum`). Estimated by OLS
with **44-lag Newey–West HAC** standard errors (2× the 22-day overlap of the
forward target). The variance risk premium is `VP = X²/12 − CV`, where `CV`
is the model's fitted conditional variance.

## Files

| File | Role |
|---|---|
| `data_prep.py` | **Library.** `load_sp500_returns()` (continuous ES front-month from the equity-futures parquets), `load_vix()` (VIX Index from `VolatilityIndexData.csv`), `load_variance_swap()` (SPX 1-month VS fair strike from `EquityIndexVarianceSwapData.csv`), `compute_rv_components()` (RV1/RV5/RV22), `build_panel()` (full VIX panel with forward target and 1-day-lagged predictors) |
| `har_model.py` | **Library.** `estimate_har()` (full-sample OLS + NW SEs, adj-R², IS RMSE), `out_of_sample_forecast()` (train/test split, Mincer–Zarnowitz R², OOS RMSE/MAE/MAPE) — both take an `xcol` argument (default `VIX2_lag`) selecting the implied-variance predictor column — `_nw_se()` (NW HAC SE helper reused repo-wide), `NW_LAGS = 44` |
| `run_replication.py` | **Entry point (writes the figures).** Runs the VIX formulation on the full history (OOS split 2005-07-15, matching the paper's 75% split) and the VS formulation on the VS-available sample (Nov 2008+, OOS split at the 75th-percentile row). Renders one summary PNG per formulation |
| `run_comparison.py` | **Entry point (stdout only).** Head-to-head VIX vs VS on the common VS-restricted sample: in-sample and OOS estimates (via `har_model` with `xcol='VS2_lag'` / `'VIX2_lag'` on a panel built from `data_prep` loaders), Diebold–Mariano equal-MSE test, forecast-encompassing regression, combined two-predictor model, VP descriptive stats, and VP → forward-SPX-return predictability regressions at 1/3/12-month horizons |
| `output/` | Generated figures (see below) |

The two entry points are independent (neither consumes the other's output)
and can be run in any order. Note their samples differ: `run_replication.py`
estimates the VIX model on the **full VIX history (1990+)**, while
`run_comparison.py` restricts both models to the **VS-available sample
(2008+)** so they are directly comparable.

Implementation note: `run_replication.py` builds its VS panel with the VS
predictor stored in a column named `VIX2_lag` so that `har_model.estimate_har`
runs unchanged — the column name is structural, the values are VS²/12.

## Outputs

| File | Content |
|---|---|
| `output/bh_vix_summary.png` | HAR-RV-VIX: top panel = fitted VRP time series (positive/negative shading, long-run mean ± 1σ, crisis shading); bottom panel = results table (IS adj-R²/RMSE, OOS MZ-R²/RMSE, per-variable β / NW-SE / t-stat with the paper's coefficients for reference) |
| `output/bh_vs_summary.png` | Same layout for HAR-RV-VS |

`run_comparison.py` prints all of its results (no files written).

## How to run

```bash
python Bekaert_Hoerova_Replication/run_replication.py   # writes the two PNGs
python Bekaert_Hoerova_Replication/run_comparison.py    # prints the VIX-vs-VS comparison
```

Paths are anchored to the script location (`Path(__file__).parent.parent /
"data"`), so any working directory works. Requires `data/`:
`EquityFuture_security_meta.parquet`, `EquityFuture_historical.parquet`,
`VolatilityIndexData.csv`, `EquityIndexVarianceSwapData.csv`.

Dependencies: `numpy`, `pandas` (≥ 2.2 — the month-end resample alias `'ME'`
is used), `matplotlib`, `statsmodels`, `pyarrow`. See the repo-root
`requirements.txt`.
