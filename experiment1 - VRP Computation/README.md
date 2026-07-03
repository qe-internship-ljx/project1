# experiment1 — VRP Computation

Production-grade daily **variance risk premium (VRP)** extraction, extending
the `bh_replication` Bekaert & Hoerova (2014) baseline with day-by-day
re-estimated HAR forecasts. This experiment produces the VRP time series that
`experiment2 - Return Regression` (and `experiment3 - Nasdaq`) consume.

## Purpose

The BH replication estimates the HAR model **once on the full sample**, so
its fitted VRP uses information that was not available in real time — fine
for validating the paper, unusable as a trading signal. This experiment's
purpose is to turn the academic VRP into a **point-in-time signal**: on every
trading day the HAR forecast is re-estimated using only data available up to
that day (with a 22-day gap so the last training label is fully realized),
exactly as a live system would have computed it. Along the way it answers
three questions that determine whether the signal is trustworthy downstream:
(1) does HAR forecast accuracy survive genuine out-of-sample, day-by-day
re-estimation (vs a naive martingale baseline)? (2) how sensitive is the VRP
to the estimation scheme (1000-day rolling vs expanding window) and to the
implied-variance source (VIX² vs SPX variance swaps)? (3) are the HAR
coefficients stable enough over time for the signal to mean the same thing
across regimes? The resulting daily `VP` series is the organising signal of
the whole project — every timing backtest in experiments 2 and 3 trades on
the output of this loop.

Definitions (all in monthly %² units):

- **Implied variance** `IVar = VIX²/12` (full history from 1990), with a
  parallel `IVar = VS²/12` variant from SPX 1-month variance swaps (Nov 2008+).
- **Conditional variance** `CV` = the HAR Model-8 forecast of the next 22-day
  realized variance, re-estimated every day on past data only.
- **VRP** (column `VP`): `VP = IVar − CV`.

## Pipeline (`main()`)

1. **Panel build** — front-month ES returns → RV components (`RV1/RV5/RV22`,
   daily-squared-return proxy) joined with `VIX²/12`; forward target
   `RV22_fwd = RV22.shift(-22)` and 1-day-lagged predictors. Loaders are
   imported from `bh_replication` (`data_prep`, `har_model`), not duplicated.
2. **Static HAR fits** — `estimate_har` on the paper sample (1990–2010) and
   the full sample, plus fixed-split OOS forecasts (split 2005-07-15, the
   paper's 75% point); coefficients printed against the paper's Table 3.
3. **In-sample VRP** — `extract_vrp` attaches fitted `CV` and `VP`.
4. **Rolling production loop** — `production_loop(panel, window=1000)`: for
   each day *t*, fit HAR by OLS on the trailing 1000 trading days ending
   **22 days before** *t* (the gap guarantees the last training label is fully
   realized — no look-ahead), predict `RV22_fwd(t)`, record
   `y_actual, y_hat, error, CV, IVar, VP` and (optionally) per-step betas and
   NW(44) t-stats.
5. **Metrics** — RMSE / MAE / MAPE / Mincer-Zarnowitz R² for the HAR OOS
   forecasts vs a martingale baseline (`E[RV_{t+22}] = RV22(t)`, BTZ Model 30).
6. **Expanding production loop** — `production_loop_expanding`: training
   window anchored at 1990-01-02 and growing daily; OOS predictions from
   2006-01-01, same 22-day gap. This loop's `VP` is the signal used by
   experiment2.
7. **VS parallel run** — the same rolling loop on the `VS²/12` panel;
   overlap means and VIX/VS VRP correlation are printed.

## Files

| File | Role |
|---|---|
| `experiment.py` | The full pipeline (entry point `main()`). Config constants at the top: `PAPER_START/END/SPLIT`, `ROLL_WIN = 1000`, `EXP_TRAIN_START = 1990-01-02`, `EXP_OOS_START = 2006-01-01`, `PAPER_COEFS`. Reusable pieces: `production_loop`, `production_loop_expanding`, `plot_combined_vrp_summary` (also imported by `experiment3 - Euro`) |
| `output/production_loop_rolling.csv` | Rolling-loop record, one row per trading day: `date, y_actual, y_hat, error, CV, IVar, VP` |
| `output/production_loop_expanding.csv` | Same columns for the expanding loop (OOS from 2006) — **the file experiment2/experiment3-Nasdaq read** |
| `output/vrp_experiment_summary_rolling.png` | 4-panel summary (rolling loop): VRP series with mean ± 1σ and crisis shading; IS adj-R² vs trailing-252d OOS MZ-R²; HAR betas over time; NW(44) t-stats over time |
| `output/vrp_experiment_summary_expanding.png` | Same 4 panels for the expanding loop |

## How to run

```bash
python "experiment1 - VRP Computation/experiment.py"
```

No CLI flags — change `ROLL_WIN` / `EXP_*` constants to alter window schemes;
both loops run on every invocation (the rolling loop over the full panel is
the runtime bottleneck, several minutes). Paths are file-anchored, so any
working directory works, but the repo layout must be intact (needs sibling
`bh_replication/` and `data/`).

Reads: `data/EquityFuture_*.parquet`, `data/VolatilityIndexData.csv`,
`data/EquityIndexVarianceSwapData.csv`.
Dependencies: `numpy`, `pandas`, `matplotlib`, `statsmodels`, `pyarrow`.

Downstream: run this **before** `experiment2 - Return Regression` and
`experiment3 - Nasdaq`, which read `output/production_loop_expanding.csv`
(experiment2 also reads `production_loop_rolling.parquet`/`.csv`).
