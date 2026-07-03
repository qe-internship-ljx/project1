# Asensio_Replication — Asensio (2013) VIX Futures Flow Regressions

Replication of **Asensio (2013), "The VIX-VIX Futures Puzzle", Table 7** —
the linear retail-flow parameter baseline. The experiment asks whether VIX
futures *flow* variables (open interest, volume) are related to the
*forward-month volatility premium* priced into the VIX futures curve, with a
focus on the post-crisis sub-period **April 2009 – February 2012** (the ETF
inflow window, N ≈ 147 weekly observations).

This replication is standalone: it shares only the `data/` folder with the
rest of the repo and is not part of the VRP/timing pipeline.

## Pipeline

`asensio_flow_replication.py` is self-contained; `main()` runs:

1. **Load** — VX futures prices/OI/volume from the two
   `VolatilityIndexFuture_*` parquets (business-day time-to-expiry per
   contract), VIX spot from `VolatilityIndexData.csv`.
2. **Daily panel** — front-month price, total OI/volume summed across the
   strip, `vix_basis = front − spot`, and a 252-day rolling-mean OI trend.
3. **Premium measures** — from the first seven contracts (F1..F7):
   term-structure slope `ts_slope = (F7−F1)/F1` (the primary Y),
   absolute spread `ts_spread = F7−F1`, and the arbitrage-profit proxy
   `r_arb` per the paper's equations 15–17 (short near/next-term, long the
   equal-weight 7-contract strip).
4. **Weekly resample** — Friday sampling via `.resample("W-FRI").last()`.
5. **Six OLS specifications** (each with 5-lag Newey–West HAC SEs,
   full-sample and post-crisis, plus OI/1000-scaled and log variants):
   - Spec 1: `vix_basis ~ rolling_oi` (level)
   - Spec 2: `Δr_arb ~ Δtotal_oi` (Asensio Table 7 exact, first differences)
   - Spec 3: `Δvix_basis ~ Δtotal_oi`
   - Spec 4: `Δr_arb ~ Δrolling_oi` (smoothed flow)
   - Spec 5 (**primary**): `ts_slope ~ total_oi`
   - Spec 6: `ts_spread ~ total_oi`
6. **Report** — each result is printed against the hard-coded paper targets
   (`PAPER_TARGETS`), followed by a summary comparison table, a weekly panel
   CSV and two figures.

The regressions are contemporaneous (descriptive replication of Table 7),
not predictive trading signals.

## Files and key functions

| Item | Role |
|---|---|
| `asensio_flow_replication.py` | The entire experiment. Config constants at the top: `POST_CRISIS_START/END`, `NW_LAGS = 5`, `PAPER_TARGETS`. Key functions: `load_vx_all`, `load_vix_spot`, `build_daily_panel`, `build_arb_profit_daily`, `build_term_structure_slope`, `resample_weekly`, `ols_nw` (OLS + HAC), `make_scatter_plot`, `make_oi_timeseries_plot`, `main` |
| `output/asensio_flow_panel.csv` | Weekly (W-FRI) panel: `front_price, vix_spot, vix_basis, total_oi, rolling_oi, ts_slope, ts_spread, log_ts` (git-ignored; regenerated on run) |
| `output/asensio_flow_replication.png` | 3-panel scatter + OLS fit with β / NW-t / R² / N annotations: Spec 5, Spec 5 scaled, Spec 2 (post-crisis window) |
| `output/asensio_oi_timeseries.png` | 3 stacked panels over Apr 2009 – Feb 2012: total OI bars + 252d rolling mean, term-structure slope, VIX basis |

## How to run

```bash
python Asensio_Replication/asensio_flow_replication.py
```

No CLI arguments (behaviour is set by the module constants). Paths are
anchored to the script location, so any working directory works. Reads
`data/VolatilityIndexFuture_security_meta.parquet`,
`data/VolatilityIndexFuture_historical.parquet`,
`data/VolatilityIndexData.csv`. Requires the sibling `bh_replication`
directory on disk (`ols_nw` delegates to its shared `_nw_se` HAC helper).

Dependencies: `numpy`, `pandas`, `matplotlib`, `statsmodels`, `pyarrow`.
