# experiment3 — Euro (Euro Stoxx 50 / VSTOXX Port)

Port of the `experiment2 - Return Regression` timing pipeline to the
**European market**: every signal is re-derived from European instruments and
the dependent return series is the **Euro Stoxx 50 future** (curve group
`FX`). Signal mapping vs the US pipeline:

| US (experiment2) | Euro equivalent |
|---|---|
| ES front-month return | Euro Stoxx 50 front-month return |
| VIX (implied variance) | **V2X Index** (`IVar = V2X²/12`) |
| VIX futures term slope | **VSTOXX futures** (curve group `DI`) daily cross-sectional slope |
| VVIX MA5 | **VV2TX** (VSTOXX-of-VSTOXX) 5-day MA |
| VRP from experiment1's output | Euro VRP **recomputed here** by running experiment1's `production_loop` (1000-day rolling HAR) on V2X²/12 — there is no precomputed Euro VRP |

The regression/backtest engine itself is not duplicated: this directory hosts
`cross_market.py`, a thin market-agnostic driver that calls experiment2's
`regressions.py` / `base_strategies.py` / `leveraged_strategies.py` /
`helpers.py` on any panel with the standard column contract
(`VP, term_slope, vvix_ma5, fwd_20d, daily_ret`). The same driver is imported
by `experiment3 - Nasdaq`.

## Files

| File | Role |
|---|---|
| `cross_market.py` | **Shared driver (used by Euro and Nasdaq).** `load_front_month(curve_group)` builds the continuous front-month equity-future series (nearest-expiry contract per date, price level rebuilt from cumulated returns rebased to 1000) for any curve group ("FX" Euro Stoxx, "NN" NQ). `_redirect(out_root, cache_dir)` repoints the experiment2 modules' `OUTPUT`/`CACHE_DIR` globals at the calling market's folders (so US caches are never touched — cache tags are keyed by predictor name only and would collide across markets). `run_univariate` / `run_bivariate` run one model through all 3 base variants × 4 deltas + all 3 leveraged variants and plot. `run_all(panel, out_root, cache_dir, vv_label)` is the pipeline entry: buy-and-hold benchmark, then VRP, vol-of-vol, VRP+Term Slope, VRP+vol-of-vol, and the leveraged-asymmetric comparison figure |
| `euro_experiment.py` | **Euro adapter + entry point.** Loaders: `load_v2x_spot()`, `load_vstoxx_futures()` (with `ttm_years`), `load_vv2tx()`. Euro VRP: `run_euro_vrp_summary()` reuses `experiment1`'s `production_loop` + `plot_combined_vrp_summary` on a Euro panel (`build_euro_vrp_panel`) to produce the experiment1-style CSV/summary plot, and **returns the loop output, whose `VP/CV/IVar` columns feed the trading panel directly** — one code path for both. `compute_vstoxx_term_slope()` mirrors the FH daily cross-sectional slope on VSTOXX futures. `__main__` assembles the panel via experiment2's `build_master_panel` and calls `cross_market.run_all` |
| `output/` | Euro artifacts (see below) |

## Pipeline (`euro_experiment.py` `__main__`)

1. Load Euro Stoxx front-month (`cross_market.load_front_month("FX")`) + V2X spot.
2. `run_euro_vrp_summary` → experiment1-style rolling VRP loop on Euro data →
   `output/production_loop_rolling.csv` + `output/vrp_experiment_summary_rolling.png`;
   its returned loop output provides `VP/CV/IVar` for the trading panel.
3. `compute_vstoxx_term_slope(load_vstoxx_futures())` → `term_slope`.
4. `load_vv2tx().rolling(5).mean()` → `vvix_ma5` column (labelled "VV2TX MA5").
5. `build_master_panel(...)` → `fwd_20d`, `daily_ret`, signal columns.
6. `cross_market.run_all(panel, out_root=output, cache_dir=output/regression_cache,
   vv_label="VV2TX MA5")`.

## Outputs (`output/`)

```
output/
├── production_loop_rolling.csv            Euro VRP production-loop record
├── vrp_experiment_summary_rolling.png     experiment1-style 4-panel VRP summary
├── regression_cache/                      betas/positions parquet cache (git-ignored)
└── plots/
    ├── VRP/ · VV2TX MA5/ · VRP + Term Slope/ · VRP + VV2TX MA5/
    │     6 PNGs per model: {symmetric,asymmetric,base_return_shift}_<model>.png
    │     + leveraged_{...}_<model>.png
    └── comparisons/leveraged_asymmetric_comparison.png
```

## How to run

```bash
python "experiment3 - Euro/euro_experiment.py"
```

Self-sufficient with respect to VRP (recomputes it), but requires the sibling
directories `experiment2 - Return Regression`, `experiment1 - VRP Computation`
(code, not output) and `bh_replication` on disk — imports are wired through
`sys.path` inserts. Reads `data/EquityFuture_*.parquet`,
`data/VolatilityIndexData.csv`, `data/VolatilityIndexFuture_*.parquet`.

The cache-staleness warning from experiment2 applies here too: after any
loader change, delete `output/regression_cache/` before rerunning.

Dependencies: `numpy`, `pandas`, `matplotlib`, `statsmodels`, `pyarrow`.
