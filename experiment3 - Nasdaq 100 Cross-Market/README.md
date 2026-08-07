# experiment3 — Nasdaq 100 Cross-Market (NASDAQ-100 Port)

Port of the `experiment2 - Predictive Return Regression and Timing Backtests`
timing pipeline to the **NASDAQ-100 E-mini (NQ) future** (curve group `NN`). Unlike the Euro port,
all signals stay **US-based** — only the dependent return series changes:

## Purpose

A robustness test that isolates the **traded index** while holding the
signals fixed. The VIX complex is defined on S&P 500 options, so
experiment2's result could be an SPX-specific effect; if instead the VRP and
its companions proxy for a *broad* equity risk premium, they should also time
a closely related but distinct index. Swapping only the dependent return
series to the NASDAQ-100 (higher beta, tech-concentrated, but strongly
correlated with the S&P) tests exactly that: predictability that survives the
swap points to a market-wide risk-compensation mechanism rather than an
artifact of regressing SPX-derived signals on SPX returns. Together with the
Euro port (which changes *both* signals and index), this brackets how far the
experiment2 result generalises.

| Input | Source |
|---|---|
| Dependent return `fwd_20d` / `daily_ret` | NQ front-month (loaded here) |
| VRP (`VP`) | `experiment1 - HAR VRP Computation/output/production_loop_expanding.csv` (expanding-window US VRP) via experiment2's `load_vrp_series_expanding()` |
| Term slope | `Fassas_Hourvouliades_Replication.compute_vix_term_slope()` on VIX futures |
| VVIX MA5 | experiment2's `compute_vvix_ma5(load_vvix())` |

The regression/backtest engine is delegated entirely to
`cross_market.run_all` (which lives in
`experiment3 - Euro Stoxx 50 Cross-Market/cross_market.py` and drives experiment2's `regressions` / `base_strategies` /
`leveraged_strategies` modules with outputs and caches redirected into this
directory).

## Files

| File | Role |
|---|---|
| `nasdaq_experiment.py` | The entire adapter: `__main__` loads the NQ front-month via `cross_market.load_front_month("NN")` (nearest-expiry contract per date, price level rebuilt from cumulated returns rebased to 1000), loads the US signals, builds the panel via experiment2's `build_master_panel`, and calls `cross_market.run_all(..., vv_label="VVIX MA5")` |
| `output/` | NASDAQ artifacts (see below) |

## Outputs (`output/`)

```
output/
├── regression_cache/                 betas/positions parquet cache (git-ignored)
└── plots/
    └── VRP/ · VVIX MA5/ · VRP + Term Slope/ · VRP + VVIX MA5/
          6 PNGs per model: {symmetric,asymmetric,base_return_shift}_<model>.png
          + leveraged_{symmetric,asymmetric,base_return_shift}_<model>.png
```

Models, threshold variants (δ ∈ {0.2%, 0.5%, 0.75%, 1.0%}), the |t| > 1.65
gate, leverage levels (±1..±4), transaction costs and the 3%-risk-free Sharpe
are all identical to experiment2 — see that README for the methodology.

## How to run

```bash
# Prerequisite: experiment1 must have been run (its expanding-loop CSV is read)
python "experiment1 - HAR VRP Computation/experiment.py"

python "experiment3 - Nasdaq 100 Cross-Market/nasdaq_experiment.py"
```

Requires the sibling directories
`experiment2 - Predictive Return Regression and Timing Backtests`,
`experiment3 - Euro Stoxx 50 Cross-Market` (for `cross_market.py`),
`Fassas_Hourvouliades_Replication` and `Bekaert_Hoerova_Replication` on disk (wired via
`sys.path` inserts). Reads
`data/EquityFuture_*.parquet`, `data/VolatilityIndexData.csv`,
`data/VolatilityIndexFuture_*.parquet`.

The cache-staleness warning from experiment2 applies here too: after any
loader change (or after regenerating experiment1's output), delete
`output/regression_cache/` before rerunning.

Dependencies: `numpy`, `pandas`, `matplotlib`, `statsmodels`, `pyarrow`.
