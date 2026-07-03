# experiment2 — Return Regression (Equity Index Timing Engine)

Systematic long/short timing of the **S&P 500 E-mini (ES) future** using
signals extracted from volatility markets. Two coupled stages:

1. **Predictive regressions** — expanding-window OLS predicting the ES
   **forward 20-day return** (`fwd_20d`) from each signal, tracking the
   coefficient and its Newey–West t-stat through time (plus in-sample R² and
   Campbell–Thompson OOS R²).
2. **t-gated backtests** — the daily prediction `ŷ` drives positions only
   when **all** slope coefficients pass `|t| > 1.65` (`T_THRESH`). Positions
   come in two families — unit (±1) "base" and multi-level (±1..±4)
   "leveraged" — each under three threshold modes. Every strategy is
   simulated net of transaction costs and compared with buy-and-hold.

## Signals

| Panel column | Signal | Source |
|---|---|---|
| `VP` | Variance risk premium `IVar − CV` | `experiment1 - VRP Computation/output/production_loop_rolling.parquet` (`.csv` fallback) via `load_vrp_series()` |
| `term_slope` | VIX futures term-structure slope β₀(t) | `fh_replication.compute_vix_term_slope()` (daily cross-sectional OLS `price ~ TtM`) |
| `vvix_ma5`, `vvix_ma10` | 5/10-day SMA of VVIX (vol-of-vol), converted to monthly units (÷√12) | `VolatilityIndexData.csv` |
| `open_interest` | Total ES open interest, 252-day rolling mean (removes roll sawtooth) | equity-futures parquets |
| `trend_q` | Price / 200-day SMA trend quotient | ES front-month |
| `vix`, `vix_basis`, `vol_trend` | Additional baselines available in the panel | various |

Target: `fwd_20d` = cumulative 20-day forward ES return. OOS predictions
start at `OOS_START = 2012-01-01` with a strict `OOS_GAP = 20`-day gap between
the last training label and the prediction row (no overlap look-ahead);
minimum training window 500 rows; NW HAC with `NW_LAGS = 20`.

## Strategy rules

Threshold grid `DELTAS = [0.2%, 0.5%, 0.75%, 1.0%]`; `µ` is the trailing
500-day (`RW`) mean of realized `fwd_20d`, lagged by the OOS gap.

| Mode | Base (±1) rule | Leveraged (±1..±4) sizing |
|---|---|---|
| symmetric | long if `ŷ > δ`, short if `ŷ < −δ` | level from `|ŷ|` vs the four thresholds |
| asymmetric | long if `ŷ − µ > δ`, short if `−ŷ > δ` | long level from `ŷ − µ`, short level from `|ŷ|` when `ŷ < 0` |
| base-return-shift (rolling-µ) | long if `ŷ − µ > δ`, short if `µ − ŷ > δ` | level from `|ŷ − µ|` |

Simulation (`helpers.simulate_strategy`): position set at close *t* earns the
day *t+1* return; transaction cost 0.05% per unit position change, capped at
0.05% per day. Performance (`compute_performance_stats`): annualised
return/vol, **Sharpe with a 3% annualised risk-free rate**, max drawdown,
trade count.

## Files

| File | Role |
|---|---|
| `helpers.py` | **Shared library (no main).** Data loaders (`load_vrp_series`, `load_vrp_series_expanding`, `load_es_front_month`, `load_es_open_interest`, `load_vvix`, `load_vix_spot`, `load_vix_basis`, `load_vix_futures_term_structure`), signal builders (`compute_vvix_ma5/ma10`, `compute_trend_quotient`), `build_master_panel`, `simulate_strategy`, `compute_performance_stats`, `compute_buy_and_hold`. Also imported by both `experiment3` ports |
| `regressions.py` | Panel builder `build_panel` (all predictors) + expanding-window engines `compute_betas` / `compute_betas_bivariate` / `compute_betas_trivariate`, ŷ helpers, `in_sample_r2`, `oos_cumulative_r2`, rolling-µ helper, shared constants (`OOS_START`, `OOS_GAP`, `NW_LAGS`, `MIN_WIN`, `RW`). Its `main()` pre-computes every beta cache and prints an R² summary |
| `base_strategies.py` | Unit-position strategies (`run_ew*`, `run_ew_asym*`, `run_ew_rolmu*`) and the canonical multi-panel plot helpers; `main()` renders all base plots. `--t` flag overrides the t-gate |
| `leveraged_strategies.py` | Multi-level position strategies (`run_ew_leveraged_*`, `run_ew_biv_leveraged_*`) and their 4-panel figures; `main()` renders all leveraged plots + the comparison figure. `--t` supported |
| `plot.py` | One-shot orchestrator: `base_strategies.main()` + `leveraged_strategies.main()` + the Sharpe summary table + the VRP-vs-VVIX scatter |

Import order: `helpers → regressions → base_strategies →
leveraged_strategies → plot`. External imports: `bh_replication/har_model._nw_se`
and `fh_replication.fh_replication.compute_vix_term_slope`.

## Outputs (`output/`)

```
output/
├── regression_cache/          parquet cache of betas and positions (git-ignored)
└── plots/
    ├── VRP/ · VVIX MA5/ · VVIX MA10/ · VRP + VVIX MA5/ · VRP + VVIX MA10/
    │   · VRP + Term Slope/ · VRP + Open Interest/
    │       6 PNGs per model: {symmetric,asymmetric,base_return_shift}_<model>.png
    │       + leveraged_{symmetric,asymmetric,base_return_shift}_<model>.png
    ├── comparisons/leveraged_asymmetric_vvix_vs_vrp_vvix.png
    ├── sharpe_table_extended.png       (7 models × 6 strategy variants)
    └── scatter_vrp_vs_vvix_ma5.png
```

Each strategy PNG stacks: cumulative net return (log scale) vs buy-and-hold
per δ, position/exposure panels, the NW t-stat + normalised beta evolution,
and the predicted-return band; COVID/2022 regimes are shaded. Signals whose
coefficients only become significant after 2020 have their stats rebased to
the activation date so legends reflect the effective track record.

Cache naming: `betas_EW[biv|triv]_<preds>_fwd_20d_oos2012-01-01.parquet` for
coefficients; `pos_EW[asym|rolmu|biv|_leveraged_*]_<preds>_..._t165_...parquet`
for positions (`t165` encodes the 1.65 t-gate, `d…bps` the threshold, `rw500`
the rolling-µ window).

## How to run

```bash
cd "experiment2 - Return Regression"
python regressions.py           # 1. pre-warm the beta caches (slow, expanding OLS)
python base_strategies.py       # 2. base strategy plots        [--t 1.28 to change gate]
python leveraged_strategies.py  # 3. leveraged plots + comparison
python plot.py                  # or: 2+3+Sharpe table+scatter in one shot
```

Prerequisite: run `experiment1 - VRP Computation/experiment.py` first (the
VRP loaders read its `output/`). Also needs sibling `bh_replication/`,
`fh_replication/` and `data/`.

**Cache warning:** `regression_cache/` files are keyed by parameter names
only, with no data hash — if you change any loader in `helpers.py` (or
refresh `data/` or experiment1's output), **delete
`output/regression_cache/` and rerun `regressions.py`**, otherwise stale
betas/positions are silently reused.

Dependencies: `numpy`, `pandas`, `matplotlib`, `scipy`, `statsmodels`,
`pyarrow`.
