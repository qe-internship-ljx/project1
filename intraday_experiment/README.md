# intraday_experiment — VRP with 5-Minute Intraday Realized Variance

The `bh_replication` and `experiment1` pipelines were constrained to a
**daily squared-return proxy** for realized variance. This experiment removes
that constraint: daily RV is built from **5-minute intraday ES futures
returns** (sum of squared within-day log returns), then fed through the same
HAR-RV machinery, so the results quantify the R² penalty of the daily proxy.

Intraday data availability starts **2016-01-04** (vs the 1990s for the daily
baselines), so samples are shorter.

RV construction (`load_intraday_rv`): per date, keep only the front-month
(nearest-expiry) ES contract's 5-minute bins with a CLOSE price; compute
within-day log returns (`×100`); daily RV = Σ(log_ret²). The first bin of
each day has no within-day predecessor, so the overnight gap return is
excluded. Aggregates follow the repo convention: `RV1 = daily·22`,
`RV5 = 5d mean·22`, `RV22 = 22d sum` (monthly %² units).

## Files

| File | Role |
|---|---|
| `experiment.py` | **Entry point and shared library.** `load_intraday_rv()` and `load_vix_ivar()` (VIX²/12) are imported by the other two scripts. `build_panel()` joins RV + IVar with the forward target and lags; `production_loop(panel, window=500)` is the day-by-day rolling OLS with the 22-day no-look-ahead gap (mirroring experiment1's design, with a 500-day window suited to the shorter sample); `plot_combined_vrp_summary()` renders the 4-panel summary. `main()` runs the loop and writes the CSV + PNG |
| `run_bh_intraday.py` | **Entry point.** BH-style single-fit replication on intraday RV: `build_vix_panel()` (VIX²/12) and `build_vs_panel()` (VS²/12 from SPX 1-month variance swaps; values stored under the structural column name `VIX2_lag` so `har_model.estimate_har` runs unchanged). Full-sample in-sample fit + 75/25 OOS split via `bh_replication/har_model.py` (`estimate_har`, `out_of_sample_forecast`), one 2-panel summary PNG per formulation (in-sample fitted VRP series + IS/OOS statistics table) |
| `plot_daily_rv.py` | **Entry point (diagnostic).** Plots the daily intraday RV series with a 22-day rolling mean and crisis bands; prints summary stats |
| `output/` | Generated artifacts (see below) |

Cross-directory imports: `experiment.py` imports `_nw_se` / `NW_LAGS` and
`run_bh_intraday.py` imports `estimate_har` / `out_of_sample_forecast` /
`NW_LAGS` from `bh_replication/har_model.py`; both auxiliary scripts import
from `experiment.py` (wired via `sys.path`).

## Outputs (`output/`)

| File | Content | Producer |
|---|---|---|
| `production_loop_intraday.csv` | Per-day rolling-loop record: `date, y_actual, y_hat, error, CV, IVar, VP` | `experiment.py` |
| `vrp_experiment_summary_intraday.png` | 4 panels: VRP series with mean ± 1σ; IS adj-R² + trailing-252d OOS MZ-R²; rolling HAR betas; NW(44) t-stats | `experiment.py` |
| `bh_intraday_vix_summary.png` | 2-panel BH-style summary, VIX²/12 formulation | `run_bh_intraday.py` |
| `bh_intraday_vs_summary.png` | Same layout, VS²/12 formulation | `run_bh_intraday.py` |
| `daily_rv_intraday.png` | Daily RV series + 22-day rolling mean + crisis bands | `plot_daily_rv.py` |

## How to run

```bash
cd intraday_experiment
python experiment.py          # rolling-loop VRP → CSV + 4-panel PNG
python run_bh_intraday.py     # BH VIX & VS single-fit summaries (2 PNGs)
python plot_daily_rv.py       # daily RV diagnostic PNG
```

Order-independent; each script is self-contained. Paths are file-anchored, so
any working directory works; the repo layout must be intact (needs sibling
`bh_replication/` and `data/`).

Reads: `data/es_intraday_sorted.csv` (large),
`data/EquityFuture_security_meta.parquet`, `data/VolatilityIndexData.csv`,
`data/EquityIndexVarianceSwapData.csv`.
Dependencies: `numpy`, `pandas`, `matplotlib`, `statsmodels`, `pyarrow`.
