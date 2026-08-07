# Volatility Risk Premium — VIX-Derived Signals for Equity Index Timing

Research monorepo studying whether signals extracted from the options-implied
volatility complex (VIX, VIX futures, VVIX, variance swaps) can time equity
index futures. The organising signal is the **variance risk premium (VRP)** —
the gap between risk-neutral (implied) and physical (forecast) variance —
estimated with a HAR model in the tradition of Bekaert & Hoerova (2014).

Each top-level directory is a self-contained experiment or paper replication
with its own `README.md` (pipeline, file roles, outputs, run guide) and its own
`output/` folder. Shared logic is imported across directories via `sys.path`
insertion rather than duplicated.

## Repository layout

| Directory | What it is |
|---|---|
| `Bekaert_Hoerova_Replication/` | Bekaert & Hoerova (2014) HAR-RV-VIX "Model 8" baseline replication; also the shared library (`data_prep.py`, `har_model.py`) used by the experiments |
| `Fassas_Hourvouliades_Replication/` | Fassas & Hourvouliades (2019) VIX-futures curve geometry replication; exports the daily term-structure slope signal |
| `Simon_Campasano_Replication/` | Simon & Campasano (2014) VIX futures basis / carry-trade replication (standalone Python package) |
| `Asensio_Replication/` | Asensio (2013) "VIX-VIX Futures Puzzle" Table 7 flow-variable regressions (standalone) |
| `experiment1 - HAR VRP Computation/` | Production-grade daily VRP extraction: rolling/expanding HAR loops producing the VRP series consumed downstream |
| `experiment1 - Intraday Variant/` | Variant of experiment1 with realized variance built from 5-minute intraday ES data (removes the daily-squared-return proxy constraint of the baselines) |
| `experiment2 - Predictive Return Regression and Timing Backtests/` | The core timing engine: expanding-window predictive regressions of ES 20-day forward returns on VRP/VVIX/term-slope/OI signals + t-gated long/short backtests |
| `experiment3 - Euro Stoxx 50 Cross-Market/` | Experiment-2 pipeline ported to Euro Stoxx 50 / VSTOXX / VV2TX (also hosts the shared `cross_market.py` driver) |
| `experiment3 - Nasdaq 100 Cross-Market/` | Experiment-2 pipeline ported to NASDAQ-100 (NQ) with US signals |
| `data/` | Shared raw inputs (parquet/CSV; git-ignored) |
| `report/` | LaTeX write-up of the project (git-ignored) |

## Dependency graph and run order

```
Bekaert_Hoerova_Replication ──(data_prep, har_model)──► experiment1 ──(VRP CSVs)──► experiment2
Fassas_Hourvouliades_Replication ──(compute_vix_term_slope)───────────────────────► experiment2
experiment2 (helpers, regressions, strategies) ──► experiment3 (Euro Stoxx 50, Nasdaq 100)
Bekaert_Hoerova_Replication (har_model) ──► experiment1 - Intraday Variant
Simon_Campasano_Replication, Asensio_Replication ── standalone
```

(Directories are abbreviated by their `experimentN` prefix above.)

To reproduce the full US pipeline, run in this order:

```bash
pip install -r requirements.txt

python Bekaert_Hoerova_Replication/run_replication.py
python Fassas_Hourvouliades_Replication/fh_replication.py
python "experiment1 - HAR VRP Computation/experiment.py"     # writes the VRP series experiment2 needs
cd "experiment2 - Predictive Return Regression and Timing Backtests"
python regressions.py && python plot.py
```

The cross-market ports and side experiments can then be run independently
(`experiment3 - Nasdaq 100 Cross-Market` requires experiment1's output;
`experiment3 - Euro Stoxx 50 Cross-Market` recomputes its own VRP). The
replications (`Simon_Campasano_Replication`, `Asensio_Replication`,
`Bekaert_Hoerova_Replication` comparisons, and `experiment1 - Intraday Variant`)
have no cross-dependencies beyond `data/` and the shared
`Bekaert_Hoerova_Replication` modules — see each directory's README.

All scripts anchor paths to their own file location (`Path(__file__).parent`),
so the working directory does not matter as long as the repo layout is intact.

## Data

All inputs live in `data/` (git-ignored, must be provisioned separately):

| File | Content |
|---|---|
| `EquityFuture_historical.parquet` + `EquityFuture_security_meta.parquet` | Daily equity index futures (ES = S&P 500 E-mini, NN = NASDAQ-100 E-mini, FX = Euro Stoxx 50): prices, returns, open interest, expiries |
| `VolatilityIndexFuture_historical.parquet` + `VolatilityIndexFuture_security_meta.parquet` | Daily volatility index futures (VX = VIX futures, DI = VSTOXX futures): prices, OI, volume, last trade dates |
| `VolatilityIndexData.csv` | Daily volatility index levels (VIX, VVIX, V2X, VV2TX, …) |
| `EquityIndexVarianceSwapData.csv` | SPX variance swap fair strikes by tenor (from Nov 2008) |
| `es_intraday_sorted.csv` | 5-minute intraday ES futures bars (from 2016) |
| `intraday_eq_vx_data.csv` | Intraday equity/VX data (currently not referenced by any script) |
| `column_descriptions.csv`, `*_curve_meta.parquet` | Metadata |

## Conventions

- Variance units: monthly %² (`VIX²/12`; daily squared returns × 22 scalings).
- Newey–West HAC standard errors throughout (44 lags for 22-day-overlap HAR
  targets, 20 lags for 20-day-overlap return regressions, 5 lags for weekly data).
- Sharpe ratios are computed with a **3% annualised risk-free rate**.
- Heavy expanding-window computations are cached under
  `*/output/regression_cache/` (git-ignored). The caches key on parameter names
  only — **after changing any data loader, delete the cache directory and rerun**.
