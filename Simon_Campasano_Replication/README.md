# Simon_Campasano_Replication — Simon & Campasano (2014) VIX Futures Basis Trading

Replication of **Simon & Campasano (2014), "The VIX Futures Basis: Evidence
and Trading Strategies"**, structured as a Python package.

The paper's two core empirical claims are reproduced, then its trading
strategy is simulated:

1. The VIX futures **basis does not forecast spot-VIX changes** (Eq. 1).
2. The basis **does forecast VIX-futures price changes** (Eq. 2) — futures
   roll predictably toward spot, a harvestable carry premium.
3. Strategy: **short VIX futures in steep contango / long in backwardation**,
   with equity risk delta-hedged via ES futures using a dynamically estimated
   hedge ratio (Eqs. 3–4).

Methodological deviation from the paper: daily **settlement prices** are used
instead of the paper's intraday 3:00–3:15 PM CST synchronous quotes. Two
analysis regimes are produced: **paper period** (2006–2011 window, trades
2007–2011) and **full timeframe** (2004–2026) to test out-of-sample
persistence.

## Files

Library modules (imported, never run directly):

| Module | Role |
|---|---|
| `data.py` | Raw loaders: `load_vix_futures()` (VX parquets + last-trade dates), `load_vix_spot()`, `load_es_futures()` |
| `panel.py` | `build_daily_panel()` — master daily panel: front/second/trade-eligible (≥ 10 business days to settlement) VX contracts, front ES, `front_basis`, `trade_basis`, `daily_roll = trade_basis / trade_tts`, contango/backwardation flags |
| `regressions.py` | Section II / Exhibit 4: `build_monthly_data()` (one obs per month, tracking each front contract to settlement) and `run_regressions()` — Eq. 1 (`Δvix ~ basis`) and Eq. 2 (`Δfutures ~ basis`) on full / contango / backwardation subsamples, with Durbin–Watson stats |
| `hedge_ratio.py` | Section III: `compute_oos_hedge_ratios()` — daily expanding-window refit of Eq. 3 (`Δvix_fut ~ es_ret + es_ret·tts`), then Eq. 4 converts betas into an ES contract hedge ratio using the prior day's price/TtM; plus a full-sample in-sample fit for reporting |
| `simulator.py` | Trade state machine: enter short when `daily_roll > 0.10`, long when `< −0.10`; exit when the roll crosses ±0.05 or after 9 business days. Realistic per-contract costs (VIX RT $65, ES RT $15.50). Defines the `Trade` dataclass with hedged/unhedged/roll P&L fields |
| `metrics.py` | Sortino ratio, downside deviation, decile tables, bootstrap p-value; Exhibit 5 and Exhibit 7 printers |
| `plots.py` | Four P&L figures + `plot_all()` |

Entry points:

| Script | Role |
|---|---|
| `run_analysis.py` | **Primary orchestrator.** Part A: paper-period exhibits (1, 2, 4), Eq. 3, OOS hedge ratios, 2007–2011 simulation, Exhibits 5/7, comparison vs the paper's headline numbers. Part B (`run_full_timeframe`): the same machinery over 2005–2026 with year-by-year and sub-period breakdowns. Uses relative imports — must be run with `-m` |
| `diagnose_short.py` | Standalone diagnostic ("why does the short strategy weaken outside 2006–2011?"): P&L waterfalls by era, entry-roll size, VIX-move distributions, hedge effectiveness, roll-vs-cost coverage, contango depth by VIX bucket. Prints tables only |

Internal import graph: `panel → data`; `metrics`/`plots` → `simulator`;
`run_analysis` orchestrates everything. No circular imports.

## Outputs

Console: all exhibit tables (Exhibits 1, 2, 4, 5, 7, Eq. 3) are printed, not
saved. Figures (four per regime, written by `plots.plot_all`):

```
output/
├── paper_period/      fig1_cumulative_pnl.png   cumulative hedged P&L + VIX subplot
│                      fig2_monthly_pnl.png      monthly P&L bars by direction + net line
│                      fig3_trade_scatter.png    per-trade hedged P&L scatter
│                      fig4_annual_pnl.png       annual bars + cumulative line
└── full_timeframe/    same four filenames for the 2005–2026 run
```

## How to run

From the **repo root** (the package uses relative imports):

```bash
python -m Simon_Campasano_Replication.run_analysis     # full replication, ~all exhibits + 8 PNGs
python -m Simon_Campasano_Replication.diagnose_short   # short-leg diagnostic tables
```

Reads `data/VolatilityIndexFuture_*.parquet`, `data/VolatilityIndexData.csv`,
`data/EquityFuture_*.parquet` (paths resolved relative to the source files).
Dependencies: `numpy`, `pandas`, `statsmodels`, `matplotlib`, `pyarrow`.
