"""
Performance metrics and exhibit formatting (Exhibits 5, 7, 8).

The paper uses Sortino ratios instead of Sharpe ratios because P&L
distributions are non-normal and the strategy penalises only downside risk.

  Sortino = mean_PnL / semi_std

where semi_std is the semi-standard deviation of negative outcomes,
with gains set to zero before calculation (minimum acceptable return = 0).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .simulator import Trade


# ????????????????????????????????????????????????????????????????????????????
# Core statistics
# ????????????????????????????????????????????????????????????????????????????

def semi_std(values: np.ndarray) -> float:
    """
    Semi-standard deviation using losses only (gains treated as zero).
    Matches the paper: minimum acceptable return = 0.
    """
    losses = np.where(values < 0, values, 0.0)
    return float(np.sqrt(np.mean(losses ** 2)))


def sortino_ratio(values: np.ndarray) -> float:
    mean = float(np.mean(values))
    denom = semi_std(values)
    return mean / denom if denom > 1e-9 else np.nan


def decile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q * 100))


def randomisation_pvalue(
    actual_mean: float,
    all_pnls: np.ndarray,
    n_trades: int,
    avg_hold_days: float,
    n_iter: int = 10_000,
    rng_seed: int = 42,
) -> float:
    """
    Bootstrap p-value: fraction of random-entry simulations that achieve
    a higher mean P&L than the actual strategy.

    Mirrors the paper's randomisation test (Lo et al., 2000 approach):
    randomly draw entry days (with replacement) for the same number of
    trades, hold for avg_hold_days, compute the mean P&L, repeat 10,000x.

    Parameters
    ----------
    actual_mean   : mean P&L of the actual strategy
    all_pnls      : 1-D array of per-trade P&Ls from which random draws
                    are made (entire sample, not just actual trades)
    n_trades      : number of trades in the actual strategy
    avg_hold_days : average hold duration (used to select pool; here
                    we just resample from the empirical distribution)
    n_iter        : number of Monte-Carlo replications
    """
    rng = np.random.default_rng(rng_seed)
    count_better = 0
    for _ in range(n_iter):
        sample_means = rng.choice(all_pnls, size=n_trades, replace=True).mean()
        if sample_means >= actual_mean:
            count_better += 1
    return count_better / n_iter


# ????????????????????????????????????????????????????????????????????????????
# Summary stats for one group of trades
# ????????????????????????????????????????????????????????????????????????????

def trade_stats(
    trades: List[Trade],
    direction: str,
    pnl_field: str = "pnl_hedged",
    all_pnls: Optional[np.ndarray] = None,
) -> Dict:
    """
    Compute summary statistics for a list of trades.

    Parameters
    ----------
    trades     : list of Trade objects
    direction  : 'short' or 'long' (used for filtering)
    pnl_field  : 'pnl_hedged', 'pnl_unhedged', 'pnl_es_hedge', or 'pnl_roll'
    all_pnls   : pool of P&Ls for the randomisation test; if None, the
                 test is skipped and pvalue = NaN
    """
    sub = [t for t in trades if t.direction == direction]
    if not sub:
        return {}

    pnls = np.array([getattr(t, pnl_field) for t in sub], dtype=float)
    n    = len(pnls)
    mean = float(np.mean(pnls))
    ssd  = semi_std(pnls)
    sort = sortino_ratio(pnls)
    wins = int(np.sum(pnls > 0))

    hold = np.array([t.hold_days for t in sub if t.hold_days is not None], dtype=float)
    avg_hold = float(np.mean(hold)) if len(hold) else np.nan

    pvalue = np.nan
    if all_pnls is not None and n > 0:
        pvalue = randomisation_pvalue(mean, all_pnls, n, avg_hold)

    return {
        "direction":   direction,
        "pnl_field":   pnl_field,
        "n_trades":    n,
        "mean_pnl":    mean,
        "pvalue":      pvalue,
        "semi_std":    ssd,
        "sortino":     sort,
        "pct90":       decile(pnls, 0.90),
        "pct10":       decile(pnls, 0.10),
        "winners":     wins,
        "losers":      n - wins,
        "avg_hold":    avg_hold,
    }


# ????????????????????????????????????????????????????????????????????????????
# Exhibit printers
# ????????????????????????????????????????????????????????????????????????????

_FIELDS = [
    ("pnl_hedged",   "Hedged P&L"),
    ("pnl_unhedged", "Unhedged P&L"),
    ("pnl_es_hedge", "S&P Hedge P&L"),
    ("pnl_roll",     "Roll P&L"),
]


def print_exhibit5(trades: List[Trade], label: str = "Full sample") -> None:
    """Replicate the layout of Exhibit 5 (or Exhibit 8 sub-periods)."""
    # Build pool of all hedged P&Ls for randomisation test
    all_hedged = np.array([t.pnl_hedged for t in trades], dtype=float)

    w = 72
    print("=" * w)
    print(f"EXHIBIT 5 -- Trading strategy P&L summary  [{label}]")
    print(f"  Entry: |daily_roll| > 0.10;  Exit: |daily_roll| < 0.05 or 9 biz-days")
    print(f"  Costs: avg bid-ask spread + $3 brokerage per contract (round-trip)")
    print("=" * w)

    for direction in ("short", "long"):
        print(f"\n{'SHORT' if direction=='short' else 'LONG'} VIX FUTURES TRADES")
        print("-" * w)
        header = f"{'Metric':<26}"
        for _, col_label in _FIELDS:
            header += f"{col_label:>12}"
        print(header)
        print("-" * w)

        rows_data = {}
        for field, col_label in _FIELDS:
            pool = all_hedged if field == "pnl_hedged" else None
            stats = trade_stats(trades, direction, field, pool)
            rows_data[field] = stats

        # Pick n_trades from any field
        ref = rows_data["pnl_hedged"]
        n   = ref.get("n_trades", 0)
        avg_hold = ref.get("avg_hold", np.nan)

        def fmt(v):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return "      --"
            return f"{v:>+12,.0f}" if abs(v) >= 1 else f"{v:>12.3f}"

        metrics_order = [
            ("mean_pnl",  "Mean P&L"),
            ("pvalue",    "P-value"),
            ("semi_std",  "Semi Std Dev"),
            ("pct90",     "90th pctile"),
            ("pct10",     "10th pctile"),
            ("sortino",   "Sortino Ratio"),
        ]
        for attr, mname in metrics_order:
            line = f"  {mname:<24}"
            for field, _ in _FIELDS:
                v = rows_data[field].get(attr, np.nan)
                if attr == "pvalue":
                    line += f"  ({v:.3f})" if not np.isnan(v) else "      --"
                elif attr == "sortino":
                    line += f"    {v:>7.2f}" if not np.isnan(v) else "      --"
                else:
                    line += fmt(v)
            print(line)

        print(f"  {'Winners/Losers':<24}"
              f"{ref.get('winners',0):>5}/{ref.get('losers',0):<5}"
              f"{'':>7}{rows_data['pnl_unhedged'].get('winners',0):>5}/{rows_data['pnl_unhedged'].get('losers',0):<5}")
        print(f"  {'N trades':<24}{n:>12}")
        print(f"  {'Avg hold (days)':<24}{avg_hold:>12.1f}")

    print()


def print_exhibit7(trades: List[Trade]) -> None:
    """Print cumulative P&L summary (Exhibit 7)."""
    df = (
        pd.DataFrame([{
            "date": t.exit_date,
            "pnl":  t.pnl_hedged,
        } for t in trades if t.exit_date is not None])
        .sort_values("date")
    )
    if df.empty:
        print("No completed trades.")
        return
    df["cum_pnl"] = df["pnl"].cumsum()
    total = df["cum_pnl"].iloc[-1]
    n     = len(df)
    drawdown = (df["cum_pnl"].cummax() - df["cum_pnl"]).max()
    print("=" * 60)
    print("EXHIBIT 7 -- Cumulative P&L (hedged, all trades)")
    print("=" * 60)
    print(f"  Total gain   :  ${total:>10,.0f}")
    print(f"  N trades     :  {n}")
    print(f"  Max drawdown :  ${drawdown:>10,.0f}")
    print()
    # Year-by-year breakdown
    df["year"] = df["date"].dt.year
    print(f"  {'Year':<8} {'N':>5} {'Annual P&L':>12} {'Cum P&L':>12}")
    print("  " + "-" * 40)
    for yr, g in df.groupby("year"):
        print(f"  {yr:<8} {len(g):>5} {g['pnl'].sum():>+12,.0f} "
              f"{g['cum_pnl'].iloc[-1]:>+12,.0f}")
    print()
