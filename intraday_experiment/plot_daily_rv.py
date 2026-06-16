"""
intraday_experiment/plot_daily_rv.py
=====================================
Plot daily realized variance from 5-minute intraday ES futures over time.

Outputs:  ./output/daily_rv_intraday.png
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT   = Path(__file__).parent
OUTPUT = ROOT / "output"
OUTPUT.mkdir(exist_ok=True)

sys.path.insert(0, str(ROOT))
from experiment import load_intraday_rv  # noqa: E402

CRISIS_BANDS = [
    ("GFC",    "2007-09-01", "2009-06-30", "#d62728"),
    ("COVID",  "2020-02-01", "2020-06-30", "#9467bd"),
    ("Hiking", "2022-01-01", "2023-12-31", "#ff7f0e"),
]


def main() -> None:
    print("Loading intraday RV…", flush=True)
    rv = load_intraday_rv()

    # Raw daily RV in daily %²-units; RV1 = daily_rv × 22
    daily_rv   = rv["RV1"] / 22
    rolling_22 = daily_rv.rolling(22).mean()

    # Convert index to datetime for matplotlib compatibility
    x = pd.to_datetime(daily_rv.index)

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.fill_between(x, daily_rv.values, alpha=0.35, color="steelblue", linewidth=0)
    ax.plot(x, daily_rv.values, color="steelblue", linewidth=0.6,
            alpha=0.7, label="Daily RV")
    ax.plot(x, rolling_22.values, color="navy", linewidth=1.8,
            label="22-day rolling mean")

    ymax = ax.get_ylim()[1]

    for label, start, end, color in CRISIS_BANDS:
        t0, t1 = pd.Timestamp(start), pd.Timestamp(end)
        ax.axvspan(t0, t1, color=color, alpha=0.08, zorder=0)
        mid = t0 + (t1 - t0) / 2
        ax.text(mid, ymax * 0.97, label,
                ha="center", va="top", fontsize=7.5, color=color, alpha=0.85)

    ax.set_title("Daily Realized Variance from Intraday 5-min ES Futures",
                 fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Daily RV  (%²)", fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.tick_params(axis="x", rotation=45)
    ax.legend(fontsize=10, loc="upper right")
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    out = OUTPUT / "daily_rv_intraday.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}", flush=True)

    print(f"\nSummary stats (daily RV, %²):")
    print(f"  n obs : {daily_rv.count():,}")
    print(f"  mean  : {daily_rv.mean():.4f}")
    print(f"  median: {daily_rv.median():.4f}")
    print(f"  max   : {daily_rv.max():.4f}  ({daily_rv.idxmax().date()})")
    print(f"  min   : {daily_rv.min():.4f}  ({daily_rv.idxmin().date()})")


if __name__ == "__main__":
    main()
