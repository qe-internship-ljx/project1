"""
P&L visualisations for the Simon & Campasano (2014) replication.

Produces four figures saved to vix_basis/output/:
  fig1_cumulative_pnl.png   -- cumulative hedged P&L vs VIX level
  fig2_monthly_pnl.png      -- monthly P&L bars (short / long / net)
  fig3_trade_scatter.png    -- per-trade P&L scatter coloured by direction
  fig4_annual_pnl.png       -- annual P&L bars with drawdown annotation

Call:  plot_all(trades, panel, out_dir)
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe in all envs)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from .simulator import Trade, trades_to_dataframe

# ── colour palette ────────────────────────────────────────────────────────
C_SHORT  = "#e05c5c"   # red
C_LONG   = "#4c8bcc"   # blue
C_NET    = "#2ca02c"   # green
C_VIX    = "#888888"   # grey
C_FILL   = "#d4ecd4"   # light green fill under cum-PnL
C_DD     = "#f5c0c0"   # light red fill for drawdown periods


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _trade_monthly(df: pd.DataFrame, field: str = "pnl_hedged") -> pd.DataFrame:
    """Aggregate per-trade P&L to calendar months using exit_date."""
    df = df.dropna(subset=[field, "exit_date"]).copy()
    df["month"] = df["exit_date"].dt.to_period("M")
    return (
        df.groupby(["month", "direction"])[field]
        .sum()
        .unstack(fill_value=0)
        .reindex(columns=["short", "long"], fill_value=0)
        .assign(net=lambda d: d.get("short", 0) + d.get("long", 0))
    )


# ─────────────────────────────────────────────────────────────────────────
# Figure 1 – Cumulative P&L vs VIX
# ─────────────────────────────────────────────────────────────────────────

def fig_cumulative(
    df: pd.DataFrame,
    panel: pd.DataFrame,
    out_path: Path,
    label: str = "",
) -> None:
    """
    Top panel  : cumulative hedged P&L ($ total, 1 contract per trade)
                 filled area shows drawdown from the running maximum.
    Bottom panel: spot VIX level for context.
    """
    df = df.dropna(subset=["pnl_hedged", "exit_date"]).sort_values("exit_date")
    if df.empty:
        return

    # Cumulative P&L on exit dates
    dates  = df["exit_date"].values
    cum    = df["pnl_hedged"].cumsum().values
    peak   = np.maximum.accumulate(cum)

    # Daily VIX
    vix = panel[["date", "vix_spot"]].dropna().sort_values("date")

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 8),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
    )

    # -- P&L panel --
    ax1.fill_between(dates, 0, cum, where=(cum >= 0),
                     color=C_FILL, alpha=0.6, step="post")
    ax1.fill_between(dates, 0, cum, where=(cum < 0),
                     color=C_DD, alpha=0.5, step="post")
    ax1.fill_between(dates, cum, peak,
                     color=C_DD, alpha=0.35, step="post", label="Drawdown")
    ax1.step(dates, cum, color=C_NET, linewidth=1.8, where="post",
             label=f"Cumulative hedged P&L  (total: ${cum[-1]:+,.0f})")
    ax1.axhline(0, color="black", linewidth=0.6, linestyle="--")

    # Annotate final value
    ax1.annotate(f"${cum[-1]:+,.0f}",
                 xy=(dates[-1], cum[-1]),
                 xytext=(10, 0), textcoords="offset points",
                 va="center", fontsize=9, color=C_NET)

    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"${x/1000:+.0f}k"))
    ax1.set_ylabel("Cumulative P&L (USD)", fontsize=10)
    ax1.legend(fontsize=9, loc="upper left")
    ax1.set_xlim(dates[0], dates[-1])
    ax1.grid(axis="y", linewidth=0.4, alpha=0.5)
    ax1.set_xticklabels([])
    title = f"Simon & Campasano (2014) Replication — Cumulative Hedged P&L"
    if label:
        title += f"\n{label}"
    ax1.set_title(title, fontsize=11)

    # -- VIX panel --
    ax2.fill_between(vix["date"], vix["vix_spot"],
                     alpha=0.25, color=C_VIX)
    ax2.plot(vix["date"], vix["vix_spot"],
             color=C_VIX, linewidth=0.9, label="VIX spot")
    ax2.set_ylabel("VIX", fontsize=10)
    ax2.set_xlim(dates[0], dates[-1])
    ax2.grid(axis="y", linewidth=0.4, alpha=0.5)
    ax2.legend(fontsize=9, loc="upper left")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────
# Figure 2 – Monthly P&L bars
# ─────────────────────────────────────────────────────────────────────────

def fig_monthly(
    df: pd.DataFrame,
    out_path: Path,
    label: str = "",
) -> None:
    """
    Stacked monthly bar chart: short P&L (red) + long P&L (blue).
    Net monthly P&L line overlaid.
    """
    monthly = _trade_monthly(df)
    if monthly.empty:
        return

    idx   = [p.to_timestamp() for p in monthly.index]
    short = monthly.get("short", pd.Series(0, index=monthly.index)).values
    long_ = monthly.get("long",  pd.Series(0, index=monthly.index)).values
    net   = monthly["net"].values

    fig, ax = plt.subplots(figsize=(16, 5))

    bar_w = 20  # days

    # Positive short bars (short gain = positive P&L on the trade, which means
    # VIX fell) plotted up; negative short bars plotted down
    ax.bar(idx, np.where(short >= 0, short, 0),
           width=bar_w, color=C_SHORT, alpha=0.75, label="Short P&L")
    ax.bar(idx, np.where(short < 0, short, 0),
           width=bar_w, color=C_SHORT, alpha=0.75)
    ax.bar(idx, np.where(long_ >= 0, long_, 0),
           width=bar_w, bottom=np.where(short >= 0, short, 0),
           color=C_LONG, alpha=0.75, label="Long P&L")
    ax.bar(idx, np.where(long_ < 0, long_, 0),
           width=bar_w, bottom=np.where(short < 0, short, 0),
           color=C_LONG, alpha=0.75)

    ax.plot(idx, net, color=C_NET, linewidth=1.5,
            marker="o", markersize=2.5, label="Net monthly P&L")
    ax.axhline(0, color="black", linewidth=0.6)

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"${x/1000:+.0f}k"))
    ax.set_ylabel("Monthly P&L (USD)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    title = "Monthly P&L by Strategy Leg"
    if label:
        title += f"  |  {label}"
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────
# Figure 3 – Per-trade P&L scatter
# ─────────────────────────────────────────────────────────────────────────

def fig_trade_scatter(
    df: pd.DataFrame,
    panel: pd.DataFrame,
    out_path: Path,
    label: str = "",
) -> None:
    """
    Top panel  : scatter of per-trade hedged P&L over time.
                 Red = short, Blue = long.
    Bottom panel: VIX spot for context.
    """
    df = df.dropna(subset=["pnl_hedged", "entry_date"]).copy()
    if df.empty:
        return

    vix = panel[["date", "vix_spot"]].dropna().sort_values("date")

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 7),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
    )

    for direction, colour in [("short", C_SHORT), ("long", C_LONG)]:
        sub = df[df["direction"] == direction]
        if sub.empty:
            continue
        ax1.scatter(
            sub["entry_date"], sub["pnl_hedged"],
            color=colour, alpha=0.55, s=30,
            label=f"{direction.capitalize()}  (n={len(sub)},"
                  f" mean ${sub['pnl_hedged'].mean():+,.0f})",
            zorder=3,
        )

    ax1.axhline(0, color="black", linewidth=0.6, linestyle="--")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"${x/1000:+.0f}k"))
    ax1.set_ylabel("Hedged P&L per trade (USD)", fontsize=10)
    ax1.legend(fontsize=9)
    ax1.grid(linewidth=0.4, alpha=0.4)
    ax1.set_xticklabels([])
    title = "Per-Trade Hedged P&L"
    if label:
        title += f"  |  {label}"
    ax1.set_title(title, fontsize=11)

    ax2.fill_between(vix["date"], vix["vix_spot"], alpha=0.25, color=C_VIX)
    ax2.plot(vix["date"], vix["vix_spot"], color=C_VIX, linewidth=0.9)
    ax2.set_ylabel("VIX", fontsize=10)
    ax2.grid(axis="y", linewidth=0.4, alpha=0.5)

    # Align x-axis
    x_min = df["entry_date"].min()
    x_max = df["entry_date"].max()
    for ax in (ax1, ax2):
        ax.set_xlim(x_min - pd.Timedelta(days=60),
                    x_max + pd.Timedelta(days=60))

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────
# Figure 4 – Annual P&L bars
# ─────────────────────────────────────────────────────────────────────────

def fig_annual(
    df: pd.DataFrame,
    out_path: Path,
    label: str = "",
) -> None:
    """
    Grouped bar chart: annual hedged P&L for short, long, and net.
    Running cumulative P&L line on right axis.
    """
    df = df.dropna(subset=["pnl_hedged", "entry_date"]).copy()
    df["year"] = df["entry_date"].dt.year

    years  = sorted(df["year"].unique())
    x      = np.arange(len(years))
    width  = 0.28

    short_pnl = [df[(df["year"] == y) & (df["direction"] == "short")]["pnl_hedged"].sum() for y in years]
    long_pnl  = [df[(df["year"] == y) & (df["direction"] == "long")]["pnl_hedged"].sum()  for y in years]
    net_pnl   = [s + l for s, l in zip(short_pnl, long_pnl)]
    cum_pnl   = np.cumsum(net_pnl)

    fig, ax1 = plt.subplots(figsize=(14, 5))
    ax2 = ax1.twinx()

    ax1.bar(x - width, short_pnl, width, color=C_SHORT, alpha=0.8, label="Short P&L")
    ax1.bar(x,         long_pnl,  width, color=C_LONG,  alpha=0.8, label="Long P&L")
    ax1.bar(x + width, net_pnl,   width,
            color=[C_NET if v >= 0 else C_DD for v in net_pnl],
            alpha=0.85, label="Net P&L")
    ax1.axhline(0, color="black", linewidth=0.6)

    ax2.plot(x, cum_pnl, color="black", linewidth=1.8,
             marker="D", markersize=4, label="Cumulative P&L (RHS)")
    ax2.axhline(0, color="black", linewidth=0.4, linestyle=":")

    fmt = mticker.FuncFormatter(lambda v, _: f"${v/1000:+.0f}k")
    ax1.yaxis.set_major_formatter(fmt)
    ax2.yaxis.set_major_formatter(fmt)

    ax1.set_xticks(x)
    ax1.set_xticklabels([str(y) for y in years], fontsize=9)
    ax1.set_ylabel("Annual P&L (USD)", fontsize=10)
    ax2.set_ylabel("Cumulative P&L (USD)", fontsize=10)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper left")
    ax1.grid(axis="y", linewidth=0.4, alpha=0.5)

    title = "Annual Hedged P&L — Short / Long / Net"
    if label:
        title += f"  |  {label}"
    ax1.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────
# Master entry point
# ─────────────────────────────────────────────────────────────────────────

def plot_all(
    trades: List[Trade],
    panel: pd.DataFrame,
    out_dir: Path | str,
    label: str = "",
) -> list[Path]:
    """
    Generate all four figures and save to out_dir.
    Returns list of saved file paths.
    """
    out_dir = _ensure_dir(Path(out_dir))
    df = trades_to_dataframe(trades)
    saved = []

    specs = [
        ("fig1_cumulative_pnl.png",  lambda p: fig_cumulative(df, panel, p, label)),
        ("fig2_monthly_pnl.png",     lambda p: fig_monthly(df, p, label)),
        ("fig3_trade_scatter.png",   lambda p: fig_trade_scatter(df, panel, p, label)),
        ("fig4_annual_pnl.png",      lambda p: fig_annual(df, p, label)),
    ]
    for fname, fn in specs:
        path = out_dir / fname
        fn(path)
        saved.append(path)
        print(f"  Saved: {path}")

    return saved
