"""
plot_perf_stats.py
==================
Performance statistics chart for Base VRP, Model A, and Model C
across delta thresholds {0.1%, 0.5%, 1.0%, 2.0%}, plus Buy-and-Hold baseline.

Panels (one row per metric):
  Row 1 — Total realised return (%)
  Row 2 — Annualised return (%)
  Row 3 — Sharpe ratio (0% risk-free)
  Row 4 — Maximum drawdown (%)
  Row 5 — Average position (+1=always long, 0=always flat, -1=always short)

Each panel has 4 grouped bars (one per model/baseline), with the 4 delta
variants shown side-by-side within each group. Buy-and-hold is a horizontal
reference line on each metric panel.
"""

import sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

ROOT   = Path(__file__).parent
OUTPUT = ROOT / "output"
sys.path.insert(0, str(ROOT.parent / "bh_replication"))
sys.path.insert(0, str(ROOT))
from experiment2 import (
    load_vrp_series, load_es_front_month, load_vix_futures_term_structure,
    load_vvix, compute_vix_term_slope, compute_trend_quotient, compute_vvix_ma5,
    build_master_panel, run_rolling_regression_positions,
    compute_buy_and_hold, simulate_strategy, compute_performance_stats,
)

# ── Build panel ─────────────────────────────────────────────────────────────
print("Building panel...")
vrp      = load_vrp_series()
es       = load_es_front_month()
vx_df    = load_vix_futures_term_structure()
vvix     = load_vvix()
slope    = compute_vix_term_slope(vx_df)
trend_q  = compute_trend_quotient(es)
vvix_ma5 = compute_vvix_ma5(vvix)
panel    = build_master_panel(vrp, es, slope, trend_q, vvix_ma5)
panel    = panel[panel.index >= "2006-03-06"].copy()
daily_ret = panel["daily_ret"].dropna()

# ── Generate positions and stats ────────────────────────────────────────────
DELTAS    = [0.002, 0.005, 0.0075, 0.010]
DELTA_LBL = ["d=0.2%", "d=0.5%", "d=0.75%", "d=1.0%"]
MODELS    = ["Base", "Model_A", "Model_C"]
MODEL_LBL = {
    "Base":    "Base\n(VRP only)",
    "Model_A": "Model A\n(VRP+Slope)",
    "Model_C": "Model C\n(VRP+VVIX)",
}

all_stats = {}   # (model, delta_idx) → stats dict
all_pos   = {}

for m in MODELS:
    for di, delta in enumerate(DELTAS):
        print(f"  {MODELS.index(m)*4+di+1:02d}/12  {m}  {DELTA_LBL[di]}...")
        pos = run_rolling_regression_positions(panel, m, delta)
        sim = simulate_strategy(pos, daily_ret)
        st  = compute_performance_stats(sim, f"{m}_{DELTA_LBL[di]}")
        # Extra stat: average position
        st["avg_position"] = float(pos.mean())
        st["pct_long"]     = float((pos == 1).mean()) * 100
        st["pct_short"]    = float((pos ==-1).mean()) * 100
        st["pct_flat"]     = float((pos == 0).mean()) * 100
        all_stats[(m, di)] = st
        all_pos[(m, di)]   = pos

# Buy-and-hold
bah_pos = compute_buy_and_hold(daily_ret)
bah_sim = simulate_strategy(bah_pos, daily_ret)
bah_st  = compute_performance_stats(bah_sim, "Buy-and-Hold")
bah_st["avg_position"] = float(bah_pos.mean())
bah_st["pct_long"]  = 100.0
bah_st["pct_short"] = 0.0
bah_st["pct_flat"]  = 0.0

# ── Layout helpers ───────────────────────────────────────────────────────────
# Per model: one bar cluster, within it one bar per delta
N_MODELS   = len(MODELS)
N_DELTAS   = len(DELTAS)
BAR_W      = 0.18
GROUP_GAP  = 0.85          # x-distance between model group centres
DELTA_OFF  = np.linspace(-(N_DELTAS-1)/2, (N_DELTAS-1)/2, N_DELTAS) * BAR_W

MODEL_BASE_COLOR = {
    "Base":    "#2171b5",
    "Model_A": "#238b45",
    "Model_C": "#6a51a3",
}
# 4 shades per model: dark → light (matching comparison chart)
MODEL_PALETTE = {
    "Base":    ["#08306b", "#2171b5", "#6baed6", "#c6dbef"],
    "Model_A": ["#00441b", "#238b45", "#74c476", "#c7e9c0"],
    "Model_C": ["#3f007d", "#6a51a3", "#9e9ac8", "#dadaeb"],
}
BAH_COLOR  = "#d62728"
BAH_ALPHA  = 0.55

GROUP_X = np.arange(N_MODELS) * GROUP_GAP   # x-centre for each model group

METRICS = [
    ("total_ret",    "Total Return (%)",            100, False),
    ("ann_ret",      "Annualised Return (%)",        100, False),
    ("sharpe",       "Sharpe Ratio\n(0% risk-free)",   1, False),
    ("max_dd",       "Max Drawdown (%)",             100, True ),
    ("avg_position", "Average Position\n(-1 short / 0 flat / +1 long)", 1, False),
]

# ── Figure ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(
    len(METRICS), 1,
    figsize=(13, 18),
    gridspec_kw={"hspace": 0.55},
)

legend_handles = []

for ax, (metric, ylabel, scale, invert) in zip(axes, METRICS):
    # Draw one bar cluster per model
    for mi, m in enumerate(MODELS):
        x_centre = GROUP_X[mi]
        for di in range(N_DELTAS):
            val  = all_stats[(m, di)][metric] * scale
            col  = MODEL_PALETTE[m][di]
            xpos = x_centre + DELTA_OFF[di]
            bar  = ax.bar(xpos, val, width=BAR_W * 0.92,
                          color=col, edgecolor="white", linewidth=0.4,
                          label=DELTA_LBL[di] if mi == 0 else "_nolegend_")
            # Value label on bar
            v_str = (f"{val:.1f}%" if metric not in ("sharpe","avg_position")
                     else f"{val:.3f}" if metric == "sharpe"
                     else f"{val:.3f}")
            offset_y = 0.5 if val >= 0 else -0.5
            ax.text(xpos, val + offset_y, v_str,
                    ha="center", va="bottom" if val >= 0 else "top",
                    fontsize=5.5, color="#333333", rotation=90)

    # Buy-and-hold reference line
    bah_val = bah_st[metric] * scale
    ax.axhline(bah_val, color=BAH_COLOR, lw=1.4, ls="--", alpha=BAH_ALPHA,
               label=f"Buy-and-Hold ({bah_val:.1f}" +
                     ("%" if metric not in ("sharpe","avg_position") else "") + ")")

    # Zero line
    ax.axhline(0, color="black", lw=0.5)

    # x-axis: model group labels
    ax.set_xticks(GROUP_X)
    ax.set_xticklabels([MODEL_LBL[m] for m in MODELS], fontsize=10)
    ax.set_xlim(GROUP_X[0] - GROUP_GAP * 0.55, GROUP_X[-1] + GROUP_GAP * 0.55)

    ax.set_ylabel(ylabel, fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{v:.0f}%" if metric not in ("sharpe","avg_position") else f"{v:.2f}"
    ))

    # Invert y for drawdown so deeper is visually lower
    if invert:
        ax.invert_yaxis()
        ax.set_title("Max Drawdown — lower bar = worse", fontsize=9, pad=3)
    else:
        ax.set_title(ylabel.replace("\n", " "), fontsize=9, pad=3)

    # Legend inside first metric panel only; reference line in all
    if metric == METRICS[0][0]:
        delta_handles = [
            matplotlib.patches.Patch(color=MODEL_PALETTE["Base"][di],
                                     label=DELTA_LBL[di])
            for di in range(N_DELTAS)
        ]
        bah_handle = matplotlib.lines.Line2D(
            [], [], color=BAH_COLOR, lw=1.4, ls="--", label="Buy-and-Hold (ref)"
        )
        ax.legend(handles=delta_handles + [bah_handle],
                  title="Threshold (d)", fontsize=8, title_fontsize=8,
                  loc="upper right", ncol=3)

    ax.grid(axis="y", alpha=0.25, lw=0.6)
    ax.spines[["top","right"]].set_visible(False)

# ── Average position panel: add stacked long/flat/short annotation ──────────
ax_pos_panel = axes[4]
for mi, m in enumerate(MODELS):
    x_centre = GROUP_X[mi]
    for di in range(N_DELTAS):
        st   = all_stats[(m, di)]
        xpos = x_centre + DELTA_OFF[di]
        # Annotate with long%/flat%/short% below the bar
        ax_pos_panel.text(
            xpos,
            min(all_stats[(m, di)]["avg_position"] - 0.03, -0.05),
            f"L{st['pct_long']:.0f}/S{st['pct_short']:.0f}",
            ha="center", va="top", fontsize=5, color="#555555", rotation=90,
        )

fig.suptitle(
    "Experiment 2 — Performance Statistics by Model and Threshold\n"
    "Base VRP (univariate)  |  Model A (VRP + VIX Slope)  |  Model C (VRP + VVIX MA5)\n"
    "Rolling 500-day OLS  ·  Significance gate |t| > 1.28  ·  0.05% slippage per trade",
    fontsize=11, y=0.995,
)

out = OUTPUT / "perf_stats.png"
fig.savefig(out, dpi=160, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")
