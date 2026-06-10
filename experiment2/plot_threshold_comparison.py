"""
plot_threshold_comparison.py
=============================
Side-by-side comparison of the Base (univariate VRP) rolling regression
strategy at two significance thresholds: |t| > 1.96 (95% CI) vs |t| > 1.28 (80% CI).

Each threshold panel shows:
  - Cumulative net returns for all 4 delta variants
  - 4 position history sub-panels (one per delta)

Outputs:
  output/base_threshold_comparison.png
"""

import sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
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

# ── Build panel ───────────────────────────────────────────────────────────────
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

DELTAS      = [0.002, 0.005, 0.0075, 0.010]
DELTA_LBL   = ["d = 0.2%", "d = 0.5%", "d = 0.75%", "d = 1.0%"]
THRESHOLDS  = [1.96, 1.28]
T_LABELS    = {1.96: "|t| > 1.96  (95% CI)", 1.28: "|t| > 1.28  (80% CI)"}
PALETTE     = ["#08306b", "#2171b5", "#6baed6", "#9ecae1"]
LINESTYLE   = ["-", "--", "-.", ":"]
BAH_COLOR   = "#d62728"

# ── Compute all positions ─────────────────────────────────────────────────────
data = {}   # (t_thresh, di) -> (st, sim, pos)
for t in THRESHOLDS:
    for di, delta in enumerate(DELTAS):
        print(f"  t={t}  {DELTA_LBL[di]}...")
        pos = run_rolling_regression_positions(panel, "Base", delta, t_threshold=t)
        sim = simulate_strategy(pos, daily_ret)
        st  = compute_performance_stats(sim, DELTA_LBL[di])
        st["avg_position"] = float(pos.mean())
        st["pct_long"]     = float((pos == 1).mean()) * 100
        st["pct_short"]    = float((pos ==-1).mean()) * 100
        st["pct_flat"]     = float((pos == 0).mean()) * 100
        data[(t, di)] = (st, sim, pos)

bah_pos = compute_buy_and_hold(daily_ret)
bah_sim = simulate_strategy(bah_pos, daily_ret)
bah_st  = compute_performance_stats(bah_sim, "Buy-and-Hold")

def shade_crises(ax, s, e):
    for a, b in [("2008-09-01","2009-06-01"),
                 ("2020-02-01","2020-06-01"),
                 ("2022-01-01","2022-12-31")]:
        a, b = pd.Timestamp(a), pd.Timestamp(b)
        if b > s and a < e:
            ax.axvspan(max(a, s), min(b, e), alpha=0.08, color="grey", lw=0)

s_dt = daily_ret.index[0]
e_dt = daily_ret.index[-1]

# ════════════════════════════════════════════════════════════════════════════
# Figure: 2 columns (one per threshold), 5 rows (return + 4 position panels)
# ════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(22, 18))
fig.suptitle(
    "Base Model (Univariate VRP) — Significance Threshold Comparison\n"
    "Rolling 500-day OLS  ·  0.05% slippage per trade  ·  "
    "Left: |t| > 1.96 (95% CI)  |  Right: |t| > 1.28 (80% CI)",
    fontsize=12, y=0.998,
)

gs = gridspec.GridSpec(
    5, 2,
    height_ratios=[2.8, 1, 1, 1, 1],
    hspace=0.08, wspace=0.06,
    top=0.94, bottom=0.04, left=0.06, right=0.98,
)

for col, t in enumerate(THRESHOLDS):
    ax_ret  = fig.add_subplot(gs[0, col])
    ax_poss = [fig.add_subplot(gs[i+1, col], sharex=ax_ret) for i in range(4)]

    # Return panel
    shade_crises(ax_ret, s_dt, e_dt)
    ax_ret.plot(bah_sim["cum_net"].index, bah_sim["cum_net"].values,
                color=BAH_COLOR, lw=1.2, ls="-.", alpha=0.55,
                label=f"Buy-and-Hold  [SR={bah_st['sharpe']:.2f}  "
                      f"Total={bah_st['total_ret']*100:.0f}%]")
    for di in range(len(DELTAS)):
        st, sim, pos = data[(t, di)]
        ax_ret.plot(sim["cum_net"].index, sim["cum_net"].values,
                    color=PALETTE[di], lw=1.8 - di*0.15,
                    ls=LINESTYLE[di], alpha=0.92,
                    label=(f"{DELTA_LBL[di]}  "
                           f"[SR={st['sharpe']:.2f}  "
                           f"DD={st['max_dd']*100:.1f}%  "
                           f"Total={st['total_ret']*100:.0f}%  "
                           f"Trades={st['n_trades']}]"))

    ax_ret.axhline(1, color="black", lw=0.4, ls=":")
    ax_ret.set_yscale("log")
    ax_ret.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{v:.1f}x"))
    if col == 0:
        ax_ret.set_ylabel("Cumulative Net Return (log)", fontsize=9)
    ax_ret.set_title(T_LABELS[t], fontsize=11, pad=5)
    ax_ret.legend(fontsize=7.5, loc="upper left")
    plt.setp(ax_ret.get_xticklabels(), visible=False)

    # Position panels
    for di, ax_p in enumerate(ax_poss):
        st, sim, pos = data[(t, di)]
        shade_crises(ax_p, s_dt, e_dt)
        ax_p.fill_between(pos.index, pos.where(pos == 1, 0), 0,
                          color=PALETTE[di], alpha=0.75)
        ax_p.fill_between(pos.index, pos.where(pos == -1, 0), 0,
                          color=PALETTE[di], alpha=0.35, hatch="///")
        ax_p.axhline(0, color="black", lw=0.4)
        ax_p.set_ylim(-1.5, 1.5)
        ax_p.set_yticks([-1, 0, 1])
        ax_p.set_yticklabels(["Short", "Flat", "Long"], fontsize=7)
        if col == 0:
            ax_p.set_ylabel(DELTA_LBL[di], fontsize=8.5, rotation=0,
                            ha="right", va="center", labelpad=55,
                            color=PALETTE[di])
        ann = (f"Long {st['pct_long']:.1f}%   Short {st['pct_short']:.1f}%   "
               f"Flat {st['pct_flat']:.1f}%")
        ax_p.text(0.01, 0.88, ann, transform=ax_p.transAxes,
                  fontsize=7, va="top", color=PALETTE[di])
        if di < 3:
            plt.setp(ax_p.get_xticklabels(), visible=False)

    ax_poss[-1].xaxis.set_major_locator(mdates.YearLocator(2))
    ax_poss[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

out = OUTPUT / "base_threshold_comparison.png"
fig.savefig(out, dpi=155, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out}")

# ── Console summary ───────────────────────────────────────────────────────────
for t in THRESHOLDS:
    print(f"\n--- Base VRP  {T_LABELS[t]} ---")
    print(f"{'Delta':<12} {'Ann.Ret':>8} {'Sharpe':>8} {'MaxDD':>8} "
          f"{'Total':>9} {'Trades':>7} {'Long%':>7} {'Short%':>7}")
    print("-" * 72)
    for di in range(len(DELTAS)):
        st, _, _ = data[(t, di)]
        print(f"{DELTA_LBL[di]:<12} "
              f"{st['ann_ret']*100:>7.2f}% "
              f"{st['sharpe']:>8.3f} "
              f"{st['max_dd']*100:>7.1f}% "
              f"{st['total_ret']*100:>8.1f}% "
              f"{st['n_trades']:>7} "
              f"{st['pct_long']:>6.1f}% "
              f"{st['pct_short']:>6.1f}%")
