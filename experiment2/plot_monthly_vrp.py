"""
plot_monthly_vrp.py
===================
Monthly VRP signal strategy:
  - At each month-end, rolling 60-month OLS: VRP -> next-month return
  - Significance gate |t| > 1.28  (NW 3 lags, non-overlapping monthly obs)
  - delta thresholds: [0.2%, 0.5%, 0.75%, 1.0%] of monthly return

Outputs:
  output/monthly_vrp.png   — cumulative returns + position history
  output/monthly_vrp_perf.png — performance stats vs buy-and-hold
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
    build_master_panel, run_monthly_vrp_positions,
    compute_buy_and_hold, simulate_strategy, compute_performance_stats,
)

# ── Build panel ──────────────────────────────────────────────────────────────
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

DELTAS    = [0.002, 0.005, 0.0075, 0.010]
DELTA_LBL = ["d = 0.2%", "d = 0.5%", "d = 0.75%", "d = 1.0%"]
PALETTE   = ["#08306b", "#2171b5", "#6baed6", "#9ecae1"]
LINESTYLE = ["-", "--", "-.", ":"]
BAH_COLOR = "#d62728"

# ── Compute positions and sims ───────────────────────────────────────────────
sims      = []
positions = []
for di, delta in enumerate(DELTAS):
    print(f"  {DELTA_LBL[di]}...")
    pos = run_monthly_vrp_positions(panel, delta)
    sim = simulate_strategy(pos, daily_ret)
    st  = compute_performance_stats(sim, DELTA_LBL[di])
    st["avg_position"] = float(pos.mean())
    st["pct_long"]     = float((pos == 1).mean()) * 100
    st["pct_short"]    = float((pos ==-1).mean()) * 100
    st["pct_flat"]     = float((pos == 0).mean()) * 100
    positions.append(pos)
    sims.append((st, sim))

bah_pos = compute_buy_and_hold(daily_ret)
bah_sim = simulate_strategy(bah_pos, daily_ret)
bah_st  = compute_performance_stats(bah_sim, "Buy-and-Hold")
bah_st["avg_position"] = 1.0
bah_st["pct_long"]  = 100.0
bah_st["pct_short"] = 0.0
bah_st["pct_flat"]  = 0.0

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
# Figure 1: Returns + position panels
# ════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(15, 16))
fig.suptitle(
    "Monthly VRP Signal Strategy\n"
    "Rolling 60-month OLS: VRP → next-month return  |  Significance gate |t| > 1.28  |  "
    "0.05% slippage per trade\n"
    "Position set at each month-end, held constant through following month",
    fontsize=11, y=0.995,
)
gs = gridspec.GridSpec(
    5, 1,
    height_ratios=[2.8, 1, 1, 1, 1],
    hspace=0.08,
    top=0.93, bottom=0.04, left=0.08, right=0.97,
)
ax_ret  = fig.add_subplot(gs[0])
ax_poss = [fig.add_subplot(gs[i+1], sharex=ax_ret) for i in range(4)]

# Return panel
shade_crises(ax_ret, s_dt, e_dt)
cum_bah = bah_sim["cum_net"]
ax_ret.plot(cum_bah.index, cum_bah.values,
            color=BAH_COLOR, lw=1.2, ls="-.", alpha=0.55,
            label=f"Buy-and-Hold  [SR={bah_st['sharpe']:.2f}  "
                  f"DD={bah_st['max_dd']*100:.1f}%  "
                  f"Total={bah_st['total_ret']*100:.0f}%]")

for di, (st, sim) in enumerate(sims):
    cum = sim["cum_net"]
    ax_ret.plot(cum.index, cum.values,
                color=PALETTE[di], lw=1.8 - di*0.15,
                ls=LINESTYLE[di], alpha=0.92,
                label=(f"{DELTA_LBL[di]}  "
                       f"[SR={st['sharpe']:.2f}  "
                       f"DD={st['max_dd']*100:.1f}%  "
                       f"Total={st['total_ret']*100:.0f}%  "
                       f"Trades={st['n_trades']}]"))

ax_ret.axhline(1, color="black", lw=0.4, ls=":")
ax_ret.set_yscale("log")
ax_ret.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1f}x"))
ax_ret.set_ylabel("Cumulative Net Return (log)", fontsize=9)
ax_ret.legend(fontsize=8, loc="upper left")
plt.setp(ax_ret.get_xticklabels(), visible=False)

# Position panels
for di, (ax_p, pos) in enumerate(zip(ax_poss, positions)):
    shade_crises(ax_p, s_dt, e_dt)
    ax_p.fill_between(pos.index, pos.where(pos == 1, 0), 0,
                      color=PALETTE[di], alpha=0.75)
    ax_p.fill_between(pos.index, pos.where(pos == -1, 0), 0,
                      color=PALETTE[di], alpha=0.35, hatch="///")
    ax_p.axhline(0, color="black", lw=0.4)
    ax_p.set_ylim(-1.5, 1.5)
    ax_p.set_yticks([-1, 0, 1])
    ax_p.set_yticklabels(["Short", "Flat", "Long"], fontsize=7)
    ax_p.set_ylabel(DELTA_LBL[di], fontsize=8.5, rotation=0,
                    ha="right", va="center", labelpad=55, color=PALETTE[di])
    st = sims[di][0]
    ann = (f"Long {st['pct_long']:.1f}%   Short {st['pct_short']:.1f}%   "
           f"Flat {st['pct_flat']:.1f}%   AvgPos={st['avg_position']:+.3f}")
    ax_p.text(0.01, 0.88, ann, transform=ax_p.transAxes,
              fontsize=7.5, va="top", color=PALETTE[di])
    if di < 3:
        plt.setp(ax_p.get_xticklabels(), visible=False)

ax_poss[-1].xaxis.set_major_locator(mdates.YearLocator(2))
ax_poss[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

out1 = OUTPUT / "monthly_vrp.png"
fig.savefig(out1, dpi=155, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out1}")

# ════════════════════════════════════════════════════════════════════════════
# Figure 2: Performance stats bar chart
# ════════════════════════════════════════════════════════════════════════════
METRICS = [
    ("total_ret",    "Total Return (%)",                        100, False),
    ("ann_ret",      "Annualised Return (%)",                   100, False),
    ("sharpe",       "Sharpe Ratio (3% risk-free)",               1, False),
    ("max_dd",       "Max Drawdown (%)",                         100, True ),
    ("avg_position", "Average Position\n(+1 long / 0 flat / -1 short)", 1, False),
]

N_D   = len(DELTAS)
BAR_W = 0.55
x     = np.arange(N_D)

fig2, axes = plt.subplots(len(METRICS), 1, figsize=(10, 16),
                           gridspec_kw={"hspace": 0.55})
fig2.suptitle(
    "Monthly VRP Signal — Performance Statistics\n"
    "Rolling 60-month OLS  |  |t| > 1.28  |  0.05% slippage",
    fontsize=11, y=0.995,
)

for ax, (metric, ylabel, scale, invert) in zip(axes, METRICS):
    vals = [sims[di][0][metric] * scale for di in range(N_D)]
    bars = ax.bar(x, vals, width=BAR_W, color=PALETTE, edgecolor="white", lw=0.5)
    for xi, v in zip(x, vals):
        fmt = f"{v:.1f}%" if metric not in ("sharpe", "avg_position") else f"{v:.3f}"
        ax.text(xi, v + (0.3 if v >= 0 else -0.3), fmt,
                ha="center", va="bottom" if v >= 0 else "top",
                fontsize=8, color="#333333")
    bah_val = bah_st[metric] * scale
    ax.axhline(bah_val, color=BAH_COLOR, lw=1.4, ls="--", alpha=0.6,
               label=f"Buy-and-Hold ({bah_val:.1f}" +
                     ("%" if metric not in ("sharpe", "avg_position") else "") + ")")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(DELTA_LBL, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.legend(fontsize=8, loc="upper right")
    if invert:
        ax.invert_yaxis()
    ax.grid(axis="y", alpha=0.25, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)

    if metric == "avg_position":
        for di in range(N_D):
            st = sims[di][0]
            ax.text(di, min(vals[di] - 0.03, -0.04),
                    f"L{st['pct_long']:.0f}/S{st['pct_short']:.0f}",
                    ha="center", va="top", fontsize=7, color="#555555")

out2 = OUTPUT / "monthly_vrp_perf.png"
fig2.savefig(out2, dpi=155, bbox_inches="tight")
plt.close(fig2)
print(f"Saved: {out2}")

# ── Console summary ───────────────────────────────────────────────────────────
print("\n--- Monthly VRP Strategy Performance ---")
print(f"{'Delta':<10} {'Ann.Ret':>8} {'Sharpe':>8} {'MaxDD':>8} "
      f"{'Total':>9} {'Trades':>7} {'Long%':>7} {'Short%':>7}")
print("-" * 70)
for di, (st, _) in enumerate(sims):
    print(f"{DELTA_LBL[di]:<10} "
          f"{st['ann_ret']*100:>7.2f}% "
          f"{st['sharpe']:>8.3f} "
          f"{st['max_dd']*100:>7.1f}% "
          f"{st['total_ret']*100:>8.1f}% "
          f"{st['n_trades']:>7} "
          f"{st['pct_long']:>6.1f}% "
          f"{st['pct_short']:>6.1f}%")
bah_tot = bah_st['total_ret']*100
print(f"\nBuy-and-Hold: Ann={bah_st['ann_ret']*100:.2f}%  "
      f"SR={bah_st['sharpe']:.3f}  DD={bah_st['max_dd']*100:.1f}%  "
      f"Total={bah_tot:.1f}%")
