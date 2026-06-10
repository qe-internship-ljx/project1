"""
plot_model_detail.py
====================
One figure per model (Base VRP, Model A, Model C).
Each figure:
  Row 1 : Cumulative net returns — all 4 delta thresholds + buy-and-hold ref
  Rows 2-5: Position over time for each delta (d=0.1%, 0.5%, 1.0%, 2.0%)

Outputs:
  output/detail_base.png
  output/detail_model_a.png
  output/detail_model_c.png
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

# ── Positions and sims ───────────────────────────────────────────────────────
DELTAS    = [0.002, 0.005, 0.0075, 0.010]
DELTA_LBL = ["d = 0.2%", "d = 0.5%", "d = 0.75%", "d = 1.0%"]
MODELS    = ["Base", "Model_A", "Model_C"]

MODEL_META = {
    "Base":    dict(label="Base — Univariate VRP",
                    subtitle="Signal: VRP only  |  Long when predicted 20d return > d, Short when < -d",
                    fname="detail_base.png",
                    palette=["#08306b","#2171b5","#6baed6","#c6dbef"],
                    pos_color="#2171b5"),
    "Model_A": dict(label="Model A — VRP + VIX Term Structure Slope",
                    subtitle="Signals: VRP + slope  |  Both must have |t| > 1.96 in rolling window",
                    fname="detail_model_a.png",
                    palette=["#00441b","#238b45","#74c476","c7e9c0"],
                    pos_color="#238b45"),
    "Model_C": dict(label="Model C — VRP + VVIX 5-day MA",
                    subtitle="Signals: VRP + VVIX MA5  |  Both must have |t| > 1.96 in rolling window",
                    fname="detail_model_c.png",
                    palette=["#3f007d","#6a51a3","#9e9ac8","#dadaeb"],
                    pos_color="#6a51a3"),
}
# fix Model_A palette entry (no leading #)
MODEL_META["Model_A"]["palette"][3] = "#c7e9c0"

BAH_COLOR = "#d62728"

# Pre-compute BAH
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

# ── Draw one figure per model ────────────────────────────────────────────────
for m in MODELS:
    meta    = MODEL_META[m]
    palette = meta["palette"]
    print(f"\nModel: {meta['label']}")

    # Generate positions and simulations for all 4 deltas
    sims = []
    positions = []
    for di, delta in enumerate(DELTAS):
        print(f"  {DELTA_LBL[di]}...")
        pos = run_rolling_regression_positions(panel, m, delta)
        sim = simulate_strategy(pos, daily_ret)
        st  = compute_performance_stats(sim, DELTA_LBL[di])
        st["avg_position"] = float(pos.mean())
        st["pct_long"]  = float((pos == 1).mean()) * 100
        st["pct_short"] = float((pos ==-1).mean()) * 100
        st["pct_flat"]  = float((pos == 0).mean()) * 100
        positions.append(pos)
        sims.append((st, sim))

    # ── Figure: 5 rows (return + 4 position panels) ─────────────────────────
    fig = plt.figure(figsize=(15, 16))
    fig.suptitle(
        f"{meta['label']}\n{meta['subtitle']}\n"
        "Rolling 500-day OLS  ·  Significance gate |t| > 1.28  ·  0.05% slippage per trade",
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

    s_dt = daily_ret.index[0]
    e_dt = daily_ret.index[-1]

    # ── Return panel ─────────────────────────────────────────────────────────
    shade_crises(ax_ret, s_dt, e_dt)

    # Buy-and-hold reference
    cum_bah = bah_sim["cum_net"]
    ax_ret.plot(cum_bah.index, cum_bah.values,
                color=BAH_COLOR, lw=1.2, ls="-.", alpha=0.55,
                label=f"Buy-and-Hold  [SR={bah_st['sharpe']:.2f}  "
                      f"DD={bah_st['max_dd']*100:.1f}%  "
                      f"Total={bah_st['total_ret']*100:.0f}%]")

    for di, (st, sim) in enumerate(sims):
        cum = sim["cum_net"]
        ax_ret.plot(
            cum.index, cum.values,
            color=palette[di], lw=1.8 - di * 0.15,
            ls=["-","--","-.",":" ][di], alpha=0.92,
            label=(f"{DELTA_LBL[di]}  "
                   f"[SR={st['sharpe']:.2f}  "
                   f"DD={st['max_dd']*100:.1f}%  "
                   f"Total={st['total_ret']*100:.0f}%  "
                   f"Trades={st['n_trades']}]"),
        )

    ax_ret.axhline(1, color="black", lw=0.4, ls=":")
    ax_ret.set_yscale("log")
    ax_ret.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{v:.1f}x")
    )
    ax_ret.set_ylabel("Cumulative Net Return (log)", fontsize=9)
    ax_ret.legend(fontsize=8, loc="upper left")
    plt.setp(ax_ret.get_xticklabels(), visible=False)

    # ── Position panels ───────────────────────────────────────────────────────
    for di, (ax_p, pos) in enumerate(zip(ax_poss, positions)):
        shade_crises(ax_p, s_dt, e_dt)

        # Long fills
        ax_p.fill_between(
            pos.index, pos.where(pos == 1, 0), 0,
            color=palette[di], alpha=0.75, label="Long (+1)",
        )
        # Short fills
        ax_p.fill_between(
            pos.index, pos.where(pos == -1, 0), 0,
            color=palette[di], alpha=0.35, hatch="///",
            label="Short (-1)",
        )
        ax_p.axhline(0, color="black", lw=0.4)
        ax_p.set_ylim(-1.5, 1.5)
        ax_p.set_yticks([-1, 0, 1])
        ax_p.set_yticklabels(["Short", "Flat", "Long"], fontsize=7)

        st = sims[di][0]
        ax_p.set_ylabel(DELTA_LBL[di], fontsize=8.5, rotation=0,
                        ha="right", va="center", labelpad=52,
                        color=palette[di])

        # Annotation: position breakdown + avg position
        ann = (f"Long {st['pct_long']:.1f}%   "
               f"Short {st['pct_short']:.1f}%   "
               f"Flat {st['pct_flat']:.1f}%   "
               f"AvgPos={st['avg_position']:+.3f}")
        ax_p.text(0.01, 0.88, ann, transform=ax_p.transAxes,
                  fontsize=7.5, va="top", color=palette[di])

        if di < 3:
            plt.setp(ax_p.get_xticklabels(), visible=False)

    ax_poss[-1].xaxis.set_major_locator(mdates.YearLocator(2))
    ax_poss[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    out = OUTPUT / meta["fname"]
    fig.savefig(out, dpi=155, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")

print("\nAll done.")
