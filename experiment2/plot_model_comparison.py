"""
plot_model_comparison.py
========================
Standalone plot: cumulative returns and position history for
  - Base model (univariate VRP)
  - Model A    (VRP + VIX term structure slope)
  - Model C    (VRP + VVIX 5-day MA)

All three use rolling 500-day OLS positions with δ ∈ {0.5%, 1.0%, 2.0%}.
Buy-and-hold is included as a passive reference line on the return chart.
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
from matplotlib.lines import Line2D

ROOT   = Path(__file__).parent
OUTPUT = ROOT / "output"
sys.path.insert(0, str(ROOT.parent / "bh_replication"))
sys.path.insert(0, str(ROOT))
from experiment2 import (
    load_vrp_series, load_es_front_month, load_vix_futures_term_structure,
    load_vvix, compute_vix_term_slope, compute_trend_quotient, compute_vvix_ma5,
    build_master_panel, run_rolling_regression_positions,
    compute_buy_and_hold, simulate_strategy, compute_performance_stats, ROLL_WIN,
)

# ── Build panel ────────────────────────────────────────────────────────────
print("Building panel …")
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

# ── Generate positions ─────────────────────────────────────────────────────
DELTAS   = [0.002, 0.005, 0.0075, 0.010]
DELTA_LBL = ["d=0.2%", "d=0.5%", "d=0.75%", "d=1.0%"]
MODELS   = ["Base", "Model_A", "Model_C"]
MODEL_LBL = {
    "Base":    "Base  (VRP only)",
    "Model_A": "Model A (VRP + Term Slope)",
    "Model_C": "Model C (VRP + VVIX MA5)",
}

all_pos = {}   # (model, delta_idx) → position Series
all_sim = {}   # (model, delta_idx) → simulation DataFrame

for m in MODELS:
    for di, delta in enumerate(DELTAS):
        key = (m, di)
        print(f"  Generating positions: {MODEL_LBL[m]}  {DELTA_LBL[di]} …")
        pos = run_rolling_regression_positions(panel, m, delta)
        sim = simulate_strategy(pos, daily_ret, label=f"{m} {DELTA_LBL[di]}")
        all_pos[key] = pos
        all_sim[key] = sim

# Buy-and-hold reference
bah_pos = compute_buy_and_hold(daily_ret)
bah_sim = simulate_strategy(bah_pos, daily_ret, label="Buy-and-Hold")

print("Computing performance stats …")
stats = {}
for key, sim in all_sim.items():
    stats[key] = compute_performance_stats(sim, sim["position"].name or str(key))

# ── Colours / line styles ──────────────────────────────────────────────────
MODEL_COLOR = {
    "Base":    ("#08306b", "#2171b5", "#6baed6", "#c6dbef"),  # blues dark→light
    "Model_A": ("#00441b", "#238b45", "#74c476", "#c7e9c0"),  # greens dark→light
    "Model_C": ("#3f007d", "#6a51a3", "#9e9ac8", "#dadaeb"),  # purples dark→light
}
DELTA_STYLE = ["-", "--", "-.", ":"]
DELTA_LW    = [1.8, 1.5, 1.4, 1.3]

BAH_COLOR = "#d62728"   # red

def _shade(ax, s, e):
    for gs, ge in [("2008-09-01","2009-06-01"),
                   ("2020-02-01","2020-06-01"),
                   ("2022-01-01","2022-12-31")]:
        gs, ge = pd.Timestamp(gs), pd.Timestamp(ge)
        if ge > s and gs < e:
            ax.axvspan(max(gs, s), min(ge, e), alpha=0.07, color="grey", lw=0)

# ── Figure layout ──────────────────────────────────────────────────────────
# 5 rows: [return (tall)] + [pos Base] + [pos A] + [pos C] + [stats table]
fig = plt.figure(figsize=(17, 18))
gs  = gridspec.GridSpec(
    5, 1,
    height_ratios=[3.5, 1, 1, 1, 1.1],
    hspace=0.10,
    top=0.94, bottom=0.02, left=0.08, right=0.97,
)

ax_ret   = fig.add_subplot(gs[0])
ax_pos   = {
    "Base":    fig.add_subplot(gs[1], sharex=ax_ret),
    "Model_A": fig.add_subplot(gs[2], sharex=ax_ret),
    "Model_C": fig.add_subplot(gs[3], sharex=ax_ret),
}
ax_table = fig.add_subplot(gs[4])

s = daily_ret.index[0]
e = daily_ret.index[-1]

# ── Row 1: Cumulative returns ──────────────────────────────────────────────
_shade(ax_ret, s, e)
ax_ret.plot(bah_sim["cum_net"].index, bah_sim["cum_net"].values,
            color=BAH_COLOR, lw=1.2, ls="-.", alpha=0.6, label="Buy-and-Hold (ref)")

for m in MODELS:
    cols = MODEL_COLOR[m]
    for di in range(4):
        key = (m, di)
        cum = all_sim[key]["cum_net"]
        st  = stats[key]
        lbl = (f"{MODEL_LBL[m]}  {DELTA_LBL[di]}"
               f"  [SR={st['sharpe']:.2f}, DD={st['max_dd']*100:.1f}%]")
        ax_ret.plot(cum.index, cum.values,
                    color=cols[di], lw=DELTA_LW[di], ls=DELTA_STYLE[di],
                    alpha=0.9, label=lbl)

ax_ret.axhline(1, color="black", lw=0.5, ls=":")
ax_ret.set_ylabel("Cumulative Net Return", fontsize=10)
ax_ret.set_title(
    "Experiment 2 — Rolling Regression Strategies: Base VRP / Model A / Model C\n"
    "Net of 0.05% slippage per trade · 500-day rolling OLS · Significance gate |t| > 1.28",
    fontsize=11,
)
ax_ret.legend(fontsize=7.5, ncol=2, loc="upper left")
ax_ret.set_yscale("log")
ax_ret.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
    lambda x, _: f"{x:.1f}×"
))

# ── Rows 2–4: Position histories ──────────────────────────────────────────
POS_LABELS = {
    "Base":    "Base\n(VRP only)",
    "Model_A": "Model A\n(VRP+Slope)",
    "Model_C": "Model C\n(VRP+VVIX)",
}

for m, ax in ax_pos.items():
    _shade(ax, s, e)
    cols = MODEL_COLOR[m]
    for di in range(4):
        pos = all_pos[(m, di)]
        # Stack positions vertically with slight offsets so all three δ are visible
        offset = (di - 1) * 0.05
        long_mask  = pos == 1
        short_mask = pos == -1
        ax.fill_between(pos.index, pos.where(long_mask,  0) + offset,
                        offset, color=cols[di], alpha=0.55,
                        label=DELTA_LBL[di] if di == 0 else "_")
        ax.fill_between(pos.index, pos.where(short_mask, 0) + offset,
                        offset, color=cols[di], alpha=0.55)

    ax.axhline(0, color="black", lw=0.4)
    ax.set_ylim(-1.8, 1.8)
    ax.set_ylabel(POS_LABELS[m], fontsize=9, rotation=0, ha="right", va="center",
                  labelpad=48)
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels(["Short", "Flat", "Long"], fontsize=7)

    # Trade count annotations
    for di in range(4):
        pos = all_pos[(m, di)]
        n_trades = int((pos.diff().abs() > 0).sum())
        ax.text(0.01, 0.94 - di * 0.22,
                f"{DELTA_LBL[di]}: {n_trades} trades  "
                f"Long {(pos==1).mean()*100:.1f}%  "
                f"Short {(pos==-1).mean()*100:.1f}%",
                transform=ax.transAxes, fontsize=6.5,
                color=MODEL_COLOR[m][di])

# ── Shared x-axis formatting ───────────────────────────────────────────────
ax_pos["Model_C"].xaxis.set_major_locator(mdates.YearLocator(2))
ax_pos["Model_C"].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
for hide_ax in [ax_ret, ax_pos["Base"], ax_pos["Model_A"]]:
    plt.setp(hide_ax.get_xticklabels(), visible=False)

# ── Performance table in dedicated axes ───────────────────────────────────
bah_s = compute_performance_stats(bah_sim, "Buy-and-Hold (ref)")
table_rows = [["Strategy", "Ann.Ret", "Ann.Vol", "Sharpe", "Max DD", "Trades"]]
table_rows.append([
    bah_s["label"],
    f"{bah_s['ann_ret']*100:.2f}%",
    f"{bah_s['ann_vol']*100:.2f}%",
    f"{bah_s['sharpe']:.3f}",
    f"{bah_s['max_dd']*100:.1f}%",
    str(bah_s["n_trades"]),
])
for m in MODELS:
    for di in range(4):
        s_obj = stats[(m, di)]
        table_rows.append([
            f"{MODEL_LBL[m]}  {DELTA_LBL[di]}",
            f"{s_obj['ann_ret']*100:.2f}%",
            f"{s_obj['ann_vol']*100:.2f}%",
            f"{s_obj['sharpe']:.3f}",
            f"{s_obj['max_dd']*100:.1f}%",
            str(s_obj["n_trades"]),
        ])

ax_table.axis("off")
col_labels = table_rows[0]
cell_data  = table_rows[1:]

# Row colours: BAH in red-tinted, each model block in its lightest tint
BLOCK_TINT = {"Base": "#deebf7", "Model_A": "#e5f5e0", "Model_C": "#efedf5"}
row_colors = [["#fde0d0"] * 6]   # BAH
for m in MODELS:
    for _ in range(4):
        row_colors.append([BLOCK_TINT[m]] * 6)

tbl = ax_table.table(
    cellText=cell_data,
    colLabels=col_labels,
    cellLoc="center",
    loc="center",
    cellColours=row_colors,
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
tbl.scale(1, 1.35)
ax_table.set_title("Performance Summary (net of 0.05% slippage, 3% risk-free Sharpe)",
                   fontsize=9, pad=4)

out = OUTPUT / "model_comparison.png"
plt.savefig(out, dpi=160, bbox_inches="tight")
plt.close()
print(f"\nSaved: {out}")
