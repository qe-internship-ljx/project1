"""
generate_all_plots.py
=====================
Master script — generates all experiment 2 plots and organises them into:

  output/univariate_vrp/
      base_daily_t128.png
      base_daily_t196.png
      base_threshold_comparison.png
      monthly_vrp.png
      monthly_vrp_perf.png

  output/bivariate/
      model_a_t128.png  /  model_a_t196.png  /  model_a_threshold_comparison.png
      model_c_t128.png  /  model_c_t196.png  /  model_c_threshold_comparison.png

  output/comparisons/
      all_models_t128.png  /  all_models_t196.png
      perf_stats_t128.png  /  perf_stats_t196.png

Out-of-sample guarantee (daily rolling regression):
  Training window for prediction at row i uses rows [i-window-20 : i-20],
  so the most recent training label fwd_ret_20[i-21] covers days i-20..i-1
  — all fully realised before prediction date i.
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
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

ROOT    = Path(__file__).parent
OUTPUT  = ROOT / "output"
DIR_UNI = OUTPUT / "univariate_vrp"
DIR_BIV = OUTPUT / "bivariate"
DIR_CMP = OUTPUT / "comparisons"
for d in [DIR_UNI, DIR_BIV, DIR_CMP]:
    d.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT.parent / "bh_replication"))
sys.path.insert(0, str(ROOT))
from experiment2 import (
    load_vrp_series, load_es_front_month, load_vix_futures_term_structure,
    load_vvix, compute_vix_term_slope, compute_trend_quotient, compute_vvix_ma5,
    build_master_panel, run_rolling_regression_positions,
    run_monthly_vrp_positions,
    compute_buy_and_hold, simulate_strategy, compute_performance_stats,
)

# ── Build panel once ──────────────────────────────────────────────────────────
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

DELTAS     = [0.002, 0.005, 0.0075, 0.010]
DELTA_LBL  = ["d = 0.2%", "d = 0.5%", "d = 0.75%", "d = 1.0%"]
THRESHOLDS = [1.96, 1.28]
MODELS     = ["Base", "Model_A", "Model_C"]
LINESTYLE  = ["-", "--", "-.", ":"]
BAH_COLOR  = "#d62728"

MODEL_META = {
    "Base":    dict(label="Base — Univariate VRP",
                    subtitle="Daily rolling 500-day OLS  |  Signal: VRP",
                    palette=["#08306b","#2171b5","#6baed6","#9ecae1"]),
    "Model_A": dict(label="Model A — VRP + VIX Term Structure Slope",
                    subtitle="Daily rolling 500-day OLS  |  Signals: VRP + slope",
                    palette=["#00441b","#238b45","#74c476","#c7e9c0"]),
    "Model_C": dict(label="Model C — VRP + VVIX 5-day MA",
                    subtitle="Daily rolling 500-day OLS  |  Signals: VRP + VVIX MA5",
                    palette=["#3f007d","#6a51a3","#9e9ac8","#dadaeb"]),
}
MODEL_LBL = {"Base": "Base (VRP only)",
             "Model_A": "Model A (VRP+Slope)",
             "Model_C": "Model C (VRP+VVIX)"}

def t_label(t):
    ci = "95" if t == 1.96 else "80"
    return f"|t| > {t:.2f}  ({ci}% CI)"

# ── BAH reference ─────────────────────────────────────────────────────────────
bah_pos = compute_buy_and_hold(daily_ret)
bah_sim = simulate_strategy(bah_pos, daily_ret)
bah_st  = compute_performance_stats(bah_sim, "Buy-and-Hold")
bah_st.update(avg_position=1.0, pct_long=100.0, pct_short=0.0, pct_flat=0.0)

s_dt = daily_ret.index[0]
e_dt = daily_ret.index[-1]

def shade(ax):
    for a, b in [("2008-09-01","2009-06-01"),
                 ("2020-02-01","2020-06-01"),
                 ("2022-01-01","2022-12-31")]:
        a, b = pd.Timestamp(a), pd.Timestamp(b)
        if b > s_dt and a < e_dt:
            ax.axvspan(max(a, s_dt), min(b, e_dt), alpha=0.08, color="grey", lw=0)

def setup_year_axis(axes, interval=2):
    """Show year labels on every panel."""
    for ax in axes:
        ax.xaxis.set_major_locator(mdates.YearLocator(interval))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        plt.setp(ax.get_xticklabels(), visible=True, fontsize=7)
        ax.tick_params(axis="x", which="major", labelsize=7, pad=2)

# ── Compute positions (loads from cache where available) ──────────────────────
print("Computing positions (loads from cache where available)...")
POS = {}
SIM = {}
for t in THRESHOLDS:
    for m in MODELS:
        for di, delta in enumerate(DELTAS):
            print(f"  {m}  {t_label(t)}  {DELTA_LBL[di]}...")
            pos = run_rolling_regression_positions(panel, m, delta, t_threshold=t)
            sim = simulate_strategy(pos, daily_ret)
            st  = compute_performance_stats(sim, f"{m} {DELTA_LBL[di]}")
            st["avg_position"] = float(pos.mean())
            st["pct_long"]     = float((pos == 1).mean()) * 100
            st["pct_short"]    = float((pos == -1).mean()) * 100
            st["pct_flat"]     = float((pos == 0).mean()) * 100
            POS[(m, t, di)] = pos
            SIM[(m, t, di)] = (st, sim)

# ════════════════════════════════════════════════════════════════════════════
# HELPER: individual model detail (5-row: return + 4 position panels)
# ════════════════════════════════════════════════════════════════════════════
def plot_model_detail(model, t_threshold, out_path):
    meta    = MODEL_META[model]
    palette = meta["palette"]

    fig = plt.figure(figsize=(15, 17))
    fig.suptitle(
        f"{meta['label']}\n"
        f"{meta['subtitle']}  |  Significance gate {t_label(t_threshold)}\n"
        "Strictly out-of-sample  ·  Training labels end 20 days before prediction  ·  "
        "0.05% slippage per trade",
        fontsize=11, y=0.998,
    )
    gs = gridspec.GridSpec(5, 1, height_ratios=[2.8, 1, 1, 1, 1],
                           hspace=0.35, top=0.93, bottom=0.04,
                           left=0.08, right=0.97)
    ax_ret  = fig.add_subplot(gs[0])
    ax_poss = [fig.add_subplot(gs[i + 1]) for i in range(4)]

    # Set shared x-range manually (no sharex — keeps tick labels independent)
    xlim = (s_dt, e_dt)
    for ax in [ax_ret] + ax_poss:
        ax.set_xlim(*xlim)

    # Return panel
    shade(ax_ret)
    ax_ret.plot(bah_sim["cum_net"].index, bah_sim["cum_net"].values,
                color=BAH_COLOR, lw=1.2, ls="-.", alpha=0.55,
                label=f"Buy-and-Hold  [SR={bah_st['sharpe']:.2f}  "
                      f"DD={bah_st['max_dd']*100:.1f}%  "
                      f"Total={bah_st['total_ret']*100:.0f}%]")
    for di in range(4):
        st, sim = SIM[(model, t_threshold, di)]
        ax_ret.plot(sim["cum_net"].index, sim["cum_net"].values,
                    color=palette[di], lw=1.8 - di * 0.15,
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

    # Position panels
    for di, ax_p in enumerate(ax_poss):
        st, _ = SIM[(model, t_threshold, di)]
        pos   = POS[(model, t_threshold, di)]
        shade(ax_p)
        ax_p.fill_between(pos.index, pos.where(pos == 1, 0), 0,
                          color=palette[di], alpha=0.75)
        ax_p.fill_between(pos.index, pos.where(pos == -1, 0), 0,
                          color=palette[di], alpha=0.35, hatch="///")
        ax_p.axhline(0, color="black", lw=0.4)
        ax_p.set_ylim(-1.5, 1.5)
        ax_p.set_yticks([-1, 0, 1])
        ax_p.set_yticklabels(["Short", "Flat", "Long"], fontsize=7)
        ax_p.set_ylabel(DELTA_LBL[di], fontsize=8.5, rotation=0,
                        ha="right", va="center", labelpad=55, color=palette[di])
        ann = (f"Long {st['pct_long']:.1f}%   Short {st['pct_short']:.1f}%   "
               f"Flat {st['pct_flat']:.1f}%   AvgPos={st['avg_position']:+.3f}")
        ax_p.text(0.01, 0.88, ann, transform=ax_p.transAxes,
                  fontsize=7.5, va="top", color=palette[di])

    setup_year_axis([ax_ret] + ax_poss)
    fig.savefig(out_path, dpi=155, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ════════════════════════════════════════════════════════════════════════════
# HELPER: threshold comparison for one model (two columns)
# ════════════════════════════════════════════════════════════════════════════
def plot_threshold_comparison(model, out_path):
    meta    = MODEL_META[model]
    palette = meta["palette"]

    fig = plt.figure(figsize=(22, 18))
    fig.suptitle(
        f"{meta['label']} — Significance Threshold Comparison\n"
        f"{meta['subtitle']}  ·  0.05% slippage  ·  Strictly out-of-sample\n"
        "Left: |t| > 1.96 (95% CI)   |   Right: |t| > 1.28 (80% CI)",
        fontsize=12, y=0.998,
    )
    gs = gridspec.GridSpec(5, 2, height_ratios=[2.8, 1, 1, 1, 1],
                           hspace=0.35, wspace=0.06,
                           top=0.93, bottom=0.04, left=0.06, right=0.98)
    xlim = (s_dt, e_dt)

    for col, t in enumerate(THRESHOLDS):
        ax_ret  = fig.add_subplot(gs[0, col])
        ax_poss = [fig.add_subplot(gs[i + 1, col]) for i in range(4)]
        for ax in [ax_ret] + ax_poss:
            ax.set_xlim(*xlim)

        shade(ax_ret)
        ax_ret.plot(bah_sim["cum_net"].index, bah_sim["cum_net"].values,
                    color=BAH_COLOR, lw=1.2, ls="-.", alpha=0.55,
                    label=f"Buy-and-Hold  [SR={bah_st['sharpe']:.2f}  "
                          f"Total={bah_st['total_ret']*100:.0f}%]")
        for di in range(4):
            st, sim = SIM[(model, t, di)]
            ax_ret.plot(sim["cum_net"].index, sim["cum_net"].values,
                        color=palette[di], lw=1.8 - di * 0.15,
                        ls=LINESTYLE[di], alpha=0.92,
                        label=(f"{DELTA_LBL[di]}  "
                               f"[SR={st['sharpe']:.2f}  "
                               f"DD={st['max_dd']*100:.1f}%  "
                               f"Total={st['total_ret']*100:.0f}%  "
                               f"Trades={st['n_trades']}]"))
        ax_ret.axhline(1, color="black", lw=0.4, ls=":")
        ax_ret.set_yscale("log")
        ax_ret.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1f}x"))
        if col == 0:
            ax_ret.set_ylabel("Cumulative Net Return (log)", fontsize=9)
        ax_ret.set_title(t_label(t), fontsize=11, pad=5)
        ax_ret.legend(fontsize=7.5, loc="upper left")

        for di, ax_p in enumerate(ax_poss):
            st, _ = SIM[(model, t, di)]
            pos   = POS[(model, t, di)]
            shade(ax_p)
            ax_p.fill_between(pos.index, pos.where(pos == 1, 0), 0,
                              color=palette[di], alpha=0.75)
            ax_p.fill_between(pos.index, pos.where(pos == -1, 0), 0,
                              color=palette[di], alpha=0.35, hatch="///")
            ax_p.axhline(0, color="black", lw=0.4)
            ax_p.set_ylim(-1.5, 1.5)
            ax_p.set_yticks([-1, 0, 1])
            ax_p.set_yticklabels(["Short", "Flat", "Long"], fontsize=7)
            if col == 0:
                ax_p.set_ylabel(DELTA_LBL[di], fontsize=8.5, rotation=0,
                                ha="right", va="center", labelpad=55,
                                color=palette[di])
            ann = (f"Long {st['pct_long']:.1f}%   Short {st['pct_short']:.1f}%   "
                   f"Flat {st['pct_flat']:.1f}%")
            ax_p.text(0.01, 0.88, ann, transform=ax_p.transAxes,
                      fontsize=7, va="top", color=palette[di])

        setup_year_axis([ax_ret] + ax_poss)

    fig.savefig(out_path, dpi=155, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ════════════════════════════════════════════════════════════════════════════
# HELPER: all-models comparison
# ════════════════════════════════════════════════════════════════════════════
def plot_comparison(t_threshold, out_path):
    fig = plt.figure(figsize=(17, 18))
    fig.suptitle(
        f"Experiment 2 — All Models Comparison  |  {t_label(t_threshold)}\n"
        "Base VRP (univariate)  |  Model A (VRP + VIX Slope)  |  Model C (VRP + VVIX MA5)\n"
        "Rolling 500-day OLS  ·  Strictly out-of-sample  ·  0.05% slippage  ·  "
        "Deltas: 0.2%, 0.5%, 0.75%, 1.0%",
        fontsize=11, y=0.998,
    )
    gs = gridspec.GridSpec(5, 1, height_ratios=[3.5, 1, 1, 1, 1.1],
                           hspace=0.35, top=0.94, bottom=0.02,
                           left=0.08, right=0.97)
    ax_ret = fig.add_subplot(gs[0])
    ax_pos = {m: fig.add_subplot(gs[i + 1]) for i, m in enumerate(MODELS)}
    ax_tbl = fig.add_subplot(gs[4])
    xlim   = (s_dt, e_dt)
    for ax in [ax_ret] + list(ax_pos.values()):
        ax.set_xlim(*xlim)

    shade(ax_ret)
    ax_ret.plot(bah_sim["cum_net"].index, bah_sim["cum_net"].values,
                color=BAH_COLOR, lw=1.2, ls="-.", alpha=0.6,
                label="Buy-and-Hold (ref)")
    for m in MODELS:
        cols = MODEL_META[m]["palette"]
        for di in range(4):
            st, sim = SIM[(m, t_threshold, di)]
            ax_ret.plot(sim["cum_net"].index, sim["cum_net"].values,
                        color=cols[di], lw=[1.8,1.5,1.4,1.3][di],
                        ls=LINESTYLE[di], alpha=0.9,
                        label=(f"{MODEL_LBL[m]}  {DELTA_LBL[di]}"
                               f"  [SR={st['sharpe']:.2f}  DD={st['max_dd']*100:.1f}%]"))
    ax_ret.axhline(1, color="black", lw=0.5, ls=":")
    ax_ret.set_ylabel("Cumulative Net Return (log scale)", fontsize=10)
    ax_ret.set_yscale("log")
    ax_ret.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:.1f}x"))
    ax_ret.legend(fontsize=7.5, ncol=2, loc="upper left")

    POS_LABELS = {"Base":    "Base\n(VRP only)",
                  "Model_A": "Model A\n(VRP+Slope)",
                  "Model_C": "Model C\n(VRP+VVIX)"}
    for m, ax in ax_pos.items():
        shade(ax)
        cols = MODEL_META[m]["palette"]
        for di in range(4):
            pos = POS[(m, t_threshold, di)]
            off = (di - 1.5) * 0.05
            ax.fill_between(pos.index, pos.where(pos ==  1, 0) + off, off,
                            color=cols[di], alpha=0.55)
            ax.fill_between(pos.index, pos.where(pos == -1, 0) + off, off,
                            color=cols[di], alpha=0.55)
            n_tr = int((pos.diff().abs() > 0).sum())
            ax.text(0.01, 0.94 - di * 0.22,
                    f"{DELTA_LBL[di]}: {n_tr} trades  "
                    f"Long {(pos==1).mean()*100:.1f}%  "
                    f"Short {(pos==-1).mean()*100:.1f}%",
                    transform=ax.transAxes, fontsize=6.5, color=cols[di])
        ax.axhline(0, color="black", lw=0.4)
        ax.set_ylim(-1.8, 1.8)
        ax.set_ylabel(POS_LABELS[m], fontsize=9, rotation=0,
                      ha="right", va="center", labelpad=48)
        ax.set_yticks([-1, 0, 1])
        ax.set_yticklabels(["Short", "Flat", "Long"], fontsize=7)

    setup_year_axis([ax_ret] + list(ax_pos.values()))

    BLOCK_TINT = {"Base": "#deebf7", "Model_A": "#e5f5e0", "Model_C": "#efedf5"}
    rows = [["Strategy", "Ann.Ret", "Ann.Vol", "Sharpe", "Max DD", "Trades"]]
    rows.append(["Buy-and-Hold",
                 f"{bah_st['ann_ret']*100:.2f}%",
                 f"{bah_st['ann_vol']*100:.2f}%",
                 f"{bah_st['sharpe']:.3f}",
                 f"{bah_st['max_dd']*100:.1f}%", "—"])
    for m in MODELS:
        for di in range(4):
            st, _ = SIM[(m, t_threshold, di)]
            rows.append([f"{MODEL_LBL[m]}  {DELTA_LBL[di]}",
                         f"{st['ann_ret']*100:.2f}%",
                         f"{st['ann_vol']*100:.2f}%",
                         f"{st['sharpe']:.3f}",
                         f"{st['max_dd']*100:.1f}%",
                         str(st["n_trades"])])
    row_colors = [["#fde0d0"] * 6]
    for m in MODELS:
        for _ in range(4):
            row_colors.append([BLOCK_TINT[m]] * 6)
    ax_tbl.axis("off")
    tbl = ax_tbl.table(cellText=rows[1:], colLabels=rows[0],
                       cellLoc="center", loc="center", cellColours=row_colors)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.35)
    ax_tbl.set_title(
        f"Performance Summary  |  {t_label(t_threshold)}  |  "
        "Net of 0.05% slippage  |  0% risk-free Sharpe",
        fontsize=9, pad=4)

    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ════════════════════════════════════════════════════════════════════════════
# HELPER: performance stats bar chart
# ════════════════════════════════════════════════════════════════════════════
def plot_perf_stats(t_threshold, out_path):
    METRICS = [
        ("total_ret",    "Total Return (%)",                              100, False),
        ("ann_ret",      "Annualised Return (%)",                         100, False),
        ("sharpe",       "Sharpe Ratio (0% risk-free)",                     1, False),
        ("max_dd",       "Max Drawdown (%)",                               100, True ),
        ("avg_position", "Average Position\n(-1 short / 0 flat / +1 long)", 1, False),
    ]
    N_D = 4
    BAR_W = 0.18
    GROUP_GAP = 0.85
    DELTA_OFF = np.linspace(-(N_D - 1) / 2, (N_D - 1) / 2, N_D) * BAR_W
    GROUP_X   = np.arange(len(MODELS)) * GROUP_GAP

    fig, axes = plt.subplots(len(METRICS), 1, figsize=(13, 18),
                             gridspec_kw={"hspace": 0.55})
    fig.suptitle(
        f"Experiment 2 — Performance Statistics  |  {t_label(t_threshold)}\n"
        "Base VRP (univariate)  |  Model A (VRP + VIX Slope)  |  Model C (VRP + VVIX MA5)\n"
        "Rolling 500-day OOS OLS  ·  0.05% slippage per trade",
        fontsize=11, y=0.998,
    )

    for ax, (metric, ylabel, scale, invert) in zip(axes, METRICS):
        for mi, m in enumerate(MODELS):
            cols = MODEL_META[m]["palette"]
            for di in range(N_D):
                st, _ = SIM[(m, t_threshold, di)]
                val  = st[metric] * scale
                xpos = GROUP_X[mi] + DELTA_OFF[di]
                ax.bar(xpos, val, width=BAR_W * 0.92, color=cols[di],
                       edgecolor="white", lw=0.4,
                       label=DELTA_LBL[di] if mi == 0 else "_nolegend_")
                fmt = (f"{val:.1f}%" if metric not in ("sharpe", "avg_position")
                       else f"{val:.3f}")
                off = 0.5 if val >= 0 else -0.5
                ax.text(xpos, val + off, fmt, ha="center",
                        va="bottom" if val >= 0 else "top",
                        fontsize=5.5, color="#333333", rotation=90)

        bv = bah_st[metric] * scale
        ax.axhline(bv, color=BAH_COLOR, lw=1.4, ls="--", alpha=0.55,
                   label=f"Buy-and-Hold ({bv:.1f}"
                         + ("%" if metric not in ("sharpe", "avg_position") else "") + ")")
        ax.axhline(0, color="black", lw=0.5)
        ax.set_xticks(GROUP_X)
        ax.set_xticklabels([MODEL_LBL[m] for m in MODELS], fontsize=10)
        ax.set_xlim(GROUP_X[0] - GROUP_GAP * 0.55, GROUP_X[-1] + GROUP_GAP * 0.55)
        ax.set_ylabel(ylabel, fontsize=9)
        if invert:
            ax.invert_yaxis()
        if metric == METRICS[0][0]:
            handles = [mpatches.Patch(color=MODEL_META["Base"]["palette"][di],
                                      label=DELTA_LBL[di]) for di in range(N_D)]
            handles += [mlines.Line2D([], [], color=BAH_COLOR, lw=1.4, ls="--",
                                      label="Buy-and-Hold")]
            ax.legend(handles=handles, title="Threshold (d)", fontsize=8,
                      title_fontsize=8, loc="upper right", ncol=3)
        if metric == "avg_position":
            for mi, m in enumerate(MODELS):
                for di in range(N_D):
                    st, _ = SIM[(m, t_threshold, di)]
                    xpos = GROUP_X[mi] + DELTA_OFF[di]
                    ax.text(xpos, min(st["avg_position"] - 0.03, -0.05),
                            f"L{st['pct_long']:.0f}/S{st['pct_short']:.0f}",
                            ha="center", va="top", fontsize=5,
                            color="#555555", rotation=90)
        ax.grid(axis="y", alpha=0.25, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)

    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ════════════════════════════════════════════════════════════════════════════
# HELPER: monthly VRP signal
# ════════════════════════════════════════════════════════════════════════════
def plot_monthly(out_detail, out_perf):
    M_PAL = ["#08306b", "#2171b5", "#6baed6", "#9ecae1"]
    m_sims, m_poss = [], []
    for di, delta in enumerate(DELTAS):
        print(f"  Monthly  {DELTA_LBL[di]}...")
        pos = run_monthly_vrp_positions(panel, delta)
        sim = simulate_strategy(pos, daily_ret)
        st  = compute_performance_stats(sim, DELTA_LBL[di])
        st["avg_position"] = float(pos.mean())
        st["pct_long"]  = float((pos == 1).mean()) * 100
        st["pct_short"] = float((pos == -1).mean()) * 100
        st["pct_flat"]  = float((pos == 0).mean()) * 100
        m_poss.append(pos)
        m_sims.append((st, sim))

    xlim = (s_dt, e_dt)

    # Detail figure
    fig = plt.figure(figsize=(15, 17))
    fig.suptitle(
        "Monthly VRP Signal — Base Model (Univariate VRP)\n"
        "Rolling 60-month OLS: VRP → next-month return  |  "
        "Significance gate |t| > 1.28  (NW 3 lags, non-overlapping monthly obs)\n"
        "Strictly out-of-sample  ·  Position set at month-end, held for following month  ·  "
        "0.05% slippage",
        fontsize=11, y=0.998,
    )
    gs = gridspec.GridSpec(5, 1, height_ratios=[2.8, 1, 1, 1, 1],
                           hspace=0.35, top=0.93, bottom=0.04,
                           left=0.08, right=0.97)
    ax_ret  = fig.add_subplot(gs[0])
    ax_poss = [fig.add_subplot(gs[i + 1]) for i in range(4)]
    for ax in [ax_ret] + ax_poss:
        ax.set_xlim(*xlim)

    shade(ax_ret)
    ax_ret.plot(bah_sim["cum_net"].index, bah_sim["cum_net"].values,
                color=BAH_COLOR, lw=1.2, ls="-.", alpha=0.55,
                label=f"Buy-and-Hold  [SR={bah_st['sharpe']:.2f}  "
                      f"DD={bah_st['max_dd']*100:.1f}%  "
                      f"Total={bah_st['total_ret']*100:.0f}%]")
    for di, (st, sim) in enumerate(m_sims):
        ax_ret.plot(sim["cum_net"].index, sim["cum_net"].values,
                    color=M_PAL[di], lw=1.8 - di * 0.15, ls=LINESTYLE[di], alpha=0.92,
                    label=(f"{DELTA_LBL[di]}  [SR={st['sharpe']:.2f}  "
                           f"DD={st['max_dd']*100:.1f}%  "
                           f"Total={st['total_ret']*100:.0f}%  "
                           f"Trades={st['n_trades']}]"))
    ax_ret.axhline(1, color="black", lw=0.4, ls=":")
    ax_ret.set_yscale("log")
    ax_ret.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1f}x"))
    ax_ret.set_ylabel("Cumulative Net Return (log)", fontsize=9)
    ax_ret.legend(fontsize=8, loc="upper left")

    for di, (ax_p, pos) in enumerate(zip(ax_poss, m_poss)):
        st = m_sims[di][0]
        shade(ax_p)
        ax_p.fill_between(pos.index, pos.where(pos == 1, 0), 0,
                          color=M_PAL[di], alpha=0.75)
        ax_p.fill_between(pos.index, pos.where(pos == -1, 0), 0,
                          color=M_PAL[di], alpha=0.35, hatch="///")
        ax_p.axhline(0, color="black", lw=0.4)
        ax_p.set_ylim(-1.5, 1.5)
        ax_p.set_yticks([-1, 0, 1])
        ax_p.set_yticklabels(["Short", "Flat", "Long"], fontsize=7)
        ax_p.set_ylabel(DELTA_LBL[di], fontsize=8.5, rotation=0,
                        ha="right", va="center", labelpad=55, color=M_PAL[di])
        ann = (f"Long {st['pct_long']:.1f}%   Short {st['pct_short']:.1f}%   "
               f"Flat {st['pct_flat']:.1f}%   AvgPos={st['avg_position']:+.3f}")
        ax_p.text(0.01, 0.88, ann, transform=ax_p.transAxes,
                  fontsize=7.5, va="top", color=M_PAL[di])

    setup_year_axis([ax_ret] + ax_poss)
    fig.savefig(out_detail, dpi=155, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_detail.name}")

    # Perf bar chart
    METRICS = [
        ("total_ret",    "Total Return (%)",               100, False),
        ("ann_ret",      "Annualised Return (%)",           100, False),
        ("sharpe",       "Sharpe Ratio (0% risk-free)",       1, False),
        ("max_dd",       "Max Drawdown (%)",                100, True ),
        ("avg_position", "Avg Position (+1 long / -1 short)", 1, False),
    ]
    fig2, axes = plt.subplots(len(METRICS), 1, figsize=(10, 16),
                              gridspec_kw={"hspace": 0.55})
    fig2.suptitle(
        "Monthly VRP Signal — Performance Statistics\n"
        "Rolling 60-month OOS OLS  |  Significance gate |t| > 1.28  |  0.05% slippage",
        fontsize=11, y=0.998,
    )
    xpos = np.arange(4)
    for ax, (metric, ylabel, scale, invert) in zip(axes, METRICS):
        vals = [m_sims[di][0][metric] * scale for di in range(4)]
        ax.bar(xpos, vals, width=0.55, color=M_PAL, edgecolor="white", lw=0.5)
        for xi, v in zip(xpos, vals):
            fmt = f"{v:.1f}%" if metric not in ("sharpe", "avg_position") else f"{v:.3f}"
            ax.text(xi, v + (0.3 if v >= 0 else -0.3), fmt,
                    ha="center", va="bottom" if v >= 0 else "top",
                    fontsize=8, color="#333333")
        bv = bah_st[metric] * scale
        ax.axhline(bv, color=BAH_COLOR, lw=1.4, ls="--", alpha=0.6,
                   label=f"Buy-and-Hold ({bv:.1f}"
                         + ("%" if metric not in ("sharpe", "avg_position") else "") + ")")
        ax.axhline(0, color="black", lw=0.5)
        ax.set_xticks(xpos)
        ax.set_xticklabels(DELTA_LBL, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(fontsize=8, loc="upper right")
        if invert:
            ax.invert_yaxis()
        ax.grid(axis="y", alpha=0.25, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        if metric == "avg_position":
            for di in range(4):
                st = m_sims[di][0]
                ax.text(di, min(vals[di] - 0.03, -0.04),
                        f"L{st['pct_long']:.0f}/S{st['pct_short']:.0f}",
                        ha="center", va="top", fontsize=7, color="#555555")
    fig2.savefig(out_perf, dpi=155, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved: {out_perf.name}")


# ════════════════════════════════════════════════════════════════════════════
# RUN ALL
# ════════════════════════════════════════════════════════════════════════════
print("\n--- univariate_vrp/ ---")
plot_model_detail("Base", 1.96, DIR_UNI / "base_daily_t196.png")
plot_model_detail("Base", 1.28, DIR_UNI / "base_daily_t128.png")
plot_threshold_comparison("Base", DIR_UNI / "base_threshold_comparison.png")
plot_monthly(DIR_UNI / "monthly_vrp.png", DIR_UNI / "monthly_vrp_perf.png")

print("\n--- bivariate/ ---")
for m, tag in [("Model_A", "model_a"), ("Model_C", "model_c")]:
    plot_model_detail(m, 1.96, DIR_BIV / f"{tag}_t196.png")
    plot_model_detail(m, 1.28, DIR_BIV / f"{tag}_t128.png")
    plot_threshold_comparison(m, DIR_BIV / f"{tag}_threshold_comparison.png")

print("\n--- comparisons/ ---")
plot_comparison(1.96, DIR_CMP / "all_models_t196.png")
plot_comparison(1.28, DIR_CMP / "all_models_t128.png")
plot_perf_stats(1.96, DIR_CMP / "perf_stats_t196.png")
plot_perf_stats(1.28, DIR_CMP / "perf_stats_t128.png")

print("\nAll done.")
