"""
expanding_window_eval.py  (file kept as fixed_split_eval.py for import compatibility)
======================================================================================
Two expanding-window OOS evaluation designs for Base, Model A, Model C.

  1. Expanding window — fixed threshold (EW)
     Training grows from >= 500 days to full history; at each day t the OLS is
     re-estimated on sub.iloc[0 : t-20].  Position gated on NW t-stats (all
     betas |t| > 1.28).  Threshold: long when y_hat > delta, short < -delta.

  2. Expanding window — rolling-mu threshold (EW-rolmu)
     Same OLS and t-gate, but the threshold is centred on the 500-day rolling
     mean of realised 20-day returns: long when y_hat > mu + delta,
     short when y_hat < mu - delta.

OOS guarantee
-------------
  fwd_ret_20[t]  = cumulative return from day t+1 to t+20 (realised at t+20).
  Training window at prediction row i: sub.iloc[0 : i-20].
    Last label = fwd_ret_20[i-21], which uses prices ending at day i-1. ✓
  Rolling mu     = mean of fwd_ret_20 over the same training slice. ✓
  Position pos[i] applied to daily_ret[i+1] via pos.shift(1) in
    simulate_strategy. ✓
  Performance stats computed on OOS-only slice (>= OOS_START). ✓

Plots saved to output/expanding_window/ and output/expanding_window_rolmu/.
"""

import sys, warnings, shutil
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

ROOT    = Path(__file__).parent
OUTPUT  = ROOT / "output"
DIR_FS  = OUTPUT / "expanding_window"
DIR_FS.mkdir(parents=True, exist_ok=True)

CACHE_DIR = OUTPUT / "regression_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Remove legacy folders if they still exist under old names
for _old in [OUTPUT / "fixed_split", OUTPUT / "fixed_split_rolmu"]:
    if _old.exists():
        shutil.rmtree(_old)
        print(f"  Removed legacy folder: {_old.name}")

sys.path.insert(0, str(ROOT.parent / "bh_replication"))
sys.path.insert(0, str(ROOT))

from statsmodels.api import OLS, add_constant
from experiment2 import (
    load_vrp_series, load_es_front_month, load_vix_futures_term_structure,
    load_vvix, compute_vix_term_slope, compute_trend_quotient, compute_vvix_ma5,
    build_master_panel,
    run_rolling_regression_positions,          # for comparison
    compute_buy_and_hold, simulate_strategy, compute_performance_stats,
)
from har_model import _nw_se

# ── Constants ─────────────────────────────────────────────────────────────────
TRAIN_END  = "2011-12-31"
OOS_START  = "2012-01-01"
MIN_WIN    = 500          # expanding window minimum
T_THRESH   = 1.28
NW_LAGS    = 20

DELTAS    = [0.002, 0.005, 0.0075, 0.010]
DELTA_LBL = ["d=0.2%", "d=0.5%", "d=0.75%", "d=1.0%"]
MODELS    = ["Base", "Model_A", "Model_C"]

MODEL_FEATURES = {
    "Base":       ["VP"],
    "Model_A":    ["VP", "term_slope"],
    "Model_C":    ["VP", "vvix_ma5"],
    "Model_VVIX": ["vvix_ma5"],
}
MODEL_LABEL = {
    "Base":       "Base — Univariate VRP",
    "Model_A":    "Model A — VRP + Term Slope",
    "Model_C":    "Model C — VRP + VVIX MA5",
    "Model_VVIX": "Model VVIX — Univariate VVIX MA5",
}
MODEL_PALETTE = {
    "Base":       ["#08306b", "#2171b5", "#6baed6", "#9ecae1"],
    "Model_A":    ["#00441b", "#238b45", "#74c476", "#c7e9c0"],
    "Model_C":    ["#3f007d", "#6a51a3", "#9e9ac8", "#dadaeb"],
    "Model_VVIX": ["#7f2704", "#d94801", "#fd8d3c", "#fdbe85"],
}
LINESTYLE = ["-", "--", "-.", ":"]
BAH_COLOR = "#d62728"

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

s_dt = daily_ret.index[0]
e_dt = daily_ret.index[-1]
oos_dt = pd.Timestamp(OOS_START)

bah_pos = compute_buy_and_hold(daily_ret)
bah_sim = simulate_strategy(bah_pos, daily_ret)
bah_st  = compute_performance_stats(bah_sim, "Buy-and-Hold")
bah_st.update(avg_position=1.0, pct_long=100.0, pct_short=0.0, pct_flat=0.0)


# ════════════════════════════════════════════════════════════════════════════
# 1. FIXED SPLIT
# ════════════════════════════════════════════════════════════════════════════

def run_fixed_split(panel, model, delta):
    """
    Single OLS fit on training data; fixed coefficients applied to OOS.

    OOS gap: the last training label fwd_ret_20[j] covers days j+1..j+20.
    To avoid any overlap with the OOS period, training is cut 20 rows before
    the first OOS row, so the last label's 20-day window closes before OOS_START.

    Returns (daily positions over full index, fitted res, nw SEs, t-stats,
             n_train, n_oos).
    """
    feat_cols = MODEL_FEATURES[model]
    sub = panel.dropna(subset=feat_cols + ["fwd_ret_20"]).copy()

    oos_start_row = sub.index.searchsorted(pd.Timestamp(OOS_START))
    # Last safe training row: oos_start_row - 21 (label ends at oos_start_row - 1)
    train = sub.iloc[: oos_start_row - 20]
    oos   = sub.iloc[oos_start_row :]

    X_tr = add_constant(train[feat_cols], has_constant="skip")
    res  = OLS(train["fwd_ret_20"], X_tr).fit()
    nw   = _nw_se(res, nlags=NW_LAGS)
    t_stats = res.params.values[1:] / nw[1:]

    # Vectorised prediction on OOS rows
    X_oos = add_constant(oos[feat_cols], has_constant="skip")
    y_hat = res.predict(X_oos)
    oos_rmse = float(np.sqrt(np.mean((y_hat.values - oos["fwd_ret_20"].values) ** 2)))

    pos = pd.Series(0.0, index=panel.index, name=f"pos_FS_{model}_d{delta}")
    pos.loc[oos.index] = np.where(y_hat >  delta,  1.0,
                          np.where(y_hat < -delta, -1.0, 0.0))
    return pos, res, nw, t_stats, len(train), len(oos), oos_rmse


# ════════════════════════════════════════════════════════════════════════════
# 2. EXPANDING WINDOW
# ════════════════════════════════════════════════════════════════════════════

def run_expanding_window(panel, model, delta, t_threshold=T_THRESH):
    """
    Expanding window starting OOS from OOS_START.
    Training uses all history from the start up to (i - 20) rows before prediction.
    Predictions only made for dates >= OOS_START, so the full pre-2012 history
    is used for training but no positions are taken before OOS_START.
    """
    feat_cols = MODEL_FEATURES[model]
    tag = f"pos_EW_{model}_d{int(delta*10000)}bps_t{int(t_threshold*100)}_oos{OOS_START}.parquet"
    cache_path = CACHE_DIR / tag
    if cache_path.exists():
        s = pd.read_parquet(cache_path).squeeze()
        s.name = f"pos_EW_{model}_d{delta}"
        return s

    sub = panel.dropna(subset=feat_cols + ["fwd_ret_20"]).copy()
    N   = len(sub)
    pos = pd.Series(0.0, index=sub.index, name=f"pos_EW_{model}_d{delta}")

    # First prediction row: first date >= OOS_START (or MIN_WIN+20, whichever is later)
    oos_idx  = sub.index.searchsorted(pd.Timestamp(OOS_START))
    start_i  = max(MIN_WIN + 20, oos_idx)

    for i in range(start_i, N):
        train = sub.iloc[0 : i - 20]          # all history up to i-21
        X_tr  = add_constant(train[feat_cols], has_constant="skip")
        res   = OLS(train["fwd_ret_20"], X_tr).fit()
        nw    = _nw_se(res, nlags=NW_LAGS)
        t_stats = res.params.values[1:] / nw[1:]
        if not np.all(np.abs(t_stats) > t_threshold):
            continue
        test_row = sub.iloc[[i]][feat_cols].copy()
        test_row.insert(0, "const", 1.0)
        y_hat = float(res.predict(test_row).iloc[0])
        if   y_hat >  delta: pos.iloc[i] =  1.0
        elif y_hat < -delta: pos.iloc[i] = -1.0

    pos.to_frame().to_parquet(cache_path)
    return pos


def run_expanding_window_rolmu(panel, model, delta, t_threshold=T_THRESH,
                                rolling_window=500):
    """
    Same as run_expanding_window but with a rolling-mean-adjusted threshold.
    At prediction day i:
        mu   = mean of fwd_ret_20 over the last `rolling_window` training rows
        long  when y_hat > mu + delta
        short when y_hat < mu - delta
    This benchmarks the predicted return against the recent historical average
    rather than against a fixed zero-centred band.
    """
    feat_cols = MODEL_FEATURES[model]
    tag = (f"pos_EW_rolmu_{model}_d{int(delta*10000)}bps"
           f"_t{int(t_threshold*100)}_rw{rolling_window}_oos{OOS_START}.parquet")
    cache_path = CACHE_DIR / tag
    if cache_path.exists():
        s = pd.read_parquet(cache_path).squeeze()
        s.name = f"pos_EWrm_{model}_d{delta}"
        return s

    sub = panel.dropna(subset=feat_cols + ["fwd_ret_20"]).copy()
    N   = len(sub)
    pos = pd.Series(0.0, index=sub.index, name=f"pos_EWrm_{model}_d{delta}")

    oos_idx = sub.index.searchsorted(pd.Timestamp(OOS_START))
    start_i = max(MIN_WIN + 20, oos_idx)

    fwd = sub["fwd_ret_20"].values   # pre-extract for speed

    for i in range(start_i, N):
        train = sub.iloc[0 : i - 20]
        X_tr  = add_constant(train[feat_cols], has_constant="skip")
        res   = OLS(train["fwd_ret_20"], X_tr).fit()
        nw    = _nw_se(res, nlags=NW_LAGS)
        t_stats = res.params.values[1:] / nw[1:]
        if not np.all(np.abs(t_stats) > t_threshold):
            continue

        # Rolling-mean threshold: use last rolling_window obs from training set
        lo = max(0, i - 20 - rolling_window)
        mu = float(np.mean(fwd[lo : i - 20]))

        test_row = sub.iloc[[i]][feat_cols].copy()
        test_row.insert(0, "const", 1.0)
        y_hat = float(res.predict(test_row).iloc[0])

        if   y_hat > mu + delta: pos.iloc[i] =  1.0
        elif y_hat < mu - delta: pos.iloc[i] = -1.0

    pos.to_frame().to_parquet(cache_path)
    return pos


# ── Pre-compute all positions ─────────────────────────────────────────────────
print("Computing fixed-split positions (fast)...")
FS  = {}   # (model, di) -> (pos, t_stats, n_train, n_oos)
FS_SIM = {}

for m in MODELS:
    for di, delta in enumerate(DELTAS):
        pos, res, nw, t_stats, n_tr, n_oos, oos_rmse = run_fixed_split(panel, m, delta)
        sim = simulate_strategy(pos, daily_ret)
        st  = compute_performance_stats(sim, f"FS_{m}_{DELTA_LBL[di]}")
        oos_pos = pos[pos.index >= OOS_START]
        st["avg_position"] = float(oos_pos.mean())
        st["pct_long"]  = float((oos_pos ==  1).mean()) * 100
        st["pct_short"] = float((oos_pos == -1).mean()) * 100
        st["pct_flat"]  = float((oos_pos ==  0).mean()) * 100
        FS[(m, di)]     = (pos, res, nw, t_stats, n_tr, n_oos, oos_rmse)
        FS_SIM[(m, di)] = (st, sim)

print("Computing expanding-window positions (slow, cached after first run)...")
EW  = {}
EW_SIM = {}
for m in MODELS:
    for di, delta in enumerate(DELTAS):
        print(f"  EW  {m}  {DELTA_LBL[di]}...")
        pos = run_expanding_window(panel, m, delta)
        sim = simulate_strategy(pos, daily_ret)
        # Stats computed on OOS slice only — pre-OOS positions are all 0 and
        # including them in the denominator would understate Sharpe/vol.
        sim_oos = sim[sim.index >= OOS_START]
        st  = compute_performance_stats(sim_oos, f"EW_{m}_{DELTA_LBL[di]}")
        oos_pos_ew = pos[pos.index >= OOS_START]
        st["avg_position"] = float(oos_pos_ew.mean())
        st["pct_long"]  = float((oos_pos_ew ==  1).mean()) * 100
        st["pct_short"] = float((oos_pos_ew == -1).mean()) * 100
        st["pct_flat"]  = float((oos_pos_ew ==  0).mean()) * 100
        EW[(m, di)]     = pos
        EW_SIM[(m, di)] = (st, sim)

print("Loading rolling-window (t=1.28) from cache for comparison...")
RW  = {}
RW_SIM = {}
for m in MODELS:
    for di, delta in enumerate(DELTAS):
        pos = run_rolling_regression_positions(panel, m, delta, t_threshold=1.28)
        sim = simulate_strategy(pos, daily_ret)
        st  = compute_performance_stats(sim, f"RW_{m}_{DELTA_LBL[di]}")
        st["avg_position"] = float(pos.mean())
        st["pct_long"]  = float((pos ==  1).mean()) * 100
        st["pct_short"] = float((pos == -1).mean()) * 100
        st["pct_flat"]  = float((pos ==  0).mean()) * 100
        RW[(m, di)]     = pos
        RW_SIM[(m, di)] = (st, sim)


# ════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════

def shade(ax):
    for a, b in [("2008-09-01","2009-06-01"),
                 ("2020-02-01","2020-06-01"),
                 ("2022-01-01","2022-12-31")]:
        a, b = pd.Timestamp(a), pd.Timestamp(b)
        if b > s_dt and a < e_dt:
            ax.axvspan(max(a, s_dt), min(b, e_dt), alpha=0.08, color="grey", lw=0)

def setup_year_axis(axes, interval=2):
    for ax in axes:
        ax.xaxis.set_major_locator(mdates.YearLocator(interval))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        plt.setp(ax.get_xticklabels(), visible=True, fontsize=7)
        ax.tick_params(axis="x", which="major", labelsize=7, pad=2)

def perf_label(st):
    return (f"SR={st['sharpe']:+.2f}  "
            f"ret={st['ann_ret']*100:+.1f}%  "
            f"DD={st['max_dd']*100:.1f}%  "
            f"trades={st['n_trades']}")

def star(t):
    a = abs(t)
    if a > 2.576: return "***"
    if a > 1.960: return "**"
    if a > 1.645: return "*"
    return ""

def oos_cumret(sim_df, start=OOS_START):
    """Cumulative net return rebased to 1.0 at `start`."""
    net = sim_df["net_pnl"]
    s = net[net.index >= start]
    return (1 + s).cumprod()


# ════════════════════════════════════════════════════════════════════════════
# PLOT A: Per-model fixed-split detail
# ════════════════════════════════════════════════════════════════════════════

def plot_fixed_split_detail(model, out_path):
    pal = MODEL_PALETTE[model]
    _, res0, nw0, t_stats_d0, n_tr, n_oos, oos_rmse = FS[(model, 0)]  # same fit for all deltas

    fig = plt.figure(figsize=(15, 18))
    fig.suptitle(
        f"{MODEL_LABEL[model]} — Fixed Train/Test Split\n"
        f"Train: 2006-03-06 → {TRAIN_END}  ({n_tr} days)   "
        f"OOS: {OOS_START} → present  ({n_oos} days)\n"
        "Coefficients frozen at training time · Position = sign(ŷ) when |ŷ| > δ · "
        "No per-day t-stat gate in OOS · 0.05% slippage",
        fontsize=10, y=0.998,
    )
    gs = gridspec.GridSpec(6, 1, height_ratios=[0.55, 2.5, 1, 1, 1, 1],
                           hspace=0.4, top=0.93, bottom=0.03,
                           left=0.09, right=0.97)

    # ── Training stats panel ──────────────────────────────────────────────────
    ax_info = fig.add_subplot(gs[0])
    ax_info.axis("off")
    feat_cols = MODEL_FEATURES[model]
    names = ["const"] + feat_cols

    lines = [
        f"  Training regression  n={n_tr}  R²={res0.rsquared:.4f}  "
        f"OOS RMSE={oos_rmse:.5f}  NW-HAC {NW_LAGS} lags  ·  "
        f"No per-day |t| gate in OOS (δ-only filter):"
    ]
    for j, name in enumerate(names):
        coef = res0.params.iloc[j]
        se   = nw0[j]
        tv   = coef / se if se > 0 else 0
        lines.append(f"    {name:>14s}:  β = {coef:+.5f}   t = {tv:+.3f}{star(tv)}")
    ax_info.text(0.01, 0.95, "\n".join(lines), transform=ax_info.transAxes,
                 fontsize=8, va="top", family="monospace",
                 bbox=dict(fc="#f7f7f7", ec="#cccccc", lw=0.8, pad=4))

    # ── Return panel (OOS only, rebased to 1.0 at OOS_START) ─────────────────
    ax_ret = fig.add_subplot(gs[1])
    xlim = (oos_dt, e_dt)
    ax_ret.set_xlim(*xlim)
    shade(ax_ret)

    bah_oos = oos_cumret(bah_sim)
    ax_ret.plot(bah_oos.index, bah_oos.values,
                color=BAH_COLOR, lw=1.2, ls="-.", alpha=0.55,
                label=f"Buy-and-Hold  [{perf_label(bah_st)}]")
    for di in range(4):
        st, sim = FS_SIM[(model, di)]
        cum = oos_cumret(sim)
        ax_ret.plot(cum.index, cum.values,
                    color=pal[di], lw=1.8 - di*0.15, ls=LINESTYLE[di], alpha=0.92,
                    label=f"{DELTA_LBL[di]}  [{perf_label(st)}]")

    ax_ret.axhline(1, color="black", lw=0.4, ls=":")
    ax_ret.set_yscale("log")
    ax_ret.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"{v:.1f}x"))
    ax_ret.set_ylabel("Cumulative Net Return (log)", fontsize=9)
    ax_ret.legend(fontsize=7.5, loc="upper left")

    # ── Position panels (OOS only) ────────────────────────────────────────────
    ax_poss = [fig.add_subplot(gs[i+2]) for i in range(4)]
    for ax in [ax_ret] + ax_poss:
        ax.set_xlim(*xlim)

    for di, ax_p in enumerate(ax_poss):
        st, _ = FS_SIM[(model, di)]
        pos_full, *_ = FS[(model, di)]
        pos = pos_full[pos_full.index >= OOS_START]   # clip to OOS
        shade(ax_p)
        ax_p.fill_between(pos.index, pos.where(pos ==  1, 0), 0,
                          color=pal[di], alpha=0.75)
        ax_p.fill_between(pos.index, pos.where(pos == -1, 0), 0,
                          color=pal[di], alpha=0.35, hatch="///")
        ax_p.axhline(0, color="black", lw=0.4)
        ax_p.set_ylim(-1.5, 1.5)
        ax_p.set_yticks([-1, 0, 1])
        ax_p.set_yticklabels(["Short", "Flat", "Long"], fontsize=7)
        ax_p.set_ylabel(DELTA_LBL[di], fontsize=8.5, rotation=0,
                        ha="right", va="center", labelpad=55, color=pal[di])
        ann = (f"Long {(pos==1).mean()*100:.1f}%  "
               f"Short {(pos==-1).mean()*100:.1f}%  "
               f"Flat {(pos==0).mean()*100:.1f}%  "
               f"AvgPos={pos.mean():+.3f}")
        ax_p.text(0.01, 0.88, ann, transform=ax_p.transAxes,
                  fontsize=7.5, va="top", color=pal[di])

    setup_year_axis([ax_ret] + ax_poss)
    fig.savefig(out_path, dpi=155, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ════════════════════════════════════════════════════════════════════════════
# PLOT B: Per-model expanding-window detail
# ════════════════════════════════════════════════════════════════════════════

def plot_expanding_detail(model, out_path, ew_dict=None, ew_sim_dict=None,
                          extra_title=""):
    _ew     = ew_dict     if ew_dict     is not None else EW
    _ew_sim = ew_sim_dict if ew_sim_dict is not None else EW_SIM
    pal = MODEL_PALETTE[model]

    fig = plt.figure(figsize=(15, 18))
    fig.suptitle(
        f"{MODEL_LABEL[model]} — Expanding Window{extra_title}\n"
        f"OOS from {OOS_START}; training grows daily (all prior history); "
        f"OOS gap = 20 days  ·  0.05% slippage  ·  Series rebased to 1.0 at {OOS_START}",
        fontsize=10, y=0.998,
    )
    gs = gridspec.GridSpec(6, 1, height_ratios=[0.55, 2.5, 1, 1, 1, 1],
                           hspace=0.4, top=0.93, bottom=0.03,
                           left=0.09, right=0.97)

    # ── Training stats panel (initial fit = same training data as Fixed Split) ─
    ax_info = fig.add_subplot(gs[0])
    ax_info.axis("off")
    _, res0, nw0, _, n_tr, n_oos, oos_rmse = FS[(model, 0)]
    feat_cols_info = MODEL_FEATURES[model]
    names = ["const"] + feat_cols_info
    lines = [
        f"  Pre-OOS training regression (n={n_tr})  R²={res0.rsquared:.4f}  "
        f"OOS RMSE={oos_rmse:.5f}  NW-HAC {NW_LAGS} lags  ·  "
        f"Daily |t| > {T_THRESH:.2f} gate (all betas jointly) in OOS:"
    ]
    for j, name in enumerate(names):
        coef = res0.params.iloc[j]
        se   = nw0[j]
        tv   = coef / se if se > 0 else 0
        lines.append(f"    {name:>14s}:  β = {coef:+.5f}   t = {tv:+.3f}{star(tv)}")
    ax_info.text(0.01, 0.95, "\n".join(lines), transform=ax_info.transAxes,
                 fontsize=8, va="top", family="monospace",
                 bbox=dict(fc="#f7f7f7", ec="#cccccc", lw=0.8, pad=4))

    ax_ret  = fig.add_subplot(gs[1])
    ax_poss = [fig.add_subplot(gs[i+2]) for i in range(4)]
    xlim = (oos_dt, e_dt)
    for ax in [ax_ret] + ax_poss:
        ax.set_xlim(*xlim)

    shade(ax_ret)
    bah_oos = oos_cumret(bah_sim)
    ax_ret.plot(bah_oos.index, bah_oos.values,
                color=BAH_COLOR, lw=1.2, ls="-.", alpha=0.55,
                label=f"Buy-and-Hold  [{perf_label(bah_st)}]")
    for di in range(4):
        st, sim = _ew_sim[(model, di)]
        cum = oos_cumret(sim)
        ax_ret.plot(cum.index, cum.values,
                    color=pal[di], lw=1.8 - di*0.15, ls=LINESTYLE[di], alpha=0.92,
                    label=f"{DELTA_LBL[di]}  [{perf_label(st)}]")

    ax_ret.axhline(1, color="black", lw=0.4, ls=":")
    ax_ret.set_yscale("log")
    ax_ret.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"{v:.1f}x"))
    ax_ret.set_ylabel("Cumulative Net Return (log)", fontsize=9)
    ax_ret.legend(fontsize=7.5, loc="upper left")

    for di, ax_p in enumerate(ax_poss):
        st, _ = _ew_sim[(model, di)]
        pos_full = _ew[(model, di)]
        pos = pos_full[pos_full.index >= OOS_START]   # clip to OOS
        shade(ax_p)
        ax_p.fill_between(pos.index, pos.where(pos ==  1, 0), 0,
                          color=pal[di], alpha=0.75)
        ax_p.fill_between(pos.index, pos.where(pos == -1, 0), 0,
                          color=pal[di], alpha=0.35, hatch="///")
        ax_p.axhline(0, color="black", lw=0.4)
        ax_p.set_ylim(-1.5, 1.5)
        ax_p.set_yticks([-1, 0, 1])
        ax_p.set_yticklabels(["Short", "Flat", "Long"], fontsize=7)
        ax_p.set_ylabel(DELTA_LBL[di], fontsize=8.5, rotation=0,
                        ha="right", va="center", labelpad=55, color=pal[di])
        ann = (f"Long {(pos==1).mean()*100:.1f}%  "
               f"Short {(pos==-1).mean()*100:.1f}%  "
               f"Flat {(pos==0).mean()*100:.1f}%  "
               f"AvgPos={pos.mean():+.3f}")
        ax_p.text(0.01, 0.88, ann, transform=ax_p.transAxes,
                  fontsize=7.5, va="top", color=pal[di])

    setup_year_axis([ax_ret] + ax_poss)
    fig.savefig(out_path, dpi=155, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ════════════════════════════════════════════════════════════════════════════
# PLOT C: Three-way comparison per model (FS vs EW vs Rolling, best delta)
# ════════════════════════════════════════════════════════════════════════════

def best_delta(sim_dict, model):
    """Return delta index with highest Sharpe for this model."""
    return max(range(4), key=lambda di: sim_dict[(model, di)][0]["sharpe"])

def plot_method_comparison(model, out_path):
    pal = MODEL_PALETTE[model]
    fig, axes = plt.subplots(3, 1, figsize=(14, 13),
                             gridspec_kw={"height_ratios": [2.5, 2.5, 2.5],
                                          "hspace": 0.45})
    fig.suptitle(
        f"{MODEL_LABEL[model]} — Rolling vs Expanding Window vs Fixed Split\n"
        f"All series rebased to 1.0 at {OOS_START}  ·  |t| > {T_THRESH:.2f} gate (EW & Rolling)  ·  "
        "0.05% slippage",
        fontsize=10, y=0.998,
    )
    xlim = (oos_dt, e_dt)   # all panels show OOS period only

    method_data = [
        ("Rolling window (500 days, t=1.28)", RW_SIM, ["#08306b","#2171b5","#6baed6","#9ecae1"]),
        ("Expanding window (min 500 days)",   EW_SIM, ["#7f2704","#d94801","#fd8d3c","#fdbe85"]),
        ("Fixed split (train 2006–2011)",     FS_SIM, ["#00441b","#238b45","#74c476","#c7e9c0"]),
    ]

    bah_oos = oos_cumret(bah_sim)

    for ax, (title, sim_dict, cols) in zip(axes, method_data):
        ax.set_xlim(*xlim)
        shade(ax)
        ax.plot(bah_oos.index, bah_oos.values,
                color=BAH_COLOR, lw=1.1, ls="-.", alpha=0.45, label="Buy-and-Hold")

        for di in range(4):
            st, sim = sim_dict[(model, di)]
            cum = oos_cumret(sim)
            ax.plot(cum.index, cum.values,
                    color=cols[di], lw=1.8 - di*0.15, ls=LINESTYLE[di], alpha=0.9,
                    label=f"{DELTA_LBL[di]}  [SR={st['sharpe']:+.2f}  "
                          f"ret={st['ann_ret']*100:+.1f}%  "
                          f"DD={st['max_dd']*100:.1f}%]")

        ax.axhline(1, color="black", lw=0.4, ls=":")
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"{v:.1f}x"))
        ax.set_ylabel("Cum. Return (log)", fontsize=8)
        ax.set_title(title, fontsize=9, loc="left", pad=4)
        ax.legend(fontsize=7, loc="upper left", ncol=2)

        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        plt.setp(ax.get_xticklabels(), visible=True, fontsize=7)

    fig.savefig(out_path, dpi=155, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ════════════════════════════════════════════════════════════════════════════
# PLOT D: Overall performance table — all models × all methods × best delta
# ════════════════════════════════════════════════════════════════════════════

def plot_summary_table(out_path, ew_sim_dict=None, method_label="Expanding Window", models=None):
    _ew_sim  = ew_sim_dict if ew_sim_dict is not None else EW_SIM
    _models  = models if models is not None else MODELS
    fig, ax = plt.subplots(figsize=(16, 5))
    fig.suptitle(
        f"Performance Summary — {method_label}\n"
        "Best delta (highest OOS Sharpe) shown per method × model  ·  "
        "Net of 0.05% slippage  ·  0% risk-free",
        fontsize=11,
    )
    ax.axis("off")

    headers = ["Model", "Method", "Best δ",
               "Ann. Ret", "Ann. Vol", "Sharpe", "Max DD", "Trades",
               "Long%", "Short%", "Flat%"]

    TINT = {
        "Base":        "#deebf7",
        "Model_A":     "#e5f5e0",
        "Model_C":     "#efedf5",
        "Model_VVIX":  "#feedde",
    }

    rows, colors = [], []

    rows.append(["Buy-and-Hold", "—", "—",
                 f"{bah_st['ann_ret']*100:.2f}%",
                 f"{bah_st['ann_vol']*100:.2f}%",
                 f"{bah_st['sharpe']:.3f}",
                 f"{bah_st['max_dd']*100:.1f}%",
                 "—", "100%", "0%", "0%"])
    colors.append(["#fde0d0"] * len(headers))

    method_triples = [
        (method_label, _ew_sim),
    ]

    for m in _models:
        for mname, sim_dict in method_triples:
            di_best = best_delta(sim_dict, m)
            st, _ = sim_dict[(m, di_best)]
            rows.append([
                MODEL_LABEL[m].split("—")[0].strip(),
                mname,
                DELTA_LBL[di_best],
                f"{st['ann_ret']*100:.2f}%",
                f"{st['ann_vol']*100:.2f}%",
                f"{st['sharpe']:.3f}",
                f"{st['max_dd']*100:.1f}%",
                str(st["n_trades"]),
                f"{st['pct_long']:.1f}%",
                f"{st['pct_short']:.1f}%",
                f"{st['pct_flat']:.1f}%",
            ])
            colors.append([TINT[m]] * len(headers))

    tbl = ax.table(cellText=rows, colLabels=headers,
                   cellLoc="center", loc="center",
                   cellColours=colors)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.6)

    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ════════════════════════════════════════════════════════════════════════════
# PLOT E: Post-2020 comparison — all models × both methods, best δ each
# ════════════════════════════════════════════════════════════════════════════

POST_START = "2020-01-01"

def _post_stats(sim_df, label, start=POST_START):
    """Compute performance stats over [start, end] only, rebasing cum_net."""
    s = sim_df[sim_df.index >= start].copy()
    s["cum_net"] = (1 + s["net_pnl"]).cumprod()
    return compute_performance_stats(s, label)

def _rebase(sim_df, start=POST_START):
    net = sim_df["net_pnl"]
    s = net[net.index >= start]
    return (1 + s).cumprod()

def plot_post2020_comparison(out_path, ew_dict=None, ew_sim_dict=None, models=None):
    _ew     = ew_dict     if ew_dict     is not None else EW
    _ew_sim = ew_sim_dict if ew_sim_dict is not None else EW_SIM
    _models = models if models is not None else MODELS

    # Start at the first date any model in the list takes a non-zero position
    first_dates = []
    for m in _models:
        di  = best_delta(_ew_sim, m)
        pos = _ew[(m, di)]
        active = pos[pos != 0]
        if len(active):
            first_dates.append(active.index.min())
    START = min(first_dates) if first_dates else pd.Timestamp(POST_START)
    start = START.strftime("%Y-%m-%d")

    _MODEL_COLOR = {
        "Base":        "#08306b",
        "Model_A":     "#006d2c",
        "Model_C":     "#3f007d",
        "Model_VVIX":  "#7f2704",
    }

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.suptitle(
        f"Performance from First Signal Activation ({start}) — Best δ  ·  Expanding Window\n"
        f"Rebased to 1.0 at {start} (first date any model takes a position)  ·  "
        f"|t| > {T_THRESH:.2f} gate  ·  0.05% slippage",
        fontsize=10,
    )

    # Buy-and-Hold
    bah_post = _post_stats(bah_sim, "BaH", start)
    bah_cum  = _rebase(bah_sim, start)
    ax.plot(bah_cum.index, bah_cum.values,
            color=BAH_COLOR, lw=1.8, ls="-.", alpha=0.65,
            label=(f"Buy-and-Hold  "
                   f"[SR={bah_post['sharpe']:+.2f}  "
                   f"ret={bah_post['ann_ret']*100:+.1f}%  "
                   f"DD={bah_post['max_dd']*100:.1f}%]"))

    def _pos_lbls(pos_series):
        p = pos_series[pos_series.index >= start]
        return int((p == 1).mean() * 100), int((p == -1).mean() * 100)

    for m in _models:
        col = _MODEL_COLOR.get(m, "#333333")
        short_lbl = MODEL_LABEL[m].split("—")[1].strip()

        di_ew = best_delta(_ew_sim, m)
        _, sim_ew = _ew_sim[(m, di_ew)]
        st_post_ew = _post_stats(sim_ew, f"EW_{m}", start)
        cum_ew = _rebase(sim_ew, start)
        pos_ew = _ew[(m, di_ew)]
        pL_ew, pS_ew = _pos_lbls(pos_ew)
        ax.plot(cum_ew.index, cum_ew.values,
                color=col, lw=2.2, ls="-", alpha=0.92,
                label=(f"{short_lbl} ({DELTA_LBL[di_ew]})  "
                       f"[SR={st_post_ew['sharpe']:+.2f}  "
                       f"ret={st_post_ew['ann_ret']*100:+.1f}%  "
                       f"DD={st_post_ew['max_dd']*100:.1f}%  "
                       f"L{pL_ew}%/S{pS_ew}%]"))

    # ── Shade crises and format ────────────────────────────────────────
    for a, b in [("2020-02-01","2020-06-01"), ("2022-01-01","2022-12-31")]:
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), alpha=0.07, color="grey", lw=0)

    ax.axhline(1, color="black", lw=0.5, ls=":")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2f}x"))
    ax.set_ylabel("Cumulative Net Return (log, rebased to 1.0)", fontsize=10)
    ax.set_xlim(START, e_dt)
    ax.legend(fontsize=7.5, loc="upper left", ncol=1,
              framealpha=0.92, edgecolor="#cccccc")
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax.get_xticklabels(), visible=True, fontsize=9)
    ax.grid(axis="y", alpha=0.2, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)

    fig.savefig(out_path, dpi=155, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ════════════════════════════════════════════════════════════════════════════
# PLOT F: Secondary-variable beta + t-stat evolution over time (Models A & C)
# ════════════════════════════════════════════════════════════════════════════

def compute_ew_betas(model):
    """
    Record how all coefficients and NW t-stats evolve as the expanding window
    grows. Cached per model (betas don't depend on delta/threshold variant).
    Returns DataFrame indexed by daily prediction dates (>= OOS_START) with
    columns [beta_VP, se_VP, t_VP] for Base, plus [beta_sec, se_sec, t_sec]
    for bivariate models.
    """
    feat_cols = MODEL_FEATURES[model]
    bivariate = len(feat_cols) >= 2

    cache_path = CACHE_DIR / f"betas_EW_{model}_oos{OOS_START}.parquet"
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        # Back-fill Base cache if it predates the univariate support
        if not bivariate and "beta_VP" not in df.columns:
            cache_path.unlink()
        else:
            return df

    print(f"    Computing beta time series for {model} (one-time)...")
    sub     = panel.dropna(subset=feat_cols + ["fwd_ret_20"]).copy()
    N       = len(sub)
    oos_idx = sub.index.searchsorted(pd.Timestamp(OOS_START))
    start_i = max(MIN_WIN + 20, oos_idx)

    records = []
    for i in range(start_i, N):
        train = sub.iloc[0 : i - 20]
        X_tr  = add_constant(train[feat_cols], has_constant="skip")
        res   = OLS(train["fwd_ret_20"], X_tr).fit()
        nw    = _nw_se(res, nlags=NW_LAGS)
        b_alpha, se_alpha = float(res.params.iloc[0]), float(nw[0])
        b_vp,    se_vp    = float(res.params.iloc[1]), float(nw[1])
        rec = {
            "beta_alpha": b_alpha,
            "se_alpha":   se_alpha,
            "t_alpha":    b_alpha / se_alpha if se_alpha > 0 else 0.0,
            "beta_VP":    b_vp,
            "se_VP":      se_vp,
            "t_VP":       b_vp / se_vp if se_vp > 0 else 0.0,
        }
        if bivariate:
            b_sec, se_sec = float(res.params.iloc[2]), float(nw[2])
            rec.update({
                "beta_sec": b_sec,
                "se_sec":   se_sec,
                "t_sec":    b_sec / se_sec if se_sec > 0 else 0.0,
            })
        records.append(rec)

    df = pd.DataFrame(records, index=sub.index[start_i:])
    df.to_parquet(cache_path)
    return df


FEAT_DISPLAY = {
    "VP":         "VRP",
    "term_slope": "Term Slope",
    "vvix_ma5":   "VVIX MA5",
}

def plot_ew_beta_evolution(out_path, models=None):
    """
    N-row × 2-col figure (one row per model).  Left column: β ± 2 NW-SE band.
    Right column: NW t-statistics with ±T_THRESH gate lines.
    Betas are identical for both threshold variants (same OLS training).
    """
    if models is None:
        models = ["Base", "Model_A", "Model_C"]
    SEC_COLOR = {"Model_A": "#006d2c", "Model_C": "#3f007d", "Model_VVIX": "#7f2704"}

    n_rows = len(models)
    fig, axes = plt.subplots(n_rows, 2, figsize=(16, 5 * n_rows),
                             gridspec_kw={"wspace": 0.15, "hspace": 0.38},
                             squeeze=False)
    fig.suptitle(
        "Expanding Window — Coefficient Estimates and NW t-statistics Over Time\n"
        f"Training = all history up to t−20  ·  NW-HAC {NW_LAGS} lags  ·  "
        f"Position gate: all betas |t| > {T_THRESH:.2f}  ·  Shaded band = ± 2 NW-SE",
        fontsize=10,
    )

    for row, model in enumerate(models):
        betas     = compute_ew_betas(model)
        bivariate = "beta_sec" in betas.columns
        pal_vp    = MODEL_PALETTE[model][0]
        sec_col   = SEC_COLOR.get(model, "darkorange")
        feat_cols = MODEL_FEATURES[model]
        prim_name = FEAT_DISPLAY.get(feat_cols[0], feat_cols[0])
        sec_name  = FEAT_DISPLAY.get(feat_cols[1], feat_cols[1]) if bivariate else ""

        ax_b = axes[row, 0]   # beta panel
        ax_t = axes[row, 1]   # t-stat panel

        # ── Beta panel ─────────────────────────────────────────────────
        # Signal betas on left axis; alpha (intercept) on right axis
        # (alpha lives in return space ~±0.01 while signal betas can be much
        #  smaller or larger depending on regressor scale — separate axes keep
        #  the signal betas readable).
        b_vp  = betas["beta_VP"]
        se_vp = betas["se_VP"]
        ax_b.plot(b_vp.index, b_vp.values, color=pal_vp, lw=1.2, label=f"β({prim_name})")
        ax_b.fill_between(b_vp.index, b_vp - 2*se_vp, b_vp + 2*se_vp,
                          color=pal_vp, alpha=0.15)

        if bivariate:
            b_sec  = betas["beta_sec"]
            se_sec = betas["se_sec"]
            ax_b.plot(b_sec.index, b_sec.values, color=sec_col, lw=1.2,
                      label=f"β({sec_name})")
            ax_b.fill_between(b_sec.index, b_sec - 2*se_sec, b_sec + 2*se_sec,
                              color=sec_col, alpha=0.12)

        ax_b.axhline(0, color="black", lw=0.4, ls=":")
        ax_b.set_title(f"{MODEL_LABEL[model]} — β ± 2 NW-SE", fontsize=9, loc="left")
        ax_b.set_ylabel("Signal β", fontsize=9)
        ax_b.grid(axis="y", alpha=0.2, lw=0.6)
        ax_b.spines["top"].set_visible(False)

        # Alpha on twin right axis
        ax_b2 = ax_b.twinx()
        b_alpha  = betas["beta_alpha"]
        se_alpha = betas["se_alpha"]
        ax_b2.plot(b_alpha.index, b_alpha.values, color="grey", lw=1.0,
                   ls=":", alpha=0.85, label="α (intercept)")
        ax_b2.fill_between(b_alpha.index, b_alpha - 2*se_alpha, b_alpha + 2*se_alpha,
                           color="grey", alpha=0.08)
        ax_b2.set_ylabel("α", fontsize=9, color="grey")
        ax_b2.tick_params(axis="y", labelcolor="grey", labelsize=7)
        ax_b2.spines["top"].set_visible(False)

        # Combined legend
        h1, l1 = ax_b.get_legend_handles_labels()
        h2, l2 = ax_b2.get_legend_handles_labels()
        ax_b.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")

        # ── t-stat panel ───────────────────────────────────────────────
        t_alpha = betas["t_alpha"]
        t_vp    = betas["t_VP"]
        ax_t.plot(t_alpha.index, t_alpha.values, color="grey", lw=1.0,
                  ls=":", alpha=0.7, label="t(α)")
        ax_t.plot(t_vp.index, t_vp.values, color=pal_vp, lw=1.0, ls="--",
                  alpha=0.8, label=f"t({prim_name})")

        if bivariate:
            t_sec = betas["t_sec"]
            ax_t.plot(t_sec.index, t_sec.values, color=sec_col, lw=1.5,
                      label=f"t({sec_name})")

        ax_t.axhline( T_THRESH, color="firebrick", lw=1.2, ls="--",
                     label=f"|t| = {T_THRESH:.2f} (gate)")
        ax_t.axhline(-T_THRESH, color="firebrick", lw=1.2, ls="--")
        ax_t.axhline(0, color="black", lw=0.5, ls=":")
        ax_t.fill_between(t_vp.index, -T_THRESH, T_THRESH,
                          color="firebrick", alpha=0.05, label="Below gate → flat")
        ax_t.set_title(f"{MODEL_LABEL[model]} — NW t-statistics", fontsize=9, loc="left")
        ax_t.set_ylabel("NW t-statistic", fontsize=9)
        ax_t.legend(fontsize=8, loc="upper left")
        ax_t.grid(axis="y", alpha=0.2, lw=0.6)
        ax_t.spines[["top", "right"]].set_visible(False)

        # ── Shared x-axis formatting ────────────────────────────────────
        for ax in [ax_b, ax_t]:
            ax.set_xlim(pd.Timestamp(OOS_START), e_dt)
            ax.xaxis.set_major_locator(mdates.YearLocator(2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            plt.setp(ax.get_xticklabels(), visible=True, fontsize=8)
            for a_s, b_s in [("2020-02-01","2020-06-01"), ("2022-01-01","2022-12-31")]:
                ax.axvspan(pd.Timestamp(a_s), pd.Timestamp(b_s),
                           alpha=0.07, color="grey", lw=0)

    fig.savefig(out_path, dpi=155, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ════════════════════════════════════════════════════════════════════════════
# RUN ALL
# ════════════════════════════════════════════════════════════════════════════
print("\n--- expanding_window/ detail plots ---")
for m, tag in [("Base","base"), ("Model_A","model_a"), ("Model_C","model_c")]:
    plot_expanding_detail(m, DIR_FS / f"{tag}_expanding.png")

print("\n--- expanding_window/ summary ---")
plot_summary_table(DIR_FS / "performance_summary.png")

print("\n--- expanding_window/ post-activation comparison ---")
plot_post2020_comparison(DIR_FS / "post2020_comparison.png")

print("\n--- expanding_window/ beta evolution ---")
plot_ew_beta_evolution(DIR_FS / "ew_beta_evolution.png")

print("\nAll done.")


# ════════════════════════════════════════════════════════════════════════════
# ROLLING-MU THRESHOLD: rerun EW with threshold = rolling_avg(fwd_ret_20) ± delta
# Threshold benchmarks the prediction against the 500-day rolling historical
# average rather than a fixed zero-centred band.
# ════════════════════════════════════════════════════════════════════════════

DIR_ROLMU = OUTPUT / "expanding_window_rolmu"
DIR_ROLMU.mkdir(parents=True, exist_ok=True)

print("\nComputing expanding-window (rolling-mu threshold) positions...")
EW_RM     = {}
EW_RM_SIM = {}
for m in MODELS:
    for di, delta in enumerate(DELTAS):
        print(f"  EW-rolmu  {m}  {DELTA_LBL[di]}...")
        pos = run_expanding_window_rolmu(panel, m, delta)
        sim = simulate_strategy(pos, daily_ret)
        sim_oos = sim[sim.index >= OOS_START]
        st  = compute_performance_stats(sim_oos, f"EWrm_{m}_{DELTA_LBL[di]}")
        oos_pos = pos[pos.index >= OOS_START]
        st["avg_position"] = float(oos_pos.mean())
        st["pct_long"]  = float((oos_pos ==  1).mean()) * 100
        st["pct_short"] = float((oos_pos == -1).mean()) * 100
        st["pct_flat"]  = float((oos_pos ==  0).mean()) * 100
        EW_RM[(m, di)]     = pos
        EW_RM_SIM[(m, di)] = (st, sim)

print("\n--- expanding_window_rolmu/ detail plots ---")
for m, tag in [("Base","base"), ("Model_A","model_a"), ("Model_C","model_c")]:
    plot_expanding_detail(
        m, DIR_ROLMU / f"{tag}_expanding_rolmu.png",
        ew_dict=EW_RM, ew_sim_dict=EW_RM_SIM,
        extra_title=" · Threshold = rolling-avg(20d return) ± δ",
    )

print("\n--- expanding_window_rolmu/ summary ---")
plot_summary_table(
    DIR_ROLMU / "performance_summary_rolmu.png",
    ew_sim_dict=EW_RM_SIM,
    method_label="EW rolling-mu threshold",
)

print("\n--- expanding_window_rolmu/ post-activation comparison ---")
plot_post2020_comparison(
    DIR_ROLMU / "post2020_comparison_rolmu.png",
    ew_dict=EW_RM, ew_sim_dict=EW_RM_SIM,
)

print("\n--- expanding_window_rolmu/ beta evolution ---")
# Betas are identical for both threshold variants (same OLS training data).
# Copy the standard EW beta plot into the rolmu folder for completeness.
plot_ew_beta_evolution(DIR_ROLMU / "ew_beta_evolution.png")

print("\nAll done (rolling-mu).")


# ════════════════════════════════════════════════════════════════════════════
# UNIVARIATE VVIX MODEL — expanding window, rolling-mu threshold
# Motivated by the finding that Model C is dominated by VVIX: β_VVIX × VVIX
# contributes ~38–187× more to ŷ than β_VRP × VRP.
# This isolates the VVIX signal cleanly with no VRP in the regression.
# ════════════════════════════════════════════════════════════════════════════

DIR_VVIX = OUTPUT / "expanding_window_vvix"
DIR_VVIX.mkdir(parents=True, exist_ok=True)

# FS run (single-fit, for info box in plot_expanding_detail)
pos_v, res_v, nw_v, ts_v, n_tr_v, n_oos_v, rmse_v = run_fixed_split(
    panel, "Model_VVIX", DELTAS[0])
FS[("Model_VVIX", 0)] = (pos_v, res_v, nw_v, ts_v, n_tr_v, n_oos_v, rmse_v)

print("\nComputing univariate VVIX (rolling-mu) positions...")
EW_VVIX     = {}
EW_VVIX_SIM = {}
for di, delta in enumerate(DELTAS):
    print(f"  EW-rolmu  Model_VVIX  {DELTA_LBL[di]}...")
    pos = run_expanding_window_rolmu(panel, "Model_VVIX", delta)
    sim = simulate_strategy(pos, daily_ret)
    sim_oos = sim[sim.index >= OOS_START]
    st = compute_performance_stats(sim_oos, f"EWrm_VVIX_{DELTA_LBL[di]}")
    oos_pos = pos[pos.index >= OOS_START]
    st["avg_position"] = float(oos_pos.mean())
    st["pct_long"]  = float((oos_pos ==  1).mean()) * 100
    st["pct_short"] = float((oos_pos == -1).mean()) * 100
    st["pct_flat"]  = float((oos_pos ==  0).mean()) * 100
    EW_VVIX[(di)]     = pos
    EW_VVIX_SIM[(di)] = (st, sim)

# Wrap in (model, di) keys so plot functions work unchanged
EW_VVIX_DICT     = {("Model_VVIX", di): EW_VVIX[di]     for di in range(4)}
EW_VVIX_SIM_DICT = {("Model_VVIX", di): EW_VVIX_SIM[di] for di in range(4)}

print("\n--- expanding_window_vvix/ detail plot ---")
plot_expanding_detail(
    "Model_VVIX", DIR_VVIX / "model_vvix_expanding_rolmu.png",
    ew_dict=EW_VVIX_DICT, ew_sim_dict=EW_VVIX_SIM_DICT,
    extra_title=" · Threshold = rolling-avg(20d return) ± δ",
)

print("\n--- expanding_window_vvix/ summary ---")
plot_summary_table(
    DIR_VVIX / "performance_summary_vvix.png",
    ew_sim_dict=EW_VVIX_SIM_DICT,
    method_label="EW VVIX univariate (rolling-mu threshold)",
    models=["Model_VVIX"],
)

print("\n--- expanding_window_vvix/ post-activation comparison ---")
plot_post2020_comparison(
    DIR_VVIX / "post2020_comparison_vvix.png",
    ew_dict=EW_VVIX_DICT, ew_sim_dict=EW_VVIX_SIM_DICT,
    models=["Model_VVIX"],
)

print("\n--- expanding_window_vvix/ beta evolution ---")
plot_ew_beta_evolution(
    DIR_VVIX / "ew_beta_evolution_vvix.png",
    models=["Model_VVIX"],
)

print("\nAll done (univariate VVIX).")
