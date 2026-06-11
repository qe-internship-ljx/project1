"""
horizon_regression.py
=====================
Expanding-window OOS regression analyses at alternative horizons.

  (A) fwd_ret_40 ~ VRP            (40-day horizon, OOS gap = 40, NW 40 lags)
  (B) fwd_ret_20 ~ VVIX MA5      (20-day horizon, OOS gap = 20, NW 20 lags)
  (C) fwd_ret_20 ~ vol_trend     (20-day horizon, OOS gap = 20, NW 20 lags)
       vol_trend = ln(RV_5d / RV_22d), symmetric +-delta threshold
  (D) fwd_ret_20 ~ VIX spot      (20-day horizon, OOS gap = 20, NW 20 lags)
  (E) fwd_ret_20 ~ term_slope    (20-day horizon, OOS gap = 20, NW 20 lags)
  (F) fwd_ret_20 ~ vrp_vvix      (20-day horizon, OOS gap = 20, NW 20 lags)
       vrp_vvix = VP * vvix_ma5

Methodology matches fixed_split_eval.py:
  - OOS from 2012-01-01, min training = 500 obs
  - Per-day |t| > 1.28 gate on predictor beta (flat when fails)
  - 0.05% slippage, cumulative return rebased to 1.0 at OOS start
  - Positions and betas cached in output/regression_cache/

Each analysis produces a plot per the spec in plot.md:
  Panel 1  — cumulative OOS net return (log), BaH + 4 deltas
  Panel 2  — NW t-stat (left) + beta (right twin) over time
  Panels 3-6 — position over time, one per delta

Outputs:
  output/expanding_window/VRP/vrp_40d_ew.png
  output/expanding_window/VVIX MA5/vvix_20d_ew.png
  output/expanding_window/poor_correlation/Vol Trend/voltrend_20d_ew.png
  output/expanding_window/poor_correlation/VIX/vix_20d_ew.png
  output/expanding_window/Term Slope/termslope_20d_ew.png
"""

import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

from statsmodels.api import OLS, add_constant

ROOT      = Path(__file__).parent
OUTPUT    = ROOT / "output"
CACHE_DIR = OUTPUT / "regression_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT.parent / "bh_replication"))
from har_model import _nw_se

sys.path.insert(0, str(ROOT))
from experiment2 import (
    load_vrp_series, load_es_front_month, load_vvix, compute_vvix_ma5,
    load_vix_spot, load_vix_futures_term_structure,
    compute_buy_and_hold, simulate_strategy, compute_performance_stats,
)
from fh_replication.fh_replication import compute_vix_term_slope

OOS_START = "2012-01-01"
MIN_WIN   = 500
T_THRESH  = 1.28
DELTAS    = [0.002, 0.005, 0.0075, 0.010]
DELTA_LBL = ["d=0.2%", "d=0.5%", "d=0.75%", "d=1.0%"]
BAH_COLOR = "#d62728"


def build_panel(vrp, es, vvix_ma5, vix_spot, term_slope):
    ret   = es["returns"]
    panel = pd.DataFrame({
        "VP":         vrp["VP"],
        "vvix_ma5":   vvix_ma5,
        "vix":        vix_spot,
        "term_slope": term_slope,
        "daily_ret":  ret,
    })
    for h in [20, 40]:
        fwd = (ret + 1).rolling(h).apply(np.prod, raw=True).shift(-h) - 1
        panel[f"fwd_{h}d"] = fwd
    # vol_trend = ln(RV_5d / RV_22d); RV annualised via sqrt(252)
    r2   = ret ** 2
    rv5  = np.sqrt(r2.rolling(5).mean()  * 252)
    rv22 = np.sqrt(r2.rolling(22).mean() * 252)
    panel["vol_trend"] = np.log(rv5 / rv22)
    panel["vrp_vvix"]  = panel["VP"] * panel["vvix_ma5"]
    panel = panel.dropna(subset=["VP", "vvix_ma5", "term_slope"])
    return panel[panel.index >= "2006-03-06"]


def run_ew(panel, predictor, fwd_col, oos_gap, nw_lags, delta):
    """Expanding-window positions for a univariate regression at one delta."""
    tag = (f"pos_EW_{predictor}_{fwd_col}_d{int(delta*10000)}bps"
           f"_t{int(T_THRESH*100)}_oos{OOS_START}.parquet")
    cache = CACHE_DIR / tag
    if cache.exists():
        return pd.read_parquet(cache).squeeze().rename(f"pos_{predictor}_{delta}")

    sub = panel.dropna(subset=[predictor, fwd_col]).copy()
    N   = len(sub)
    pos = pd.Series(0.0, index=sub.index, name=f"pos_{predictor}_{delta}")

    oos_idx = sub.index.searchsorted(pd.Timestamp(OOS_START))
    start_i = max(MIN_WIN + oos_gap, oos_idx)

    for i in range(start_i, N):
        train = sub.iloc[0 : i - oos_gap]
        X_tr  = add_constant(train[[predictor]], has_constant="skip")
        res   = OLS(train[fwd_col], X_tr).fit()
        nw    = _nw_se(res, nlags=nw_lags)
        t_val = float(res.params.iloc[1]) / float(nw[1])
        if abs(t_val) <= T_THRESH:
            continue
        test = sub.iloc[[i]][[predictor]].copy()
        test.insert(0, "const", 1.0)
        y_hat = float(res.predict(test).iloc[0])
        if   y_hat >  delta: pos.iloc[i] =  1.0
        elif y_hat < -delta: pos.iloc[i] = -1.0

    pos.to_frame().to_parquet(cache)
    return pos


def compute_betas(panel, predictor, fwd_col, oos_gap, nw_lags):
    """Record NW t-stat (and beta) of predictor at each prediction day."""
    tag   = f"betas_EW_{predictor}_{fwd_col}_oos{OOS_START}.parquet"
    cache = CACHE_DIR / tag
    if cache.exists():
        return pd.read_parquet(cache)

    print(f"    Computing beta time series for {predictor} -> {fwd_col}...")
    sub     = panel.dropna(subset=[predictor, fwd_col]).copy()
    N       = len(sub)
    oos_idx = sub.index.searchsorted(pd.Timestamp(OOS_START))
    start_i = max(MIN_WIN + oos_gap, oos_idx)

    records = []
    for i in range(start_i, N):
        train = sub.iloc[0 : i - oos_gap]
        X_tr  = add_constant(train[[predictor]], has_constant="skip")
        res   = OLS(train[fwd_col], X_tr).fit()
        nw    = _nw_se(res, nlags=nw_lags)
        b, se = float(res.params.iloc[1]), float(nw[1])
        records.append({"beta": b, "se": se, "t_stat": b / se if se > 0 else 0.0})

    df = pd.DataFrame(records, index=sub.index[start_i:])
    df.to_parquet(cache)
    return df


def oos_cumret(sim, start=OOS_START):
    net = sim["net_pnl"]
    s   = net[net.index >= start]
    return (1 + s).cumprod()


def _shade(ax, start, end):
    for a, b in [("2020-02-01", "2020-06-01"), ("2022-01-01", "2022-12-31")]:
        a, b = pd.Timestamp(a), pd.Timestamp(b)
        if b > start and a < end:
            ax.axvspan(max(a, start), min(b, end), alpha=0.08, color="grey", lw=0)


def plot_2panel(pred_label, horizon_label, oos_gap, nw_lags,
                color_palette, sim_dict, betas_df, bah_sim, bah_st, out_path):
    oos_dt = pd.Timestamp(OOS_START)
    e_dt   = betas_df.index[-1]
    xlim   = (oos_dt, e_dt)
    n_deltas = len(DELTAS)

    # height ratios: return (tall), t-stat (medium), one panel per delta
    h_ratios = [2.5, 1.2] + [1.0] * n_deltas
    fig, axes = plt.subplots(
        2 + n_deltas, 1, figsize=(14, 11 + 2.2 * n_deltas), sharex=True,
        gridspec_kw={"height_ratios": h_ratios, "hspace": 0.35},
    )
    ax_ret = axes[0]
    ax_t   = axes[1]
    ax_pos = axes[2:]

    fig.suptitle(
        f"{pred_label} -> {horizon_label} Forward Return  "
        f"(Expanding Window, OOS from {OOS_START})\n"
        f"Training grows daily; OOS gap = {oos_gap} days; "
        f"NW-HAC {nw_lags} lags; |t| > {T_THRESH:.2f} gate; 0.05% slippage",
        fontsize=10, y=0.998,
    )
    fig.subplots_adjust(top=0.955, bottom=0.03, left=0.10, right=0.93)

    # ── Cumulative return ──────────────────────────────────────────────────────
    ax_ret.set_xlim(*xlim)
    _shade(ax_ret, oos_dt, e_dt)

    bah_oos = oos_cumret(bah_sim)
    ax_ret.plot(bah_oos.index, bah_oos.values,
                color=BAH_COLOR, lw=1.5, ls="-.", alpha=0.6,
                label=(f"Buy-and-Hold  "
                       f"[SR={bah_st['sharpe']:+.2f}  "
                       f"ret={bah_st['ann_ret']*100:+.1f}%  "
                       f"DD={bah_st['max_dd']*100:.1f}%]"))

    for di, (delta, lbl) in enumerate(zip(DELTAS, DELTA_LBL)):
        st, sim = sim_dict[di]
        cum     = oos_cumret(sim)
        pos_oos = sim["position"][sim.index >= OOS_START]
        pL = float((pos_oos == 1).mean() * 100)
        pS = float((pos_oos == -1).mean() * 100)
        ax_ret.plot(cum.index, cum.values,
                    color=color_palette[di], lw=1.8, alpha=0.9,
                    label=(f"{lbl}  "
                           f"[SR={st['sharpe']:+.2f}  "
                           f"ret={st['ann_ret']*100:+.1f}%  "
                           f"DD={st['max_dd']*100:.1f}%  "
                           f"L{pL:.0f}%/S{pS:.0f}%]"))

    ax_ret.axhline(1, color="black", lw=0.4, ls=":")
    ax_ret.set_yscale("log")
    ax_ret.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2f}x"))
    ax_ret.set_ylabel("Cumulative Net Return (log)", fontsize=9)
    ax_ret.legend(fontsize=8, loc="upper left", framealpha=0.92)
    ax_ret.grid(axis="y", alpha=0.2, lw=0.6)
    ax_ret.spines[["top", "right"]].set_visible(False)

    # ── t-stat + beta over time ────────────────────────────────────────────────
    ax_t.set_xlim(*xlim)
    _shade(ax_t, oos_dt, e_dt)

    t_series = betas_df["t_stat"]
    b_series = betas_df["beta"]

    ax_t.plot(t_series.index, t_series.values,
              color=color_palette[0], lw=1.0, alpha=0.85,
              label=f"NW t-stat of {pred_label} ({nw_lags}-lag HAC)")
    ax_t.fill_between(t_series.index, -T_THRESH, T_THRESH,
                      color="firebrick", alpha=0.05, label="Below gate (flat zone)")
    ax_t.axhline( T_THRESH, color="firebrick", lw=1.2, ls="--",
                 label=f"|t| = {T_THRESH:.2f} gate")
    ax_t.axhline(-T_THRESH, color="firebrick", lw=1.2, ls="--")
    ax_t.axhline(0, color="black", lw=0.5, ls=":")
    ax_t.set_ylabel(f"NW t-stat ({pred_label})", fontsize=9)
    ax_t.grid(axis="y", alpha=0.2, lw=0.6)
    ax_t.spines["top"].set_visible(False)

    ax_t2 = ax_t.twinx()
    ax_t2.plot(b_series.index, b_series.values,
               color="dimgrey", lw=1.0, ls="--", alpha=0.60,
               label=f"Beta ({pred_label})")
    ax_t2.axhline(0, color="dimgrey", lw=0.4, ls=":")
    ax_t2.set_ylabel(f"Beta ({pred_label})", fontsize=8, color="dimgrey")
    ax_t2.tick_params(axis="y", labelcolor="dimgrey", labelsize=7)
    ax_t2.spines["top"].set_visible(False)

    lines1, labs1 = ax_t.get_legend_handles_labels()
    lines2, labs2 = ax_t2.get_legend_handles_labels()
    ax_t.legend(lines1 + lines2, labs1 + labs2, fontsize=8, loc="upper left")

    # ── Position panels (one per delta) ───────────────────────────────────────
    for di, (delta, lbl, ax_p) in enumerate(zip(DELTAS, DELTA_LBL, ax_pos)):
        _, sim = sim_dict[di]
        pos = sim["position"][sim.index >= OOS_START]
        ax_p.set_xlim(*xlim)
        _shade(ax_p, oos_dt, e_dt)
        ax_p.fill_between(pos.index, pos.where(pos ==  1, 0), 0,
                          color=color_palette[di], alpha=0.75, label="Long")
        ax_p.fill_between(pos.index, pos.where(pos == -1, 0), 0,
                          color=color_palette[di], alpha=0.30, hatch="///", label="Short")
        ax_p.axhline(0, color="black", lw=0.4)
        ax_p.set_ylim(-1.5, 1.5)
        ax_p.set_yticks([-1, 0, 1])
        ax_p.set_yticklabels(["Short", "Flat", "Long"], fontsize=8)
        ax_p.set_ylabel(lbl, fontsize=9, rotation=0,
                        ha="right", va="center", labelpad=56, color=color_palette[di])
        pL = float((pos == 1).mean() * 100)
        pS = float((pos == -1).mean() * 100)
        pF = float((pos == 0).mean() * 100)
        ax_p.text(0.01, 0.97,
                  f"Long {pL:.1f}%  Short {pS:.1f}%  Flat {pF:.1f}%  "
                  f"AvgPos={float(pos.mean()):+.3f}",
                  transform=ax_p.transAxes, fontsize=7.5, va="top",
                  color=color_palette[di])
        ax_p.spines[["top", "right"]].set_visible(False)

    for ax in [ax_ret, ax_t] + list(ax_pos):
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.tick_params(axis="x", which="major", labelbottom=True, labelsize=7, pad=2)

    fig.savefig(out_path, dpi=155, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def main():
    print("=" * 72)
    print("  Horizon Regression — Expanding Window")
    print("  (A) VRP        -> 40-day forward return")
    print("  (B) VVIX MA5   -> 20-day forward return")
    print("  (C) vol_trend  -> 20-day forward return")
    print("  (D) VIX spot   -> 20-day forward return")
    print("  (E) term_slope -> 20-day forward return")
    print("=" * 72)

    print("\n[1] Loading data...")
    vrp        = load_vrp_series()
    es         = load_es_front_month()
    vvix_ma5   = compute_vvix_ma5(load_vvix())
    vix_spot   = load_vix_spot()
    term_slope = compute_vix_term_slope(load_vix_futures_term_structure())
    panel      = build_panel(vrp, es, vvix_ma5, vix_spot, term_slope)
    print(f"    {len(panel):,} obs  "
          f"[{panel.index.min().date()} - {panel.index.max().date()}]")

    daily_ret = panel["daily_ret"].dropna()
    bah_pos   = compute_buy_and_hold(daily_ret)
    bah_sim_  = simulate_strategy(bah_pos, daily_ret)
    bah_st    = compute_performance_stats(
        bah_sim_[bah_sim_.index >= OOS_START], "Buy-and-Hold"
    )

    # ── (A) VRP -> 40-day ────────────────────────────────────────────────────
    print("\n[2] VRP -> 40-day expanding-window positions...")
    vrp_sims = {}
    for di, (delta, lbl) in enumerate(zip(DELTAS, DELTA_LBL)):
        print(f"    {lbl}...", end="  ", flush=True)
        pos = run_ew(panel, "VP", "fwd_40d", oos_gap=40, nw_lags=40, delta=delta)
        sim = simulate_strategy(pos, daily_ret)
        st  = compute_performance_stats(sim[sim.index >= OOS_START], f"EW_VRP40d_{lbl}")
        vrp_sims[di] = (st, sim)
        p   = pos[pos.index >= OOS_START]
        print(f"SR={st['sharpe']:+.3f}  L={float((p==1).mean())*100:.1f}%  "
              f"F={float((p==0).mean())*100:.1f}%")

    vrp_betas = compute_betas(panel, "VP", "fwd_40d", oos_gap=40, nw_lags=40)

    out_vrp = OUTPUT / "expanding_window" / "VRP"
    out_vrp.mkdir(parents=True, exist_ok=True)
    plot_2panel(
        pred_label="VRP", horizon_label="40-day",
        oos_gap=40, nw_lags=40,
        color_palette=["#08306b", "#2171b5", "#4292c6", "#6baed6"],
        sim_dict=vrp_sims, betas_df=vrp_betas,
        bah_sim=bah_sim_, bah_st=bah_st,
        out_path=out_vrp / "vrp_40d_ew.png",
    )

    # ── (B) VVIX MA5 -> 20-day ──────────────────────────────────────────────
    print("\n[3] VVIX MA5 -> 20-day expanding-window positions...")
    vvix_sims = {}
    for di, (delta, lbl) in enumerate(zip(DELTAS, DELTA_LBL)):
        print(f"    {lbl}...", end="  ", flush=True)
        pos = run_ew(panel, "vvix_ma5", "fwd_20d", oos_gap=20, nw_lags=20, delta=delta)
        sim = simulate_strategy(pos, daily_ret)
        st  = compute_performance_stats(sim[sim.index >= OOS_START], f"EW_VVIX20d_{lbl}")
        vvix_sims[di] = (st, sim)
        p   = pos[pos.index >= OOS_START]
        print(f"SR={st['sharpe']:+.3f}  L={float((p==1).mean())*100:.1f}%  "
              f"F={float((p==0).mean())*100:.1f}%")

    vvix_betas = compute_betas(panel, "vvix_ma5", "fwd_20d", oos_gap=20, nw_lags=20)

    out_vvix = OUTPUT / "expanding_window" / "VVIX MA5"
    out_vvix.mkdir(parents=True, exist_ok=True)
    plot_2panel(
        pred_label="VVIX MA5", horizon_label="20-day",
        oos_gap=20, nw_lags=20,
        color_palette=["#3f007d", "#6a51a3", "#807dba", "#9e9ac8"],
        sim_dict=vvix_sims, betas_df=vvix_betas,
        bah_sim=bah_sim_, bah_st=bah_st,
        out_path=out_vvix / "vvix_20d_ew.png",
    )

    # ── (C) vol_trend -> 20-day ─────────────────────────────────────────────
    print("\n[4] vol_trend -> 20-day expanding-window positions...")
    vt_sims = {}
    for di, (delta, lbl) in enumerate(zip(DELTAS, DELTA_LBL)):
        print(f"    {lbl}...", end="  ", flush=True)
        pos = run_ew(panel, "vol_trend", "fwd_20d", oos_gap=20, nw_lags=20, delta=delta)
        sim = simulate_strategy(pos, daily_ret)
        st  = compute_performance_stats(sim[sim.index >= OOS_START], f"EW_VT20d_{lbl}")
        vt_sims[di] = (st, sim)
        p   = pos[pos.index >= OOS_START]
        print(f"SR={st['sharpe']:+.3f}  L={float((p==1).mean())*100:.1f}%  "
              f"F={float((p==0).mean())*100:.1f}%")

    vt_betas = compute_betas(panel, "vol_trend", "fwd_20d", oos_gap=20, nw_lags=20)

    out_vt = OUTPUT / "expanding_window" / "poor_correlation" / "Vol Trend"
    out_vt.mkdir(parents=True, exist_ok=True)
    plot_2panel(
        pred_label="Vol Trend [ln(RV5/RV22)]", horizon_label="20-day",
        oos_gap=20, nw_lags=20,
        color_palette=["#00441b", "#238b45", "#41ab5d", "#74c476"],
        sim_dict=vt_sims, betas_df=vt_betas,
        bah_sim=bah_sim_, bah_st=bah_st,
        out_path=out_vt / "voltrend_20d_ew.png",
    )

    # ── (D) VIX spot -> 20-day ──────────────────────────────────────────────
    print("\n[5] VIX spot -> 20-day expanding-window positions...")
    vix_sims = {}
    for di, (delta, lbl) in enumerate(zip(DELTAS, DELTA_LBL)):
        print(f"    {lbl}...", end="  ", flush=True)
        pos = run_ew(panel, "vix", "fwd_20d", oos_gap=20, nw_lags=20, delta=delta)
        sim = simulate_strategy(pos, daily_ret)
        st  = compute_performance_stats(sim[sim.index >= OOS_START], f"EW_VIX20d_{lbl}")
        vix_sims[di] = (st, sim)
        p   = pos[pos.index >= OOS_START]
        print(f"SR={st['sharpe']:+.3f}  L={float((p==1).mean())*100:.1f}%  "
              f"F={float((p==0).mean())*100:.1f}%")

    vix_betas = compute_betas(panel, "vix", "fwd_20d", oos_gap=20, nw_lags=20)

    out_vix = OUTPUT / "expanding_window" / "poor_correlation" / "VIX"
    out_vix.mkdir(parents=True, exist_ok=True)
    plot_2panel(
        pred_label="VIX", horizon_label="20-day",
        oos_gap=20, nw_lags=20,
        color_palette=["#7f0000", "#cb181d", "#ef3b2c", "#fc9272"],
        sim_dict=vix_sims, betas_df=vix_betas,
        bah_sim=bah_sim_, bah_st=bah_st,
        out_path=out_vix / "vix_20d_ew.png",
    )

    # ── (E) term_slope -> 20-day ─────────────────────────────────────────────
    print("\n[6] term_slope -> 20-day expanding-window positions...")
    ts_sims = {}
    for di, (delta, lbl) in enumerate(zip(DELTAS, DELTA_LBL)):
        print(f"    {lbl}...", end="  ", flush=True)
        pos = run_ew(panel, "term_slope", "fwd_20d", oos_gap=20, nw_lags=20, delta=delta)
        sim = simulate_strategy(pos, daily_ret)
        st  = compute_performance_stats(sim[sim.index >= OOS_START], f"EW_TS20d_{lbl}")
        ts_sims[di] = (st, sim)
        p   = pos[pos.index >= OOS_START]
        print(f"SR={st['sharpe']:+.3f}  L={float((p==1).mean())*100:.1f}%  "
              f"F={float((p==0).mean())*100:.1f}%")

    ts_betas = compute_betas(panel, "term_slope", "fwd_20d", oos_gap=20, nw_lags=20)

    out_ts = OUTPUT / "expanding_window" / "Term Slope"
    out_ts.mkdir(parents=True, exist_ok=True)
    plot_2panel(
        pred_label="Term Slope", horizon_label="20-day",
        oos_gap=20, nw_lags=20,
        color_palette=["#7f2704", "#d94801", "#fd8d3c", "#fdbe85"],
        sim_dict=ts_sims, betas_df=ts_betas,
        bah_sim=bah_sim_, bah_st=bah_st,
        out_path=out_ts / "termslope_20d_ew.png",
    )

    # ── (F) vrp_vvix -> 20-day ──────────────────────────────────────────────
    print("\n[7] VRP x VVIX MA5 -> 20-day expanding-window positions...")
    vv_sims = {}
    for di, (delta, lbl) in enumerate(zip(DELTAS, DELTA_LBL)):
        print(f"    {lbl}...", end="  ", flush=True)
        pos = run_ew(panel, "vrp_vvix", "fwd_20d", oos_gap=20, nw_lags=20, delta=delta)
        sim = simulate_strategy(pos, daily_ret)
        st  = compute_performance_stats(sim[sim.index >= OOS_START], f"EW_VRPVVIX20d_{lbl}")
        vv_sims[di] = (st, sim)
        p   = pos[pos.index >= OOS_START]
        print(f"SR={st['sharpe']:+.3f}  L={float((p==1).mean())*100:.1f}%  "
              f"F={float((p==0).mean())*100:.1f}%")

    vv_betas = compute_betas(panel, "vrp_vvix", "fwd_20d", oos_gap=20, nw_lags=20)

    out_vv = OUTPUT / "expanding_window" / "VRP x VVIX"
    out_vv.mkdir(parents=True, exist_ok=True)
    plot_2panel(
        pred_label="VRP x VVIX MA5", horizon_label="20-day",
        oos_gap=20, nw_lags=20,
        color_palette=["#4d3000", "#8c5a00", "#cc8400", "#ffb84d"],
        sim_dict=vv_sims, betas_df=vv_betas,
        bah_sim=bah_sim_, bah_st=bah_st,
        out_path=out_vv / "vrp_vvix_20d_ew.png",
    )

    print("\nDone.")
    print("  output/expanding_window/VRP/vrp_40d_ew.png")
    print("  output/expanding_window/VVIX MA5/vvix_20d_ew.png")
    print("  output/expanding_window/poor_correlation/Vol Trend/voltrend_20d_ew.png")
    print("  output/expanding_window/poor_correlation/VIX/vix_20d_ew.png")
    print("  output/expanding_window/Term Slope/termslope_20d_ew.png")
    print("  output/expanding_window/VRP x VVIX/vrp_vvix_20d_ew.png")
    print("=" * 72)


if __name__ == "__main__":
    main()
