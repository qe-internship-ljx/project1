"""
horizon_regression.py
=====================
Expanding-window OOS regression analyses at alternative horizons.

  (A) fwd_ret_40 ~ VRP                 (40-day, OOS gap=40, NW 40 lags)  symmetric + asymmetric
  (B) fwd_ret_20 ~ VVIX MA5            (20-day, OOS gap=20, NW 20 lags)  symmetric only
  (C) fwd_ret_20 ~ vol_trend           (20-day, OOS gap=20, NW 20 lags)  [poor corr baseline]
       vol_trend = ln(RV_5d / RV_22d)
  (D) fwd_ret_20 ~ VIX spot            (20-day, OOS gap=20, NW 20 lags)  [poor corr baseline]
  (E) fwd_ret_20 ~ term_slope          (20-day, OOS gap=20, NW 20 lags)  [poor corr baseline]
  (G) fwd_ret_20 ~ VP + open_interest  (20-day, OOS gap=20, NW 20 lags)  symmetric + asymmetric

Methodology matches fixed_split_eval.py:
  - OOS from 2012-01-01, min training = 500 obs
  - Per-day |t| > 1.28 gate on predictor beta (flat when fails); both betas for bivariate
  - 0.05% slippage, cumulative return rebased to 1.0 at OOS start
  - Positions and betas cached in output/regression_cache/

Asymmetric threshold rule:
  Long  if ŷ > μ₅₀₀  (rolling 500-day mean of actual forward returns in training)
  Short if ŷ < 0
  Flat  otherwise

Stat-window rule:
  When any strategy on a plot first activates after 2020-01-01, ALL legend statistics
  (including buy-and-hold) are computed from that activation date for consistency.

OOS R²: Campbell-Thompson (2008) — 1 - SS_res / SS_tot vs prevailing historical mean.
  No t-stat gate applied; pure forecast accuracy. Cached per model.

Outputs:
  output/expanding_window/VRP 40-day/symmetric_VRP_40d.png
  output/expanding_window/VRP 40-day/asymmetric_VRP_40d.png
  output/expanding_window/VVIX MA5/symmetric_VVIX_MA5.png
  output/expanding_window/poor_correlation/Vol Trend/symmetric_Vol_Trend.png
  output/expanding_window/poor_correlation/VIX/symmetric_VIX.png
  output/expanding_window/poor_correlation/Term Slope/symmetric_Term_Slope.png
  output/expanding_window/VRP + Open Interest/symmetric_VRP_+_Open_Interest.png
  output/expanding_window/VRP + Open Interest/asymmetric_VRP_+_Open_Interest.png
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
    load_es_open_interest,
)
from fh_replication.fh_replication import compute_vix_term_slope

OOS_START = "2012-01-01"
MIN_WIN   = 500
T_THRESH  = 1.28
DELTAS    = [0.002, 0.005, 0.0075, 0.010]
DELTA_LBL = ["d=0.2%", "d=0.5%", "d=0.75%", "d=1.0%"]
BAH_COLOR = "#d62728"


def build_panel(vrp, es, vvix_ma5, vix_spot, term_slope, oi):
    ret   = es["returns"]
    panel = pd.DataFrame({
        "VP":             vrp["VP"],
        "vvix_ma5":       vvix_ma5,
        "vix":            vix_spot,
        "term_slope":     term_slope,
        "open_interest":  oi,
        "daily_ret":      ret,
    })
    for h in [20, 40]:
        fwd = (ret + 1).rolling(h).apply(np.prod, raw=True).shift(-h) - 1
        panel[f"fwd_{h}d"] = fwd
    r2   = ret ** 2
    rv5  = np.sqrt(r2.rolling(5).mean()  * 252)
    rv22 = np.sqrt(r2.rolling(22).mean() * 252)
    panel["vol_trend"] = np.log(rv5 / rv22)
    panel = panel.dropna(subset=["VP", "vvix_ma5", "term_slope"])
    return panel[panel.index >= "2006-03-06"]


# ── OOS R² (Campbell-Thompson 2008) ──────────────────────────────────────────

def compute_oos_r2(panel, predictor, fwd_col, oos_gap, nw_lags):
    """OOS R² vs prevailing mean. No t-stat gate. Cached."""
    tag   = f"r2oos_EW_{predictor}_{fwd_col}_oos{OOS_START}.txt"
    cache = CACHE_DIR / tag
    if cache.exists():
        return float(cache.read_text().strip())

    print(f"    Computing OOS R² for {predictor} -> {fwd_col}...", flush=True)
    sub     = panel.dropna(subset=[predictor, fwd_col]).copy()
    N       = len(sub)
    fwd_arr = sub[fwd_col].values
    oos_idx = sub.index.searchsorted(pd.Timestamp(OOS_START))
    start_i = max(MIN_WIN + oos_gap, oos_idx)

    y_hat_l, y_mean_l, y_act_l = [], [], []
    for i in range(start_i, N):
        train = sub.iloc[0 : i - oos_gap]
        X_tr  = add_constant(train[[predictor]], has_constant="skip")
        res   = OLS(train[fwd_col], X_tr).fit()
        test  = sub.iloc[[i]][[predictor]].copy()
        test.insert(0, "const", 1.0)
        y_hat_l.append(float(res.predict(test).iloc[0]))
        y_mean_l.append(float(np.mean(fwd_arr[0 : i - oos_gap])))
        y_act_l.append(fwd_arr[i])

    ya, yh, ym = np.array(y_act_l), np.array(y_hat_l), np.array(y_mean_l)
    r2 = 1.0 - np.sum((ya - yh) ** 2) / np.sum((ya - ym) ** 2)
    cache.write_text(str(r2))
    return r2


def compute_oos_r2_bivariate(panel, pred1, pred2, fwd_col, oos_gap, nw_lags):
    """OOS R² (bivariate) vs prevailing mean. No t-stat gate. Cached."""
    tag   = f"r2oos_EWbiv_{pred1}_{pred2}_{fwd_col}_oos{OOS_START}.txt"
    cache = CACHE_DIR / tag
    if cache.exists():
        return float(cache.read_text().strip())

    print(f"    Computing OOS R² for {pred1} + {pred2} -> {fwd_col}...", flush=True)
    sub     = panel.dropna(subset=[pred1, pred2, fwd_col]).copy()
    N       = len(sub)
    fwd_arr = sub[fwd_col].values
    oos_idx = sub.index.searchsorted(pd.Timestamp(OOS_START))
    start_i = max(MIN_WIN + oos_gap, oos_idx)

    y_hat_l, y_mean_l, y_act_l = [], [], []
    for i in range(start_i, N):
        train = sub.iloc[0 : i - oos_gap]
        X_tr  = add_constant(train[[pred1, pred2]], has_constant="skip")
        res   = OLS(train[fwd_col], X_tr).fit()
        test  = sub.iloc[[i]][[pred1, pred2]].copy()
        test.insert(0, "const", 1.0)
        y_hat_l.append(float(res.predict(test).iloc[0]))
        y_mean_l.append(float(np.mean(fwd_arr[0 : i - oos_gap])))
        y_act_l.append(fwd_arr[i])

    ya, yh, ym = np.array(y_act_l), np.array(y_hat_l), np.array(y_mean_l)
    r2 = 1.0 - np.sum((ya - yh) ** 2) / np.sum((ya - ym) ** 2)
    cache.write_text(str(r2))
    return r2


# ── Position computation: univariate symmetric ────────────────────────────────

def run_ew(panel, predictor, fwd_col, oos_gap, nw_lags, delta):
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


# ── Position computation: bivariate symmetric ─────────────────────────────────

def run_ew_bivariate(panel, pred1, pred2, fwd_col, oos_gap, nw_lags, delta):
    """Both betas must pass the |t| > T_THRESH gate."""
    tag = (f"pos_EWbiv_{pred1}_{pred2}_{fwd_col}_d{int(delta*10000)}bps"
           f"_t{int(T_THRESH*100)}_oos{OOS_START}.parquet")
    cache = CACHE_DIR / tag
    if cache.exists():
        return pd.read_parquet(cache).squeeze().rename(f"pos_{pred1}_{pred2}_{delta}")

    sub = panel.dropna(subset=[pred1, pred2, fwd_col]).copy()
    N   = len(sub)
    pos = pd.Series(0.0, index=sub.index, name=f"pos_{pred1}_{pred2}_{delta}")

    oos_idx = sub.index.searchsorted(pd.Timestamp(OOS_START))
    start_i = max(MIN_WIN + oos_gap, oos_idx)

    for i in range(start_i, N):
        train = sub.iloc[0 : i - oos_gap]
        X_tr  = add_constant(train[[pred1, pred2]], has_constant="skip")
        res   = OLS(train[fwd_col], X_tr).fit()
        nw    = _nw_se(res, nlags=nw_lags)
        t1    = float(res.params.iloc[1]) / float(nw[1])
        t2    = float(res.params.iloc[2]) / float(nw[2])
        if abs(t1) <= T_THRESH or abs(t2) <= T_THRESH:
            continue
        test = sub.iloc[[i]][[pred1, pred2]].copy()
        test.insert(0, "const", 1.0)
        y_hat = float(res.predict(test).iloc[0])
        if   y_hat >  delta: pos.iloc[i] =  1.0
        elif y_hat < -delta: pos.iloc[i] = -1.0

    pos.to_frame().to_parquet(cache)
    return pos


# ── Position computation: univariate asymmetric ───────────────────────────────

def run_ew_asym(panel, predictor, fwd_col, oos_gap, nw_lags):
    """Long if ŷ > mu500, short if ŷ < 0. mu500 computed from training data only."""
    tag = (f"pos_EWasym_{predictor}_{fwd_col}"
           f"_t{int(T_THRESH*100)}_oos{OOS_START}.parquet")
    cache = CACHE_DIR / tag
    if cache.exists():
        return pd.read_parquet(cache).squeeze().rename(f"pos_asym_{predictor}")

    sub = panel.dropna(subset=[predictor, fwd_col]).copy()
    N   = len(sub)
    pos = pd.Series(0.0, index=sub.index, name=f"pos_asym_{predictor}")

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
        actual_in_train = sub[fwd_col].iloc[max(0, i - oos_gap - 500) : i - oos_gap]
        mu500 = float(actual_in_train.mean()) if len(actual_in_train) > 0 else 0.0
        if   y_hat >  mu500: pos.iloc[i] =  1.0
        elif y_hat <  0.0:   pos.iloc[i] = -1.0

    pos.to_frame().to_parquet(cache)
    return pos


# ── Position computation: bivariate asymmetric ────────────────────────────────

def run_ew_asym_bivariate(panel, pred1, pred2, fwd_col, oos_gap, nw_lags):
    """Both betas must pass gate. Long if ŷ > mu500, short if ŷ < 0."""
    tag = (f"pos_EWasym_{pred1}_{pred2}_{fwd_col}"
           f"_t{int(T_THRESH*100)}_oos{OOS_START}.parquet")
    cache = CACHE_DIR / tag
    if cache.exists():
        return pd.read_parquet(cache).squeeze().rename(f"pos_asym_{pred1}_{pred2}")

    sub = panel.dropna(subset=[pred1, pred2, fwd_col]).copy()
    N   = len(sub)
    pos = pd.Series(0.0, index=sub.index, name=f"pos_asym_{pred1}_{pred2}")

    oos_idx = sub.index.searchsorted(pd.Timestamp(OOS_START))
    start_i = max(MIN_WIN + oos_gap, oos_idx)

    for i in range(start_i, N):
        train = sub.iloc[0 : i - oos_gap]
        X_tr  = add_constant(train[[pred1, pred2]], has_constant="skip")
        res   = OLS(train[fwd_col], X_tr).fit()
        nw    = _nw_se(res, nlags=nw_lags)
        t1    = float(res.params.iloc[1]) / float(nw[1])
        t2    = float(res.params.iloc[2]) / float(nw[2])
        if abs(t1) <= T_THRESH or abs(t2) <= T_THRESH:
            continue
        test = sub.iloc[[i]][[pred1, pred2]].copy()
        test.insert(0, "const", 1.0)
        y_hat = float(res.predict(test).iloc[0])
        actual_in_train = sub[fwd_col].iloc[max(0, i - oos_gap - 500) : i - oos_gap]
        mu500 = float(actual_in_train.mean()) if len(actual_in_train) > 0 else 0.0
        if   y_hat >  mu500: pos.iloc[i] =  1.0
        elif y_hat <  0.0:   pos.iloc[i] = -1.0

    pos.to_frame().to_parquet(cache)
    return pos


# ── Beta time series: univariate ─────────────────────────────────────────────

def compute_betas(panel, predictor, fwd_col, oos_gap, nw_lags):
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


# ── Beta time series: bivariate ───────────────────────────────────────────────

def compute_betas_bivariate(panel, pred1, pred2, fwd_col, oos_gap, nw_lags):
    tag   = f"betas_EWbiv_{pred1}_{pred2}_{fwd_col}_oos{OOS_START}.parquet"
    cache = CACHE_DIR / tag
    if cache.exists():
        return pd.read_parquet(cache)

    print(f"    Computing bivariate betas for {pred1} + {pred2} -> {fwd_col}...")
    sub     = panel.dropna(subset=[pred1, pred2, fwd_col]).copy()
    N       = len(sub)
    oos_idx = sub.index.searchsorted(pd.Timestamp(OOS_START))
    start_i = max(MIN_WIN + oos_gap, oos_idx)

    records = []
    for i in range(start_i, N):
        train = sub.iloc[0 : i - oos_gap]
        X_tr  = add_constant(train[[pred1, pred2]], has_constant="skip")
        res   = OLS(train[fwd_col], X_tr).fit()
        nw    = _nw_se(res, nlags=nw_lags)
        b1, se1 = float(res.params.iloc[1]), float(nw[1])
        b2, se2 = float(res.params.iloc[2]), float(nw[2])
        records.append({
            "beta_1":   b1,  "se_1":   se1,
            "t_stat_1": b1 / se1 if se1 > 0 else 0.0,
            "beta_2":   b2,  "se_2":   se2,
            "t_stat_2": b2 / se2 if se2 > 0 else 0.0,
        })

    df = pd.DataFrame(records, index=sub.index[start_i:])
    df.to_parquet(cache)
    return df


# ── Helpers ───────────────────────────────────────────────────────────────────

def oos_cumret(sim, start=OOS_START):
    net = sim["net_pnl"]
    s   = net[net.index >= start]
    return (1 + s).cumprod()


def _shade(ax, start, end):
    for a, b in [("2020-02-01", "2020-06-01"), ("2022-01-01", "2022-12-31")]:
        a, b = pd.Timestamp(a), pd.Timestamp(b)
        if b > start and a < end:
            ax.axvspan(max(a, start), min(b, end), alpha=0.08, color="grey", lw=0)


def _stat_window(sim_dict_or_sim, bah_sim):
    """Compute shared stat window for all curves on a plot.

    For a dict of (st, sim) per delta, finds the earliest first activation
    across all deltas. If after 2020-01-01, returns (_rebase_start, _stat_start,
    _stat_lbl). Otherwise returns (None, OOS_START timestamp, "").
    """
    if isinstance(sim_dict_or_sim, dict):
        candidates = []
        for _, sim in sim_dict_or_sim.values():
            p = sim["position"][sim.index >= OOS_START]
            a = p[p != 0]
            if len(a):
                candidates.append(a.index.min())
        activation = min(candidates) if candidates else None
    else:
        sim = sim_dict_or_sim
        p = sim["position"][sim.index >= OOS_START]
        a = p[p != 0]
        activation = a.index.min() if len(a) else None

    rebase_start = None
    if activation is not None and activation > pd.Timestamp("2020-01-01"):
        idx       = bah_sim.index
        act_iloc  = idx.searchsorted(activation)
        rebase_start = idx[min(act_iloc + 1, len(idx) - 1)]

    stat_start = rebase_start if rebase_start is not None else pd.Timestamp(OOS_START)
    stat_lbl   = (f" · stats from {stat_start.strftime('%Y-%m-%d')}"
                  if rebase_start is not None else "")
    return rebase_start, stat_start, stat_lbl


# ── Plotting: symmetric multi-delta (univariate) ──────────────────────────────

def plot_2panel(pred_label, horizon_label, oos_gap, nw_lags,
                color_palette, sim_dict, betas_df, bah_sim, out_path,
                r2_oos=None):
    oos_dt = pd.Timestamp(OOS_START)
    e_dt   = betas_df.index[-1]
    xlim   = (oos_dt, e_dt)
    n_deltas = len(DELTAS)

    _rebase_start, _stat_start, _stat_lbl = _stat_window(sim_dict, bah_sim)

    r2_str = f"  R²_OOS = {r2_oos:+.4f}" if r2_oos is not None else ""

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
        f"(Expanding Window, OOS from {OOS_START}){r2_str}\n"
        f"Training grows daily; OOS gap = {oos_gap} days; "
        f"NW-HAC {nw_lags} lags; |t| > {T_THRESH:.2f} gate; 0.05% slippage",
        fontsize=10, y=0.998,
    )
    fig.subplots_adjust(top=0.955, bottom=0.03, left=0.10, right=0.93)

    ax_ret.set_xlim(*xlim)
    _shade(ax_ret, oos_dt, e_dt)

    _bah_st_plot = compute_performance_stats(
        bah_sim[bah_sim.index >= _stat_start], "BaH_plot")
    bah_oos = oos_cumret(bah_sim)
    ax_ret.plot(bah_oos.index, bah_oos.values,
                color=BAH_COLOR, lw=1.5, ls="-.", alpha=0.6,
                label=(f"Buy-and-Hold{_stat_lbl}  "
                       f"[SR={_bah_st_plot['sharpe']:+.2f}  "
                       f"ret={_bah_st_plot['ann_ret']*100:+.1f}%  "
                       f"DD={_bah_st_plot['max_dd']*100:.1f}%]"))

    if _rebase_start is not None:
        _bah_from   = bah_sim["net_pnl"][bah_sim.index >= _rebase_start]
        _bah_act    = (1 + _bah_from).cumprod()
        _bah_act_st = compute_performance_stats(
            bah_sim[bah_sim.index >= _rebase_start], "BaH_act")
        ax_ret.plot(
            _bah_act.index, _bah_act.values,
            color=BAH_COLOR, lw=1.2, ls=":", alpha=0.85,
            label=(f"Buy-and-Hold from {_rebase_start.strftime('%Y-%m-%d')}  "
                   f"[SR={_bah_act_st['sharpe']:+.2f}  "
                   f"ret={_bah_act_st['ann_ret']*100:+.1f}%  "
                   f"DD={_bah_act_st['max_dd']*100:.1f}%]"),
        )

    for di, (delta, lbl) in enumerate(zip(DELTAS, DELTA_LBL)):
        _, sim = sim_dict[di]
        cum    = oos_cumret(sim)
        st_plot = compute_performance_stats(
            sim[sim.index >= _stat_start], f"EW_{di}_plot")
        pos_stat = sim["position"][sim.index >= _stat_start]
        pL = float((pos_stat == 1).mean() * 100)
        pS = float((pos_stat == -1).mean() * 100)
        ax_ret.plot(cum.index, cum.values,
                    color=color_palette[di], lw=1.8, alpha=0.9,
                    label=(f"{lbl}  "
                           f"[SR={st_plot['sharpe']:+.2f}  "
                           f"ret={st_plot['ann_ret']*100:+.1f}%  "
                           f"DD={st_plot['max_dd']*100:.1f}%  "
                           f"L{pL:.0f}%/S{pS:.0f}%]"))

    ax_ret.axhline(1, color="black", lw=0.4, ls=":")
    ax_ret.set_yscale("log")
    ax_ret.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2f}x"))
    ax_ret.set_ylabel("Cumulative Net Return (log)", fontsize=9)
    ax_ret.legend(fontsize=8, loc="upper left", framealpha=0.92)
    ax_ret.grid(axis="y", alpha=0.2, lw=0.6)
    ax_ret.spines[["top", "right"]].set_visible(False)

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


# ── Plotting: symmetric multi-delta (bivariate) ───────────────────────────────

def plot_2panel_bivariate(pred1_label, pred2_label, horizon_label, oos_gap, nw_lags,
                          color_palette, sim_dict, betas_df, bah_sim, out_path,
                          r2_oos=None):
    oos_dt = pd.Timestamp(OOS_START)
    e_dt   = betas_df.index[-1]
    xlim   = (oos_dt, e_dt)
    n_deltas = len(DELTAS)

    _rebase_start, _stat_start, _stat_lbl = _stat_window(sim_dict, bah_sim)

    r2_str = f"  R²_OOS = {r2_oos:+.4f}" if r2_oos is not None else ""

    h_ratios = [2.5, 1.2] + [1.0] * n_deltas
    fig, axes = plt.subplots(
        2 + n_deltas, 1, figsize=(14, 11 + 2.2 * n_deltas), sharex=True,
        gridspec_kw={"height_ratios": h_ratios, "hspace": 0.35},
    )
    ax_ret = axes[0]
    ax_t   = axes[1]
    ax_pos = axes[2:]

    fig.suptitle(
        f"{pred1_label} + {pred2_label} -> {horizon_label} Forward Return  "
        f"(Expanding Window, OOS from {OOS_START}){r2_str}\n"
        f"Training grows daily; OOS gap = {oos_gap} days; "
        f"NW-HAC {nw_lags} lags; |t| > {T_THRESH:.2f} gate (both betas); 0.05% slippage",
        fontsize=10, y=0.998,
    )
    fig.subplots_adjust(top=0.955, bottom=0.03, left=0.10, right=0.93)

    ax_ret.set_xlim(*xlim)
    _shade(ax_ret, oos_dt, e_dt)

    _bah_st_plot = compute_performance_stats(
        bah_sim[bah_sim.index >= _stat_start], "BaH_plot")
    bah_oos = oos_cumret(bah_sim)
    ax_ret.plot(bah_oos.index, bah_oos.values,
                color=BAH_COLOR, lw=1.5, ls="-.", alpha=0.6,
                label=(f"Buy-and-Hold{_stat_lbl}  "
                       f"[SR={_bah_st_plot['sharpe']:+.2f}  "
                       f"ret={_bah_st_plot['ann_ret']*100:+.1f}%  "
                       f"DD={_bah_st_plot['max_dd']*100:.1f}%]"))

    if _rebase_start is not None:
        _bah_from   = bah_sim["net_pnl"][bah_sim.index >= _rebase_start]
        _bah_act    = (1 + _bah_from).cumprod()
        _bah_act_st = compute_performance_stats(
            bah_sim[bah_sim.index >= _rebase_start], "BaH_act")
        ax_ret.plot(
            _bah_act.index, _bah_act.values,
            color=BAH_COLOR, lw=1.2, ls=":", alpha=0.85,
            label=(f"Buy-and-Hold from {_rebase_start.strftime('%Y-%m-%d')}  "
                   f"[SR={_bah_act_st['sharpe']:+.2f}  "
                   f"ret={_bah_act_st['ann_ret']*100:+.1f}%  "
                   f"DD={_bah_act_st['max_dd']*100:.1f}%]"),
        )

    for di, (delta, lbl) in enumerate(zip(DELTAS, DELTA_LBL)):
        _, sim = sim_dict[di]
        cum     = oos_cumret(sim)
        st_plot = compute_performance_stats(
            sim[sim.index >= _stat_start], f"EWbiv_{di}_plot")
        pos_stat = sim["position"][sim.index >= _stat_start]
        pL = float((pos_stat == 1).mean() * 100)
        pS = float((pos_stat == -1).mean() * 100)
        ax_ret.plot(cum.index, cum.values,
                    color=color_palette[di], lw=1.8, alpha=0.9,
                    label=(f"{lbl}  "
                           f"[SR={st_plot['sharpe']:+.2f}  "
                           f"ret={st_plot['ann_ret']*100:+.1f}%  "
                           f"DD={st_plot['max_dd']*100:.1f}%  "
                           f"L{pL:.0f}%/S{pS:.0f}%]"))

    ax_ret.axhline(1, color="black", lw=0.4, ls=":")
    ax_ret.set_yscale("log")
    ax_ret.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2f}x"))
    ax_ret.set_ylabel("Cumulative Net Return (log)", fontsize=9)
    ax_ret.legend(fontsize=8, loc="upper left", framealpha=0.92)
    ax_ret.grid(axis="y", alpha=0.2, lw=0.6)
    ax_ret.spines[["top", "right"]].set_visible(False)

    ax_t.set_xlim(*xlim)
    _shade(ax_t, oos_dt, e_dt)

    t1 = betas_df["t_stat_1"]
    t2 = betas_df["t_stat_2"]
    b1 = betas_df["beta_1"]
    b2 = betas_df["beta_2"]

    ax_t.plot(t1.index, t1.values, color=color_palette[0], lw=1.0, alpha=0.85,
              label=f"NW t-stat: {pred1_label} ({nw_lags}-lag HAC)")
    ax_t.plot(t2.index, t2.values, color=color_palette[0], lw=1.0, alpha=0.60, ls="--",
              label=f"NW t-stat: {pred2_label} ({nw_lags}-lag HAC)")
    ax_t.fill_between(t1.index, -T_THRESH, T_THRESH,
                      color="firebrick", alpha=0.05, label="Below gate (flat zone)")
    ax_t.axhline( T_THRESH, color="firebrick", lw=1.2, ls="--",
                 label=f"|t| = {T_THRESH:.2f} gate")
    ax_t.axhline(-T_THRESH, color="firebrick", lw=1.2, ls="--")
    ax_t.axhline(0, color="black", lw=0.5, ls=":")
    ax_t.set_ylabel("NW t-stat", fontsize=9)
    ax_t.grid(axis="y", alpha=0.2, lw=0.6)
    ax_t.spines["top"].set_visible(False)

    ax_t2 = ax_t.twinx()
    ax_t2.plot(b1.index, b1.values, color="dimgrey", lw=1.0, ls="-", alpha=0.60,
               label=f"Beta: {pred1_label}")
    ax_t2.plot(b2.index, b2.values, color="dimgrey", lw=1.0, ls=":", alpha=0.60,
               label=f"Beta: {pred2_label}")
    ax_t2.axhline(0, color="dimgrey", lw=0.4, ls=":")
    ax_t2.set_ylabel("Beta", fontsize=8, color="dimgrey")
    ax_t2.tick_params(axis="y", labelcolor="dimgrey", labelsize=7)
    ax_t2.spines["top"].set_visible(False)

    lines1, labs1 = ax_t.get_legend_handles_labels()
    lines2, labs2 = ax_t2.get_legend_handles_labels()
    ax_t.legend(lines1 + lines2, labs1 + labs2, fontsize=8, loc="upper left")

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


# ── Plotting: asymmetric threshold (univariate) ───────────────────────────────

def plot_asym(pred_label, horizon_label, oos_gap, nw_lags,
              main_color, sim, betas_df, bah_sim, out_path,
              r2_oos=None, extra_title=""):
    oos_dt = pd.Timestamp(OOS_START)
    e_dt   = betas_df.index[-1]
    xlim   = (oos_dt, e_dt)

    _rebase_start, _stat_start, _stat_lbl = _stat_window(sim, bah_sim)

    r2_str = f"  R²_OOS = {r2_oos:+.4f}" if r2_oos is not None else ""

    h_ratios = [2.5, 1.2, 1.0]
    fig, axes = plt.subplots(
        3, 1, figsize=(14, 10), sharex=True,
        gridspec_kw={"height_ratios": h_ratios, "hspace": 0.35},
    )
    ax_ret, ax_t, ax_p = axes

    fig.suptitle(
        f"{pred_label} -> {horizon_label} Forward Return  "
        f"(Expanding Window, OOS from {OOS_START}, Asymmetric Threshold){r2_str}"
        f"{extra_title}\n"
        f"Long: ŷ > μ₅₀₀  Short: ŷ < 0  "
        f"NW-HAC {nw_lags} lags; |t| > {T_THRESH:.2f} gate; 0.05% slippage",
        fontsize=10, y=0.998,
    )
    fig.subplots_adjust(top=0.945, bottom=0.04, left=0.10, right=0.93)

    ax_ret.set_xlim(*xlim)
    _shade(ax_ret, oos_dt, e_dt)

    bah_oos      = oos_cumret(bah_sim)
    _bah_st_plot = compute_performance_stats(bah_sim[bah_sim.index >= _stat_start], "BaH_plot")
    ax_ret.plot(bah_oos.index, bah_oos.values,
                color=BAH_COLOR, lw=1.5, ls="-.", alpha=0.6,
                label=(f"Buy-and-Hold{_stat_lbl}  "
                       f"[SR={_bah_st_plot['sharpe']:+.2f}  "
                       f"ret={_bah_st_plot['ann_ret']*100:+.1f}%  "
                       f"DD={_bah_st_plot['max_dd']*100:.1f}%]"))

    if _rebase_start is not None:
        _bah_from   = bah_sim["net_pnl"][bah_sim.index >= _rebase_start]
        _bah_act    = (1 + _bah_from).cumprod()
        _bah_act_st = compute_performance_stats(
            bah_sim[bah_sim.index >= _rebase_start], "BaH_act")
        ax_ret.plot(
            _bah_act.index, _bah_act.values,
            color=BAH_COLOR, lw=1.2, ls=":", alpha=0.85,
            label=(f"Buy-and-Hold from {_rebase_start.strftime('%Y-%m-%d')}  "
                   f"[SR={_bah_act_st['sharpe']:+.2f}  "
                   f"ret={_bah_act_st['ann_ret']*100:+.1f}%  "
                   f"DD={_bah_act_st['max_dd']*100:.1f}%]"),
        )

    st      = compute_performance_stats(sim[sim.index >= _stat_start], "asym")
    cum     = oos_cumret(sim)
    pos_oos = sim["position"][sim.index >= _stat_start]
    pL = float((pos_oos == 1).mean() * 100)
    pS = float((pos_oos == -1).mean() * 100)
    ax_ret.plot(cum.index, cum.values,
                color=main_color, lw=1.8, alpha=0.9,
                label=(f"Asymmetric  "
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

    ax_t.set_xlim(*xlim)
    _shade(ax_t, oos_dt, e_dt)

    t_series = betas_df["t_stat"]
    b_series = betas_df["beta"]

    ax_t.plot(t_series.index, t_series.values,
              color=main_color, lw=1.0, alpha=0.85,
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

    pos = sim["position"][sim.index >= OOS_START]
    ax_p.set_xlim(*xlim)
    _shade(ax_p, oos_dt, e_dt)
    ax_p.fill_between(pos.index, pos.where(pos ==  1, 0), 0,
                      color=main_color, alpha=0.75, label="Long")
    ax_p.fill_between(pos.index, pos.where(pos == -1, 0), 0,
                      color=main_color, alpha=0.30, hatch="///", label="Short")
    ax_p.axhline(0, color="black", lw=0.4)
    ax_p.set_ylim(-1.5, 1.5)
    ax_p.set_yticks([-1, 0, 1])
    ax_p.set_yticklabels(["Short", "Flat", "Long"], fontsize=8)
    ax_p.set_ylabel("Position", fontsize=9)
    pF = float((pos_oos == 0).mean() * 100)
    ax_p.text(0.01, 0.97,
              f"Long {pL:.1f}%  Short {pS:.1f}%  Flat {pF:.1f}%  "
              f"AvgPos={float(pos_oos.mean()):+.3f}",
              transform=ax_p.transAxes, fontsize=7.5, va="top", color=main_color)
    ax_p.spines[["top", "right"]].set_visible(False)

    for ax in [ax_ret, ax_t, ax_p]:
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.tick_params(axis="x", which="major", labelbottom=True, labelsize=7, pad=2)

    fig.savefig(out_path, dpi=155, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ── Plotting: asymmetric threshold (bivariate) ────────────────────────────────

def plot_asym_bivariate(pred1_label, pred2_label, horizon_label, oos_gap, nw_lags,
                        main_color, sim, betas_df, bah_sim, out_path,
                        r2_oos=None, extra_title=""):
    oos_dt = pd.Timestamp(OOS_START)
    e_dt   = betas_df.index[-1]
    xlim   = (oos_dt, e_dt)

    _rebase_start, _stat_start, _stat_lbl = _stat_window(sim, bah_sim)

    r2_str = f"  R²_OOS = {r2_oos:+.4f}" if r2_oos is not None else ""

    h_ratios = [2.5, 1.2, 1.0]
    fig, axes = plt.subplots(
        3, 1, figsize=(14, 10), sharex=True,
        gridspec_kw={"height_ratios": h_ratios, "hspace": 0.35},
    )
    ax_ret, ax_t, ax_p = axes

    fig.suptitle(
        f"{pred1_label} + {pred2_label} -> {horizon_label} Forward Return  "
        f"(Expanding Window, OOS from {OOS_START}, Asymmetric Threshold){r2_str}"
        f"{extra_title}\n"
        f"Long: ŷ > μ₅₀₀  Short: ŷ < 0  "
        f"NW-HAC {nw_lags} lags; |t| > {T_THRESH:.2f} gate (both betas); 0.05% slippage",
        fontsize=10, y=0.998,
    )
    fig.subplots_adjust(top=0.945, bottom=0.04, left=0.10, right=0.93)

    ax_ret.set_xlim(*xlim)
    _shade(ax_ret, oos_dt, e_dt)

    bah_oos      = oos_cumret(bah_sim)
    _bah_st_plot = compute_performance_stats(bah_sim[bah_sim.index >= _stat_start], "BaH_plot")
    ax_ret.plot(bah_oos.index, bah_oos.values,
                color=BAH_COLOR, lw=1.5, ls="-.", alpha=0.6,
                label=(f"Buy-and-Hold{_stat_lbl}  "
                       f"[SR={_bah_st_plot['sharpe']:+.2f}  "
                       f"ret={_bah_st_plot['ann_ret']*100:+.1f}%  "
                       f"DD={_bah_st_plot['max_dd']*100:.1f}%]"))

    if _rebase_start is not None:
        _bah_from   = bah_sim["net_pnl"][bah_sim.index >= _rebase_start]
        _bah_act    = (1 + _bah_from).cumprod()
        _bah_act_st = compute_performance_stats(
            bah_sim[bah_sim.index >= _rebase_start], "BaH_act")
        ax_ret.plot(
            _bah_act.index, _bah_act.values,
            color=BAH_COLOR, lw=1.2, ls=":", alpha=0.85,
            label=(f"Buy-and-Hold from {_rebase_start.strftime('%Y-%m-%d')}  "
                   f"[SR={_bah_act_st['sharpe']:+.2f}  "
                   f"ret={_bah_act_st['ann_ret']*100:+.1f}%  "
                   f"DD={_bah_act_st['max_dd']*100:.1f}%]"),
        )

    st      = compute_performance_stats(sim[sim.index >= _stat_start], "asym_biv")
    cum     = oos_cumret(sim)
    pos_oos = sim["position"][sim.index >= _stat_start]
    pL = float((pos_oos == 1).mean() * 100)
    pS = float((pos_oos == -1).mean() * 100)
    ax_ret.plot(cum.index, cum.values,
                color=main_color, lw=1.8, alpha=0.9,
                label=(f"Asymmetric  "
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

    ax_t.set_xlim(*xlim)
    _shade(ax_t, oos_dt, e_dt)

    t1 = betas_df["t_stat_1"]
    t2 = betas_df["t_stat_2"]
    b1 = betas_df["beta_1"]
    b2 = betas_df["beta_2"]

    ax_t.plot(t1.index, t1.values, color=main_color, lw=1.0, alpha=0.85,
              label=f"NW t-stat: {pred1_label} ({nw_lags}-lag HAC)")
    ax_t.plot(t2.index, t2.values, color=main_color, lw=1.0, alpha=0.60, ls="--",
              label=f"NW t-stat: {pred2_label} ({nw_lags}-lag HAC)")
    ax_t.fill_between(t1.index, -T_THRESH, T_THRESH,
                      color="firebrick", alpha=0.05, label="Below gate (flat zone)")
    ax_t.axhline( T_THRESH, color="firebrick", lw=1.2, ls="--",
                 label=f"|t| = {T_THRESH:.2f} gate")
    ax_t.axhline(-T_THRESH, color="firebrick", lw=1.2, ls="--")
    ax_t.axhline(0, color="black", lw=0.5, ls=":")
    ax_t.set_ylabel("NW t-stat", fontsize=9)
    ax_t.grid(axis="y", alpha=0.2, lw=0.6)
    ax_t.spines["top"].set_visible(False)

    ax_t2 = ax_t.twinx()
    ax_t2.plot(b1.index, b1.values, color="dimgrey", lw=1.0, ls="-", alpha=0.60,
               label=f"Beta: {pred1_label}")
    ax_t2.plot(b2.index, b2.values, color="dimgrey", lw=1.0, ls=":", alpha=0.60,
               label=f"Beta: {pred2_label}")
    ax_t2.axhline(0, color="dimgrey", lw=0.4, ls=":")
    ax_t2.set_ylabel("Beta", fontsize=8, color="dimgrey")
    ax_t2.tick_params(axis="y", labelcolor="dimgrey", labelsize=7)
    ax_t2.spines["top"].set_visible(False)

    lines1, labs1 = ax_t.get_legend_handles_labels()
    lines2, labs2 = ax_t2.get_legend_handles_labels()
    ax_t.legend(lines1 + lines2, labs1 + labs2, fontsize=8, loc="upper left")

    pos = sim["position"][sim.index >= OOS_START]
    ax_p.set_xlim(*xlim)
    _shade(ax_p, oos_dt, e_dt)
    ax_p.fill_between(pos.index, pos.where(pos ==  1, 0), 0,
                      color=main_color, alpha=0.75, label="Long")
    ax_p.fill_between(pos.index, pos.where(pos == -1, 0), 0,
                      color=main_color, alpha=0.30, hatch="///", label="Short")
    ax_p.axhline(0, color="black", lw=0.4)
    ax_p.set_ylim(-1.5, 1.5)
    ax_p.set_yticks([-1, 0, 1])
    ax_p.set_yticklabels(["Short", "Flat", "Long"], fontsize=8)
    ax_p.set_ylabel("Position", fontsize=9)
    pF = float((pos_oos == 0).mean() * 100)
    ax_p.text(0.01, 0.97,
              f"Long {pL:.1f}%  Short {pS:.1f}%  Flat {pF:.1f}%  "
              f"AvgPos={float(pos_oos.mean()):+.3f}",
              transform=ax_p.transAxes, fontsize=7.5, va="top", color=main_color)
    ax_p.spines[["top", "right"]].set_visible(False)

    for ax in [ax_ret, ax_t, ax_p]:
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.tick_params(axis="x", which="major", labelbottom=True, labelsize=7, pad=2)

    fig.savefig(out_path, dpi=155, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("  Horizon Regression — Expanding Window")
    print("  (B) VVIX MA5        -> 20-day  (symmetric)")
    print("  (C) vol_trend       -> 20-day  (poor correlation baseline)")
    print("  (D) VIX spot        -> 20-day  (poor correlation baseline)")
    print("  (E) term_slope      -> 20-day  (poor correlation baseline)")
    print("  (G) VRP + Open Int. -> 20-day  (symmetric + asymmetric)")
    print("=" * 72)

    print("\n[1] Loading data...")
    vrp        = load_vrp_series()
    es         = load_es_front_month()
    vvix_ma5   = compute_vvix_ma5(load_vvix())
    vix_spot   = load_vix_spot()
    term_slope = compute_vix_term_slope(load_vix_futures_term_structure())
    oi         = load_es_open_interest()
    panel      = build_panel(vrp, es, vvix_ma5, vix_spot, term_slope, oi)
    print(f"    {len(panel):,} obs  "
          f"[{panel.index.min().date()} - {panel.index.max().date()}]")

    daily_ret = panel["daily_ret"].dropna()
    bah_pos   = compute_buy_and_hold(daily_ret)
    bah_sim_  = simulate_strategy(bah_pos, daily_ret)

    # ── (B) VVIX MA5 -> 20-day symmetric ────────────────────────────────────
    print("\n[2] VVIX MA5 -> 20-day expanding-window positions...")
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

    vvix_betas  = compute_betas(panel, "vvix_ma5", "fwd_20d", oos_gap=20, nw_lags=20)
    vvix_r2_oos = compute_oos_r2(panel, "vvix_ma5", "fwd_20d", oos_gap=20, nw_lags=20)
    print(f"    OOS R² (VVIX MA5 20d) = {vvix_r2_oos:+.4f}")

    out_vvix = OUTPUT / "expanding_window" / "VVIX MA5"
    out_vvix.mkdir(parents=True, exist_ok=True)
    plot_2panel(
        pred_label="VVIX MA5", horizon_label="20-day",
        oos_gap=20, nw_lags=20,
        color_palette=["#3f007d", "#6a51a3", "#807dba", "#9e9ac8"],
        sim_dict=vvix_sims, betas_df=vvix_betas,
        bah_sim=bah_sim_,
        out_path=out_vvix / "symmetric_VVIX_MA5.png",
        r2_oos=vvix_r2_oos,
    )

    # ── (C) vol_trend -> 20-day (poor correlation baseline) ─────────────────
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

    vt_betas  = compute_betas(panel, "vol_trend", "fwd_20d", oos_gap=20, nw_lags=20)
    vt_r2_oos = compute_oos_r2(panel, "vol_trend", "fwd_20d", oos_gap=20, nw_lags=20)
    print(f"    OOS R² (vol_trend 20d) = {vt_r2_oos:+.4f}")

    out_vt = OUTPUT / "expanding_window" / "poor_correlation" / "Vol Trend"
    out_vt.mkdir(parents=True, exist_ok=True)
    plot_2panel(
        pred_label="Vol Trend [ln(RV5/RV22)]", horizon_label="20-day",
        oos_gap=20, nw_lags=20,
        color_palette=["#00441b", "#238b45", "#41ab5d", "#74c476"],
        sim_dict=vt_sims, betas_df=vt_betas,
        bah_sim=bah_sim_,
        out_path=out_vt / "symmetric_Vol_Trend.png",
        r2_oos=vt_r2_oos,
    )

    # ── (D) VIX spot -> 20-day (poor correlation baseline) ──────────────────
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

    vix_betas  = compute_betas(panel, "vix", "fwd_20d", oos_gap=20, nw_lags=20)
    vix_r2_oos = compute_oos_r2(panel, "vix", "fwd_20d", oos_gap=20, nw_lags=20)
    print(f"    OOS R² (VIX 20d) = {vix_r2_oos:+.4f}")

    out_vix = OUTPUT / "expanding_window" / "poor_correlation" / "VIX"
    out_vix.mkdir(parents=True, exist_ok=True)
    plot_2panel(
        pred_label="VIX", horizon_label="20-day",
        oos_gap=20, nw_lags=20,
        color_palette=["#7f0000", "#cb181d", "#ef3b2c", "#fc9272"],
        sim_dict=vix_sims, betas_df=vix_betas,
        bah_sim=bah_sim_,
        out_path=out_vix / "symmetric_VIX.png",
        r2_oos=vix_r2_oos,
    )

    # ── (E) term_slope -> 20-day (poor correlation baseline) ────────────────
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

    ts_betas  = compute_betas(panel, "term_slope", "fwd_20d", oos_gap=20, nw_lags=20)
    ts_r2_oos = compute_oos_r2(panel, "term_slope", "fwd_20d", oos_gap=20, nw_lags=20)
    print(f"    OOS R² (term_slope 20d) = {ts_r2_oos:+.4f}")

    out_ts = OUTPUT / "expanding_window" / "poor_correlation" / "Term Slope"
    out_ts.mkdir(parents=True, exist_ok=True)
    plot_2panel(
        pred_label="Term Slope", horizon_label="20-day",
        oos_gap=20, nw_lags=20,
        color_palette=["#7f2704", "#d94801", "#fd8d3c", "#fdbe85"],
        sim_dict=ts_sims, betas_df=ts_betas,
        bah_sim=bah_sim_,
        out_path=out_ts / "symmetric_Term_Slope.png",
        r2_oos=ts_r2_oos,
    )

    # ── (G) VRP + Open Interest -> 20-day symmetric ──────────────────────────
    print("\n[7] VRP + Open Interest -> 20-day expanding-window positions (symmetric)...")
    oi_sims = {}
    for di, (delta, lbl) in enumerate(zip(DELTAS, DELTA_LBL)):
        print(f"    {lbl}...", end="  ", flush=True)
        pos = run_ew_bivariate(panel, "VP", "open_interest",
                               "fwd_20d", oos_gap=20, nw_lags=20, delta=delta)
        sim = simulate_strategy(pos, daily_ret)
        st  = compute_performance_stats(sim[sim.index >= OOS_START], f"EW_VRPOI20d_{lbl}")
        oi_sims[di] = (st, sim)
        p   = pos[pos.index >= OOS_START]
        print(f"SR={st['sharpe']:+.3f}  L={float((p==1).mean())*100:.1f}%  "
              f"S={float((p==-1).mean())*100:.1f}%  "
              f"F={float((p==0).mean())*100:.1f}%")

    oi_betas  = compute_betas_bivariate(panel, "VP", "open_interest",
                                        "fwd_20d", oos_gap=20, nw_lags=20)
    oi_r2_oos = compute_oos_r2_bivariate(panel, "VP", "open_interest",
                                         "fwd_20d", oos_gap=20, nw_lags=20)
    print(f"    OOS R² (VRP + OI 20d) = {oi_r2_oos:+.4f}")

    out_oi = OUTPUT / "expanding_window" / "VRP + Open Interest"
    out_oi.mkdir(parents=True, exist_ok=True)
    plot_2panel_bivariate(
        pred1_label="VRP", pred2_label="Open Interest", horizon_label="20-day",
        oos_gap=20, nw_lags=20,
        color_palette=["#54278f", "#756bb1", "#9e9ac8", "#cbc9e2"],
        sim_dict=oi_sims, betas_df=oi_betas,
        bah_sim=bah_sim_,
        out_path=out_oi / "symmetric_VRP_+_Open_Interest.png",
        r2_oos=oi_r2_oos,
    )

    # ── (8) Signal regressions (full-sample, NW-20 HAC SEs) ─────────────────
    print("\n[8] Signal regressions (full sample, NW-20 HAC SEs)...")
    sub = panel.dropna(subset=["VP", "vvix_ma5", "fwd_20d"]).copy()
    for y_col, x_col in [("vvix_ma5", "VP"), ("VP", "vvix_ma5")]:
        X = add_constant(sub[[x_col]], has_constant="skip")
        res = OLS(sub[y_col], X).fit()
        nw  = _nw_se(res, nlags=20)
        alpha, beta = float(res.params.iloc[0]), float(res.params.iloc[1])
        t_a,   t_b  = alpha / float(nw[0]), beta / float(nw[1])
        r2    = float(res.rsquared)
        corr  = float(sub[[y_col, x_col]].corr().iloc[0, 1])
        N     = len(sub)
        print(f"  {y_col} ~ {x_col}:  "
              f"alpha={alpha:.4f} (t={t_a:.2f})  "
              f"beta={beta:.4f} (t={t_b:.2f})  "
              f"R2={r2:.4f}  corr={corr:.3f}  N={N:,}")

    # ── (9A) VRP + Open Interest -> 20-day asymmetric ────────────────────────
    print("\n[9A] VRP + Open Interest -> 20-day, asymmetric threshold...")
    pos_asym_oi = run_ew_asym_bivariate(
        panel, "VP", "open_interest", "fwd_20d", oos_gap=20, nw_lags=20)
    sim_asym_oi = simulate_strategy(pos_asym_oi, daily_ret)
    p_asym_oi = pos_asym_oi[pos_asym_oi.index >= OOS_START]
    print(f"    L={float((p_asym_oi==1).mean())*100:.1f}%  "
          f"S={float((p_asym_oi==-1).mean())*100:.1f}%  "
          f"F={float((p_asym_oi==0).mean())*100:.1f}%")

    plot_asym_bivariate(
        pred1_label="VRP", pred2_label="Open Interest", horizon_label="20-day",
        oos_gap=20, nw_lags=20,
        main_color="#54278f",
        sim=sim_asym_oi, betas_df=oi_betas,
        bah_sim=bah_sim_,
        out_path=out_oi / "asymmetric_VRP_+_Open_Interest.png",
        r2_oos=oi_r2_oos,
    )

    print("\nDone.")
    print(f"  output/expanding_window/VVIX MA5/symmetric_VVIX_MA5.png")
    print(f"  output/expanding_window/poor_correlation/Vol Trend/symmetric_Vol_Trend.png")
    print(f"  output/expanding_window/poor_correlation/VIX/symmetric_VIX.png")
    print(f"  output/expanding_window/poor_correlation/Term Slope/symmetric_Term_Slope.png")
    print(f"  output/expanding_window/VRP + Open Interest/symmetric_VRP_+_Open_Interest.png")
    print(f"  output/expanding_window/VRP + Open Interest/asymmetric_VRP_+_Open_Interest.png")
    print("=" * 72)


if __name__ == "__main__":
    main()
