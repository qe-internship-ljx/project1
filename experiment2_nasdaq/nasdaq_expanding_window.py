"""
nasdaq_expanding_window.py
==========================
Experiment 2 — NASDAQ Edition: Expanding-Window OOS Evaluation

Replicates the VRP / VRP+TermSlope / VRP+VVIX / VVIX expanding-window
experiments from experiment2/fixed_split_eval.py, replacing the dependent
variable with NQ (NASDAQ-100 E-mini) front-month 20-day forward return.

The signals (VRP from HAR, VIX futures term-structure slope, VVIX MA5) and
the full protocol are identical to experiment2:
  - OOS start: 2012-01-01
  - Minimum training window: 500 observations
  - OOS gap (label-leakage buffer): 20 days
  - NW-HAC standard errors with 20 lags
  - |t| > 1.28 gate on every beta (flat when fails)
  - Binary position sizing: ±1 full notional or 0
  - Slippage: 0.05% per unit of position change
  - Deltas: 0.2%, 0.5%, 0.75%, 1.0%

Two threshold variants per model:
  1. Symmetric — long when ŷ > δ, short when ŷ < −δ
  2. Rolling-mu — long when ŷ > μ + δ, short when ŷ < μ − δ
     (μ = 500-day rolling mean of realised NQ 20-day returns, in-sample only)

Models evaluated:
  Base        — VRP only
  Model_A     — VRP + VIX term-structure slope
  Model_C     — VRP + VVIX MA5
  Model_VVIX  — Univariate VVIX MA5

Plots follow the specification in experiment2/plot.md exactly (6 panels:
cumulative return · t-stat/beta · 4 position panels).

Outputs saved to experiment2_nasdaq/output/expanding_window/.
All positions and beta time-series are cached in output/regression_cache/.
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

EXP2_DIR = ROOT.parent / "experiment2"
sys.path.insert(0, str(ROOT.parent / "bh_replication"))
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(EXP2_DIR))

from har_model import _nw_se
from helpers import (
    load_vrp_series,
    load_vix_futures_term_structure,
    load_vvix,
    compute_vvix_ma5,
    compute_buy_and_hold,
    simulate_strategy,
    compute_performance_stats,
)
from fh_replication.fh_replication import compute_vix_term_slope

DATA = ROOT.parent / "data"

# ── Constants ─────────────────────────────────────────────────────────────────
OOS_START  = "2012-01-01"
MIN_WIN    = 500
T_THRESH   = 1.28
NW_LAGS    = 20
DELTAS     = [0.002, 0.005, 0.0075, 0.010]
DELTA_LBL  = ["d=0.2%", "d=0.5%", "d=0.75%", "d=1.0%"]
BAH_COLOR  = "#d62728"

MODELS = ["Base", "Model_A", "Model_C", "Model_VVIX"]

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

EW_MODEL_DIR = {
    "Base":       OUTPUT / "expanding_window" / "VRP",
    "Model_A":    OUTPUT / "expanding_window" / "VRP + Term Slope",
    "Model_C":    OUTPUT / "expanding_window" / "VRP + VVIX MA5",
    "Model_VVIX": OUTPUT / "expanding_window" / "VVIX MA5",
}
DIR_EW_CMP = OUTPUT / "expanding_window" / "comparisons"

for _d in [*EW_MODEL_DIR.values(), DIR_EW_CMP]:
    _d.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING — NQ Front-Month Futures
# ═══════════════════════════════════════════════════════════════════════════

def load_nq_front_month() -> pd.DataFrame:
    """
    Build a continuous NASDAQ-100 E-mini (NQ) daily series.
    Front-month: on each date, take the contract with the earliest expiry.

    Returns DataFrame indexed by date with columns [price, price_level, returns].
    price_level is the reconstructed continuous price level (cumulative product
    of daily returns, rebased to 1000 at the start of available data).
    """
    sec_meta = pd.read_parquet(DATA / "EquityFuture_security_meta.parquet")
    hist     = pd.read_parquet(DATA / "EquityFuture_historical.parquet")

    nq_tickers = sec_meta[sec_meta["curve_group"] == "NN"]["security"].tolist()
    nq = hist[hist["security"].isin(nq_tickers)].copy()
    nq["date"] = pd.to_datetime(nq["date"])

    meta_nq = sec_meta[sec_meta["curve_group"] == "NN"][
        ["security", "expiry_yearmonth"]].copy()
    meta_nq["expiry_date"] = pd.to_datetime(meta_nq["expiry_yearmonth"], format="%Y-%m")
    nq = nq.merge(meta_nq[["security", "expiry_date"]], on="security")

    # Front-month selection: earliest expiry on each date
    nq = nq.sort_values(["date", "expiry_date"])
    front = nq.groupby("date").first().reset_index()[
        ["date", "price", "returns"]].dropna(subset=["returns"])
    front = front.sort_values("date").set_index("date")

    # Reconstruct continuous price level from returns (rebased to 1000)
    ret = front["returns"].dropna()
    price_level = (1 + ret).cumprod() * 1000
    price_level.name = "price_level"

    front = front.join(price_level, how="left")
    return front[["price", "price_level", "returns"]].dropna()


# ═══════════════════════════════════════════════════════════════════════════
# PANEL CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════

def build_nq_panel(vrp: pd.DataFrame,
                   nq: pd.DataFrame,
                   term_slope: pd.Series,
                   vvix_ma5: pd.Series) -> pd.DataFrame:
    """
    Assemble daily panel with all signals and NQ forward 20-day return target.

    Forward return: (NQ_{t+20} - NQ_t) / NQ_t, computed as the 20-day forward
    rolling product of (1 + daily_NQ_return).
    """
    ret = nq["returns"]
    fwd_20 = (ret + 1).rolling(20).apply(np.prod, raw=True).shift(-20) - 1
    fwd_20.name = "fwd_ret_20"

    panel = pd.DataFrame({
        "VP":          vrp["VP"],
        "CV":          vrp["CV"],
        "IVar":        vrp["IVar"],
        "term_slope":  term_slope,
        "vvix_ma5":    vvix_ma5,
        "fwd_ret_20":  fwd_20,
        "daily_ret":   ret.rename("daily_ret"),
        "price_level": nq["price_level"],
    }).dropna(subset=["VP", "term_slope", "vvix_ma5"])

    # Restrict to full-signal overlap period (VVIX available from 2006-03-06)
    panel = panel[panel.index >= "2006-03-06"]
    return panel.sort_index()


# ═══════════════════════════════════════════════════════════════════════════
# EXPANDING-WINDOW POSITION GENERATION
# ═══════════════════════════════════════════════════════════════════════════

def run_expanding_window(panel, model, delta, t_threshold=T_THRESH):
    """
    Expanding window OOS positions for a given model and delta.

    At each prediction row i (>= OOS_START):
      - Training: panel.iloc[0 : i - 20]  (OOS gap of 20 rows)
      - Fit OLS; compute NW t-stats with 20 lags
      - If all |t| > t_threshold: predict ŷ; gate position on delta
      - Else: flat (0)

    Positions cached to CACHE_DIR.
    """
    feat_cols = MODEL_FEATURES[model]
    tag = (f"pos_EW_NQ_{model}_d{int(delta*10000)}bps"
           f"_t{int(t_threshold*100)}_oos{OOS_START}.parquet")
    cache_path = CACHE_DIR / tag
    if cache_path.exists():
        s = pd.read_parquet(cache_path).squeeze()
        s.name = f"pos_EW_NQ_{model}_d{delta}"
        return s

    sub = panel.dropna(subset=feat_cols + ["fwd_ret_20"]).copy()
    N   = len(sub)
    pos = pd.Series(0.0, index=sub.index, name=f"pos_EW_NQ_{model}_d{delta}")

    oos_idx = sub.index.searchsorted(pd.Timestamp(OOS_START))
    start_i = max(MIN_WIN + 20, oos_idx)

    for i in range(start_i, N):
        train  = sub.iloc[0 : i - 20]
        X_tr   = add_constant(train[feat_cols], has_constant="skip")
        res    = OLS(train["fwd_ret_20"], X_tr).fit()
        nw     = _nw_se(res, nlags=NW_LAGS)
        t_vals = res.params.values[1:] / nw[1:]
        if not np.all(np.abs(t_vals) > t_threshold):
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
    Same as run_expanding_window but threshold is shifted by the rolling mean
    of realised 20-day NQ returns over the last `rolling_window` training rows:
        long  when ŷ > μ + δ
        short when ŷ < μ − δ
    """
    feat_cols = MODEL_FEATURES[model]
    tag = (f"pos_EW_NQ_rolmu_{model}_d{int(delta*10000)}bps"
           f"_t{int(t_threshold*100)}_rw{rolling_window}_oos{OOS_START}.parquet")
    cache_path = CACHE_DIR / tag
    if cache_path.exists():
        s = pd.read_parquet(cache_path).squeeze()
        s.name = f"pos_EWrm_NQ_{model}_d{delta}"
        return s

    sub = panel.dropna(subset=feat_cols + ["fwd_ret_20"]).copy()
    N   = len(sub)
    pos = pd.Series(0.0, index=sub.index, name=f"pos_EWrm_NQ_{model}_d{delta}")
    fwd = sub["fwd_ret_20"].values

    oos_idx = sub.index.searchsorted(pd.Timestamp(OOS_START))
    start_i = max(MIN_WIN + 20, oos_idx)

    for i in range(start_i, N):
        train  = sub.iloc[0 : i - 20]
        X_tr   = add_constant(train[feat_cols], has_constant="skip")
        res    = OLS(train["fwd_ret_20"], X_tr).fit()
        nw     = _nw_se(res, nlags=NW_LAGS)
        t_vals = res.params.values[1:] / nw[1:]
        if not np.all(np.abs(t_vals) > t_threshold):
            continue

        lo = max(0, i - 20 - rolling_window)
        mu = float(np.mean(fwd[lo : i - 20]))

        test_row = sub.iloc[[i]][feat_cols].copy()
        test_row.insert(0, "const", 1.0)
        y_hat = float(res.predict(test_row).iloc[0])
        if   y_hat > mu + delta: pos.iloc[i] =  1.0
        elif y_hat < mu - delta: pos.iloc[i] = -1.0

    pos.to_frame().to_parquet(cache_path)
    return pos


def compute_ew_betas(panel, model):
    """
    Record how all coefficients and NW t-stats evolve as the expanding window
    grows. Cached per model — betas do not depend on delta or threshold variant.

    Returns DataFrame indexed by daily prediction dates (>= OOS_START) with
    columns [beta_VP, se_VP, t_VP] for Base, plus [beta_sec, se_sec, t_sec]
    for bivariate models, and r2_insample for all.
    If cached parquet lacks r2_insample, recompute.
    """
    feat_cols = MODEL_FEATURES[model]
    bivariate = len(feat_cols) >= 2

    cache_path = CACHE_DIR / f"betas_EW_NQ_{model}_oos{OOS_START}.parquet"
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        if "r2_insample" in df.columns:
            return df
        cache_path.unlink()   # recompute — missing r2_insample column

    print(f"    Computing beta time series for NQ {model} (one-time)...")
    sub     = panel.dropna(subset=feat_cols + ["fwd_ret_20"]).copy()
    N       = len(sub)
    oos_idx = sub.index.searchsorted(pd.Timestamp(OOS_START))
    start_i = max(MIN_WIN + 20, oos_idx)

    records = []
    for i in range(start_i, N):
        train  = sub.iloc[0 : i - 20]
        X_tr   = add_constant(train[feat_cols], has_constant="skip")
        res    = OLS(train["fwd_ret_20"], X_tr).fit()
        nw     = _nw_se(res, nlags=NW_LAGS)
        b_alpha, se_alpha = float(res.params.iloc[0]), float(nw[0])
        b_vp,    se_vp    = float(res.params.iloc[1]), float(nw[1])
        rec = {
            "beta_alpha":  b_alpha,
            "se_alpha":    se_alpha,
            "t_alpha":     b_alpha / se_alpha if se_alpha > 0 else 0.0,
            "beta_VP":     b_vp,
            "se_VP":       se_vp,
            "t_VP":        b_vp / se_vp if se_vp > 0 else 0.0,
            "r2_insample": float(res.rsquared),
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


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

FEAT_DISPLAY = {
    "VP":         "VRP",
    "term_slope": "Term Slope",
    "vvix_ma5":   "VVIX MA5",
}


def _shade(ax, s_dt, e_dt):
    for a, b in [("2008-09-01", "2009-06-01"),
                 ("2020-02-01", "2020-06-01"),
                 ("2022-01-01", "2022-12-31")]:
        a, b = pd.Timestamp(a), pd.Timestamp(b)
        if b > s_dt and a < e_dt:
            ax.axvspan(max(a, s_dt), min(b, e_dt), alpha=0.08, color="grey", lw=0)


def oos_cumret(sim_df, start=OOS_START):
    """Cumulative net return rebased to 1.0 at `start`."""
    net = sim_df["net_pnl"]
    s   = net[net.index >= start]
    return (1 + s).cumprod()


def perf_lbl(st):
    return (f"SR={st['sharpe']:+.2f}  "
            f"ret={st['ann_ret']*100:+.1f}%  "
            f"DD={st['max_dd']*100:.1f}%")


# ═══════════════════════════════════════════════════════════════════════════
# PLOT — Expanding-Window Detail (6 panels, per plot.md spec)
# ═══════════════════════════════════════════════════════════════════════════

def plot_expanding_detail(model, out_path, ew_dict, ew_sim_dict,
                          bah_sim, s_dt, e_dt, extra_title=""):
    """
    6-panel plot per experiment2/plot.md:
      Panel 1  — cumulative net return (log scale)
      Panel 2  — NW t-stat (left) + beta (right twin) over time
      Panels 3-6 — position over time, one per delta
    """
    pal       = MODEL_PALETTE[model]
    feat_cols = MODEL_FEATURES[model]
    bivariate = len(feat_cols) >= 2
    pred_labels = [FEAT_DISPLAY.get(f, f) for f in feat_cols]
    pred_str    = " + ".join(pred_labels)

    n_d = len(DELTAS)
    fig, axes = plt.subplots(
        2 + n_d, 1,
        figsize=(14, 11 + 2.2 * n_d),
        sharex=True,
        gridspec_kw={"height_ratios": [2.5, 1.2] + [1.0] * n_d, "hspace": 0.35},
    )
    fig.subplots_adjust(top=0.955, bottom=0.03, left=0.10, right=0.93)

    # Compute OOS R² (Goyal-Welch) using betas to reconstruct y_hat
    _betas_nq = compute_ew_betas(panel, model)
    _sub_nq   = panel.dropna(subset=feat_cols + ["fwd_ret_20"]).copy()
    _idx_nq   = _betas_nq.index.intersection(_sub_nq.index)
    _oos_r2_nq = float("nan")
    if len(_idx_nq) > 0:
        _y_act_nq  = _sub_nq.loc[_idx_nq, "fwd_ret_20"]
        _y_all_nq  = _sub_nq["fwd_ret_20"]
        _prev_m_nq = _y_all_nq.expanding().mean().shift(21).loc[_idx_nq]
        _alpha_nq  = _betas_nq.loc[_idx_nq, "beta_alpha"]
        _x_prim_nq = _sub_nq.loc[_idx_nq, feat_cols[0]]
        _b_prim_nq = _betas_nq.loc[_idx_nq, "beta_VP"]
        _y_hat_nq  = _alpha_nq + _b_prim_nq * _x_prim_nq
        if bivariate and "beta_sec" in _betas_nq.columns:
            _x_sec_nq  = _sub_nq.loc[_idx_nq, feat_cols[1]]
            _b_sec_nq  = _betas_nq.loc[_idx_nq, "beta_sec"]
            _y_hat_nq  = _y_hat_nq + _b_sec_nq * _x_sec_nq
        _ss_res_nq = float(((_y_act_nq - _y_hat_nq) ** 2).sum())
        _ss_tot_nq = float(((_y_act_nq - _prev_m_nq) ** 2).sum())
        if _ss_tot_nq > 0:
            _oos_r2_nq = 1.0 - _ss_res_nq / _ss_tot_nq
    _oos_r2_nq_str = f"\nOOS R² = {_oos_r2_nq:.4f}" if not np.isnan(_oos_r2_nq) else ""

    fig.suptitle(
        f"{pred_str} -> NQ 20-day Forward Return  "
        f"(Expanding Window, OOS from {OOS_START}){extra_title}{_oos_r2_nq_str}\n"
        f"Training grows daily; OOS gap = 20 days; "
        f"NW-HAC {NW_LAGS} lags; |t| > {T_THRESH:.2f} gate; 0.05% slippage",
        fontsize=10, y=0.998,
    )

    ax_ret  = axes[0]
    ax_tb   = axes[1]
    ax_poss = axes[2:]
    oos_dt  = pd.Timestamp(OOS_START)
    xlim    = (oos_dt, e_dt)

    # ── Panel 1: Cumulative Net Return ────────────────────────────────────
    ax_ret.set_xlim(*xlim)
    _shade(ax_ret, oos_dt, e_dt)

    # Find earliest activation across all deltas
    _activation = None
    for di in range(n_d):
        _p = ew_dict[(model, di)]
        _active = _p[(_p.index >= OOS_START) & (_p != 0)]
        if len(_active):
            _cand = _active.index.min()
            if _activation is None or _cand < _activation:
                _activation = _cand

    # Rebase start: use activation+1 if activation is post-2020
    _rebase_start = None
    if _activation is not None and _activation > pd.Timestamp("2020-01-01"):
        _bah_idx  = bah_sim.index
        _act_iloc = _bah_idx.searchsorted(_activation)
        _rebase_start = _bah_idx[min(_act_iloc + 1, len(_bah_idx) - 1)]

    _stat_start = _rebase_start if _rebase_start is not None else pd.Timestamp(OOS_START)

    # Main B&H line
    _bah_st_plot = compute_performance_stats(
        bah_sim[bah_sim.index >= _stat_start], "BaH_plot")
    bah_oos = oos_cumret(bah_sim)
    _stat_lbl = (f" · stats from {_stat_start.strftime('%Y-%m-%d')}"
                 if _rebase_start is not None else "")
    ax_ret.plot(bah_oos.index, bah_oos.values,
                color=BAH_COLOR, lw=1.5, ls="-.", alpha=0.6,
                label=(f"Buy-and-Hold (NQ){_stat_lbl}  "
                       f"[SR={_bah_st_plot['sharpe']:+.2f}  "
                       f"ret={_bah_st_plot['ann_ret']*100:+.1f}%  "
                       f"DD={_bah_st_plot['max_dd']*100:.1f}%]"))

    # Second B&H line rebased to activation+1 if post-2020
    if _rebase_start is not None:
        _bah_from   = bah_sim["net_pnl"][bah_sim.index >= _rebase_start]
        _bah_act    = (1 + _bah_from).cumprod()
        _bah_act_st = compute_performance_stats(
            bah_sim[bah_sim.index >= _rebase_start], "BaH_act")
        ax_ret.plot(_bah_act.index, _bah_act.values,
                    color=BAH_COLOR, lw=1.2, ls=":", alpha=0.85,
                    label=(f"Buy-and-Hold from {_rebase_start.strftime('%Y-%m-%d')}  "
                           f"[SR={_bah_act_st['sharpe']:+.2f}  "
                           f"ret={_bah_act_st['ann_ret']*100:+.1f}%  "
                           f"DD={_bah_act_st['max_dd']*100:.1f}%]"))

    for di in range(n_d):
        _, sim_df = ew_sim_dict[(model, di)]
        st_plot   = compute_performance_stats(
            sim_df[sim_df.index >= _stat_start],
            f"EW_NQ_{model}_{DELTA_LBL[di]}_plot")
        cum     = oos_cumret(sim_df)
        pos_oos = ew_dict[(model, di)]
        pos_oos = pos_oos[pos_oos.index >= _stat_start]
        pL = float((pos_oos ==  1).mean() * 100)
        pS = float((pos_oos == -1).mean() * 100)
        ax_ret.plot(cum.index, cum.values,
                    color=pal[di], lw=1.8, alpha=0.9,
                    label=(f"{DELTA_LBL[di]}  "
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

    # ── Panel 2: NW t-stat and Beta Over Time ────────────────────────────
    betas = compute_ew_betas(panel, model)
    ax_tb.set_xlim(*xlim)
    _shade(ax_tb, oos_dt, e_dt)

    t_prim = betas["t_VP"]
    b_prim = betas["beta_VP"]
    ax_tb.plot(t_prim.index, t_prim.values,
               color=pal[0], lw=1.0, alpha=0.85,
               label=f"NW t-stat ({pred_labels[0]})")
    if bivariate:
        ax_tb.plot(betas["t_sec"].index, betas["t_sec"].values,
                   color=pal[1], lw=1.0, alpha=0.85, ls="--",
                   label=f"NW t-stat ({pred_labels[1]})")

    ax_tb.fill_between(t_prim.index, -T_THRESH, T_THRESH,
                       color="firebrick", alpha=0.05, label="Below gate (flat zone)")
    ax_tb.axhline( T_THRESH, color="firebrick", lw=1.2, ls="--",
                  label=f"|t| = {T_THRESH:.2f} gate")
    ax_tb.axhline(-T_THRESH, color="firebrick", lw=1.2, ls="--")
    ax_tb.axhline(0, color="black", lw=0.5, ls=":")
    ax_tb.set_ylabel(f"NW t-stat ({pred_str})", fontsize=9)
    ax_tb.grid(axis="y", alpha=0.2, lw=0.6)
    ax_tb.spines["top"].set_visible(False)

    ax_tb2 = ax_tb.twinx()
    ax_tb2.plot(b_prim.index, b_prim.values,
                color="dimgrey", lw=1.0, ls="--", alpha=0.60,
                label=f"Beta ({pred_labels[0]})")
    if bivariate:
        ax_tb2.plot(betas["beta_sec"].index, betas["beta_sec"].values,
                    color="dimgrey", lw=1.0, ls=":", alpha=0.50,
                    label=f"Beta ({pred_labels[1]})")
    ax_tb2.axhline(0, color="dimgrey", lw=0.4, ls=":")
    ax_tb2.set_ylabel(f"Beta ({pred_str})", fontsize=8, color="dimgrey")
    ax_tb2.tick_params(axis="y", labelcolor="dimgrey", labelsize=7)
    ax_tb2.spines["top"].set_visible(False)

    # In-sample R² on the right axis (shared with beta axis = ax_tb2)
    if "r2_insample" in betas.columns:
        r2_s = betas["r2_insample"]
        ax_tb2.plot(r2_s.index, r2_s.values,
                    color="forestgreen", lw=0.9, ls=":", alpha=0.75,
                    label="In-sample R²")

    lines1, labs1 = ax_tb.get_legend_handles_labels()
    lines2, labs2 = ax_tb2.get_legend_handles_labels()
    ax_tb.legend(lines1 + lines2, labs1 + labs2, fontsize=8, loc="upper left")

    # ── Panels 3–6: Position Over Time ───────────────────────────────────
    for di, ax_p in enumerate(ax_poss):
        pos_full = ew_dict[(model, di)]
        pos = pos_full[pos_full.index >= OOS_START]
        _shade(ax_p, oos_dt, e_dt)
        ax_p.fill_between(pos.index, pos.where(pos ==  1, 0), 0,
                          color=pal[di], alpha=0.75)
        ax_p.fill_between(pos.index, pos.where(pos == -1, 0), 0,
                          color=pal[di], alpha=0.30, hatch="///")
        ax_p.axhline(0, color="black", lw=0.4)
        ax_p.set_ylim(-1.5, 1.5)
        ax_p.set_yticks([-1, 0, 1])
        ax_p.set_yticklabels(["Short", "Flat", "Long"], fontsize=8)
        ax_p.set_ylabel(DELTA_LBL[di], fontsize=9, rotation=0,
                        ha="right", va="center", labelpad=56, color=pal[di])
        pL = float((pos ==  1).mean() * 100)
        pS = float((pos == -1).mean() * 100)
        pF = float((pos ==  0).mean() * 100)
        ax_p.text(0.01, 0.97,
                  f"Long {pL:.1f}%  Short {pS:.1f}%  Flat {pF:.1f}%  "
                  f"AvgPos={float(pos.mean()):+.3f}",
                  transform=ax_p.transAxes, fontsize=7.5, va="top",
                  color=pal[di])
        ax_p.spines[["top", "right"]].set_visible(False)

    for ax in [ax_ret, ax_tb] + list(ax_poss):
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.tick_params(axis="x", which="major", labelbottom=True, labelsize=7, pad=2)

    fig.savefig(out_path, dpi=155, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ═══════════════════════════════════════════════════════════════════════════
# PLOT — Post-2020 Cross-Model Comparison
# ═══════════════════════════════════════════════════════════════════════════

def plot_post2020_comparison(out_path, ew_dict, ew_sim_dict, bah_sim, e_dt,
                              models=None):
    """
    Single-panel plot: all models × best delta, rebased from first secondary-
    signal activation.
    """
    if models is None:
        models = MODELS

    _MODEL_COLOR = {
        "Base":       "#08306b",
        "Model_A":    "#006d2c",
        "Model_C":    "#3f007d",
        "Model_VVIX": "#7f2704",
    }

    def best_delta(sim_dict, m):
        return max(range(4), key=lambda di: sim_dict[(m, di)][0]["sharpe"])

    # Find reference start: first date Model_A or Model_C takes a position
    _activation_models = [m for m in ["Model_A", "Model_C"] if (m, 0) in ew_sim_dict]
    first_dates = []
    for m in _activation_models:
        di  = best_delta(ew_sim_dict, m)
        pos = ew_dict[(m, di)]
        active = pos[pos != 0]
        if len(active):
            first_dates.append(active.index.min())
    START = min(first_dates) if first_dates else pd.Timestamp("2020-01-01")
    start = START.strftime("%Y-%m-%d")

    def _rebase(sim_df):
        net = sim_df["net_pnl"]
        s   = net[net.index >= start]
        return (1 + s).cumprod()

    def _post_stats(sim_df, label):
        s = sim_df[sim_df.index >= start].copy()
        s["cum_net"] = (1 + s["net_pnl"]).cumprod()
        return compute_performance_stats(s, label)

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.suptitle(
        f"NQ — Performance from First Secondary-Signal Activation ({start})  ·  "
        f"Best δ  ·  Expanding Window\n"
        f"Rebased to 1.0 at {start}  ·  |t| > {T_THRESH:.2f} gate  ·  "
        f"0.05% slippage",
        fontsize=10,
    )

    bah_post = _post_stats(bah_sim, "BaH")
    bah_cum  = _rebase(bah_sim)
    ax.plot(bah_cum.index, bah_cum.values,
            color=BAH_COLOR, lw=1.8, ls="-.", alpha=0.65,
            label=(f"Buy-and-Hold (NQ)  "
                   f"[SR={bah_post['sharpe']:+.2f}  "
                   f"ret={bah_post['ann_ret']*100:+.1f}%  "
                   f"DD={bah_post['max_dd']*100:.1f}%]"))

    def _pos_lbls(pos_series):
        p = pos_series[pos_series.index >= start]
        return int((p == 1).mean() * 100), int((p == -1).mean() * 100)

    _SHORT = {
        "Base":       "Univariate VRP",
        "Model_A":    "VRP + Term Slope",
        "Model_C":    "VRP + VVIX MA5",
        "Model_VVIX": "Univariate VVIX MA5",
    }
    for m in models:
        col = _MODEL_COLOR.get(m, "#333333")
        short_lbl = _SHORT.get(m, m)
        di_best = best_delta(ew_sim_dict, m)
        _, sim_ew = ew_sim_dict[(m, di_best)]
        st_post = _post_stats(sim_ew, f"EW_NQ_{m}")
        cum_ew  = _rebase(sim_ew)
        pos_ew  = ew_dict[(m, di_best)]
        pL, pS  = _pos_lbls(pos_ew)
        ax.plot(cum_ew.index, cum_ew.values,
                color=col, lw=2.2, ls="-", alpha=0.92,
                label=(f"{short_lbl} ({DELTA_LBL[di_best]})  "
                       f"[SR={st_post['sharpe']:+.2f}  "
                       f"ret={st_post['ann_ret']*100:+.1f}%  "
                       f"DD={st_post['max_dd']*100:.1f}%  "
                       f"L{pL}%/S{pS}%]"))

    for a, b in [("2020-02-01", "2020-06-01"), ("2022-01-01", "2022-12-31")]:
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


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

print("=" * 72)
print("  Experiment 2 — NASDAQ Edition: Expanding-Window OOS Evaluation")
print("  Dependent variable: NQ (NASDAQ-100 E-mini) 20-day forward return")
print("=" * 72)

# ── Load data ──────────────────────────────────────────────────────────────
print("\n[1] Loading data...")
vrp      = load_vrp_series()
nq       = load_nq_front_month()
vx_df    = load_vix_futures_term_structure()
vvix_raw = load_vvix()
slope    = compute_vix_term_slope(vx_df)
vvix_ma5 = compute_vvix_ma5(vvix_raw)

print(f"    VRP:  {vrp.index.min().date()} – {vrp.index.max().date()} ({len(vrp):,} obs)")
print(f"    NQ:   {nq.index.min().date()} – {nq.index.max().date()} ({len(nq):,} obs)")
print(f"    VVIX: {vvix_raw.index.min().date()} – {vvix_raw.index.max().date()}")

# ── Build panel ────────────────────────────────────────────────────────────
print("\n[2] Building NQ signal panel...")
panel = build_nq_panel(vrp, nq, slope, vvix_ma5)
print(f"    Panel: {panel.index.min().date()} – {panel.index.max().date()} "
      f"({len(panel):,} obs)")
print(f"    Signal coverage: VRP positive {(panel['VP'] > 0).mean()*100:.1f}%  "
      f"| contango {(panel['term_slope'] > 0).mean()*100:.1f}%  "
      f"| VVIX MA5 < 95 on {(panel['vvix_ma5'] < 95).mean()*100:.1f}%")

daily_ret = panel["daily_ret"].dropna()
s_dt = daily_ret.index[0]
e_dt = daily_ret.index[-1]
oos_dt = pd.Timestamp(OOS_START)

# ── Buy-and-Hold benchmark (NQ) ────────────────────────────────────────────
bah_pos = compute_buy_and_hold(daily_ret)
bah_sim = simulate_strategy(bah_pos, daily_ret)
bah_st  = compute_performance_stats(bah_sim[bah_sim.index >= OOS_START], "Buy-and-Hold")
print(f"\n    NQ Buy-and-Hold (OOS):  SR={bah_st['sharpe']:+.3f}  "
      f"ret={bah_st['ann_ret']*100:+.1f}%  MaxDD={bah_st['max_dd']*100:.1f}%")

# ═══════════════════════════════════════════════════════════════════════════
# 1. SYMMETRIC (FIXED) THRESHOLD — EXPANDING WINDOW
# ═══════════════════════════════════════════════════════════════════════════

print("\n[3] Computing expanding-window positions (symmetric threshold)...")
EW     = {}
EW_SIM = {}

for m in MODELS:
    for di, delta in enumerate(DELTAS):
        print(f"  EW  {m}  {DELTA_LBL[di]}...", end="  ", flush=True)
        pos = run_expanding_window(panel, m, delta)
        sim = simulate_strategy(pos, daily_ret)
        sim_oos = sim[sim.index >= OOS_START]
        st  = compute_performance_stats(sim_oos, f"EW_NQ_{m}_{DELTA_LBL[di]}")
        oos_pos = pos[pos.index >= OOS_START]
        st["avg_position"] = float(oos_pos.mean())
        st["pct_long"]  = float((oos_pos ==  1).mean()) * 100
        st["pct_short"] = float((oos_pos == -1).mean()) * 100
        st["pct_flat"]  = float((oos_pos ==  0).mean()) * 100
        EW[(m, di)]     = pos
        EW_SIM[(m, di)] = (st, sim)
        print(f"SR={st['sharpe']:+.3f}  "
              f"L={st['pct_long']:.0f}%  S={st['pct_short']:.0f}%  F={st['pct_flat']:.0f}%")

# ── Symmetric detail plots ─────────────────────────────────────────────────
print("\n[4] Generating symmetric threshold plots...")
for m in MODELS:
    fname = ("symmetric_" +
             EW_MODEL_DIR[m].name.replace(" ", "_").replace("+", "+") + ".png")
    plot_expanding_detail(
        m,
        EW_MODEL_DIR[m] / fname,
        ew_dict=EW, ew_sim_dict=EW_SIM,
        bah_sim=bah_sim, s_dt=s_dt, e_dt=e_dt,
    )

print("\n[4b] Generating symmetric cross-model comparison plot...")
plot_post2020_comparison(
    DIR_EW_CMP / "symmetric_comparisons.png",
    ew_dict=EW, ew_sim_dict=EW_SIM,
    bah_sim=bah_sim, e_dt=e_dt,
)

# ═══════════════════════════════════════════════════════════════════════════
# 2. ROLLING-MU THRESHOLD — EXPANDING WINDOW
# ═══════════════════════════════════════════════════════════════════════════

print("\n[5] Computing expanding-window positions (rolling-mu threshold)...")
EW_RM     = {}
EW_RM_SIM = {}

for m in MODELS:
    for di, delta in enumerate(DELTAS):
        print(f"  EW-rolmu  {m}  {DELTA_LBL[di]}...", end="  ", flush=True)
        pos = run_expanding_window_rolmu(panel, m, delta)
        sim = simulate_strategy(pos, daily_ret)
        sim_oos = sim[sim.index >= OOS_START]
        st  = compute_performance_stats(sim_oos, f"EWrm_NQ_{m}_{DELTA_LBL[di]}")
        oos_pos = pos[pos.index >= OOS_START]
        st["avg_position"] = float(oos_pos.mean())
        st["pct_long"]  = float((oos_pos ==  1).mean()) * 100
        st["pct_short"] = float((oos_pos == -1).mean()) * 100
        st["pct_flat"]  = float((oos_pos ==  0).mean()) * 100
        EW_RM[(m, di)]     = pos
        EW_RM_SIM[(m, di)] = (st, sim)
        print(f"SR={st['sharpe']:+.3f}  "
              f"L={st['pct_long']:.0f}%  S={st['pct_short']:.0f}%  F={st['pct_flat']:.0f}%")

# ── Rolling-mu detail plots ────────────────────────────────────────────────
print("\n[6] Generating rolling-mu threshold plots...")
for m in MODELS:
    fname = ("base_return_shift_" +
             EW_MODEL_DIR[m].name.replace(" ", "_").replace("+", "+") + ".png")
    plot_expanding_detail(
        m,
        EW_MODEL_DIR[m] / fname,
        ew_dict=EW_RM, ew_sim_dict=EW_RM_SIM,
        bah_sim=bah_sim, s_dt=s_dt, e_dt=e_dt,
        extra_title=" · Threshold = rolling-avg(20d NQ return) ± δ",
    )

print("\n[6b] Generating rolling-mu cross-model comparison plot...")
plot_post2020_comparison(
    DIR_EW_CMP / "base_return_shift_comparisons.png",
    ew_dict=EW_RM, ew_sim_dict=EW_RM_SIM,
    bah_sim=bah_sim, e_dt=e_dt,
)

# ═══════════════════════════════════════════════════════════════════════════
# PERFORMANCE SUMMARY
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("  PERFORMANCE SUMMARY — Expanding Window (OOS: 2012-01-01 onwards)")
print("  NQ (NASDAQ-100 E-mini) front-month 20-day forward return")
print("=" * 72)

def best_delta_idx(sim_dict, m):
    return max(range(4), key=lambda di: sim_dict[(m, di)][0]["sharpe"])

print(f"\n{'Model':<20} {'Method':<20} {'Best d':<10} "
      f"{'SR':>6} {'Ret%':>8} {'DD%':>8} {'L%':>6} {'S%':>6} {'F%':>6}")
print("-" * 90)

bah_oos = bah_st
print(f"  {'Buy-and-Hold (NQ)':<18} {'---':<20} {'---':<10} "
      f"{bah_oos['sharpe']:>6.3f} "
      f"{bah_oos['ann_ret']*100:>7.2f}% "
      f"{bah_oos['max_dd']*100:>7.1f}%  100%    0%    0%")

MODEL_SHORT = {
    "Base":       "Univariate VRP",
    "Model_A":    "VRP + Term Slope",
    "Model_C":    "VRP + VVIX MA5",
    "Model_VVIX": "Univariate VVIX MA5",
}

for m in MODELS:
    for method_label, sim_dict in [("Symmetric", EW_SIM), ("Rolling-mu", EW_RM_SIM)]:
        di = best_delta_idx(sim_dict, m)
        st, _ = sim_dict[(m, di)]
        short_lbl = MODEL_SHORT[m]
        print(f"  {short_lbl:<18} {method_label:<20} {DELTA_LBL[di]:<10} "
              f"{st['sharpe']:>6.3f} "
              f"{st['ann_ret']*100:>7.2f}% "
              f"{st['max_dd']*100:>7.1f}% "
              f"{st['pct_long']:>5.0f}% "
              f"{st['pct_short']:>5.0f}% "
              f"{st['pct_flat']:>5.0f}%")

print("\nAll outputs saved to:", OUTPUT / "expanding_window")
print("=" * 72)
