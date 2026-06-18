"""
regressions.py
==============
All expanding-window OLS regression computation for every model.

Models covered (ignoring poor-correlation baselines and intraday VRP):
  Univariate:  VRP · VVIX MA5 · VVIX MA10
  Bivariate:   VRP + VVIX MA5 · VRP + VVIX MA10 ·
               VRP + Term Slope · VRP + Open Interest
  Trivariate:  VRP + VVIX MA5 + Term Slope

Strategy types cached per model:
  Unit-position:   sym (×4 deltas) · asym · rolmu
  Unbound (±1..4): sym · asym · rolmu   (univariate + bivariate + trivariate)
  Betas (OOS):     univariate · bivariate · trivariate
  OOS R²:          univariate · bivariate

Running main() pre-populates output/regression_cache/ so subsequent calls
to base_strategies.py and leveraged_strategies.py hit cache every time.
"""

import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path
import numpy as np
import pandas as pd

from statsmodels.api import OLS, add_constant

ROOT      = Path(__file__).parent
OUTPUT    = ROOT / "output"
CACHE_DIR = OUTPUT / "regression_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT.parent / "bh_replication"))
from har_model import _nw_se

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))
from experiment2 import (
    load_vrp_series, load_es_front_month, load_vvix,
    compute_vvix_ma5, compute_vvix_ma10,
    load_vix_spot, load_vix_futures_term_structure,
    load_es_open_interest,
)
from fh_replication.fh_replication import compute_vix_term_slope

# ─── Shared constants ─────────────────────────────────────────────────────────
OOS_START  = "2012-01-01"
MIN_WIN    = 500
T_THRESH   = 1.28
DELTAS     = [0.002, 0.005, 0.0075, 0.010]
DELTA_LBL  = ["d=0.2%", "d=0.5%", "d=0.75%", "d=1.0%"]
OOS_GAP    = 20
NW_LAGS    = 20
VVIX_ACT   = pd.Timestamp("2006-03-06")
RW         = 500          # rolling window for mu (base-return-shift / asymmetric)
THRESHOLDS = [0.010, 0.0075, 0.005, 0.002]   # unbound thresholds (descending)
LEVELS     = [4, 3, 2, 1]                     # matching position multipliers


# ─── Shared plot helpers (imported by base_strategies.py + leveraged_strategies.py) ─

def _shade(ax, start, end):
    for a, b in [("2020-02-01", "2020-06-01"), ("2022-01-01", "2022-12-31")]:
        a, b = pd.Timestamp(a), pd.Timestamp(b)
        if b > start and a < end:
            ax.axvspan(max(a, start), min(b, end), alpha=0.08, color="grey", lw=0)


def oos_cumret(sim, start=OOS_START):
    net = sim["net_pnl"]
    s   = net[net.index >= start]
    return (1 + s).cumprod()


# ─── Data ─────────────────────────────────────────────────────────────────────

def build_panel(vrp, es, vvix_ma5, vix_spot, term_slope, oi):
    ret   = es["returns"]
    panel = pd.DataFrame({
        "VP":            vrp["VP"],
        "vvix_ma5":      vvix_ma5,
        "vix":           vix_spot if vix_spot is not None else np.nan,
        "term_slope":    term_slope,
        "open_interest": oi,
        "daily_ret":     ret,
    })
    panel["fwd_20d"] = (ret + 1).rolling(20).apply(np.prod, raw=True).shift(-20) - 1
    r2   = ret ** 2
    rv5  = np.sqrt(r2.rolling(5).mean()  * 252)
    rv22 = np.sqrt(r2.rolling(22).mean() * 252)
    panel["vol_trend"] = np.log(rv5 / rv22)
    panel = panel.dropna(subset=["VP", "vvix_ma5", "term_slope"])
    return panel[panel.index >= "2006-03-06"]


# ─── OOS R² (Campbell-Thompson 2008) ──────────────────────────────────────────

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

    print(f"    Computing OOS R² for {pred1}+{pred2} -> {fwd_col}...", flush=True)
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


# ─── Unit-position: symmetric ─────────────────────────────────────────────────

def run_ew(panel, predictor, fwd_col, oos_gap, nw_lags, delta):
    """±1 when |ŷ| > delta and t-stat gate passes."""
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
        if abs(float(res.params.iloc[1]) / float(nw[1])) <= T_THRESH:
            continue
        test  = sub.iloc[[i]][[predictor]].copy(); test.insert(0, "const", 1.0)
        y_hat = float(res.predict(test).iloc[0])
        if   y_hat >  delta: pos.iloc[i] =  1.0
        elif y_hat < -delta: pos.iloc[i] = -1.0

    pos.to_frame().to_parquet(cache)
    return pos


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
        test  = sub.iloc[[i]][[pred1, pred2]].copy(); test.insert(0, "const", 1.0)
        y_hat = float(res.predict(test).iloc[0])
        if   y_hat >  delta: pos.iloc[i] =  1.0
        elif y_hat < -delta: pos.iloc[i] = -1.0

    pos.to_frame().to_parquet(cache)
    return pos


# ─── Unit-position: asymmetric ────────────────────────────────────────────────

def run_ew_asym(panel, predictor, fwd_col, oos_gap, nw_lags, delta=0.0):
    """Long if (ŷ-µ₅₀₀) > delta, short if -ŷ > delta."""
    d_sfx = f"_d{int(round(delta * 10000))}" if delta > 0 else ""
    tag = (f"pos_EWasym_{predictor}_{fwd_col}"
           f"_t{int(T_THRESH*100)}{d_sfx}_oos{OOS_START}.parquet")
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
        if abs(float(res.params.iloc[1]) / float(nw[1])) <= T_THRESH:
            continue
        test  = sub.iloc[[i]][[predictor]].copy(); test.insert(0, "const", 1.0)
        y_hat = float(res.predict(test).iloc[0])
        mu500 = float(sub[fwd_col].iloc[max(0, i - oos_gap - 500) : i - oos_gap].mean())
        if   (y_hat - mu500) > delta: pos.iloc[i] =  1.0
        elif       (-y_hat)  > delta: pos.iloc[i] = -1.0

    pos.to_frame().to_parquet(cache)
    return pos


def run_ew_asym_bivariate(panel, pred1, pred2, fwd_col, oos_gap, nw_lags, delta=0.0):
    """Both betas must pass gate. Long if (ŷ-µ₅₀₀) > delta, short if -ŷ > delta."""
    d_sfx = f"_d{int(round(delta * 10000))}" if delta > 0 else ""
    tag = (f"pos_EWasym_{pred1}_{pred2}_{fwd_col}"
           f"_t{int(T_THRESH*100)}{d_sfx}_oos{OOS_START}.parquet")
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
        test  = sub.iloc[[i]][[pred1, pred2]].copy(); test.insert(0, "const", 1.0)
        y_hat = float(res.predict(test).iloc[0])
        mu500 = float(sub[fwd_col].iloc[max(0, i - oos_gap - 500) : i - oos_gap].mean())
        if   (y_hat - mu500) > delta: pos.iloc[i] =  1.0
        elif       (-y_hat)  > delta: pos.iloc[i] = -1.0

    pos.to_frame().to_parquet(cache)
    return pos


# ─── Unit-position: base-return-shift ────────────────────────────────────────

def run_ew_rolmu(panel, predictor, fwd_col, oos_gap, nw_lags, delta=0.0):
    """Long if (ŷ-µ₅₀₀) > delta, short if (µ₅₀₀-ŷ) > delta."""
    d_sfx = f"_d{int(round(delta * 10000))}" if delta > 0 else ""
    tag = (f"pos_EWrolmu_{predictor}_{fwd_col}"
           f"_t{int(T_THRESH*100)}{d_sfx}_oos{OOS_START}.parquet")
    cache = CACHE_DIR / tag
    if cache.exists():
        return pd.read_parquet(cache).squeeze().rename(f"pos_rolmu_{predictor}")

    sub = panel.dropna(subset=[predictor, fwd_col]).copy()
    N   = len(sub)
    pos = pd.Series(0.0, index=sub.index, name=f"pos_rolmu_{predictor}")
    oos_idx = sub.index.searchsorted(pd.Timestamp(OOS_START))
    start_i = max(MIN_WIN + oos_gap, oos_idx)

    for i in range(start_i, N):
        train = sub.iloc[0 : i - oos_gap]
        X_tr  = add_constant(train[[predictor]], has_constant="skip")
        res   = OLS(train[fwd_col], X_tr).fit()
        nw    = _nw_se(res, nlags=nw_lags)
        if abs(float(res.params.iloc[1]) / float(nw[1])) <= T_THRESH:
            continue
        mu500 = float(sub[fwd_col].iloc[max(0, i - oos_gap - 500) : i - oos_gap].mean())
        test  = sub.iloc[[i]][[predictor]].copy(); test.insert(0, "const", 1.0)
        y_hat = float(res.predict(test).iloc[0])
        if   (y_hat - mu500) >  delta: pos.iloc[i] =  1.0
        elif (mu500 - y_hat) >  delta: pos.iloc[i] = -1.0

    pos.to_frame().to_parquet(cache)
    return pos


def run_ew_rolmu_bivariate(panel, pred1, pred2, fwd_col, oos_gap, nw_lags, delta=0.0):
    """Both betas must pass gate. Long if (ŷ-µ₅₀₀) > delta, short if (µ₅₀₀-ŷ) > delta."""
    d_sfx = f"_d{int(round(delta * 10000))}" if delta > 0 else ""
    tag = (f"pos_EWrolmu_{pred1}_{pred2}_{fwd_col}"
           f"_t{int(T_THRESH*100)}{d_sfx}_oos{OOS_START}.parquet")
    cache = CACHE_DIR / tag
    if cache.exists():
        return pd.read_parquet(cache).squeeze().rename(f"pos_rolmu_{pred1}_{pred2}")

    sub = panel.dropna(subset=[pred1, pred2, fwd_col]).copy()
    N   = len(sub)
    pos = pd.Series(0.0, index=sub.index, name=f"pos_rolmu_{pred1}_{pred2}")
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
        mu500 = float(sub[fwd_col].iloc[max(0, i - oos_gap - 500) : i - oos_gap].mean())
        test  = sub.iloc[[i]][[pred1, pred2]].copy(); test.insert(0, "const", 1.0)
        y_hat = float(res.predict(test).iloc[0])
        if   (y_hat - mu500) >  delta: pos.iloc[i] =  1.0
        elif (mu500 - y_hat) >  delta: pos.iloc[i] = -1.0

    pos.to_frame().to_parquet(cache)
    return pos


# ─── Unbound (multi-level) helpers ────────────────────────────────────────────

def _level(excess_abs):
    for thresh, lv in zip(THRESHOLDS, LEVELS):
        if excess_abs >= thresh:
            return lv
    return 0


# ─── Unbound: univariate ──────────────────────────────────────────────────────

def run_ew_unbound_sym(panel, predictor, fwd_col, oos_gap, nw_lags):
    """Level based on |ŷ|."""
    tag = (f"pos_EW_unbound_sym_{predictor}_{fwd_col}"
           f"_t{int(T_THRESH*100)}_oos{OOS_START}.parquet")
    cache = CACHE_DIR / tag
    if cache.exists():
        return pd.read_parquet(cache).squeeze().rename(f"pos_ubsym_{predictor}")

    sub = panel.dropna(subset=[predictor, fwd_col]).copy()
    pos = pd.Series(0.0, index=sub.index, name=f"pos_ubsym_{predictor}")
    oos_idx = sub.index.searchsorted(pd.Timestamp(OOS_START))
    start_i = max(MIN_WIN + oos_gap, oos_idx)

    for i in range(start_i, len(sub)):
        train = sub.iloc[0 : i - oos_gap]
        X_tr  = add_constant(train[[predictor]], has_constant="skip")
        res   = OLS(train[fwd_col], X_tr).fit()
        nw    = _nw_se(res, nlags=nw_lags)
        if abs(float(res.params.iloc[1]) / float(nw[1])) <= T_THRESH:
            continue
        test  = sub.iloc[[i]][[predictor]].copy(); test.insert(0, "const", 1.0)
        y_hat = float(res.predict(test).iloc[0])
        lv    = _level(abs(y_hat))
        if lv > 0:
            pos.iloc[i] = float(lv) if y_hat > 0 else float(-lv)

    pos.to_frame().to_parquet(cache)
    return pos


def run_ew_unbound_asym(panel, predictor, fwd_col, oos_gap, nw_lags,
                        rolling_window=500):
    """Long: level by (ŷ-µ); short: level by |ŷ| when ŷ<0."""
    tag = (f"pos_EW_unbound_asym_{predictor}_{fwd_col}"
           f"_t{int(T_THRESH*100)}_rw{rolling_window}_oos{OOS_START}.parquet")
    cache = CACHE_DIR / tag
    if cache.exists():
        return pd.read_parquet(cache).squeeze().rename(f"pos_ubasym_{predictor}")

    sub = panel.dropna(subset=[predictor, fwd_col]).copy()
    pos = pd.Series(0.0, index=sub.index, name=f"pos_ubasym_{predictor}")
    fwd = sub[fwd_col].values
    oos_idx = sub.index.searchsorted(pd.Timestamp(OOS_START))
    start_i = max(MIN_WIN + oos_gap, oos_idx)

    for i in range(start_i, len(sub)):
        train = sub.iloc[0 : i - oos_gap]
        X_tr  = add_constant(train[[predictor]], has_constant="skip")
        res   = OLS(train[fwd_col], X_tr).fit()
        nw    = _nw_se(res, nlags=nw_lags)
        if abs(float(res.params.iloc[1]) / float(nw[1])) <= T_THRESH:
            continue
        lo    = max(0, i - oos_gap - rolling_window)
        mu    = float(np.mean(fwd[lo : i - oos_gap]))
        test  = sub.iloc[[i]][[predictor]].copy(); test.insert(0, "const", 1.0)
        y_hat = float(res.predict(test).iloc[0])
        lv_long  = _level(y_hat - mu) if (y_hat - mu) > 0 else 0
        lv_short = _level(-y_hat)     if y_hat         < 0 else 0
        if lv_long > 0:
            pos.iloc[i] =  float(lv_long)
        elif lv_short > 0:
            pos.iloc[i] = -float(lv_short)

    pos.to_frame().to_parquet(cache)
    return pos


def run_ew_unbound_rolmu(panel, predictor, fwd_col, oos_gap, nw_lags,
                         rolling_window=500):
    """Level based on |ŷ - µ|."""
    tag = (f"pos_EW_rolmu_unbound_{predictor}_{fwd_col}"
           f"_t{int(T_THRESH*100)}_rw{rolling_window}_oos{OOS_START}.parquet")
    cache = CACHE_DIR / tag
    if cache.exists():
        return pd.read_parquet(cache).squeeze().rename(f"pos_ubrolmu_{predictor}")

    sub = panel.dropna(subset=[predictor, fwd_col]).copy()
    pos = pd.Series(0.0, index=sub.index, name=f"pos_ubrolmu_{predictor}")
    fwd = sub[fwd_col].values
    oos_idx = sub.index.searchsorted(pd.Timestamp(OOS_START))
    start_i = max(MIN_WIN + oos_gap, oos_idx)

    for i in range(start_i, len(sub)):
        train = sub.iloc[0 : i - oos_gap]
        X_tr  = add_constant(train[[predictor]], has_constant="skip")
        res   = OLS(train[fwd_col], X_tr).fit()
        nw    = _nw_se(res, nlags=nw_lags)
        if abs(float(res.params.iloc[1]) / float(nw[1])) <= T_THRESH:
            continue
        lo     = max(0, i - oos_gap - rolling_window)
        mu     = float(np.mean(fwd[lo : i - oos_gap]))
        test   = sub.iloc[[i]][[predictor]].copy(); test.insert(0, "const", 1.0)
        y_hat  = float(res.predict(test).iloc[0])
        excess = y_hat - mu
        lv     = _level(abs(excess))
        if lv > 0:
            pos.iloc[i] = float(lv) if excess > 0 else float(-lv)

    pos.to_frame().to_parquet(cache)
    return pos


# ─── Unbound: bivariate ───────────────────────────────────────────────────────

def run_ew_biv_unbound_sym(panel, pred1, pred2, fwd_col, oos_gap, nw_lags):
    tag = (f"pos_EW_unbound_sym_{pred1}_{pred2}_{fwd_col}"
           f"_t{int(T_THRESH*100)}_oos{OOS_START}.parquet")
    cache = CACHE_DIR / tag
    if cache.exists():
        return pd.read_parquet(cache).squeeze().rename(f"pos_ubsym_{pred1}_{pred2}")

    sub = panel.dropna(subset=[pred1, pred2, fwd_col]).copy()
    pos = pd.Series(0.0, index=sub.index, name=f"pos_ubsym_{pred1}_{pred2}")
    oos_idx = sub.index.searchsorted(pd.Timestamp(OOS_START))
    start_i = max(MIN_WIN + oos_gap, oos_idx)

    for i in range(start_i, len(sub)):
        train = sub.iloc[0 : i - oos_gap]
        X_tr  = add_constant(train[[pred1, pred2]], has_constant="skip")
        res   = OLS(train[fwd_col], X_tr).fit()
        nw    = _nw_se(res, nlags=nw_lags)
        t1    = float(res.params.iloc[1]) / float(nw[1])
        t2    = float(res.params.iloc[2]) / float(nw[2])
        if abs(t1) <= T_THRESH or abs(t2) <= T_THRESH:
            continue
        test  = sub.iloc[[i]][[pred1, pred2]].copy(); test.insert(0, "const", 1.0)
        y_hat = float(res.predict(test).iloc[0])
        lv    = _level(abs(y_hat))
        if lv > 0:
            pos.iloc[i] = float(lv) if y_hat > 0 else float(-lv)

    pos.to_frame().to_parquet(cache)
    return pos


def run_ew_biv_unbound_asym(panel, pred1, pred2, fwd_col, oos_gap, nw_lags,
                             rolling_window=500):
    tag = (f"pos_EW_unbound_asym_{pred1}_{pred2}_{fwd_col}"
           f"_t{int(T_THRESH*100)}_rw{rolling_window}_oos{OOS_START}.parquet")
    cache = CACHE_DIR / tag
    if cache.exists():
        return pd.read_parquet(cache).squeeze().rename(f"pos_ubasym_{pred1}_{pred2}")

    sub = panel.dropna(subset=[pred1, pred2, fwd_col]).copy()
    pos = pd.Series(0.0, index=sub.index, name=f"pos_ubasym_{pred1}_{pred2}")
    fwd = sub[fwd_col].values
    oos_idx = sub.index.searchsorted(pd.Timestamp(OOS_START))
    start_i = max(MIN_WIN + oos_gap, oos_idx)

    for i in range(start_i, len(sub)):
        train = sub.iloc[0 : i - oos_gap]
        X_tr  = add_constant(train[[pred1, pred2]], has_constant="skip")
        res   = OLS(train[fwd_col], X_tr).fit()
        nw    = _nw_se(res, nlags=nw_lags)
        t1    = float(res.params.iloc[1]) / float(nw[1])
        t2    = float(res.params.iloc[2]) / float(nw[2])
        if abs(t1) <= T_THRESH or abs(t2) <= T_THRESH:
            continue
        lo    = max(0, i - oos_gap - rolling_window)
        mu    = float(np.mean(fwd[lo : i - oos_gap]))
        test  = sub.iloc[[i]][[pred1, pred2]].copy(); test.insert(0, "const", 1.0)
        y_hat = float(res.predict(test).iloc[0])
        lv_long  = _level(y_hat - mu) if (y_hat - mu) > 0 else 0
        lv_short = _level(-y_hat)     if y_hat         < 0 else 0
        if lv_long > 0:
            pos.iloc[i] =  float(lv_long)
        elif lv_short > 0:
            pos.iloc[i] = -float(lv_short)

    pos.to_frame().to_parquet(cache)
    return pos


def run_ew_biv_unbound_rolmu(panel, pred1, pred2, fwd_col, oos_gap, nw_lags,
                              rolling_window=500):
    tag = (f"pos_EW_rolmu_unbound_{pred1}_{pred2}_{fwd_col}"
           f"_t{int(T_THRESH*100)}_rw{rolling_window}_oos{OOS_START}.parquet")
    cache = CACHE_DIR / tag
    if cache.exists():
        return pd.read_parquet(cache).squeeze().rename(f"pos_ubrolmu_{pred1}_{pred2}")

    sub = panel.dropna(subset=[pred1, pred2, fwd_col]).copy()
    pos = pd.Series(0.0, index=sub.index, name=f"pos_ubrolmu_{pred1}_{pred2}")
    fwd = sub[fwd_col].values
    oos_idx = sub.index.searchsorted(pd.Timestamp(OOS_START))
    start_i = max(MIN_WIN + oos_gap, oos_idx)

    for i in range(start_i, len(sub)):
        train = sub.iloc[0 : i - oos_gap]
        X_tr  = add_constant(train[[pred1, pred2]], has_constant="skip")
        res   = OLS(train[fwd_col], X_tr).fit()
        nw    = _nw_se(res, nlags=nw_lags)
        t1    = float(res.params.iloc[1]) / float(nw[1])
        t2    = float(res.params.iloc[2]) / float(nw[2])
        if abs(t1) <= T_THRESH or abs(t2) <= T_THRESH:
            continue
        lo     = max(0, i - oos_gap - rolling_window)
        mu     = float(np.mean(fwd[lo : i - oos_gap]))
        test   = sub.iloc[[i]][[pred1, pred2]].copy(); test.insert(0, "const", 1.0)
        y_hat  = float(res.predict(test).iloc[0])
        excess = y_hat - mu
        lv     = _level(abs(excess))
        if lv > 0:
            pos.iloc[i] = float(lv) if excess > 0 else float(-lv)

    pos.to_frame().to_parquet(cache)
    return pos


# ─── Unbound: trivariate ──────────────────────────────────────────────────────

def run_ew_triv_unbound_sym(panel, pred1, pred2, pred3, fwd_col, oos_gap, nw_lags):
    """All three t-stats must exceed gate; level based on |ŷ|."""
    tag = (f"pos_EW_unbound_sym_{pred1}_{pred2}_{pred3}_{fwd_col}"
           f"_t{int(T_THRESH*100)}_oos{OOS_START}.parquet")
    cache = CACHE_DIR / tag
    if cache.exists():
        return pd.read_parquet(cache).squeeze().rename(
            f"pos_ubsym_{pred1}_{pred2}_{pred3}")

    sub = panel.dropna(subset=[pred1, pred2, pred3, fwd_col]).copy()
    pos = pd.Series(0.0, index=sub.index,
                    name=f"pos_ubsym_{pred1}_{pred2}_{pred3}")
    oos_idx = sub.index.searchsorted(pd.Timestamp(OOS_START))
    start_i = max(MIN_WIN + oos_gap, oos_idx)

    for i in range(start_i, len(sub)):
        train = sub.iloc[0 : i - oos_gap]
        X_tr  = add_constant(train[[pred1, pred2, pred3]], has_constant="skip")
        res   = OLS(train[fwd_col], X_tr).fit()
        nw    = _nw_se(res, nlags=nw_lags)
        t1 = float(res.params.iloc[1]) / float(nw[1])
        t2 = float(res.params.iloc[2]) / float(nw[2])
        t3 = float(res.params.iloc[3]) / float(nw[3])
        if abs(t1) <= T_THRESH or abs(t2) <= T_THRESH or abs(t3) <= T_THRESH:
            continue
        test  = sub.iloc[[i]][[pred1, pred2, pred3]].copy()
        test.insert(0, "const", 1.0)
        y_hat = float(res.predict(test).iloc[0])
        lv    = _level(abs(y_hat))
        if lv > 0:
            pos.iloc[i] = float(lv) if y_hat > 0 else float(-lv)

    pos.to_frame().to_parquet(cache)
    return pos


def run_ew_triv_unbound_asym(panel, pred1, pred2, pred3, fwd_col, oos_gap, nw_lags,
                              rolling_window=500):
    tag = (f"pos_EW_unbound_asym_{pred1}_{pred2}_{pred3}_{fwd_col}"
           f"_t{int(T_THRESH*100)}_rw{rolling_window}_oos{OOS_START}.parquet")
    cache = CACHE_DIR / tag
    if cache.exists():
        return pd.read_parquet(cache).squeeze().rename(
            f"pos_ubasym_{pred1}_{pred2}_{pred3}")

    sub = panel.dropna(subset=[pred1, pred2, pred3, fwd_col]).copy()
    pos = pd.Series(0.0, index=sub.index,
                    name=f"pos_ubasym_{pred1}_{pred2}_{pred3}")
    fwd = sub[fwd_col].values
    oos_idx = sub.index.searchsorted(pd.Timestamp(OOS_START))
    start_i = max(MIN_WIN + oos_gap, oos_idx)

    for i in range(start_i, len(sub)):
        train = sub.iloc[0 : i - oos_gap]
        X_tr  = add_constant(train[[pred1, pred2, pred3]], has_constant="skip")
        res   = OLS(train[fwd_col], X_tr).fit()
        nw    = _nw_se(res, nlags=nw_lags)
        t1 = float(res.params.iloc[1]) / float(nw[1])
        t2 = float(res.params.iloc[2]) / float(nw[2])
        t3 = float(res.params.iloc[3]) / float(nw[3])
        if abs(t1) <= T_THRESH or abs(t2) <= T_THRESH or abs(t3) <= T_THRESH:
            continue
        lo    = max(0, i - oos_gap - rolling_window)
        mu    = float(np.mean(fwd[lo : i - oos_gap]))
        test  = sub.iloc[[i]][[pred1, pred2, pred3]].copy()
        test.insert(0, "const", 1.0)
        y_hat = float(res.predict(test).iloc[0])
        lv_long  = _level(y_hat - mu) if (y_hat - mu) > 0 else 0
        lv_short = _level(-y_hat)     if y_hat         < 0 else 0
        if lv_long > 0:
            pos.iloc[i] =  float(lv_long)
        elif lv_short > 0:
            pos.iloc[i] = -float(lv_short)

    pos.to_frame().to_parquet(cache)
    return pos


def run_ew_triv_unbound_rolmu(panel, pred1, pred2, pred3, fwd_col, oos_gap, nw_lags,
                               rolling_window=500):
    tag = (f"pos_EW_rolmu_unbound_{pred1}_{pred2}_{pred3}_{fwd_col}"
           f"_t{int(T_THRESH*100)}_rw{rolling_window}_oos{OOS_START}.parquet")
    cache = CACHE_DIR / tag
    if cache.exists():
        return pd.read_parquet(cache).squeeze().rename(
            f"pos_ubrolmu_{pred1}_{pred2}_{pred3}")

    sub = panel.dropna(subset=[pred1, pred2, pred3, fwd_col]).copy()
    pos = pd.Series(0.0, index=sub.index,
                    name=f"pos_ubrolmu_{pred1}_{pred2}_{pred3}")
    fwd = sub[fwd_col].values
    oos_idx = sub.index.searchsorted(pd.Timestamp(OOS_START))
    start_i = max(MIN_WIN + oos_gap, oos_idx)

    for i in range(start_i, len(sub)):
        train = sub.iloc[0 : i - oos_gap]
        X_tr  = add_constant(train[[pred1, pred2, pred3]], has_constant="skip")
        res   = OLS(train[fwd_col], X_tr).fit()
        nw    = _nw_se(res, nlags=nw_lags)
        t1 = float(res.params.iloc[1]) / float(nw[1])
        t2 = float(res.params.iloc[2]) / float(nw[2])
        t3 = float(res.params.iloc[3]) / float(nw[3])
        if abs(t1) <= T_THRESH or abs(t2) <= T_THRESH or abs(t3) <= T_THRESH:
            continue
        lo     = max(0, i - oos_gap - rolling_window)
        mu     = float(np.mean(fwd[lo : i - oos_gap]))
        test   = sub.iloc[[i]][[pred1, pred2, pred3]].copy()
        test.insert(0, "const", 1.0)
        y_hat  = float(res.predict(test).iloc[0])
        excess = y_hat - mu
        lv     = _level(abs(excess))
        if lv > 0:
            pos.iloc[i] = float(lv) if excess > 0 else float(-lv)

    pos.to_frame().to_parquet(cache)
    return pos


# ─── Beta time-series ─────────────────────────────────────────────────────────

def compute_betas(panel, predictor, fwd_col, oos_gap, nw_lags):
    tag   = f"betas_EW_{predictor}_{fwd_col}_oos{OOS_START}.parquet"
    cache = CACHE_DIR / tag
    if cache.exists():
        return pd.read_parquet(cache)

    print(f"    Computing betas for {predictor} -> {fwd_col}...")
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
        records.append({"alpha": float(res.params.iloc[0]), "beta": b, "se": se,
                        "t_stat": b / se if se > 0 else 0.0})

    df = pd.DataFrame(records, index=sub.index[start_i:])
    df.to_parquet(cache)
    return df


def compute_betas_bivariate(panel, pred1, pred2, fwd_col, oos_gap, nw_lags):
    tag   = f"betas_EWbiv_{pred1}_{pred2}_{fwd_col}_oos{OOS_START}.parquet"
    cache = CACHE_DIR / tag
    if cache.exists():
        df = pd.read_parquet(cache)
        if "alpha" in df.columns:
            return df
        cache.unlink()   # recompute — missing alpha column

    print(f"    Computing bivariate betas for {pred1}+{pred2} -> {fwd_col}...")
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
            "alpha":    float(res.params.iloc[0]),
            "beta_1":   b1,  "se_1":   se1,
            "t_stat_1": b1 / se1 if se1 > 0 else 0.0,
            "beta_2":   b2,  "se_2":   se2,
            "t_stat_2": b2 / se2 if se2 > 0 else 0.0,
        })

    df = pd.DataFrame(records, index=sub.index[start_i:])
    df.to_parquet(cache)
    return df


def compute_betas_trivariate(panel, pred1, pred2, pred3, fwd_col, oos_gap, nw_lags):
    tag   = f"betas_EWtriv_{pred1}_{pred2}_{pred3}_{fwd_col}_oos{OOS_START}.parquet"
    cache = CACHE_DIR / tag
    if cache.exists():
        return pd.read_parquet(cache)

    print(f"    Computing trivariate betas for {pred1}+{pred2}+{pred3} -> {fwd_col}...")
    sub     = panel.dropna(subset=[pred1, pred2, pred3, fwd_col]).copy()
    N       = len(sub)
    oos_idx = sub.index.searchsorted(pd.Timestamp(OOS_START))
    start_i = max(MIN_WIN + oos_gap, oos_idx)

    records = []
    for i in range(start_i, N):
        train = sub.iloc[0 : i - oos_gap]
        X_tr  = add_constant(train[[pred1, pred2, pred3]], has_constant="skip")
        res   = OLS(train[fwd_col], X_tr).fit()
        nw    = _nw_se(res, nlags=nw_lags)
        b1, se1 = float(res.params.iloc[1]), float(nw[1])
        b2, se2 = float(res.params.iloc[2]), float(nw[2])
        b3, se3 = float(res.params.iloc[3]), float(nw[3])
        records.append({
            "alpha":    float(res.params.iloc[0]),
            "beta_1":   b1,  "se_1":   se1,
            "t_stat_1": b1 / se1 if se1 > 0 else 0.0,
            "beta_2":   b2,  "se_2":   se2,
            "t_stat_2": b2 / se2 if se2 > 0 else 0.0,
            "beta_3":   b3,  "se_3":   se3,
            "t_stat_3": b3 / se3 if se3 > 0 else 0.0,
        })

    df = pd.DataFrame(records, index=sub.index[start_i:])
    df.to_parquet(cache)
    return df


# ─── Derived series helpers ───────────────────────────────────────────────────

def _yhat_univariate(panel, predictor, fwd_col, betas_df):
    sub = panel.dropna(subset=[predictor, fwd_col]).copy()
    idx = betas_df.index.intersection(sub.index)
    return (betas_df.loc[idx, "alpha"]
            + betas_df.loc[idx, "beta"] * sub.loc[idx, predictor]).rename("y_hat")


def _yhat_bivariate(panel, pred1, pred2, fwd_col, betas_df):
    sub = panel.dropna(subset=[pred1, pred2, fwd_col]).copy()
    idx = betas_df.index.intersection(sub.index)
    alpha = betas_df.loc[idx, "alpha"] if "alpha" in betas_df.columns else 0.0
    return (alpha
            + betas_df.loc[idx, "beta_1"] * sub.loc[idx, pred1]
            + betas_df.loc[idx, "beta_2"] * sub.loc[idx, pred2]).rename("y_hat")


def _yhat_trivariate(panel, pred1, pred2, pred3, fwd_col, betas_df):
    sub = panel.dropna(subset=[pred1, pred2, pred3, fwd_col]).copy()
    idx = betas_df.index.intersection(sub.index)
    alpha = betas_df.loc[idx, "alpha"] if "alpha" in betas_df.columns else 0.0
    return (alpha
            + betas_df.loc[idx, "beta_1"] * sub.loc[idx, pred1]
            + betas_df.loc[idx, "beta_2"] * sub.loc[idx, pred2]
            + betas_df.loc[idx, "beta_3"] * sub.loc[idx, pred3]).rename("y_hat")


def _rolling_mu(panel, fwd_col, oos_gap, rolling_window=500, predictor="VP"):
    sub = panel.dropna(subset=[predictor, fwd_col]).copy()
    return (sub[fwd_col]
            .rolling(rolling_window, min_periods=1).mean()
            .shift(oos_gap).rename("rolling_mu"))


# ─── Main: pre-populate all caches ────────────────────────────────────────────

def main():
    print("=" * 72)
    print("  regressions.py — pre-compute all expanding-window OLS caches")
    print("  Models: VRP · VVIX MA5 · VVIX MA10 · VRP+VVIX MA5 · VRP+VVIX MA10")
    print("          VRP+Term Slope · VRP+Open Interest · Trivariate")
    print("=" * 72)

    print("\n[1] Loading data...")
    vrp        = load_vrp_series()
    es         = load_es_front_month()
    vvix_raw   = load_vvix()
    vvix_ma5   = compute_vvix_ma5(vvix_raw)
    vvix_ma10  = compute_vvix_ma10(vvix_raw)
    vix_spot   = load_vix_spot()
    term_slope = compute_vix_term_slope(load_vix_futures_term_structure())
    oi         = load_es_open_interest()

    panel = build_panel(vrp, es, vvix_ma5, vix_spot, term_slope, oi)
    panel["vvix_ma10"] = vvix_ma10.reindex(panel.index)
    print(f"    {len(panel):,} obs  "
          f"[{panel.index.min().date()} - {panel.index.max().date()}]")

    FWD = "fwd_20d"

    print("\n[2] Univariate models...")
    for predictor in ["VP", "vvix_ma5", "vvix_ma10"]:
        print(f"  {predictor}")
        for delta in DELTAS:
            run_ew(panel, predictor, FWD, OOS_GAP, NW_LAGS, delta)
        run_ew_asym(panel, predictor, FWD, OOS_GAP, NW_LAGS)
        run_ew_rolmu(panel, predictor, FWD, OOS_GAP, NW_LAGS)
        run_ew_unbound_sym(panel, predictor, FWD, OOS_GAP, NW_LAGS)
        run_ew_unbound_asym(panel, predictor, FWD, OOS_GAP, NW_LAGS)
        run_ew_unbound_rolmu(panel, predictor, FWD, OOS_GAP, NW_LAGS)
        compute_betas(panel, predictor, FWD, OOS_GAP, NW_LAGS)
        compute_oos_r2(panel, predictor, FWD, OOS_GAP, NW_LAGS)

    print("\n[3] Bivariate models...")
    bivar_pairs = [
        ("VP", "vvix_ma5"),
        ("VP", "vvix_ma10"),
        ("VP", "term_slope"),
        ("VP", "open_interest"),
    ]
    for pred1, pred2 in bivar_pairs:
        print(f"  {pred1} + {pred2}")
        for delta in DELTAS:
            run_ew_bivariate(panel, pred1, pred2, FWD, OOS_GAP, NW_LAGS, delta)
        run_ew_asym_bivariate(panel, pred1, pred2, FWD, OOS_GAP, NW_LAGS)
        run_ew_rolmu_bivariate(panel, pred1, pred2, FWD, OOS_GAP, NW_LAGS)
        run_ew_biv_unbound_sym(panel, pred1, pred2, FWD, OOS_GAP, NW_LAGS)
        run_ew_biv_unbound_asym(panel, pred1, pred2, FWD, OOS_GAP, NW_LAGS)
        run_ew_biv_unbound_rolmu(panel, pred1, pred2, FWD, OOS_GAP, NW_LAGS)
        compute_betas_bivariate(panel, pred1, pred2, FWD, OOS_GAP, NW_LAGS)
        compute_oos_r2_bivariate(panel, pred1, pred2, FWD, OOS_GAP, NW_LAGS)

    print("\n[4] Trivariate model: VRP + VVIX MA5 + Term Slope...")
    run_ew_triv_unbound_sym(panel, "VP", "vvix_ma5", "term_slope", FWD, OOS_GAP, NW_LAGS)
    run_ew_triv_unbound_asym(panel, "VP", "vvix_ma5", "term_slope", FWD, OOS_GAP, NW_LAGS)
    run_ew_triv_unbound_rolmu(panel, "VP", "vvix_ma5", "term_slope", FWD, OOS_GAP, NW_LAGS)
    compute_betas_trivariate(panel, "VP", "vvix_ma5", "term_slope", FWD, OOS_GAP, NW_LAGS)

    print("\n" + "=" * 72)
    print("  Done — all regression caches updated in output/regression_cache/")
    print("=" * 72)


if __name__ == "__main__":
    main()
