"""
leveraged_strategies.py
=======================
Unbound multi-level position sizing variants for three models / three threshold
types.  All existing unit-position pipelines are untouched.

Position levels (THRESHOLDS = 0.2 / 0.5 / 0.75 / 1.0 percent):
    |excess| >= 1.0%  -> +/-4
    |excess| >= 0.75% -> +/-3
    |excess| >= 0.5%  -> +/-2
    |excess| >= 0.2%  -> +/-1
    otherwise / gate  ->   0

Excess definitions:
  symmetric:         excess = y_hat                (reference = 0)
  asymmetric:        long:  excess = y_hat - mu    (enters only if >= 0.2%)
                     short: excess = -y_hat        (y_hat must be < 0, >= 0.2%)
  base_return_shift: excess = y_hat - mu           (symmetric around rolling mu)

Models:
  VVIX MA5 -> 20-day (univariate)               -> output/expanding_window/VVIX MA5/
  VRP      -> 20-day (univariate)               -> output/expanding_window/VRP/
  VRP + VVIX MA5 -> 20-day (bivariate)         -> output/expanding_window/VRP + VVIX MA5/
  VRP + Term Slope -> 20-day (bivariate)       -> output/expanding_window/VRP + Term Slope/
  VRP + Open Interest -> 20-day (bivariate)    -> output/expanding_window/VRP + Open Interest/

Outputs (unbound_<threshold_type>_<variable>.png):
  unbound_symmetric_VVIX_MA5.png
  unbound_asymmetric_VVIX_MA5.png
  unbound_symmetric_VRP.png
  unbound_asymmetric_VRP.png
  unbound_base_return_shift_VRP.png
  unbound_symmetric_VRP_+_VVIX_MA5.png
  unbound_asymmetric_VRP_+_VVIX_MA5.png
  unbound_base_return_shift_VRP_+_VVIX_MA5.png
  unbound_symmetric_VRP_+_Term_Slope.png
  unbound_asymmetric_VRP_+_Term_Slope.png
  unbound_base_return_shift_VRP_+_Term_Slope.png
  unbound_symmetric_VRP_+_Open_Interest.png
  unbound_asymmetric_VRP_+_Open_Interest.png
  unbound_base_return_shift_VRP_+_Open_Interest.png
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
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

from statsmodels.api import OLS, add_constant

ROOT      = Path(__file__).parent
OUTPUT    = ROOT / "output"
CACHE_DIR = OUTPUT / "regression_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT.parent / "bh_replication"))
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT))

from har_model import _nw_se
from experiment2 import (
    load_vrp_series, load_es_front_month, load_vvix, compute_vvix_ma5,
    load_vix_spot, load_vix_futures_term_structure, load_es_open_interest,
    compute_buy_and_hold, simulate_strategy, compute_performance_stats,
    compute_trend_quotient, build_master_panel,
)
from fh_replication.fh_replication import compute_vix_term_slope
from horizon_regression import (
    compute_betas, compute_betas_bivariate,
    _shade, oos_cumret, OOS_START, MIN_WIN, T_THRESH,
)

# ─── Constants ───────────────────────────────────────────────────────────────
VVIX_ACT    = pd.Timestamp("2006-03-06")
OOS_GAP     = 20
NW_LAGS     = 20
RW          = 500
BAH_COLOR   = "#d62728"

THRESHOLDS = [0.010, 0.0075, 0.005, 0.002]
LEVELS     = [4, 3, 2, 1]

C_VVIX  = "#3f007d"   # VVIX MA5 — dark purple
C_VRP   = "#08306b"   # VRP      — dark blue
C_BIV   = "#3f007d"   # VRP+VVIX bivariate — purple (as Model C)
C_VRP2  = "#2171b5"   # second t-stat colour in bivariate panel
C_TERM  = "#00441b"   # VRP+Term Slope — dark green (as Model A)
C_TERM2 = "#238b45"   # second t-stat colour for term slope
C_OI    = "#54278f"   # VRP+Open Interest — dark violet
C_TRIV  = "#2e1503"   # VRP+VVIX MA5+Term Slope — dark brown (Model D palette[0])
C_TRIV2 = "#7f3c00"   # second t-stat colour for trivariate
C_TRIV3 = "#c45c00"   # third  t-stat colour for trivariate


# ─── Level helper ────────────────────────────────────────────────────────────

def _level(excess_abs: float) -> int:
    for thresh, lv in zip(THRESHOLDS, LEVELS):
        if excess_abs >= thresh:
            return lv
    return 0


# ═══════════════════════════════════════════════════════════════════════════
# UNIVARIATE position functions
# ═══════════════════════════════════════════════════════════════════════════

def run_ew_unbound_sym(panel, predictor, fwd_col, oos_gap, nw_lags):
    """Level based on |y_hat|."""
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
    """Long: level by (y_hat-mu); short: level by |y_hat| when y_hat<0."""
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
        y_hat        = float(res.predict(test).iloc[0])
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
    """Level based on |y_hat - rolling_mu|."""
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
        lo    = max(0, i - oos_gap - rolling_window)
        mu    = float(np.mean(fwd[lo : i - oos_gap]))
        test  = sub.iloc[[i]][[predictor]].copy(); test.insert(0, "const", 1.0)
        y_hat  = float(res.predict(test).iloc[0])
        excess = y_hat - mu
        lv     = _level(abs(excess))
        if lv > 0:
            pos.iloc[i] = float(lv) if excess > 0 else float(-lv)

    pos.to_frame().to_parquet(cache)
    return pos


# ═══════════════════════════════════════════════════════════════════════════
# BIVARIATE position functions  (both t-stats gate)
# ═══════════════════════════════════════════════════════════════════════════

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
        y_hat        = float(res.predict(test).iloc[0])
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
        lo    = max(0, i - oos_gap - rolling_window)
        mu    = float(np.mean(fwd[lo : i - oos_gap]))
        test  = sub.iloc[[i]][[pred1, pred2]].copy(); test.insert(0, "const", 1.0)
        y_hat  = float(res.predict(test).iloc[0])
        excess = y_hat - mu
        lv     = _level(abs(excess))
        if lv > 0:
            pos.iloc[i] = float(lv) if excess > 0 else float(-lv)

    pos.to_frame().to_parquet(cache)
    return pos


# ═══════════════════════════════════════════════════════════════════════════
# TRIVARIATE position functions  (all three t-stats gate)
# ═══════════════════════════════════════════════════════════════════════════

def run_ew_triv_unbound_sym(panel, pred1, pred2, pred3, fwd_col, oos_gap, nw_lags):
    """Level based on |y_hat|; all three t-stats must exceed gate."""
    tag = (f"pos_EW_unbound_sym_{pred1}_{pred2}_{pred3}_{fwd_col}"
           f"_t{int(T_THRESH*100)}_oos{OOS_START}.parquet")
    cache = CACHE_DIR / tag
    if cache.exists():
        return pd.read_parquet(cache).squeeze().rename(f"pos_ubsym_{pred1}_{pred2}_{pred3}")

    sub = panel.dropna(subset=[pred1, pred2, pred3, fwd_col]).copy()
    pos = pd.Series(0.0, index=sub.index, name=f"pos_ubsym_{pred1}_{pred2}_{pred3}")
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
    """Long: level by (y_hat-mu); short: level by |y_hat| when y_hat<0."""
    tag = (f"pos_EW_unbound_asym_{pred1}_{pred2}_{pred3}_{fwd_col}"
           f"_t{int(T_THRESH*100)}_rw{rolling_window}_oos{OOS_START}.parquet")
    cache = CACHE_DIR / tag
    if cache.exists():
        return pd.read_parquet(cache).squeeze().rename(f"pos_ubasym_{pred1}_{pred2}_{pred3}")

    sub = panel.dropna(subset=[pred1, pred2, pred3, fwd_col]).copy()
    pos = pd.Series(0.0, index=sub.index, name=f"pos_ubasym_{pred1}_{pred2}_{pred3}")
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
        y_hat        = float(res.predict(test).iloc[0])
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
    """Level based on |y_hat - rolling_mu|."""
    tag = (f"pos_EW_rolmu_unbound_{pred1}_{pred2}_{pred3}_{fwd_col}"
           f"_t{int(T_THRESH*100)}_rw{rolling_window}_oos{OOS_START}.parquet")
    cache = CACHE_DIR / tag
    if cache.exists():
        return pd.read_parquet(cache).squeeze().rename(f"pos_ubrolmu_{pred1}_{pred2}_{pred3}")

    sub = panel.dropna(subset=[pred1, pred2, pred3, fwd_col]).copy()
    pos = pd.Series(0.0, index=sub.index, name=f"pos_ubrolmu_{pred1}_{pred2}_{pred3}")
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
        y_hat  = float(res.predict(test).iloc[0])
        excess = y_hat - mu
        lv     = _level(abs(excess))
        if lv > 0:
            pos.iloc[i] = float(lv) if excess > 0 else float(-lv)

    pos.to_frame().to_parquet(cache)
    return pos


# ═══════════════════════════════════════════════════════════════════════════
# Derived series helpers
# ═══════════════════════════════════════════════════════════════════════════

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


def compute_betas_trivariate(panel, pred1, pred2, pred3, fwd_col, oos_gap, nw_lags):
    """Expanding-window OLS betas for three predictors. Cached by predictor tuple."""
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
        a0    = float(res.params.iloc[0])
        b1, se1 = float(res.params.iloc[1]), float(nw[1])
        b2, se2 = float(res.params.iloc[2]), float(nw[2])
        b3, se3 = float(res.params.iloc[3]), float(nw[3])
        records.append({
            "alpha":    a0,
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


# ═══════════════════════════════════════════════════════════════════════════
# Shared panel-drawing helpers
# ═══════════════════════════════════════════════════════════════════════════

def _draw_cumret(ax, sim, bah_sim, main_color, label):
    """Panel 1. Returns (_stat_start, pL, pS, avg_pos)."""
    oos_dt = pd.Timestamp(OOS_START)
    pos_oos    = sim["position"][sim.index >= OOS_START]
    active     = pos_oos[pos_oos != 0]
    _activation = active.index.min() if len(active) else None
    _rebase_start = None
    if _activation is not None and _activation > pd.Timestamp("2020-01-01"):
        idx_arr       = bah_sim.index
        act_i         = idx_arr.searchsorted(_activation)
        _rebase_start = idx_arr[min(act_i + 1, len(idx_arr) - 1)]
    _stat_start = _rebase_start if _rebase_start is not None else oos_dt
    _stat_lbl   = (f" stats from {_stat_start.strftime('%Y-%m-%d')}"
                   if _rebase_start is not None else "")

    _bah_st = compute_performance_stats(bah_sim[bah_sim.index >= _stat_start], "BaH")
    ax.plot(oos_cumret(bah_sim).index,
            oos_cumret(bah_sim).values,
            color=BAH_COLOR, lw=1.5, ls="-.", alpha=0.6,
            label=(f"Buy-and-Hold{_stat_lbl}  "
                   f"[SR={_bah_st['sharpe']:+.2f}  "
                   f"ret={_bah_st['ann_ret']*100:+.1f}%  "
                   f"DD={_bah_st['max_dd']*100:.1f}%]"))

    if _rebase_start is not None:
        _bah_from   = bah_sim["net_pnl"][bah_sim.index >= _rebase_start]
        _bah_act    = (1 + _bah_from).cumprod()
        _bah_act_st = compute_performance_stats(bah_sim[bah_sim.index >= _rebase_start], "BaH2")
        ax.plot(_bah_act.index, _bah_act.values,
                color=BAH_COLOR, lw=1.2, ls=":", alpha=0.85,
                label=(f"Buy-and-Hold from {_rebase_start.strftime('%Y-%m-%d')}  "
                       f"[SR={_bah_act_st['sharpe']:+.2f}  "
                       f"ret={_bah_act_st['ann_ret']*100:+.1f}%  "
                       f"DD={_bah_act_st['max_dd']*100:.1f}%]"))

    st      = compute_performance_stats(sim[sim.index >= _stat_start], "ub")
    p_stat  = sim["position"][sim.index >= _stat_start]
    pL, pS  = float((p_stat > 0).mean() * 100), float((p_stat < 0).mean() * 100)
    avg_pos = float(p_stat.mean())
    ax.plot(oos_cumret(sim).index, oos_cumret(sim).values,
            color=main_color, lw=1.8, alpha=0.9,
            label=(f"{label}  "
                   f"[SR={st['sharpe']:+.2f}  "
                   f"ret={st['ann_ret']*100:+.1f}%  "
                   f"DD={st['max_dd']*100:.1f}%  "
                   f"L{pL:.0f}%/S{pS:.0f}%  AvgPos={avg_pos:+.2f}]"))

    ax.axhline(1, color="black", lw=0.4, ls=":")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2f}x"))
    ax.set_ylabel("Cumulative Net Return (log)", fontsize=9)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.92)
    ax.grid(axis="y", alpha=0.2, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    return _stat_start, pL, pS, avg_pos


def _draw_tstat_univ(ax, betas_df, main_color, pred_label, nw_lags):
    """Panel 2 — univariate."""
    t_ser, b_ser = betas_df["t_stat"], betas_df["beta"]
    ax.plot(t_ser.index, t_ser.values, color=main_color, lw=1.0, alpha=0.85,
            label=f"NW t-stat ({pred_label}, {nw_lags}-lag HAC)")
    ax.fill_between(t_ser.index, -T_THRESH, T_THRESH,
                    color="firebrick", alpha=0.05, label="Below gate (flat zone)")
    for y in (T_THRESH, -T_THRESH):
        ax.axhline(y, color="firebrick", lw=1.2, ls="--")
    ax.axhline(0, color="black", lw=0.5, ls=":")
    ax.axhline(T_THRESH, color="firebrick", lw=0, label=f"|t| = {T_THRESH:.2f} gate")
    ax.set_ylabel(f"NW t-stat ({pred_label})", fontsize=9)
    ax.grid(axis="y", alpha=0.2, lw=0.6); ax.spines["top"].set_visible(False)

    ax2 = ax.twinx()
    ax2.plot(b_ser.index, b_ser.values, color="dimgrey", lw=1.0, ls="--", alpha=0.60,
             label=f"Beta ({pred_label})")
    ax2.axhline(0, color="dimgrey", lw=0.4, ls=":")
    ax2.set_ylabel(f"Beta ({pred_label})", fontsize=8, color="dimgrey")
    ax2.tick_params(axis="y", labelcolor="dimgrey", labelsize=7)
    ax2.spines["top"].set_visible(False)
    if "r2_insample" in betas_df.columns:
        ax2.plot(betas_df["r2_insample"].index, betas_df["r2_insample"].values,
                 color="forestgreen", lw=0.9, ls=":", alpha=0.75, label="In-sample R2")
    l1, lb1 = ax.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, lb1 + lb2, fontsize=8, loc="upper left")


def _draw_tstat_biv(ax, betas_df, c1, c2, pred1_label, pred2_label, nw_lags):
    """Panel 2 — bivariate (two t-stats)."""
    t1, t2 = betas_df["t_stat_1"], betas_df["t_stat_2"]
    b1, b2 = betas_df["beta_1"],   betas_df["beta_2"]
    ax.plot(t1.index, t1.values, color=c1, lw=1.0, alpha=0.85,
            label=f"NW t-stat: {pred1_label} ({nw_lags}-lag HAC)")
    ax.plot(t2.index, t2.values, color=c2, lw=1.0, alpha=0.85, ls="--",
            label=f"NW t-stat: {pred2_label} ({nw_lags}-lag HAC)")
    ax.fill_between(t1.index, -T_THRESH, T_THRESH,
                    color="firebrick", alpha=0.05, label="Below gate (flat zone)")
    for y in (T_THRESH, -T_THRESH):
        ax.axhline(y, color="firebrick", lw=1.2, ls="--")
    ax.axhline(0, color="black", lw=0.5, ls=":")
    ax.axhline(T_THRESH, color="firebrick", lw=0, label=f"|t| = {T_THRESH:.2f} gate")
    ax.set_ylabel("NW t-stat", fontsize=9)
    ax.grid(axis="y", alpha=0.2, lw=0.6); ax.spines["top"].set_visible(False)

    ax2 = ax.twinx()
    ax2.plot(b1.index, b1.values, color="dimgrey", lw=1.0, ls="-", alpha=0.60,
             label=f"Beta: {pred1_label}")
    ax2.plot(b2.index, b2.values, color="dimgrey", lw=1.0, ls=":", alpha=0.60,
             label=f"Beta: {pred2_label}")
    ax2.axhline(0, color="dimgrey", lw=0.4, ls=":")
    ax2.set_ylabel("Beta", fontsize=8, color="dimgrey")
    ax2.tick_params(axis="y", labelcolor="dimgrey", labelsize=7)
    ax2.spines["top"].set_visible(False)
    if "r2_insample" in betas_df.columns:
        ax2.plot(betas_df["r2_insample"].index, betas_df["r2_insample"].values,
                 color="forestgreen", lw=0.9, ls=":", alpha=0.75, label="In-sample R2")
    l1, lb1 = ax.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, lb1 + lb2, fontsize=8, loc="upper left")


def _draw_tstat_triv(ax, betas_df, c1, c2, c3, pred1_label, pred2_label, pred3_label, nw_lags):
    """Panel 2 — trivariate (three t-stats)."""
    t1, t2, t3 = betas_df["t_stat_1"], betas_df["t_stat_2"], betas_df["t_stat_3"]
    b1, b2, b3 = betas_df["beta_1"],   betas_df["beta_2"],   betas_df["beta_3"]
    ax.plot(t1.index, t1.values, color=c1, lw=1.0, alpha=0.85,
            label=f"NW t-stat: {pred1_label} ({nw_lags}-lag HAC)")
    ax.plot(t2.index, t2.values, color=c2, lw=1.0, alpha=0.85, ls="--",
            label=f"NW t-stat: {pred2_label} ({nw_lags}-lag HAC)")
    ax.plot(t3.index, t3.values, color=c3, lw=1.0, alpha=0.85, ls="-.",
            label=f"NW t-stat: {pred3_label} ({nw_lags}-lag HAC)")
    ax.fill_between(t1.index, -T_THRESH, T_THRESH,
                    color="firebrick", alpha=0.05, label="Below gate (flat zone)")
    for y in (T_THRESH, -T_THRESH):
        ax.axhline(y, color="firebrick", lw=1.2, ls="--")
    ax.axhline(0, color="black", lw=0.5, ls=":")
    ax.axhline(T_THRESH, color="firebrick", lw=0, label=f"|t| = {T_THRESH:.2f} gate")
    ax.set_ylabel("NW t-stat", fontsize=9)
    ax.grid(axis="y", alpha=0.2, lw=0.6)
    ax.spines["top"].set_visible(False)

    ax2 = ax.twinx()
    ax2.plot(b1.index, b1.values, color="dimgrey", lw=1.0, ls="-", alpha=0.60,
             label=f"Beta: {pred1_label}")
    ax2.plot(b2.index, b2.values, color="dimgrey", lw=1.0, ls=":", alpha=0.55,
             label=f"Beta: {pred2_label}")
    ax2.plot(b3.index, b3.values, color="dimgrey", lw=1.0, ls="-.", alpha=0.40,
             label=f"Beta: {pred3_label}")
    ax2.axhline(0, color="dimgrey", lw=0.4, ls=":")
    ax2.set_ylabel("Beta", fontsize=8, color="dimgrey")
    ax2.tick_params(axis="y", labelcolor="dimgrey", labelsize=7)
    ax2.spines["top"].set_visible(False)
    l1, lb1 = ax.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, lb1 + lb2, fontsize=8, loc="upper left")


def _draw_position(ax, sim, main_color, pL, pS, avg_pos):
    """Panel 3 — position (y-range +-4)."""
    pos_plot = sim["position"][sim.index >= OOS_START]
    ax.fill_between(pos_plot.index, pos_plot.clip(lower=0), 0,
                    color=main_color, alpha=0.70)
    ax.fill_between(pos_plot.index, pos_plot.clip(upper=0), 0,
                    color=main_color, alpha=0.30, hatch="///")
    ax.axhline(0, color="black", lw=0.4)
    ax.set_ylim(-4.5, 4.5)
    ax.set_yticks([-4, -3, -2, -1, 0, 1, 2, 3, 4])
    ax.tick_params(axis="y", labelsize=7)
    ax.set_ylabel("Position", fontsize=9, rotation=0,
                  ha="right", va="center", labelpad=56, color=main_color)
    pF = 100.0 - pL - pS
    ax.text(0.01, 0.97,
            f"Long {pL:.1f}%  Short {pS:.1f}%  Flat {pF:.1f}%  AvgPos={avg_pos:+.3f}",
            transform=ax.transAxes, fontsize=7.5, va="top", color=main_color)
    ax.spines[["top", "right"]].set_visible(False)


def _draw_predicted(ax, y_hat_ser, mu_ser, main_color, threshold_type):
    """Panel 4 — predicted return with threshold bands."""
    oos_dt    = pd.Timestamp(OOS_START)
    yh_oos    = y_hat_ser[y_hat_ser.index >= oos_dt]
    mu_oos    = mu_ser[mu_ser.index >= oos_dt] if mu_ser is not None else None
    band_cols = ["#cbc9e2", "#9e9ac8", "#807dba", "#6a51a3"]  # light -> dark

    if threshold_type == "sym":
        for tv, bc in zip(THRESHOLDS, band_cols):
            ax.fill_between(yh_oos.index, -tv * 100, tv * 100,
                            alpha=0.18, color=bc, linewidth=0)
        ax.axhline(0, color="black", lw=1.0, ls="--", alpha=0.5)
        ax.plot(yh_oos.index, yh_oos.values * 100,
                color=main_color, lw=0.8, alpha=0.85,
                label="y_hat (predicted 20d return)")
        bp = mpatches.Patch(facecolor="#6a51a3", alpha=0.4,
                            label="Threshold bands +/-0.2/0.5/0.75/1.0%")
    else:
        idx_c = yh_oos.index.intersection(mu_oos.index)
        yh, mu_ = yh_oos.loc[idx_c], mu_oos.loc[idx_c]
        if threshold_type == "asym":
            for tv, bc in zip(THRESHOLDS, band_cols):
                ax.fill_between(mu_.index,
                                mu_.values * 100, (mu_ + tv).values * 100,
                                alpha=0.15, color=bc, linewidth=0)
                ax.fill_between(yh.index, -tv * 100, 0,
                                alpha=0.08, color=bc, linewidth=0)
            ax.axhline(0, color="firebrick", lw=0.8, ls=":",
                       alpha=0.6, label="Short threshold (0)")
        else:
            for tv, bc in zip(THRESHOLDS, band_cols):
                ax.fill_between(mu_.index,
                                (mu_ - tv).values * 100,
                                (mu_ + tv).values * 100,
                                alpha=0.18, color=bc, linewidth=0)
        ax.plot(mu_.index, mu_.values * 100,
                color="black", lw=1.2, ls="--", alpha=0.75,
                label="Rolling mu (500d fwd return)")
        ax.plot(yh.index, yh.values * 100,
                color=main_color, lw=0.8, alpha=0.85,
                label="y_hat (predicted 20d return)")
        bp = mpatches.Patch(facecolor="#6a51a3", alpha=0.4,
                            label="Threshold bands")

    ax.axhline(0, color="black", lw=0.4, ls=":")
    ax.set_ylabel("Pred. Return (%)", fontsize=9)
    ax.grid(axis="y", alpha=0.2, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    h, lb = ax.get_legend_handles_labels()
    ax.legend(h + [bp], lb + [bp.get_label()], fontsize=8, loc="upper left")


def _finalize(fig, axes, out_path):
    for ax in axes:
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.tick_params(axis="x", which="major", labelbottom=True, labelsize=7, pad=2)
    fig.savefig(out_path, dpi=155, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_unbound_asymmetric_comparison(sim_vvix, sim_biv, sim_term, bah_sim, out_path,
                                        sim_vrp=None):
    """One-panel comparison: unbound-asymmetric VVIX MA5 vs VRP+VVIX MA5 vs VRP+Term Slope.

    If sim_vrp is provided, also plots the univariate VRP unbound-asymmetric curve
    rebased from 2020-01-01 (the period when it is typically active).
    """
    oos_dt = pd.Timestamp(OOS_START)
    VRP_ASYM_START = pd.Timestamp("2020-01-01")

    def _first_active(sim):
        pos = sim["position"][sim.index >= oos_dt]
        active = pos[pos != 0]
        return active.index.min() if len(active) else None

    candidates = [d for d in [_first_active(s) for s in (sim_vvix, sim_biv, sim_term)]
                  if d is not None]
    start = min(candidates) if candidates else oos_dt
    start_str = start.strftime("%Y-%m-%d")

    def _cum(sim, s=None):
        s = s or start
        net = sim["net_pnl"][sim.index >= s]
        return (1 + net).cumprod()

    def _stats(sim, label, s=None):
        s = s or start
        return compute_performance_stats(sim[sim.index >= s], label)

    def _pos_pct(sim, s=None):
        s = s or start
        p = sim["position"][sim.index >= s]
        return float((p > 0).mean() * 100), float((p < 0).mean() * 100), float(p.mean())

    bah_cum   = _cum(bah_sim);  bah_st   = _stats(bah_sim,  "BaH")
    vvix_cum  = _cum(sim_vvix); vvix_st  = _stats(sim_vvix, "VVIX_asym")
    biv_cum   = _cum(sim_biv);  biv_st   = _stats(sim_biv,  "BIV_asym")
    term_cum  = _cum(sim_term); term_st  = _stats(sim_term, "TERM_asym")

    pL_v, pS_v, avg_v = _pos_pct(sim_vvix)
    pL_b, pS_b, avg_b = _pos_pct(sim_biv)
    pL_t, pS_t, avg_t = _pos_pct(sim_term)

    all_sims = [sim_vvix, sim_biv, sim_term]
    if sim_vrp is not None:
        all_sims.append(sim_vrp)
    e_dt = max(s.index[-1] for s in all_sims)

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.suptitle(
        f"Unbound Asymmetric: VVIX MA5 vs VRP + VVIX MA5 vs VRP + Term Slope  ·  Expanding Window\n"
        f"Rebased to 1.0 at {start_str}  ·  |t| > {T_THRESH:.2f} gate  ·  "
        f"Levels ±1..4 at |excess| ≥ 0.2/0.5/0.75/1.0%  ·  0.05% slippage",
        fontsize=10,
    )

    ax.plot(bah_cum.index, bah_cum.values,
            color=BAH_COLOR, lw=1.8, ls="-.", alpha=0.65,
            label=(f"Buy-and-Hold  "
                   f"[SR={bah_st['sharpe']:+.2f}  "
                   f"ret={bah_st['ann_ret']*100:+.1f}%  "
                   f"DD={bah_st['max_dd']*100:.1f}%]"))

    ax.plot(vvix_cum.index, vvix_cum.values,
            color=C_VVIX, lw=2.2, ls="-", alpha=0.92,
            label=(f"VVIX MA5 (Unbound Asym)  "
                   f"[SR={vvix_st['sharpe']:+.2f}  "
                   f"ret={vvix_st['ann_ret']*100:+.1f}%  "
                   f"DD={vvix_st['max_dd']*100:.1f}%  "
                   f"L{pL_v:.0f}%/S{pS_v:.0f}%  AvgPos={avg_v:+.2f}]"))

    ax.plot(biv_cum.index, biv_cum.values,
            color=C_VRP, lw=2.2, ls="--", alpha=0.92,
            label=(f"VRP + VVIX MA5 (Unbound Asym)  "
                   f"[SR={biv_st['sharpe']:+.2f}  "
                   f"ret={biv_st['ann_ret']*100:+.1f}%  "
                   f"DD={biv_st['max_dd']*100:.1f}%  "
                   f"L{pL_b:.0f}%/S{pS_b:.0f}%  AvgPos={avg_b:+.2f}]"))

    ax.plot(term_cum.index, term_cum.values,
            color=C_TERM, lw=2.2, ls="-", alpha=0.92,
            label=(f"VRP + Term Slope (Unbound Asym)  "
                   f"[SR={term_st['sharpe']:+.2f}  "
                   f"ret={term_st['ann_ret']*100:+.1f}%  "
                   f"DD={term_st['max_dd']*100:.1f}%  "
                   f"L{pL_t:.0f}%/S{pS_t:.0f}%  AvgPos={avg_t:+.2f}]"))

    if sim_vrp is not None:
        vrp_cum = _cum(sim_vrp, VRP_ASYM_START)
        vrp_st  = _stats(sim_vrp, "VRP_asym", VRP_ASYM_START)
        pL_vrp, pS_vrp, avg_vrp = _pos_pct(sim_vrp, VRP_ASYM_START)
        ax.plot(vrp_cum.index, vrp_cum.values,
                color="#4292c6", lw=2.2, ls="-.", alpha=0.92,
                label=(f"VRP Univariate (Unbound Asym, post-2020)  "
                       f"[SR={vrp_st['sharpe']:+.2f}  "
                       f"ret={vrp_st['ann_ret']*100:+.1f}%  "
                       f"DD={vrp_st['max_dd']*100:.1f}%  "
                       f"L{pL_vrp:.0f}%/S{pS_vrp:.0f}%  AvgPos={avg_vrp:+.2f}]"))

    for a_dt, b_dt in [("2020-02-01", "2020-06-01"), ("2022-01-01", "2022-12-31")]:
        ax.axvspan(pd.Timestamp(a_dt), pd.Timestamp(b_dt), alpha=0.07, color="grey", lw=0)

    ax.axhline(1, color="black", lw=0.5, ls=":")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2f}x"))
    ax.set_ylabel("Cumulative Net Return (log, rebased to 1.0)", fontsize=10)
    ax.set_xlim(start, e_dt)
    ax.legend(fontsize=8.5, loc="upper left", framealpha=0.92, edgecolor="#cccccc")
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax.get_xticklabels(), visible=True, fontsize=9)
    ax.grid(axis="y", alpha=0.2, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=155, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ═══════════════════════════════════════════════════════════════════════════
# Full-figure plot functions
# ═══════════════════════════════════════════════════════════════════════════

_THRESH_LBL = {
    "sym":   "Symmetric (Unbound)",
    "asym":  "Asymmetric (Unbound)",
    "rolmu": "Base-Return-Shift (Unbound)",
}
_THRESH_DESC = {
    "sym":   "Long/Short +/-1..4 at |y_hat| >= 0.2/0.5/0.75/1.0%",
    "asym":  ("Long +1..4 at (y_hat-mu) >= 0.2..1.0%  |  "
              "Short -1..-4 at (-y_hat) >= 0.2..1.0%"),
    "rolmu": "Long/Short +/-1..4 at |y_hat - mu| >= 0.2/0.5/0.75/1.0%",
}


def plot_unbound_univariate(pred_label, horizon_label, threshold_type,
                            main_color, sim, betas_df, bah_sim,
                            y_hat_ser, mu_ser, out_path,
                            extra_title="", nw_lags=NW_LAGS):
    oos_dt = pd.Timestamp(OOS_START)
    e_dt   = betas_df.index[-1]

    fig, axes = plt.subplots(
        4, 1, figsize=(14, 15.4), sharex=True,
        gridspec_kw={"height_ratios": [2.5, 1.2, 1.0, 1.2], "hspace": 0.35},
    )
    ax_ret, ax_t, ax_p, ax_yh = axes

    fig.suptitle(
        f"{pred_label} -> {horizon_label} Forward Return  "
        f"(Expanding Window, OOS from {OOS_START})  "
        f"{_THRESH_LBL[threshold_type]}{extra_title}\n"
        f"{_THRESH_DESC[threshold_type]}  |  "
        f"NW-HAC {nw_lags} lags; |t| > {T_THRESH:.2f} gate; 0.05% slippage",
        fontsize=10, y=0.998,
    )
    fig.subplots_adjust(top=0.955, bottom=0.03, left=0.10, right=0.93)
    for ax in axes:
        ax.set_xlim(oos_dt, e_dt)
        _shade(ax, oos_dt, e_dt)

    _, pL, pS, avg_pos = _draw_cumret(ax_ret, sim, bah_sim, main_color, "Unbound +/-1..4")
    _draw_tstat_univ(ax_t, betas_df, main_color, pred_label, nw_lags)
    _draw_position(ax_p, sim, main_color, pL, pS, avg_pos)
    _draw_predicted(ax_yh, y_hat_ser, mu_ser, main_color, threshold_type)
    _finalize(fig, axes, out_path)


def plot_unbound_bivariate(pred1_label, pred2_label, horizon_label, threshold_type,
                           main_color, sim, betas_df, bah_sim,
                           y_hat_ser, mu_ser, out_path,
                           extra_title="", nw_lags=NW_LAGS):
    oos_dt = pd.Timestamp(OOS_START)
    e_dt   = betas_df.index[-1]

    fig, axes = plt.subplots(
        4, 1, figsize=(14, 15.4), sharex=True,
        gridspec_kw={"height_ratios": [2.5, 1.2, 1.0, 1.2], "hspace": 0.35},
    )
    ax_ret, ax_t, ax_p, ax_yh = axes

    fig.suptitle(
        f"{pred1_label} + {pred2_label} -> {horizon_label} Forward Return  "
        f"(Expanding Window, OOS from {OOS_START})  "
        f"{_THRESH_LBL[threshold_type]}{extra_title}\n"
        f"{_THRESH_DESC[threshold_type]}  |  "
        f"NW-HAC {nw_lags} lags; |t| > {T_THRESH:.2f} gate (both); 0.05% slippage",
        fontsize=10, y=0.998,
    )
    fig.subplots_adjust(top=0.955, bottom=0.03, left=0.10, right=0.93)
    for ax in axes:
        ax.set_xlim(oos_dt, e_dt)
        _shade(ax, oos_dt, e_dt)

    _, pL, pS, avg_pos = _draw_cumret(ax_ret, sim, bah_sim, main_color, "Unbound +/-1..4")
    _draw_tstat_biv(ax_t, betas_df, C_VRP, C_VRP2, pred1_label, pred2_label, nw_lags)
    _draw_position(ax_p, sim, main_color, pL, pS, avg_pos)
    _draw_predicted(ax_yh, y_hat_ser, mu_ser, main_color, threshold_type)
    _finalize(fig, axes, out_path)


def plot_unbound_trivariate(pred1_label, pred2_label, pred3_label,
                            horizon_label, threshold_type,
                            main_color, sim, betas_df, bah_sim,
                            y_hat_ser, mu_ser, out_path,
                            c1=None, c2=None, c3=None,
                            extra_title="", nw_lags=NW_LAGS):
    oos_dt = pd.Timestamp(OOS_START)
    e_dt   = betas_df.index[-1]
    if c1 is None: c1 = main_color
    if c2 is None: c2 = main_color
    if c3 is None: c3 = main_color

    fig, axes = plt.subplots(
        4, 1, figsize=(14, 15.4), sharex=True,
        gridspec_kw={"height_ratios": [2.5, 1.2, 1.0, 1.2], "hspace": 0.35},
    )
    ax_ret, ax_t, ax_p, ax_yh = axes

    fig.suptitle(
        f"{pred1_label} + {pred2_label} + {pred3_label} -> {horizon_label} Forward Return  "
        f"(Expanding Window, OOS from {OOS_START})  "
        f"{_THRESH_LBL[threshold_type]}{extra_title}\n"
        f"{_THRESH_DESC[threshold_type]}  |  "
        f"NW-HAC {nw_lags} lags; |t| > {T_THRESH:.2f} gate (all three); 0.05% slippage",
        fontsize=10, y=0.998,
    )
    fig.subplots_adjust(top=0.955, bottom=0.03, left=0.10, right=0.93)
    for ax in axes:
        ax.set_xlim(oos_dt, e_dt)
        _shade(ax, oos_dt, e_dt)

    _, pL, pS, avg_pos = _draw_cumret(ax_ret, sim, bah_sim, main_color, "Unbound +/-1..4")
    _draw_tstat_triv(ax_t, betas_df, c1, c2, c3,
                     pred1_label, pred2_label, pred3_label, nw_lags)
    _draw_position(ax_p, sim, main_color, pL, pS, avg_pos)
    _draw_predicted(ax_yh, y_hat_ser, mu_ser, main_color, threshold_type)
    _finalize(fig, axes, out_path)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def _report(pos, daily_ret, label):
    sim = simulate_strategy(pos, daily_ret)
    st  = compute_performance_stats(sim[sim.index >= OOS_START], label)
    p   = pos[pos.index >= OOS_START]
    print(f"    {label}: SR={st['sharpe']:+.3f}  "
          f"L%={(p>0).mean()*100:.1f}  S%={(p<0).mean()*100:.1f}  "
          f"F%={(p==0).mean()*100:.1f}  AvgPos={p.mean():+.3f}")
    return sim


def main():
    print("=" * 72)
    print("  Unbound Strategies")
    print("  VVIX MA5 / VRP / VRP+VVIX MA5 / VRP+Term Slope / VRP+OI  x  sym / asym / rolmu")
    print("=" * 72)

    print("\n[1] Loading data...")
    vrp        = load_vrp_series()
    es         = load_es_front_month()
    vvix_ma5   = compute_vvix_ma5(load_vvix())
    term_slope = compute_vix_term_slope(load_vix_futures_term_structure())
    trend_q    = compute_trend_quotient(es)
    oi         = load_es_open_interest()
    # Use build_master_panel so the regression sample matches fixed_split_eval.py's
    # Base/Model_C/Model_VVIX base-strategy plots in the VRP/, VVIX MA5/, VRP+VVIX MA5/ dirs.
    panel      = build_master_panel(vrp, es, term_slope, trend_q, vvix_ma5)
    panel      = panel[panel.index >= "2006-03-06"].copy()
    panel["open_interest"] = oi.reindex(panel.index)
    FWD        = "fwd_20d"
    print(f"    {len(panel):,} obs  "
          f"[{panel.index.min().date()} - {panel.index.max().date()}]")

    daily_ret = panel["daily_ret"].dropna()
    bah_sim   = simulate_strategy(compute_buy_and_hold(daily_ret), daily_ret)

    print("\n[2] Loading / computing betas (cached)...")
    betas_vvix      = compute_betas(panel, "vvix_ma5", FWD, oos_gap=OOS_GAP, nw_lags=NW_LAGS)
    betas_vrp       = compute_betas(panel, "VP",       FWD, oos_gap=OOS_GAP, nw_lags=NW_LAGS)
    betas_biv       = compute_betas_bivariate(panel, "VP", "vvix_ma5",      FWD,
                                              oos_gap=OOS_GAP, nw_lags=NW_LAGS)
    betas_term      = compute_betas_bivariate(panel, "VP", "term_slope",    FWD,
                                              oos_gap=OOS_GAP, nw_lags=NW_LAGS)
    betas_oi        = compute_betas_bivariate(panel, "VP", "open_interest", FWD,
                                              oos_gap=OOS_GAP, nw_lags=NW_LAGS)
    print("    Done.")

    yh_vvix      = _yhat_univariate(panel, "vvix_ma5",  FWD, betas_vvix)
    yh_vrp       = _yhat_univariate(panel, "VP",        FWD, betas_vrp)
    yh_biv       = _yhat_bivariate(panel, "VP", "vvix_ma5",      FWD, betas_biv)
    yh_term      = _yhat_bivariate(panel, "VP", "term_slope",    FWD, betas_term)
    yh_oi        = _yhat_bivariate(panel, "VP", "open_interest", FWD, betas_oi)
    mu_20d       = _rolling_mu(panel, FWD, OOS_GAP, RW)

    out_vvix      = OUTPUT / "expanding_window" / "VVIX MA5"
    out_vrp       = OUTPUT / "expanding_window" / "VRP"
    out_biv       = OUTPUT / "expanding_window" / "VRP + VVIX MA5"
    out_term      = OUTPUT / "expanding_window" / "VRP + Term Slope"
    out_oi        = OUTPUT / "expanding_window" / "VRP + Open Interest"
    for d in (out_vvix, out_vrp, out_biv, out_term, out_oi):
        d.mkdir(parents=True, exist_ok=True)

    # ── A: VVIX MA5 symmetric ────────────────────────────────────────────────
    print("\n[A] VVIX MA5 unbound symmetric...")
    pos = run_ew_unbound_sym(panel, "vvix_ma5", FWD, OOS_GAP, NW_LAGS)
    pos = pos.copy(); pos[pos.index < VVIX_ACT] = 0.0
    sim = _report(pos, daily_ret, "ub_sym_VVIX")
    plot_unbound_univariate(
        "VVIX MA5", "20-day", "sym", C_VVIX,
        sim, betas_vvix, bah_sim, yh_vvix, mu_20d,
        out_vvix / "leveraged_symmetric_VVIX_MA5.png",
        extra_title="\nFlat before VVIX activation (2006-03-06)",
    )

    # ── B: VVIX MA5 asymmetric ───────────────────────────────────────────────
    print("\n[B] VVIX MA5 unbound asymmetric...")
    pos = run_ew_unbound_asym(panel, "vvix_ma5", FWD, OOS_GAP, NW_LAGS)
    pos = pos.copy(); pos[pos.index < VVIX_ACT] = 0.0
    sim_vvix_asym = _report(pos, daily_ret, "ub_asym_VVIX")
    plot_unbound_univariate(
        "VVIX MA5", "20-day", "asym", C_VVIX,
        sim_vvix_asym, betas_vvix, bah_sim, yh_vvix, mu_20d,
        out_vvix / "leveraged_asymmetric_VVIX_MA5.png",
        extra_title="\nFlat before VVIX activation (2006-03-06)",
    )

    # ── C: VRP symmetric ─────────────────────────────────────────────────────
    print("\n[C] VRP unbound symmetric...")
    pos = run_ew_unbound_sym(panel, "VP", FWD, OOS_GAP, NW_LAGS)
    sim = _report(pos, daily_ret, "ub_sym_VRP")
    plot_unbound_univariate(
        "VRP", "20-day", "sym", C_VRP,
        sim, betas_vrp, bah_sim, yh_vrp, mu_20d,
        out_vrp / "leveraged_symmetric_VRP.png",
    )

    # ── D: VRP asymmetric ────────────────────────────────────────────────────
    print("\n[D] VRP unbound asymmetric...")
    pos = run_ew_unbound_asym(panel, "VP", FWD, OOS_GAP, NW_LAGS)
    sim_vrp_asym = _report(pos, daily_ret, "ub_asym_VRP")
    sim = sim_vrp_asym
    plot_unbound_univariate(
        "VRP", "20-day", "asym", C_VRP,
        sim, betas_vrp, bah_sim, yh_vrp, mu_20d,
        out_vrp / "leveraged_asymmetric_VRP.png",
    )

    # ── E: VRP base_return_shift ─────────────────────────────────────────────
    print("\n[E] VRP unbound base_return_shift...")
    pos = run_ew_unbound_rolmu(panel, "VP", FWD, OOS_GAP, NW_LAGS)
    sim = _report(pos, daily_ret, "ub_rolmu_VRP")
    plot_unbound_univariate(
        "VRP", "20-day", "rolmu", C_VRP,
        sim, betas_vrp, bah_sim, yh_vrp, mu_20d,
        out_vrp / "leveraged_base_return_shift_VRP.png",
    )

    # ── F: VRP+VVIX MA5 bivariate symmetric ──────────────────────────────────
    print("\n[F] VRP+VVIX MA5 unbound symmetric...")
    pos = run_ew_biv_unbound_sym(panel, "VP", "vvix_ma5", FWD, OOS_GAP, NW_LAGS)
    sim = _report(pos, daily_ret, "ub_sym_biv")
    plot_unbound_bivariate(
        "VRP", "VVIX MA5", "20-day", "sym", C_BIV,
        sim, betas_biv, bah_sim, yh_biv, mu_20d,
        out_biv / "leveraged_symmetric_VRP_+_VVIX_MA5.png",
    )

    # ── G: VRP+VVIX MA5 bivariate asymmetric ─────────────────────────────────
    print("\n[G] VRP+VVIX MA5 unbound asymmetric...")
    pos = run_ew_biv_unbound_asym(panel, "VP", "vvix_ma5", FWD, OOS_GAP, NW_LAGS)
    sim_biv_asym = _report(pos, daily_ret, "ub_asym_biv")
    plot_unbound_bivariate(
        "VRP", "VVIX MA5", "20-day", "asym", C_BIV,
        sim_biv_asym, betas_biv, bah_sim, yh_biv, mu_20d,
        out_biv / "leveraged_asymmetric_VRP_+_VVIX_MA5.png",
    )

    # ── H: VRP+VVIX MA5 bivariate base_return_shift ───────────────────────────
    print("\n[H] VRP+VVIX MA5 unbound base_return_shift...")
    pos = run_ew_biv_unbound_rolmu(panel, "VP", "vvix_ma5", FWD, OOS_GAP, NW_LAGS)
    sim = _report(pos, daily_ret, "ub_rolmu_biv")
    plot_unbound_bivariate(
        "VRP", "VVIX MA5", "20-day", "rolmu", C_BIV,
        sim, betas_biv, bah_sim, yh_biv, mu_20d,
        out_biv / "leveraged_base_return_shift_VRP_+_VVIX_MA5.png",
    )

    # ── I: VRP+Term Slope bivariate symmetric ────────────────────────────────
    print("\n[I] VRP+Term Slope unbound symmetric...")
    pos = run_ew_biv_unbound_sym(panel, "VP", "term_slope", FWD, OOS_GAP, NW_LAGS)
    sim = _report(pos, daily_ret, "ub_sym_term")
    plot_unbound_bivariate(
        "VRP", "Term Slope", "20-day", "sym", C_TERM,
        sim, betas_term, bah_sim, yh_term, mu_20d,
        out_term / "leveraged_symmetric_VRP_+_Term_Slope.png",
    )

    # ── J: VRP+Term Slope bivariate asymmetric ───────────────────────────────
    print("\n[J] VRP+Term Slope unbound asymmetric...")
    pos = run_ew_biv_unbound_asym(panel, "VP", "term_slope", FWD, OOS_GAP, NW_LAGS)
    sim_term_asym = _report(pos, daily_ret, "ub_asym_term")
    plot_unbound_bivariate(
        "VRP", "Term Slope", "20-day", "asym", C_TERM,
        sim_term_asym, betas_term, bah_sim, yh_term, mu_20d,
        out_term / "leveraged_asymmetric_VRP_+_Term_Slope.png",
    )

    # ── K: VRP+Term Slope bivariate base_return_shift ────────────────────────
    print("\n[K] VRP+Term Slope unbound base_return_shift...")
    pos = run_ew_biv_unbound_rolmu(panel, "VP", "term_slope", FWD, OOS_GAP, NW_LAGS)
    sim = _report(pos, daily_ret, "ub_rolmu_term")
    plot_unbound_bivariate(
        "VRP", "Term Slope", "20-day", "rolmu", C_TERM,
        sim, betas_term, bah_sim, yh_term, mu_20d,
        out_term / "leveraged_base_return_shift_VRP_+_Term_Slope.png",
    )

    # ── L: VRP+Open Interest bivariate symmetric ──────────────────────────────
    print("\n[L] VRP+Open Interest unbound symmetric...")
    pos = run_ew_biv_unbound_sym(panel, "VP", "open_interest", FWD, OOS_GAP, NW_LAGS)
    sim = _report(pos, daily_ret, "ub_sym_oi")
    plot_unbound_bivariate(
        "VRP", "Open Interest", "20-day", "sym", C_OI,
        sim, betas_oi, bah_sim, yh_oi, mu_20d,
        out_oi / "leveraged_symmetric_VRP_+_Open_Interest.png",
    )

    # ── M: VRP+Open Interest bivariate asymmetric ─────────────────────────────
    print("\n[M] VRP+Open Interest unbound asymmetric...")
    pos = run_ew_biv_unbound_asym(panel, "VP", "open_interest", FWD, OOS_GAP, NW_LAGS)
    sim = _report(pos, daily_ret, "ub_asym_oi")
    plot_unbound_bivariate(
        "VRP", "Open Interest", "20-day", "asym", C_OI,
        sim, betas_oi, bah_sim, yh_oi, mu_20d,
        out_oi / "leveraged_asymmetric_VRP_+_Open_Interest.png",
    )

    # ── N: VRP+Open Interest bivariate base_return_shift ─────────────────────
    print("\n[N] VRP+Open Interest unbound base_return_shift...")
    pos = run_ew_biv_unbound_rolmu(panel, "VP", "open_interest", FWD, OOS_GAP, NW_LAGS)
    sim = _report(pos, daily_ret, "ub_rolmu_oi")
    plot_unbound_bivariate(
        "VRP", "Open Interest", "20-day", "rolmu", C_OI,
        sim, betas_oi, bah_sim, yh_oi, mu_20d,
        out_oi / "leveraged_base_return_shift_VRP_+_Open_Interest.png",
    )

    # ── O–Q: VRP+VVIX MA5+Term Slope trivariate unbound ──────────────────────
    print("\n[O–Q] VRP + VVIX MA5 + Term Slope trivariate unbound strategies...")
    betas_triv = compute_betas_trivariate(panel, "VP", "vvix_ma5", "term_slope",
                                          FWD, OOS_GAP, NW_LAGS)
    yh_triv    = _yhat_trivariate(panel, "VP", "vvix_ma5", "term_slope", FWD, betas_triv)
    out_triv   = OUTPUT / "expanding_window" / "trivariate"
    out_triv.mkdir(parents=True, exist_ok=True)

    print("\n[O] VRP+VVIX MA5+Term Slope unbound symmetric...")
    pos = run_ew_triv_unbound_sym(panel, "VP", "vvix_ma5", "term_slope",
                                  FWD, OOS_GAP, NW_LAGS)
    sim = _report(pos, daily_ret, "ub_sym_triv")
    plot_unbound_trivariate(
        "VRP", "VVIX MA5", "Term Slope", "20-day", "sym", C_TRIV,
        sim, betas_triv, bah_sim, yh_triv, mu_20d,
        out_triv / "leveraged_symmetric_VRP_+_VVIX_MA5_+_Term_Slope.png",
        c1=C_TRIV, c2=C_TRIV2, c3=C_TRIV3,
    )

    print("\n[P] VRP+VVIX MA5+Term Slope unbound asymmetric...")
    pos = run_ew_triv_unbound_asym(panel, "VP", "vvix_ma5", "term_slope",
                                   FWD, OOS_GAP, NW_LAGS)
    sim_triv_asym = _report(pos, daily_ret, "ub_asym_triv")
    plot_unbound_trivariate(
        "VRP", "VVIX MA5", "Term Slope", "20-day", "asym", C_TRIV,
        sim_triv_asym, betas_triv, bah_sim, yh_triv, mu_20d,
        out_triv / "leveraged_asymmetric_VRP_+_VVIX_MA5_+_Term_Slope.png",
        c1=C_TRIV, c2=C_TRIV2, c3=C_TRIV3,
    )

    print("\n[Q] VRP+VVIX MA5+Term Slope unbound base_return_shift...")
    pos = run_ew_triv_unbound_rolmu(panel, "VP", "vvix_ma5", "term_slope",
                                    FWD, OOS_GAP, NW_LAGS)
    sim = _report(pos, daily_ret, "ub_rolmu_triv")
    plot_unbound_trivariate(
        "VRP", "VVIX MA5", "Term Slope", "20-day", "rolmu", C_TRIV,
        sim, betas_triv, bah_sim, yh_triv, mu_20d,
        out_triv / "leveraged_base_return_shift_VRP_+_VVIX_MA5_+_Term_Slope.png",
        c1=C_TRIV, c2=C_TRIV2, c3=C_TRIV3,
    )

    # ── Comparison: unbound-asymmetric VVIX MA5 vs VRP+VVIX MA5 ─────────────
    print("\n[Comparison] Unbound asymmetric VVIX MA5 vs VRP + VVIX MA5...")
    out_cmp = OUTPUT / "expanding_window" / "comparisons"
    plot_unbound_asymmetric_comparison(
        sim_vvix_asym, sim_biv_asym, sim_term_asym, bah_sim,
        out_cmp / "leveraged_asymmetric_vvix_vs_vrp_vvix.png",
        sim_vrp=sim_vrp_asym,
    )

    print("\nDone.")
    print("  output/expanding_window/VVIX MA5/leveraged_symmetric_VVIX_MA5.png")
    print("  output/expanding_window/VVIX MA5/leveraged_asymmetric_VVIX_MA5.png")
    print("  output/expanding_window/VRP/leveraged_symmetric_VRP.png")
    print("  output/expanding_window/VRP/leveraged_asymmetric_VRP.png")
    print("  output/expanding_window/VRP/leveraged_base_return_shift_VRP.png")
    print("  output/expanding_window/VRP + VVIX MA5/leveraged_symmetric_VRP_+_VVIX_MA5.png")
    print("  output/expanding_window/VRP + VVIX MA5/leveraged_asymmetric_VRP_+_VVIX_MA5.png")
    print("  output/expanding_window/VRP + VVIX MA5/leveraged_base_return_shift_VRP_+_VVIX_MA5.png")
    print("  output/expanding_window/VRP + Term Slope/leveraged_symmetric_VRP_+_Term_Slope.png")
    print("  output/expanding_window/VRP + Term Slope/leveraged_asymmetric_VRP_+_Term_Slope.png")
    print("  output/expanding_window/VRP + Term Slope/leveraged_base_return_shift_VRP_+_Term_Slope.png")
    print("  output/expanding_window/VRP + Open Interest/leveraged_symmetric_VRP_+_Open_Interest.png")
    print("  output/expanding_window/VRP + Open Interest/leveraged_asymmetric_VRP_+_Open_Interest.png")
    print("  output/expanding_window/VRP + Open Interest/leveraged_base_return_shift_VRP_+_Open_Interest.png")
    print("  output/expanding_window/comparisons/leveraged_asymmetric_vvix_vs_vrp_vvix.png")
    print("=" * 72)


if __name__ == "__main__":
    main()
