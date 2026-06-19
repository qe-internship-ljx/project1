"""
regressions.py
==============
Pure expanding-window OLS regression computation for every model.

Responsibilities:
  - Build the data panel (`build_panel`)
  - Compute OOS R² (univariate and bivariate)
  - Compute beta time-series (univariate, bivariate, trivariate)
  - Provide derived-series helpers: _yhat_univariate/bivariate/trivariate,
    _rolling_mu, _shade, oos_cumret
  - Export shared constants (OOS_START, MIN_WIN, T_THRESH, DELTAS, …)

Position / backtest logic lives in base_strategies.py (unit-position) and
leveraged_strategies.py (multi-level unbound).

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


# ─── Unbound (multi-level) helpers ────────────────────────────────────────────

def _level(excess_abs):
    for thresh, lv in zip(THRESHOLDS, LEVELS):
        if excess_abs >= thresh:
            return lv
    return 0


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
    print("  regressions.py — pre-compute betas and OOS R² caches")
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
        compute_betas_bivariate(panel, pred1, pred2, FWD, OOS_GAP, NW_LAGS)
        compute_oos_r2_bivariate(panel, pred1, pred2, FWD, OOS_GAP, NW_LAGS)

    print("\n[4] Trivariate model: VRP + VVIX MA5 + Term Slope...")
    compute_betas_trivariate(panel, "VP", "vvix_ma5", "term_slope", FWD, OOS_GAP, NW_LAGS)

    print("\n" + "=" * 72)
    print("  Done — all regression caches updated in output/regression_cache/")
    print("=" * 72)


if __name__ == "__main__":
    main()
