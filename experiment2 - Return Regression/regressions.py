"""
regressions.py
==============
Pure expanding-window OLS regression computation for every model.

Responsibilities:
  - Build the data panel (`build_panel`)
  - Compute beta time-series (univariate, bivariate, trivariate); each frame
    carries the NW t-stat columns (t_stat / t_stat_1 / …) used for the plots
  - Provide derived-series helpers: _yhat_univariate/bivariate/trivariate,
    _rolling_mu, _shade, oos_cumret
  - Export shared constants (OOS_START, MIN_WIN, DELTAS, …)

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
from helpers import (
    load_vrp_series, load_es_front_month, load_vvix,
    compute_vvix_ma5, compute_vvix_ma10,
    load_vix_spot, load_vix_futures_term_structure,
    load_es_open_interest, load_vix_basis, compute_trend_quotient,
    compute_performance_stats,
)
from fh_replication.fh_replication import compute_vix_term_slope

# ─── Shared constants ─────────────────────────────────────────────────────────
OOS_START  = "2012-01-01"
MIN_WIN    = 500
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


# ─── Shared performance-stat window ───────────────────────────────────────────
# Plots rebase a strategy's cumulative curve to its first activation when that
# happens after 2020 (e.g. a signal that only fires from the 2020 crash on), and
# report SR/return/drawdown over that same window. stat_start() / window_stats()
# expose that one rule so printed statistics and plot legends are always computed
# over the identical window.

def _first_activation(sim):
    """First date with a nonzero position on/after OOS_START (None if never)."""
    pos    = sim["position"][sim.index >= OOS_START]
    active = pos[pos != 0]
    return active.index.min() if len(active) else None


def stat_start(sims, ref_index):
    """Common start date for performance statistics.

    `sims` is a single sim DataFrame or an iterable of them. Uses the EARLIEST
    post-OOS_START activation among them; if that activation is after 2020-01-01
    the window starts the trading day after it (located on `ref_index`) — so a
    signal that only activates in 2020 is scored from 2020 onward, exactly as the
    plots rebase the curve. Otherwise the window spans the full OOS period from
    OOS_START.
    """
    if isinstance(sims, pd.DataFrame):
        sims = [sims]
    acts = [a for a in (_first_activation(s) for s in sims) if a is not None]
    activation = min(acts) if acts else None
    if activation is not None and activation > pd.Timestamp("2020-01-01"):
        i = ref_index.searchsorted(activation)
        return ref_index[min(i + 1, len(ref_index) - 1)]
    return pd.Timestamp(OOS_START)


def window_stats(sim, label, ref_index, start=None):
    """compute_performance_stats over the shared stat window (see stat_start), so
    printed numbers and plot legends are computed identically. Pass an explicit
    `start` to reuse a window already computed for a group of sims."""
    if start is None:
        start = stat_start(sim, ref_index)
    return compute_performance_stats(sim[sim.index >= start], label)


# ─── Data ─────────────────────────────────────────────────────────────────────

def build_panel(vrp, es, vvix_ma5, vvix_ma10, vix_spot, vix_basis,
                term_slope, oi, trend_q):
    ret   = es["returns"]
    r2    = ret ** 2
    rv5   = np.sqrt(r2.rolling(5).mean()  * 252)
    rv22  = np.sqrt(r2.rolling(22).mean() * 252)
    panel = pd.DataFrame({
        "VP":            vrp["VP"],
        "vvix_ma5":      vvix_ma5,
        "vvix_ma10":     vvix_ma10,
        "vix":           vix_spot,
        "vix_basis":     vix_basis,
        "term_slope":    term_slope,
        "open_interest": oi,
        "trend_q":       trend_q,
        "vol_trend":     np.log(rv5 / rv22),
        "daily_ret":     ret,
    })
    panel["fwd_20d"]     = (ret + 1).rolling(20).apply(np.prod, raw=True).shift(-20) - 1
    panel["vrp_monthly"] = (panel["VP"].resample("ME").last()
                            .reindex(panel.index).ffill())
    return panel[panel.index >= "2006-03-06"]



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


def _yhat(panel, predictors, fwd_col, betas_df):
    """Dispatch to the uni/bi/trivariate y_hat helper based on # of predictors."""
    cols = [predictors] if isinstance(predictors, str) else list(predictors)
    if len(cols) == 1:
        return _yhat_univariate(panel, cols[0], fwd_col, betas_df)
    if len(cols) == 2:
        return _yhat_bivariate(panel, cols[0], cols[1], fwd_col, betas_df)
    if len(cols) == 3:
        return _yhat_trivariate(panel, cols[0], cols[1], cols[2], fwd_col, betas_df)
    raise ValueError(f"_yhat supports 1-3 predictors, got {len(cols)}")


# ─── R² metrics ───────────────────────────────────────────────────────────────

def in_sample_r2(panel, predictors, fwd_col):
    """In-sample R² of a single OLS fit over the entire training timeframe.

    ``predictors`` may be a single column name or a list of names.
    """
    cols = [predictors] if isinstance(predictors, str) else list(predictors)
    sub  = panel.dropna(subset=[*cols, fwd_col]).copy()
    X    = add_constant(sub[cols], has_constant="skip")
    res  = OLS(sub[fwd_col], X).fit()
    return float(res.rsquared)


def oos_cumulative_r2(panel, predictors, fwd_col, betas_df, oos_gap=OOS_GAP):
    """Cumulative out-of-sample R² (Campbell-Thompson).

        R²_oos = 1 - Σ(y - ŷ_model)² / Σ(y - ŷ_mean)²

    where ŷ_model are the expanding-window model predictions (from ``betas_df``)
    and ŷ_mean is the prevailing training-sample mean of ``fwd_col`` (an
    expanding mean lagged by ``oos_gap``, matching the regression's train/test
    split). A positive value means the model beats the historical-mean benchmark.
    """
    cols = [predictors] if isinstance(predictors, str) else list(predictors)
    sub  = panel.dropna(subset=[*cols, fwd_col]).copy()

    yhat = _yhat(panel, predictors, fwd_col, betas_df)
    idx  = yhat.index
    y    = sub.loc[idx, fwd_col]
    mu   = (sub[fwd_col].expanding(min_periods=1).mean()
            .shift(oos_gap).loc[idx])

    sse_model = float(((y - yhat) ** 2).sum())
    sse_bench = float(((y - mu)   ** 2).sum())
    return 1.0 - sse_model / sse_bench if sse_bench > 0 else float("nan")


def _rolling_mu(panel, fwd_col, oos_gap, rolling_window=500, predictor="VP"):
    """Rolling mean of ``fwd_col`` over ``rolling_window`` rows, lagged ``oos_gap``.

    ``predictor`` may be a single column name or a list of names. The dropna
    subset must match the calling strategy's own ``sub`` so the rolling window
    is built over the same row set (single predictor for univariate models,
    both predictors for bivariate models).
    """
    cols = [predictor] if isinstance(predictor, str) else list(predictor)
    sub  = panel.dropna(subset=[*cols, fwd_col]).copy()
    return (sub[fwd_col]
            .rolling(rolling_window, min_periods=1).mean()
            .shift(oos_gap).rename("rolling_mu"))


# ─── Main: pre-populate all caches ────────────────────────────────────────────

def main():
    print("=" * 72)
    print("  regressions.py — pre-compute beta / t-stat caches")
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
    vix_basis  = load_vix_basis()
    term_slope = compute_vix_term_slope(load_vix_futures_term_structure())
    oi         = load_es_open_interest()
    trend_q    = compute_trend_quotient(es)

    panel = build_panel(vrp, es, vvix_ma5, vvix_ma10, vix_spot,
                        vix_basis, term_slope, oi, trend_q)
    print(f"    {len(panel):,} obs  "
          f"[{panel.index.min().date()} - {panel.index.max().date()}]")

    FWD = "fwd_20d"

    # Every panel column is a univariate predictor except the forward-return target.
    NON_PREDICTORS = {FWD, "daily_ret"}
    predictors = [c for c in panel.columns if c not in NON_PREDICTORS]

    print("\n[2] Univariate models...")
    for predictor in predictors:
        print(f"  {predictor}")
        df  = compute_betas(panel, predictor, FWD, OOS_GAP, NW_LAGS)
        row = df.iloc[-1]
        is_r2  = in_sample_r2(panel, predictor, FWD)
        oos_r2 = oos_cumulative_r2(panel, predictor, FWD, df, OOS_GAP)
        print(f"    full-period [{df.index[-1].date()}]: "
              f"beta={row['beta']:+.6f}  t-stat={row['t_stat']:+.2f}")
        print(f"    IS R²={is_r2:+.4f}  cumulative OOS R²={oos_r2:+.4f}")

    print("\n[3] Bivariate models...")
    bivar_pairs = [
        ("VP", "vvix_ma5"),
        ("VP", "vvix_ma10"),
        ("VP", "term_slope"),
        ("VP", "open_interest"),
    ]
    for pred1, pred2 in bivar_pairs:
        print(f"  {pred1} + {pred2}")
        df  = compute_betas_bivariate(panel, pred1, pred2, FWD, OOS_GAP, NW_LAGS)
        row = df.iloc[-1]
        is_r2  = in_sample_r2(panel, [pred1, pred2], FWD)
        oos_r2 = oos_cumulative_r2(panel, [pred1, pred2], FWD, df, OOS_GAP)
        print(f"    full-period [{df.index[-1].date()}]: "
              f"{pred1}: beta={row['beta_1']:+.6f} t-stat={row['t_stat_1']:+.2f}  |  "
              f"{pred2}: beta={row['beta_2']:+.6f} t-stat={row['t_stat_2']:+.2f}")
        print(f"    IS R²={is_r2:+.4f}  cumulative OOS R²={oos_r2:+.4f}")

    print("\n[4] Trivariate model: VRP + VVIX MA5 + Term Slope...")
    triv = ("VP", "vvix_ma5", "term_slope")
    df   = compute_betas_trivariate(panel, *triv, FWD, OOS_GAP, NW_LAGS)
    row  = df.iloc[-1]
    is_r2  = in_sample_r2(panel, list(triv), FWD)
    oos_r2 = oos_cumulative_r2(panel, list(triv), FWD, df, OOS_GAP)
    print(f"    full-period [{df.index[-1].date()}]: "
          f"{triv[0]}: beta={row['beta_1']:+.6f} t-stat={row['t_stat_1']:+.2f}  |  "
          f"{triv[1]}: beta={row['beta_2']:+.6f} t-stat={row['t_stat_2']:+.2f}  |  "
          f"{triv[2]}: beta={row['beta_3']:+.6f} t-stat={row['t_stat_3']:+.2f}")
    print(f"    IS R²={is_r2:+.4f}  cumulative OOS R²={oos_r2:+.4f}")

    print("\n" + "=" * 72)
    print("  Done — all regression caches updated in output/regression_cache/")
    print("=" * 72)


if __name__ == "__main__":
    main()
