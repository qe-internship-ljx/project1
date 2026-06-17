"""
intraday_vrp_vvix_regression.py
================================
Expanding-window OOS bivariate regression: Intraday VRP + VVIX MA5 -> 20-day forward return.

Intraday VRP: 500-day rolling HAR on 5-min ES realized variance
              (from intraday_experiment/output/production_loop_intraday.csv).

Training period: first available intraday VRP through end of 2020.
OOS evaluation: 2021-01-01 onwards with 20-day gap.
Gate: |t| > 1.28 (both betas must pass for bivariate).

Outputs (saved per plot.md conventions):
  experiment2/output/expanding_window/Intraday VRP + VVIX MA5/
    symmetric_Intraday_VRP_+_VVIX_MA5.png   — 4-delta symmetric threshold
    asymmetric_Intraday_VRP_+_VVIX_MA5.png  — asymmetric threshold
"""

import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

from statsmodels.api import OLS, add_constant

ROOT      = Path(__file__).parent
OUTPUT    = ROOT / "output"
CACHE_DIR = OUTPUT / "regression_cache_intraday"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT.parent / "bh_replication"))
sys.path.insert(0, str(ROOT))

from har_model import _nw_se
from experiment2 import (
    load_es_front_month, load_vvix, compute_vvix_ma5,
    compute_buy_and_hold, simulate_strategy, compute_performance_stats,
)
from horizon_regression import (
    MIN_WIN, T_THRESH, DELTAS, DELTA_LBL, BAH_COLOR,
    _shade, _stat_window,
    plot_2panel_bivariate, plot_asym_bivariate,
)
from intraday_vrp_regression import load_intraday_vrp

OOS_START = "2021-01-01"  # train through 2020, OOS from 2021
OOS_GAP   = 20
NW_LAGS   = 20

COLOR_PALETTE = ["#1b5e20", "#2e7d32", "#43a047", "#81c784"]
MAIN_COLOR    = "#1b5e20"


# ── Data loading and panel construction ───────────────────────────────────────

def build_panel(vrp: pd.Series, es, vvix_ma5: pd.Series) -> pd.DataFrame:
    ret   = es["returns"]
    panel = pd.DataFrame({
        "VP":        vrp,
        "vvix_ma5":  vvix_ma5,
        "daily_ret": ret,
    })
    panel["fwd_20d"] = (ret + 1).rolling(20).apply(np.prod, raw=True).shift(-20) - 1
    panel = panel.dropna(subset=["VP", "vvix_ma5"])
    return panel


# ── Position computation: bivariate symmetric ─────────────────────────────────

def run_ew_biv_intraday(panel, pred1, pred2, fwd_col, oos_gap, nw_lags, delta):
    """Bivariate symmetric threshold; both betas must pass |t| > T_THRESH."""
    tag = (f"pos_EWbiv_intraday_{pred1}_{pred2}_{fwd_col}_d{int(delta*10000)}bps"
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


# ── Position computation: bivariate asymmetric ────────────────────────────────

def run_ew_asym_biv_intraday(panel, pred1, pred2, fwd_col, oos_gap, nw_lags):
    """Bivariate asymmetric: long if ŷ > μ₅₀₀, short if ŷ < 0. Both betas gated."""
    tag = (f"pos_EWasym_biv_intraday_{pred1}_{pred2}_{fwd_col}"
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


# ── Beta time series: bivariate ────────────────────────────────────────────────

def compute_betas_biv_intraday(panel, pred1, pred2, fwd_col, oos_gap, nw_lags):
    tag   = f"betas_EWbiv_intraday_{pred1}_{pred2}_{fwd_col}_oos{OOS_START}.parquet"
    cache = CACHE_DIR / tag
    if cache.exists():
        return pd.read_parquet(cache)

    print(f"    Computing bivariate betas: intraday {pred1} + {pred2} -> {fwd_col}...")
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
        a0    = float(res.params.iloc[0])
        b1, se1 = float(res.params.iloc[1]), float(nw[1])
        b2, se2 = float(res.params.iloc[2]), float(nw[2])
        records.append({
            "alpha":    a0,
            "beta_1":   b1,  "se_1":   se1,
            "t_stat_1": b1 / se1 if se1 > 0 else 0.0,
            "beta_2":   b2,  "se_2":   se2,
            "t_stat_2": b2 / se2 if se2 > 0 else 0.0,
        })

    df = pd.DataFrame(records, index=sub.index[start_i:])
    df.to_parquet(cache)
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("  Intraday VRP + VVIX MA5 — Expanding Window Bivariate Regression")
    print(f"  Training: first available through 2020 | OOS: {OOS_START} onward")
    print("  Symmetric (4 deltas) + Asymmetric")
    print("=" * 72)

    print("\n[1] Loading data...")
    vrp      = load_intraday_vrp()
    es       = load_es_front_month()
    vvix_ma5 = compute_vvix_ma5(load_vvix())
    panel    = build_panel(vrp, es, vvix_ma5)
    print(f"    Panel: {len(panel):,} obs  "
          f"[{panel.index.min().date()} - {panel.index.max().date()}]")
    train_n = (panel.index < pd.Timestamp(OOS_START)).sum()
    oos_n   = (panel.index >= pd.Timestamp(OOS_START)).sum()
    print(f"    Training obs (before {OOS_START}): {train_n:,}  |  OOS obs: {oos_n:,}")

    daily_ret = panel["daily_ret"].dropna()
    bah_pos   = compute_buy_and_hold(daily_ret)
    bah_sim   = simulate_strategy(bah_pos, daily_ret)

    out_dir = OUTPUT / "expanding_window" / "Intraday VRP + VVIX MA5"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Symmetric (4 deltas) ──────────────────────────────────────────────────
    print("\n[2] Symmetric expanding-window bivariate positions (4 deltas)...")
    biv_sims = {}
    for di, (delta, lbl) in enumerate(zip(DELTAS, DELTA_LBL)):
        print(f"    {lbl}...", end="  ", flush=True)
        pos = run_ew_biv_intraday(panel, "VP", "vvix_ma5", "fwd_20d",
                                  oos_gap=OOS_GAP, nw_lags=NW_LAGS, delta=delta)
        sim = simulate_strategy(pos, daily_ret)
        st  = compute_performance_stats(sim[sim.index >= OOS_START],
                                        f"EWbiv_iVRP_VVIX_{lbl}")
        biv_sims[di] = (st, sim)
        p = pos[pos.index >= OOS_START]
        print(f"SR={st['sharpe']:+.3f}  "
              f"L={float((p==1).mean())*100:.1f}%  "
              f"S={float((p==-1).mean())*100:.1f}%  "
              f"F={float((p==0).mean())*100:.1f}%")

    betas_df = compute_betas_biv_intraday(panel, "VP", "vvix_ma5", "fwd_20d",
                                          oos_gap=OOS_GAP, nw_lags=NW_LAGS)
    print(f"    Bivariate betas computed: {len(betas_df):,} obs")

    plot_2panel_bivariate(
        pred1_label="Intraday VRP", pred2_label="VVIX MA5",
        horizon_label="20-day",
        oos_gap=OOS_GAP, nw_lags=NW_LAGS,
        color_palette=COLOR_PALETTE,
        sim_dict=biv_sims, betas_df=betas_df,
        bah_sim=bah_sim,
        out_path=out_dir / "symmetric_Intraday_VRP_+_VVIX_MA5.png",
        oos_start=OOS_START,
    )

    # ── Asymmetric ────────────────────────────────────────────────────────────
    print("\n[3] Asymmetric expanding-window bivariate positions...")
    pos_asym = run_ew_asym_biv_intraday(panel, "VP", "vvix_ma5", "fwd_20d",
                                         oos_gap=OOS_GAP, nw_lags=NW_LAGS)
    sim_asym = simulate_strategy(pos_asym, daily_ret)
    p_asym   = pos_asym[pos_asym.index >= OOS_START]
    st_asym  = compute_performance_stats(sim_asym[sim_asym.index >= OOS_START],
                                         "asym_biv_iVRP_VVIX")
    print(f"    SR={st_asym['sharpe']:+.3f}  "
          f"L={float((p_asym==1).mean())*100:.1f}%  "
          f"S={float((p_asym==-1).mean())*100:.1f}%  "
          f"F={float((p_asym==0).mean())*100:.1f}%")

    plot_asym_bivariate(
        pred1_label="Intraday VRP", pred2_label="VVIX MA5",
        horizon_label="20-day",
        oos_gap=OOS_GAP, nw_lags=NW_LAGS,
        main_color=MAIN_COLOR,
        sim=sim_asym, betas_df=betas_df,
        bah_sim=bah_sim,
        out_path=out_dir / "asymmetric_Intraday_VRP_+_VVIX_MA5.png",
        oos_start=OOS_START,
    )

    print("\nDone.")
    print(f"  {out_dir}/symmetric_Intraday_VRP_+_VVIX_MA5.png")
    print(f"  {out_dir}/asymmetric_Intraday_VRP_+_VVIX_MA5.png")
    print("=" * 72)


if __name__ == "__main__":
    main()
