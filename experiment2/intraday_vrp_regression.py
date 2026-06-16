"""
intraday_vrp_regression.py
===========================
Expanding-window OOS univariate VRP regression using the intraday-based
VRP series produced by intraday_experiment/experiment.py (500-day rolling
HAR on 5-min ES realized variance).

Methodology identical to horizon_regression.py VRP section, but the
predictor is the intraday VRP rather than the daily-return HAR VRP.

Outputs (saved per plot.md conventions):
  experiment2/output/expanding_window/Intraday VRP/
    symmetric_Intraday_VRP.png   — 4-delta symmetric threshold
    asymmetric_Intraday_VRP.png  — asymmetric threshold
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
CACHE_DIR = OUTPUT / "regression_cache_intraday"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT.parent / "bh_replication"))
sys.path.insert(0, str(ROOT))

from har_model import _nw_se
from experiment2 import (
    load_es_front_month,
    compute_buy_and_hold, simulate_strategy, compute_performance_stats,
)
from horizon_regression import (
    OOS_START, MIN_WIN, T_THRESH, DELTAS, DELTA_LBL, BAH_COLOR,
    oos_cumret, _shade, _stat_window,
    plot_2panel, plot_asym,
)

INTRADAY_VRP_CSV = ROOT.parent / "intraday_experiment" / "output" / "production_loop_intraday.csv"

VRP_COLOR_PALETTE = ["#08306b", "#2171b5", "#4292c6", "#6baed6"]
OOS_GAP  = 20
NW_LAGS  = 20


# ── Data loading ──────────────────────────────────────────────────────────────

def load_intraday_vrp() -> pd.Series:
    df = pd.read_csv(INTRADAY_VRP_CSV, parse_dates=["date"]).set_index("date")
    return df["VP"].rename("VP")


def build_panel(vrp: pd.Series, es) -> pd.DataFrame:
    ret   = es["returns"]
    panel = pd.DataFrame({
        "VP":        vrp,
        "daily_ret": ret,
    })
    for h in [20]:
        fwd = (ret + 1).rolling(h).apply(np.prod, raw=True).shift(-h) - 1
        panel[f"fwd_{h}d"] = fwd
    panel = panel.dropna(subset=["VP"])
    return panel


# ── Position computation ──────────────────────────────────────────────────────

def run_ew_intraday(panel, predictor, fwd_col, oos_gap, nw_lags, delta):
    """Symmetric threshold, cached under regression_cache_intraday/."""
    tag = (f"pos_EW_intraday_{predictor}_{fwd_col}_d{int(delta*10000)}bps"
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


def run_ew_asym_intraday(panel, predictor, fwd_col, oos_gap, nw_lags):
    """Asymmetric threshold, cached under regression_cache_intraday/."""
    tag = (f"pos_EWasym_intraday_{predictor}_{fwd_col}"
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


def compute_betas_intraday(panel, predictor, fwd_col, oos_gap, nw_lags):
    """Beta time series, cached under regression_cache_intraday/."""
    tag   = f"betas_EW_intraday_{predictor}_{fwd_col}_oos{OOS_START}.parquet"
    cache = CACHE_DIR / tag
    if cache.exists():
        return pd.read_parquet(cache)

    print(f"    Computing betas for intraday {predictor} -> {fwd_col}...")
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
        a     = float(res.params.iloc[0])
        b, se = float(res.params.iloc[1]), float(nw[1])
        records.append({"alpha": a, "beta": b, "se": se,
                        "t_stat": b / se if se > 0 else 0.0})

    df = pd.DataFrame(records, index=sub.index[start_i:])
    df.to_parquet(cache)
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("  Intraday VRP — Expanding Window Univariate Regression")
    print("  Symmetric (4 deltas) + Asymmetric")
    print("=" * 72)

    print("\n[1] Loading data...")
    vrp = load_intraday_vrp()
    es  = load_es_front_month()
    panel = build_panel(vrp, es)
    print(f"    Panel: {len(panel):,} obs  "
          f"[{panel.index.min().date()} - {panel.index.max().date()}]")
    print(f"    Intraday VRP: mean={panel['VP'].mean():.2f}  "
          f"std={panel['VP'].std():.2f}  "
          f"%>0={(panel['VP']>0).mean()*100:.1f}%")

    daily_ret = panel["daily_ret"].dropna()
    bah_pos   = compute_buy_and_hold(daily_ret)
    bah_sim   = simulate_strategy(bah_pos, daily_ret)

    out_dir = OUTPUT / "expanding_window" / "Intraday VRP"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Symmetric (4 deltas) ──────────────────────────────────────────────────
    print("\n[2] Symmetric expanding-window positions (4 deltas)...")
    vrp_sims = {}
    for di, (delta, lbl) in enumerate(zip(DELTAS, DELTA_LBL)):
        print(f"    {lbl}...", end="  ", flush=True)
        pos = run_ew_intraday(panel, "VP", "fwd_20d", oos_gap=OOS_GAP,
                              nw_lags=NW_LAGS, delta=delta)
        sim = simulate_strategy(pos, daily_ret)
        st  = compute_performance_stats(sim[sim.index >= OOS_START],
                                        f"EW_iVRP20d_{lbl}")
        vrp_sims[di] = (st, sim)
        p = pos[pos.index >= OOS_START]
        print(f"SR={st['sharpe']:+.3f}  "
              f"L={float((p==1).mean())*100:.1f}%  "
              f"S={float((p==-1).mean())*100:.1f}%  "
              f"F={float((p==0).mean())*100:.1f}%")

    betas_df = compute_betas_intraday(panel, "VP", "fwd_20d",
                                      oos_gap=OOS_GAP, nw_lags=NW_LAGS)
    print(f"    Betas computed: {len(betas_df):,} obs")

    plot_2panel(
        pred_label="Intraday VRP", horizon_label="20-day",
        oos_gap=OOS_GAP, nw_lags=NW_LAGS,
        color_palette=VRP_COLOR_PALETTE,
        sim_dict=vrp_sims, betas_df=betas_df,
        bah_sim=bah_sim,
        out_path=out_dir / "symmetric_Intraday_VRP.png",
    )

    # ── Asymmetric ────────────────────────────────────────────────────────────
    print("\n[3] Asymmetric expanding-window positions...")
    pos_asym = run_ew_asym_intraday(panel, "VP", "fwd_20d",
                                    oos_gap=OOS_GAP, nw_lags=NW_LAGS)
    sim_asym = simulate_strategy(pos_asym, daily_ret)
    p_asym   = pos_asym[pos_asym.index >= OOS_START]
    st_asym  = compute_performance_stats(sim_asym[sim_asym.index >= OOS_START],
                                         "asym_iVRP")
    print(f"    SR={st_asym['sharpe']:+.3f}  "
          f"L={float((p_asym==1).mean())*100:.1f}%  "
          f"S={float((p_asym==-1).mean())*100:.1f}%  "
          f"F={float((p_asym==0).mean())*100:.1f}%")

    plot_asym(
        pred_label="Intraday VRP", horizon_label="20-day",
        oos_gap=OOS_GAP, nw_lags=NW_LAGS,
        main_color=VRP_COLOR_PALETTE[0],
        sim=sim_asym, betas_df=betas_df,
        bah_sim=bah_sim,
        out_path=out_dir / "asymmetric_Intraday_VRP.png",
    )

    print("\nDone.")
    print(f"  {out_dir}/symmetric_Intraday_VRP.png")
    print(f"  {out_dir}/asymmetric_Intraday_VRP.png")
    print("=" * 72)


if __name__ == "__main__":
    main()
