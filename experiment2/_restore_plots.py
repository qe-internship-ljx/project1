"""
_restore_plots.py  -- run once, then delete
Restores 5 accidentally deleted plots by reusing cached positions.
"""
import sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "bh_replication"))

import horizon_regression as hr
from experiment2 import (
    load_vrp_series, load_es_front_month, load_vvix, compute_vvix_ma5,
    load_vix_spot, load_vix_futures_term_structure,
    compute_buy_and_hold, simulate_strategy, compute_performance_stats,
)
from fh_replication.fh_replication import compute_vix_term_slope
from statsmodels.api import OLS, add_constant

OOS_START    = hr.OOS_START
MIN_WIN      = hr.MIN_WIN
T_THRESH     = hr.T_THRESH
DELTAS       = hr.DELTAS
DELTA_LBL    = hr.DELTA_LBL
CACHE_DIR    = hr.CACHE_DIR
OUTPUT       = hr.OUTPUT

# ── Load data (same as hr.main) ───────────────────────────────────────────
print("Loading data...")
vrp        = load_vrp_series()
es         = load_es_front_month()
vvix_raw   = load_vvix()
vvix_ma5   = compute_vvix_ma5(vvix_raw)
vix_spot   = load_vix_spot()
term_slope = compute_vix_term_slope(load_vix_futures_term_structure())
panel      = hr.build_panel(vrp, es, vvix_ma5, vix_spot, term_slope)
daily_ret  = panel["daily_ret"].dropna()
bah_pos    = compute_buy_and_hold(daily_ret)
bah_sim_   = simulate_strategy(bah_pos, daily_ret)
bah_st     = compute_performance_stats(bah_sim_[bah_sim_.index >= OOS_START], "BaH")
print(f"  Panel: {len(panel)} obs")

# ── run_ew_rolmu (loads from cache) ───────────────────────────────────────
def run_ew_rolmu(panel, predictor, fwd_col, oos_gap, nw_lags, delta,
                 rolling_window=500):
    from har_model import _nw_se
    tag = (f"pos_EW_rolmu_{predictor}_{fwd_col}_d{int(delta*10000)}bps"
           f"_t{int(T_THRESH*100)}_rw{rolling_window}_oos{OOS_START}.parquet")
    cache = CACHE_DIR / tag
    if cache.exists():
        return pd.read_parquet(cache).squeeze().rename(f"pos_rolmu_{predictor}_{delta}")
    sub = panel.dropna(subset=[predictor, fwd_col]).copy()
    N   = len(sub)
    pos = pd.Series(0.0, index=sub.index, name=f"pos_rolmu_{predictor}_{delta}")
    fwd = sub[fwd_col].values
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
        lo  = max(0, i - oos_gap - rolling_window)
        mu  = float(np.mean(fwd[lo : i - oos_gap]))
        test = sub.iloc[[i]][[predictor]].copy()
        test.insert(0, "const", 1.0)
        y_hat = float(res.predict(test).iloc[0])
        if   y_hat > mu + delta: pos.iloc[i] =  1.0
        elif y_hat < mu - delta: pos.iloc[i] = -1.0
    pos.to_frame().to_parquet(cache)
    return pos

# ── 1. symmetric_VVIX_MA5.png ─────────────────────────────────────────────
print("\n[1] symmetric_VVIX_MA5.png")
vvix_sims = {}
for di, (delta, lbl) in enumerate(zip(DELTAS, DELTA_LBL)):
    pos = hr.run_ew(panel, "vvix_ma5", "fwd_20d", oos_gap=20, nw_lags=20, delta=delta)
    sim = simulate_strategy(pos, daily_ret)
    st  = compute_performance_stats(sim[sim.index >= OOS_START], f"EW_VVIX20d_{lbl}")
    vvix_sims[di] = (st, sim)
    p = pos[pos.index >= OOS_START]
    print(f"  {lbl}: SR={st['sharpe']:+.3f}  L={float((p==1).mean())*100:.1f}%  "
          f"S={float((p==-1).mean())*100:.1f}%  F={float((p==0).mean())*100:.1f}%")
vvix_betas = hr.compute_betas(panel, "vvix_ma5", "fwd_20d", oos_gap=20, nw_lags=20)
out_vvix = OUTPUT / "expanding_window" / "VVIX MA5"
out_vvix.mkdir(parents=True, exist_ok=True)
hr.plot_2panel(
    pred_label="VVIX MA5", horizon_label="20-day",
    oos_gap=20, nw_lags=20,
    color_palette=["#3f007d", "#6a51a3", "#807dba", "#9e9ac8"],
    sim_dict=vvix_sims, betas_df=vvix_betas,
    bah_sim=bah_sim_, bah_st=bah_st,
    out_path=out_vvix / "symmetric_VVIX_MA5.png",
)

# ── 2. base_return_shift_VVIX_MA5.png ────────────────────────────────────
print("\n[2] base_return_shift_VVIX_MA5.png")
VVIX_ACT = pd.Timestamp("2006-03-06")
vvix_rolmu_sims = {}
for di, (delta, lbl) in enumerate(zip(DELTAS, DELTA_LBL)):
    pos = run_ew_rolmu(panel, "vvix_ma5", "fwd_20d", oos_gap=20, nw_lags=20, delta=delta)
    pos = pos.copy(); pos[pos.index < VVIX_ACT] = 0.0
    sim = simulate_strategy(pos, daily_ret)
    st  = compute_performance_stats(sim[sim.index >= OOS_START], f"EW_VVIX20d_rolmu_{lbl}")
    vvix_rolmu_sims[di] = (st, sim)
    p = pos[pos.index >= OOS_START]
    print(f"  {lbl}: SR={st['sharpe']:+.3f}  L={float((p==1).mean())*100:.1f}%  "
          f"S={float((p==-1).mean())*100:.1f}%  F={float((p==0).mean())*100:.1f}%")
hr.plot_2panel(
    pred_label="VVIX MA5", horizon_label="20-day",
    oos_gap=20, nw_lags=20,
    color_palette=["#3f007d", "#6a51a3", "#807dba", "#9e9ac8"],
    sim_dict=vvix_rolmu_sims, betas_df=vvix_betas,
    bah_sim=bah_sim_, bah_st=bah_st,
    out_path=out_vvix / "base_return_shift_VVIX_MA5.png",
)

# ── 3. symmetric_Vol_Trend.png ────────────────────────────────────────────
print("\n[3] symmetric_Vol_Trend.png")
vt_sims = {}
for di, (delta, lbl) in enumerate(zip(DELTAS, DELTA_LBL)):
    pos = hr.run_ew(panel, "vol_trend", "fwd_20d", oos_gap=20, nw_lags=20, delta=delta)
    sim = simulate_strategy(pos, daily_ret)
    st  = compute_performance_stats(sim[sim.index >= OOS_START], f"EW_VT20d_{lbl}")
    vt_sims[di] = (st, sim)
vt_betas = hr.compute_betas(panel, "vol_trend", "fwd_20d", oos_gap=20, nw_lags=20)
out_vt = OUTPUT / "expanding_window" / "poor_correlation" / "Vol Trend"
out_vt.mkdir(parents=True, exist_ok=True)
hr.plot_2panel(
    pred_label="Vol Trend [ln(RV5/RV22)]", horizon_label="20-day",
    oos_gap=20, nw_lags=20,
    color_palette=["#00441b", "#238b45", "#41ab5d", "#74c476"],
    sim_dict=vt_sims, betas_df=vt_betas,
    bah_sim=bah_sim_, bah_st=bah_st,
    out_path=out_vt / "symmetric_Vol_Trend.png",
)

# ── 4. symmetric_VIX.png ─────────────────────────────────────────────────
print("\n[4] symmetric_VIX.png")
vix_sims = {}
for di, (delta, lbl) in enumerate(zip(DELTAS, DELTA_LBL)):
    pos = hr.run_ew(panel, "vix", "fwd_20d", oos_gap=20, nw_lags=20, delta=delta)
    sim = simulate_strategy(pos, daily_ret)
    st  = compute_performance_stats(sim[sim.index >= OOS_START], f"EW_VIX20d_{lbl}")
    vix_sims[di] = (st, sim)
vix_betas = hr.compute_betas(panel, "vix", "fwd_20d", oos_gap=20, nw_lags=20)
out_vix = OUTPUT / "expanding_window" / "poor_correlation" / "VIX"
out_vix.mkdir(parents=True, exist_ok=True)
hr.plot_2panel(
    pred_label="VIX", horizon_label="20-day",
    oos_gap=20, nw_lags=20,
    color_palette=["#7f0000", "#cb181d", "#ef3b2c", "#fc9272"],
    sim_dict=vix_sims, betas_df=vix_betas,
    bah_sim=bah_sim_, bah_st=bah_st,
    out_path=out_vix / "symmetric_VIX.png",
)

# ── 5. symmetric_Term_Slope.png ───────────────────────────────────────────
print("\n[5] symmetric_Term_Slope.png")
ts_sims = {}
for di, (delta, lbl) in enumerate(zip(DELTAS, DELTA_LBL)):
    pos = hr.run_ew(panel, "term_slope", "fwd_20d", oos_gap=20, nw_lags=20, delta=delta)
    sim = simulate_strategy(pos, daily_ret)
    st  = compute_performance_stats(sim[sim.index >= OOS_START], f"EW_TS20d_{lbl}")
    ts_sims[di] = (st, sim)
ts_betas = hr.compute_betas(panel, "term_slope", "fwd_20d", oos_gap=20, nw_lags=20)
out_ts = OUTPUT / "expanding_window" / "poor_correlation" / "Term Slope"
out_ts.mkdir(parents=True, exist_ok=True)
hr.plot_2panel(
    pred_label="Term Slope", horizon_label="20-day",
    oos_gap=20, nw_lags=20,
    color_palette=["#7f2704", "#d94801", "#fd8d3c", "#fdbe85"],
    sim_dict=ts_sims, betas_df=ts_betas,
    bah_sim=bah_sim_, bah_st=bah_st,
    out_path=out_ts / "symmetric_Term_Slope.png",
)

print("\nDone.")
