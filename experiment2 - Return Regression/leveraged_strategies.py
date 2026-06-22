"""
leveraged_strategies.py
=======================
Leveraged multi-level position sizing variants for three models / three threshold
types.

Position levels (THRESHOLDS = 0.2 / 0.5 / 0.75 / 1.0 percent):
    |excess| >= 1.0%  -> +/-4
    |excess| >= 0.75% -> +/-3
    |excess| >= 0.5%  -> +/-2
    |excess| >= 0.2%  -> +/-1
    otherwise / gate  ->   0

Excess definitions:
  symmetric:           excess = y_hat                (reference = 0)
  asymmetric:          long:  excess = y_hat - mu    (enters only if >= 0.2%)
                       short: excess = -y_hat        (y_hat must be < 0, >= 0.2%)
  base_return_shift:   excess = y_hat - mu           (symmetric around rolling mu)

Models:
  VVIX MA5  -> 20-day (univariate)            -> output/plots/VVIX MA5/
  VVIX MA10 -> 20-day (univariate)            -> output/plots/VVIX MA10/
  VRP       -> 20-day (univariate)            -> output/plots/VRP/
  VRP + VVIX MA5 -> 20-day (bivariate)       -> output/plots/VRP + VVIX MA5/
  VRP + VVIX MA10 -> 20-day (bivariate)      -> output/plots/VRP + VVIX MA10/
  VRP + Term Slope -> 20-day (bivariate)     -> output/plots/VRP + Term Slope/
  VRP + Open Interest -> 20-day (bivariate)  -> output/plots/VRP + Open Interest/

Position functions are defined in this file; betas, y_hat helpers and constants
are imported from regressions.py.
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

ROOT      = Path(__file__).parent
OUTPUT    = ROOT / "output"
CACHE_DIR = OUTPUT / "regression_cache"

sys.path.insert(0, str(ROOT.parent / "bh_replication"))
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT))

from helpers import (
    load_vrp_series, load_es_front_month, load_vvix, compute_vvix_ma5,
    compute_vvix_ma10,
    load_vix_futures_term_structure, load_es_open_interest,
    compute_buy_and_hold, simulate_strategy, compute_performance_stats,
    compute_trend_quotient, build_master_panel,
)
from fh_replication.fh_replication import compute_vix_term_slope
from regressions import (
    compute_betas, compute_betas_bivariate,
    _yhat_univariate, _yhat_bivariate, _rolling_mu,
    _shade, oos_cumret, stat_start, window_stats,
    OOS_START, VVIX_ACT, OOS_GAP, NW_LAGS, RW, THRESHOLDS, LEVELS,
)
# Shared Buy-and-Hold / finalisation / t-stat panel drawing live in
# base_strategies.py (the canonical plotting module).
from base_strategies import _draw_bah, _finalize, draw_tstat_beta_panel

# |t|-stat gate threshold. Lives here (and in base_strategies.py) — not in
# regressions.py — so the --t-threshold flag can rebind it and have every plot
# title and the t-stat gate band reflect the chosen value.
T_THRESH = 1.65

# ─── Color constants ──────────────────────────────────────────────────────────
BAH_COLOR = "#d62728"

C_VVIX   = "#3f007d"   # VVIX MA5 — dark purple
C_VRP    = "#08306b"   # VRP      — dark blue
C_BIV    = "#3f007d"   # VRP+VVIX bivariate — purple
C_VRP2   = "#2171b5"   # second t-stat colour in bivariate panel
C_TERM   = "#00441b"   # VRP+Term Slope — dark green
C_TERM2  = "#238b45"   # second t-stat colour for term slope
C_OI     = "#54278f"   # VRP+Open Interest — dark violet
C_MA10   = "#7a0177"   # VVIX MA10 — medium purple
C_BIV10  = "#ae017e"   # VRP+VVIX MA10 bivariate — dark magenta
C_BIV102 = "#f768a1"   # second t-stat colour for VRP+VVIX MA10 bivariate


# ═══════════════════════════════════════════════════════════════════════════
# Shared panel-drawing helpers
# ═══════════════════════════════════════════════════════════════════════════

def _draw_cumret(ax, sim, bah_sim, main_color, label):
    """Panel 1. Returns (_stat_start, pL, pS, avg_pos)."""
    oos_dt      = pd.Timestamp(OOS_START)
    _stat_start = stat_start(sim, bah_sim.index)
    _rebased    = _stat_start > oos_dt
    _stat_lbl   = (f" stats from {_stat_start.strftime('%Y-%m-%d')}"
                   if _rebased else "")

    _draw_bah(ax, bah_sim, _stat_start, _stat_lbl,
              _stat_start if _rebased else None)

    st      = window_stats(sim, "ub", bah_sim.index, start=_stat_start)
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
    """Panel 2 — univariate. Delegates to the shared canonical panel."""
    return draw_tstat_beta_panel(ax, betas_df, [pred_label], [main_color], nw_lags,
                                 t_thresh=T_THRESH)


def _draw_tstat_biv(ax, betas_df, c1, c2, pred1_label, pred2_label, nw_lags):
    """Panel 2 — bivariate. Delegates to the shared canonical panel."""
    return draw_tstat_beta_panel(ax, betas_df, [pred1_label, pred2_label],
                                 [c1, c2], nw_lags, t_thresh=T_THRESH)


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

    sim_oos  = sim[sim.index >= OOS_START]
    prev_pos = sim_oos["position"].shift(1)
    long_mask  = prev_pos > 0
    short_mask = prev_pos < 0
    long_acc  = float((sim_oos.loc[long_mask,  "gross_pnl"] > 0).mean()) if long_mask.any()  else float("nan")
    short_acc = float((sim_oos.loc[short_mask, "gross_pnl"] > 0).mean()) if short_mask.any() else float("nan")
    long_acc_str  = f"{long_acc  * 100:.1f}%" if not np.isnan(long_acc)  else "N/A"
    short_acc_str = f"{short_acc * 100:.1f}%" if not np.isnan(short_acc) else "N/A"

    ax.text(0.01, 0.97,
            f"Long {pL:.1f}%  Short {pS:.1f}%  Flat {pF:.1f}%  AvgPos={avg_pos:+.3f}",
            transform=ax.transAxes, fontsize=7.5, va="top", color=main_color)
    ax.text(0.01, 0.83,
            f"Long accuracy={long_acc_str}  Short accuracy={short_acc_str}",
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


# ─── Leveraged: position-sizing helpers ─────────────────────────────────────────

def _level(excess_abs):
    """Position multiplier (1..4) for |excess| against THRESHOLDS; 0 if below all."""
    for thresh, lv in zip(THRESHOLDS, LEVELS):
        if excess_abs >= thresh:
            return lv
    return 0


def _signed_levels(excess):
    """Vectorized signed level: sign(excess) × _level(|excess|)."""
    return np.sign(excess) * excess.abs().apply(_level)


def _asym_levels(y_hat, mu):
    """Long: _level(ŷ-µ) when ŷ>µ; Short: -_level(-ŷ) when ŷ<0 (long wins ties)."""
    lv_long  = (y_hat - mu).clip(lower=0).apply(_level).astype(float)
    lv_short = (-y_hat).clip(lower=0).apply(_level).astype(float)
    out = pd.Series(0.0, index=y_hat.index)
    out = out.mask(lv_short > 0, -lv_short)
    out = out.mask(lv_long  > 0,  lv_long)
    return out


# ─── Leveraged: univariate ──────────────────────────────────────────────────────

def run_ew_leveraged_sym(panel, predictor, fwd_col, oos_gap, nw_lags, t_thresh=None):
    """Level based on |ŷ|."""
    if t_thresh is None:
        t_thresh = T_THRESH
    tag = (f"pos_EW_leveraged_sym_{predictor}_{fwd_col}"
           f"_t{int(t_thresh*100)}_oos{OOS_START}.parquet")
    cache = CACHE_DIR / tag
    if cache.exists():
        return pd.read_parquet(cache).squeeze().rename(f"pos_ubsym_{predictor}")

    sub      = panel.dropna(subset=[predictor, fwd_col]).copy()
    betas_df = compute_betas(panel, predictor, fwd_col, oos_gap, nw_lags)
    y_hat    = _yhat_univariate(panel, predictor, fwd_col, betas_df)
    fire     = betas_df["t_stat"].abs() > t_thresh

    pos = pd.Series(0.0, index=sub.index, name=f"pos_ubsym_{predictor}")
    idx = y_hat.index[fire.reindex(y_hat.index, fill_value=False)]
    pos.loc[idx] = _signed_levels(y_hat.loc[idx])

    pos.to_frame().to_parquet(cache)
    return pos


def run_ew_leveraged_asym(panel, predictor, fwd_col, oos_gap, nw_lags,
                        rolling_window=500, t_thresh=None):
    """Long: level by (ŷ-µ); short: level by |ŷ| when ŷ<0."""
    if t_thresh is None:
        t_thresh = T_THRESH
    tag = (f"pos_EW_leveraged_asym_{predictor}_{fwd_col}"
           f"_t{int(t_thresh*100)}_rw{rolling_window}_oos{OOS_START}.parquet")
    cache = CACHE_DIR / tag
    if cache.exists():
        return pd.read_parquet(cache).squeeze().rename(f"pos_ubasym_{predictor}")

    sub      = panel.dropna(subset=[predictor, fwd_col]).copy()
    betas_df = compute_betas(panel, predictor, fwd_col, oos_gap, nw_lags)
    y_hat    = _yhat_univariate(panel, predictor, fwd_col, betas_df)
    mu       = _rolling_mu(panel, fwd_col, oos_gap, rolling_window=rolling_window,
                           predictor=predictor)
    fire     = betas_df["t_stat"].abs() > t_thresh

    pos = pd.Series(0.0, index=sub.index, name=f"pos_ubasym_{predictor}")
    idx = y_hat.index[fire.reindex(y_hat.index, fill_value=False)]
    pos.loc[idx] = _asym_levels(y_hat.loc[idx], mu.reindex(idx))

    pos.to_frame().to_parquet(cache)
    return pos


def run_ew_leveraged_rolmu(panel, predictor, fwd_col, oos_gap, nw_lags,
                         rolling_window=500, t_thresh=None):
    """Level based on |ŷ - µ|."""
    if t_thresh is None:
        t_thresh = T_THRESH
    tag = (f"pos_EW_rolmu_leveraged_{predictor}_{fwd_col}"
           f"_t{int(t_thresh*100)}_rw{rolling_window}_oos{OOS_START}.parquet")
    cache = CACHE_DIR / tag
    if cache.exists():
        return pd.read_parquet(cache).squeeze().rename(f"pos_ubrolmu_{predictor}")

    sub      = panel.dropna(subset=[predictor, fwd_col]).copy()
    betas_df = compute_betas(panel, predictor, fwd_col, oos_gap, nw_lags)
    y_hat    = _yhat_univariate(panel, predictor, fwd_col, betas_df)
    mu       = _rolling_mu(panel, fwd_col, oos_gap, rolling_window=rolling_window,
                           predictor=predictor)
    fire     = betas_df["t_stat"].abs() > t_thresh

    pos = pd.Series(0.0, index=sub.index, name=f"pos_ubrolmu_{predictor}")
    idx = y_hat.index[fire.reindex(y_hat.index, fill_value=False)]
    pos.loc[idx] = _signed_levels(y_hat.loc[idx] - mu.reindex(idx))

    pos.to_frame().to_parquet(cache)
    return pos


# ─── Leveraged: bivariate ───────────────────────────────────────────────────────

def run_ew_biv_leveraged_sym(panel, pred1, pred2, fwd_col, oos_gap, nw_lags, t_thresh=None):
    if t_thresh is None:
        t_thresh = T_THRESH
    tag = (f"pos_EW_leveraged_sym_{pred1}_{pred2}_{fwd_col}"
           f"_t{int(t_thresh*100)}_oos{OOS_START}.parquet")
    cache = CACHE_DIR / tag
    if cache.exists():
        return pd.read_parquet(cache).squeeze().rename(f"pos_ubsym_{pred1}_{pred2}")

    sub      = panel.dropna(subset=[pred1, pred2, fwd_col]).copy()
    betas_df = compute_betas_bivariate(panel, pred1, pred2, fwd_col, oos_gap, nw_lags)
    y_hat    = _yhat_bivariate(panel, pred1, pred2, fwd_col, betas_df)
    fire     = ((betas_df["t_stat_1"].abs() > t_thresh)
                & (betas_df["t_stat_2"].abs() > t_thresh))

    pos = pd.Series(0.0, index=sub.index, name=f"pos_ubsym_{pred1}_{pred2}")
    idx = y_hat.index[fire.reindex(y_hat.index, fill_value=False)]
    pos.loc[idx] = _signed_levels(y_hat.loc[idx])

    pos.to_frame().to_parquet(cache)
    return pos


def run_ew_biv_leveraged_asym(panel, pred1, pred2, fwd_col, oos_gap, nw_lags,
                             rolling_window=500, t_thresh=None):
    if t_thresh is None:
        t_thresh = T_THRESH
    tag = (f"pos_EW_leveraged_asym_{pred1}_{pred2}_{fwd_col}"
           f"_t{int(t_thresh*100)}_rw{rolling_window}_oos{OOS_START}.parquet")
    cache = CACHE_DIR / tag
    if cache.exists():
        return pd.read_parquet(cache).squeeze().rename(f"pos_ubasym_{pred1}_{pred2}")

    sub      = panel.dropna(subset=[pred1, pred2, fwd_col]).copy()
    betas_df = compute_betas_bivariate(panel, pred1, pred2, fwd_col, oos_gap, nw_lags)
    y_hat    = _yhat_bivariate(panel, pred1, pred2, fwd_col, betas_df)
    mu       = _rolling_mu(panel, fwd_col, oos_gap, rolling_window=rolling_window,
                           predictor=[pred1, pred2])
    fire     = ((betas_df["t_stat_1"].abs() > t_thresh)
                & (betas_df["t_stat_2"].abs() > t_thresh))

    pos = pd.Series(0.0, index=sub.index, name=f"pos_ubasym_{pred1}_{pred2}")
    idx = y_hat.index[fire.reindex(y_hat.index, fill_value=False)]
    pos.loc[idx] = _asym_levels(y_hat.loc[idx], mu.reindex(idx))

    pos.to_frame().to_parquet(cache)
    return pos


def run_ew_biv_leveraged_rolmu(panel, pred1, pred2, fwd_col, oos_gap, nw_lags,
                              rolling_window=500, t_thresh=None):
    if t_thresh is None:
        t_thresh = T_THRESH
    tag = (f"pos_EW_rolmu_leveraged_{pred1}_{pred2}_{fwd_col}"
           f"_t{int(t_thresh*100)}_rw{rolling_window}_oos{OOS_START}.parquet")
    cache = CACHE_DIR / tag
    if cache.exists():
        return pd.read_parquet(cache).squeeze().rename(f"pos_ubrolmu_{pred1}_{pred2}")

    sub      = panel.dropna(subset=[pred1, pred2, fwd_col]).copy()
    betas_df = compute_betas_bivariate(panel, pred1, pred2, fwd_col, oos_gap, nw_lags)
    y_hat    = _yhat_bivariate(panel, pred1, pred2, fwd_col, betas_df)
    mu       = _rolling_mu(panel, fwd_col, oos_gap, rolling_window=rolling_window,
                           predictor=[pred1, pred2])
    fire     = ((betas_df["t_stat_1"].abs() > t_thresh)
                & (betas_df["t_stat_2"].abs() > t_thresh))

    pos = pd.Series(0.0, index=sub.index, name=f"pos_ubrolmu_{pred1}_{pred2}")
    idx = y_hat.index[fire.reindex(y_hat.index, fill_value=False)]
    pos.loc[idx] = _signed_levels(y_hat.loc[idx] - mu.reindex(idx))

    pos.to_frame().to_parquet(cache)
    return pos


def plot_leveraged_asymmetric_comparison(sim_vvix, sim_biv, sim_term, bah_sim, out_path,
                                        sim_ma10=None, sim_biv10=None):
    """One-panel comparison: leveraged-asymmetric VVIX MA5/MA10 vs VRP+VVIX MA5/MA10 vs VRP+Term Slope."""
    oos_dt = pd.Timestamp(OOS_START)

    def _first_active(sim):
        pos = sim["position"][sim.index >= oos_dt]
        active = pos[pos != 0]
        return active.index.min() if len(active) else None

    all_strategy_sims = [s for s in (sim_vvix, sim_biv, sim_term, sim_ma10, sim_biv10)
                         if s is not None]
    candidates = [d for d in [_first_active(s) for s in all_strategy_sims]
                  if d is not None]
    start = max(candidates) if candidates else oos_dt
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

    e_dt = max(s.index[-1] for s in all_strategy_sims)

    bah_cum   = _cum(bah_sim);  bah_st   = _stats(bah_sim,  "BaH")
    vvix_cum  = _cum(sim_vvix); vvix_st  = _stats(sim_vvix, "VVIX5_asym")
    biv_cum   = _cum(sim_biv);  biv_st   = _stats(sim_biv,  "BIV5_asym")
    term_cum  = _cum(sim_term); term_st  = _stats(sim_term, "TERM_asym")

    pL_v, pS_v, avg_v = _pos_pct(sim_vvix)
    pL_b, pS_b, avg_b = _pos_pct(sim_biv)
    pL_t, pS_t, avg_t = _pos_pct(sim_term)

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.suptitle(
        f"Leveraged Asymmetric: VVIX MA5 / MA10  vs  VRP + VVIX MA5 / MA10  vs  VRP + Term Slope  ·  Expanding Window\n"
        f"Rebased to 1.0 at {start_str} (latest first activation)  ·  |t| > {T_THRESH:.2f} gate  ·  "
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
            label=(f"VVIX MA5 (Leveraged Asym)  "
                   f"[SR={vvix_st['sharpe']:+.2f}  "
                   f"ret={vvix_st['ann_ret']*100:+.1f}%  "
                   f"DD={vvix_st['max_dd']*100:.1f}%  "
                   f"L{pL_v:.0f}%/S{pS_v:.0f}%  AvgPos={avg_v:+.2f}]"))

    ax.plot(biv_cum.index, biv_cum.values,
            color=C_BIV, lw=2.2, ls="--", alpha=0.92,
            label=(f"VRP + VVIX MA5 (Leveraged Asym)  "
                   f"[SR={biv_st['sharpe']:+.2f}  "
                   f"ret={biv_st['ann_ret']*100:+.1f}%  "
                   f"DD={biv_st['max_dd']*100:.1f}%  "
                   f"L{pL_b:.0f}%/S{pS_b:.0f}%  AvgPos={avg_b:+.2f}]"))

    ax.plot(term_cum.index, term_cum.values,
            color=C_TERM, lw=2.2, ls="-.", alpha=0.92,
            label=(f"VRP + Term Slope (Leveraged Asym)  "
                   f"[SR={term_st['sharpe']:+.2f}  "
                   f"ret={term_st['ann_ret']*100:+.1f}%  "
                   f"DD={term_st['max_dd']*100:.1f}%  "
                   f"L{pL_t:.0f}%/S{pS_t:.0f}%  AvgPos={avg_t:+.2f}]"))

    if sim_ma10 is not None:
        ma10_cum = _cum(sim_ma10); ma10_st = _stats(sim_ma10, "VVIX10_asym")
        pL_m, pS_m, avg_m = _pos_pct(sim_ma10)
        ax.plot(ma10_cum.index, ma10_cum.values,
                color=C_MA10, lw=2.2, ls=":", alpha=0.92,
                label=(f"VVIX MA10 (Leveraged Asym)  "
                       f"[SR={ma10_st['sharpe']:+.2f}  "
                       f"ret={ma10_st['ann_ret']*100:+.1f}%  "
                       f"DD={ma10_st['max_dd']*100:.1f}%  "
                       f"L{pL_m:.0f}%/S{pS_m:.0f}%  AvgPos={avg_m:+.2f}]"))

    if sim_biv10 is not None:
        biv10_cum = _cum(sim_biv10); biv10_st = _stats(sim_biv10, "BIV10_asym")
        pL_b10, pS_b10, avg_b10 = _pos_pct(sim_biv10)
        ax.plot(biv10_cum.index, biv10_cum.values,
                color=C_BIV10, lw=2.2, ls=(0, (5, 1)), alpha=0.92,
                label=(f"VRP + VVIX MA10 (Leveraged Asym)  "
                       f"[SR={biv10_st['sharpe']:+.2f}  "
                       f"ret={biv10_st['ann_ret']*100:+.1f}%  "
                       f"DD={biv10_st['max_dd']*100:.1f}%  "
                       f"L{pL_b10:.0f}%/S{pS_b10:.0f}%  AvgPos={avg_b10:+.2f}]"))

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
    "sym":   "Symmetric (Leveraged)",
    "asym":  "Asymmetric (Leveraged)",
    "rolmu": "Base-Return-Shift (Leveraged)",
}
_THRESH_DESC = {
    "sym":   "Long/Short +/-1..4 at |y_hat| >= 0.2/0.5/0.75/1.0%",
    "asym":  ("Long +1..4 at (y_hat-mu) >= 0.2..1.0%  |  "
              "Short -1..-4 at (-y_hat) >= 0.2..1.0%"),
    "rolmu": "Long/Short +/-1..4 at |y_hat - mu| >= 0.2/0.5/0.75/1.0%",
}


def plot_leveraged_univariate(pred_label, horizon_label, threshold_type,
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

    _, pL, pS, avg_pos = _draw_cumret(ax_ret, sim, bah_sim, main_color, "Leveraged +/-1..4")
    _draw_tstat_univ(ax_t, betas_df, main_color, pred_label, nw_lags)
    _draw_position(ax_p, sim, main_color, pL, pS, avg_pos)
    _draw_predicted(ax_yh, y_hat_ser, mu_ser, main_color, threshold_type)
    _finalize(fig, axes, out_path)


def plot_leveraged_bivariate(pred1_label, pred2_label, horizon_label, threshold_type,
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

    _, pL, pS, avg_pos = _draw_cumret(ax_ret, sim, bah_sim, main_color, "Leveraged +/-1..4")
    _draw_tstat_biv(ax_t, betas_df, C_VRP, C_VRP2, pred1_label, pred2_label, nw_lags)
    _draw_position(ax_p, sim, main_color, pL, pS, avg_pos)
    _draw_predicted(ax_yh, y_hat_ser, mu_ser, main_color, threshold_type)
    _finalize(fig, axes, out_path)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def _report(pos, daily_ret, label, bah_sim):
    """Print SR / position mix over the same activation window the plot uses
    (see stat_start), so printed stats match the figure legend."""
    sim   = simulate_strategy(pos, daily_ret)
    start = stat_start(sim, bah_sim.index)
    st    = window_stats(sim, label, bah_sim.index, start=start)
    p     = pos[pos.index >= start]
    from_lbl = (f" (from {start.strftime('%Y-%m-%d')})"
                if start > pd.Timestamp(OOS_START) else "")
    print(f"    {label}{from_lbl}: SR={st['sharpe']:+.3f}  "
          f"L%={(p>0).mean()*100:.1f}  S%={(p<0).mean()*100:.1f}  "
          f"F%={(p==0).mean()*100:.1f}  AvgPos={p.mean():+.3f}")
    return sim


def main(t_threshold=None):
    """Run all leveraged-strategy simulations/plots.

    `t_threshold` overrides the |t|-stat gate used by every simulation (default
    T_THRESH = 1.65). It rebinds the module-level T_THRESH so plot titles and the
    per-strategy cache keys reflect the chosen value; cache files encode the
    threshold, so different thresholds never collide.
    """
    if t_threshold is not None:
        global T_THRESH
        T_THRESH = t_threshold

    print("=" * 72)
    print("  Leveraged (Leveraged) Strategies")
    print("  VVIX MA5 / VVIX MA10 / VRP / VRP+VVIX MA5 / VRP+VVIX MA10 /")
    print("  VRP+Term Slope / VRP+Open Interest  x  sym / asym / rolmu")
    print(f"  |t| gate threshold = {T_THRESH:.2f}")
    print("=" * 72)

    print("\n[1] Loading data...")
    vrp        = load_vrp_series()
    es         = load_es_front_month()
    vvix_raw   = load_vvix()
    vvix_ma5   = compute_vvix_ma5(vvix_raw)
    vvix_ma10  = compute_vvix_ma10(vvix_raw)
    term_slope = compute_vix_term_slope(load_vix_futures_term_structure())
    trend_q    = compute_trend_quotient(es)
    oi         = load_es_open_interest()
    panel      = build_master_panel(vrp, es, term_slope, trend_q, vvix_ma5)
    panel      = panel[panel.index >= "2006-03-06"].copy()
    panel["open_interest"] = oi.reindex(panel.index)
    panel["vvix_ma10"]    = vvix_ma10.reindex(panel.index)
    FWD        = "fwd_20d"
    print(f"    {len(panel):,} obs  "
          f"[{panel.index.min().date()} - {panel.index.max().date()}]")

    daily_ret = panel["daily_ret"].dropna()
    bah_sim   = simulate_strategy(compute_buy_and_hold(daily_ret), daily_ret)

    print("\n[2] Loading / computing betas (cached)...")
    betas_vvix  = compute_betas(panel, "vvix_ma5",  FWD, oos_gap=OOS_GAP, nw_lags=NW_LAGS)
    betas_ma10  = compute_betas(panel, "vvix_ma10", FWD, oos_gap=OOS_GAP, nw_lags=NW_LAGS)
    betas_vrp   = compute_betas(panel, "VP",        FWD, oos_gap=OOS_GAP, nw_lags=NW_LAGS)
    betas_biv   = compute_betas_bivariate(panel, "VP", "vvix_ma5",      FWD,
                                          oos_gap=OOS_GAP, nw_lags=NW_LAGS)
    betas_biv10 = compute_betas_bivariate(panel, "VP", "vvix_ma10",     FWD,
                                          oos_gap=OOS_GAP, nw_lags=NW_LAGS)
    betas_term  = compute_betas_bivariate(panel, "VP", "term_slope",    FWD,
                                          oos_gap=OOS_GAP, nw_lags=NW_LAGS)
    betas_oi    = compute_betas_bivariate(panel, "VP", "open_interest", FWD,
                                          oos_gap=OOS_GAP, nw_lags=NW_LAGS)
    print("    Done.")

    yh_vvix  = _yhat_univariate(panel, "vvix_ma5",  FWD, betas_vvix)
    yh_ma10  = _yhat_univariate(panel, "vvix_ma10", FWD, betas_ma10)
    yh_vrp   = _yhat_univariate(panel, "VP",        FWD, betas_vrp)
    yh_biv   = _yhat_bivariate(panel, "VP", "vvix_ma5",      FWD, betas_biv)
    yh_biv10 = _yhat_bivariate(panel, "VP", "vvix_ma10",     FWD, betas_biv10)
    yh_term  = _yhat_bivariate(panel, "VP", "term_slope",    FWD, betas_term)
    yh_oi    = _yhat_bivariate(panel, "VP", "open_interest", FWD, betas_oi)
    mu_20d   = _rolling_mu(panel, FWD, OOS_GAP, RW)

    out_vvix  = OUTPUT / "plots" / "VVIX MA5"
    out_ma10  = OUTPUT / "plots" / "VVIX MA10"
    out_vrp   = OUTPUT / "plots" / "VRP"
    out_biv   = OUTPUT / "plots" / "VRP + VVIX MA5"
    out_biv10 = OUTPUT / "plots" / "VRP + VVIX MA10"
    out_term  = OUTPUT / "plots" / "VRP + Term Slope"
    out_oi    = OUTPUT / "plots" / "VRP + Open Interest"
    for d in (out_vvix, out_ma10, out_vrp, out_biv, out_biv10, out_term, out_oi):
        d.mkdir(parents=True, exist_ok=True)

    # ── Strategy specs: (threshold_type, filename prefix, run_fn) ──
    # All three thresholds share one skeleton: run → (VVIX flat-mask) → report → plot.
    uni_strats = [
        ("sym",   "symmetric",         run_ew_leveraged_sym),
        ("asym",  "asymmetric",        run_ew_leveraged_asym),
        ("rolmu", "base_return_shift", run_ew_leveraged_rolmu),
    ]
    biv_strats = [
        ("sym",   "symmetric",         run_ew_biv_leveraged_sym),
        ("asym",  "asymmetric",        run_ew_biv_leveraged_asym),
        ("rolmu", "base_return_shift", run_ew_biv_leveraged_rolmu),
    ]

    sims = {}   # (model label, threshold_type) → simulated DataFrame

    # ── Univariate models: (label, column, betas, y_hat, color, out_dir, flat) ──
    # `flat` masks positions before VVIX activation (2006-03-06).
    uni_models = [
        ("VVIX MA5",  "vvix_ma5",  betas_vvix, yh_vvix, C_VVIX, out_vvix, True),
        ("VVIX MA10", "vvix_ma10", betas_ma10, yh_ma10, C_MA10, out_ma10, True),
        ("VRP",       "VP",        betas_vrp,  yh_vrp,  C_VRP,  out_vrp,  False),
    ]
    for label, col, betas, yhat, color, out_dir, flat in uni_models:
        for thr, prefix, run_fn in uni_strats:
            print(f"\n{label} leveraged {prefix}...")
            pos   = run_fn(panel, col, FWD, OOS_GAP, NW_LAGS)
            extra = ""
            if flat:
                pos = pos.copy(); pos[pos.index < VVIX_ACT] = 0.0
                extra = "\nFlat before VVIX activation (2006-03-06)"
            sim = _report(pos, daily_ret, f"ub_{thr}_{label.replace(' ', '_')}", bah_sim)
            sims[(label, thr)] = sim
            plot_leveraged_univariate(
                label, "20-day", thr, color,
                sim, betas, bah_sim, yhat, mu_20d,
                out_dir / f"leveraged_{prefix}_{label.replace(' ', '_')}.png",
                extra_title=extra,
            )

    # ── Bivariate models: (label2, column2, betas, y_hat, color, out_dir) ──
    biv_models = [
        ("VVIX MA5",      "vvix_ma5",      betas_biv,   yh_biv,   C_BIV,   out_biv),
        ("VVIX MA10",     "vvix_ma10",     betas_biv10, yh_biv10, C_BIV10, out_biv10),
        ("Term Slope",    "term_slope",    betas_term,  yh_term,  C_TERM,  out_term),
        ("Open Interest", "open_interest", betas_oi,    yh_oi,    C_OI,    out_oi),
    ]
    for label2, col2, betas, yhat, color, out_dir in biv_models:
        for thr, prefix, run_fn in biv_strats:
            print(f"\nVRP+{label2} leveraged {prefix}...")
            pos = run_fn(panel, "VP", col2, FWD, OOS_GAP, NW_LAGS)
            sim = _report(pos, daily_ret, f"ub_{thr}_VRP_{col2}", bah_sim)
            sims[(f"VRP + {label2}", thr)] = sim
            fname = f"leveraged_{prefix}_VRP_+_{label2.replace(' ', '_')}.png"
            plot_leveraged_bivariate(
                "VRP", label2, "20-day", thr, color,
                sim, betas, bah_sim, yhat, mu_20d,
                out_dir / fname,
            )

    # ── Comparison: leveraged-asymmetric VVIX MA5/MA10 vs VRP+VVIX MA5/MA10 ────
    print("\nLeveraged asymmetric VVIX MA5/MA10 vs VRP+VVIX MA5/MA10...")
    out_cmp = OUTPUT / "plots" / "comparisons"
    plot_leveraged_asymmetric_comparison(
        sims[("VVIX MA5", "asym")], sims[("VRP + VVIX MA5", "asym")],
        sims[("VRP + Term Slope", "asym")], bah_sim,
        out_cmp / "leveraged_asymmetric_vvix_vs_vrp_vvix.png",
        sim_ma10=sims[("VVIX MA10", "asym")],
        sim_biv10=sims[("VRP + VVIX MA10", "asym")],
    )

    print("\nDone.")
    print("=" * 72)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Leveraged multi-level position-sizing strategy evaluation.")
    parser.add_argument(
        "--t", type=float, default=None, metavar="T",
        help=f"|t|-stat gate threshold applied to every simulation "
             f"(default: {T_THRESH:.2f}).")
    args = parser.parse_args()
    main(t_threshold=args.t)
