"""
leveraged_strategies.py
=======================
Unbound multi-level position sizing variants for three models / three threshold
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
  VVIX MA5  -> 20-day (univariate)            -> output/expanding_window/VVIX MA5/
  VVIX MA10 -> 20-day (univariate)            -> output/expanding_window/VVIX MA10/
  VRP       -> 20-day (univariate)            -> output/expanding_window/VRP/
  VRP + VVIX MA5 -> 20-day (bivariate)       -> output/expanding_window/VRP + VVIX MA5/
  VRP + VVIX MA10 -> 20-day (bivariate)      -> output/expanding_window/VRP + VVIX MA10/
  VRP + Term Slope -> 20-day (bivariate)     -> output/expanding_window/VRP + Term Slope/
  VRP + Open Interest -> 20-day (bivariate)  -> output/expanding_window/VRP + Open Interest/
  VRP + VVIX MA5 + Term Slope (trivariate)   -> output/expanding_window/trivariate/

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
    load_vix_spot, load_vix_futures_term_structure, load_es_open_interest,
    compute_buy_and_hold, simulate_strategy, compute_performance_stats,
    compute_trend_quotient, build_master_panel,
)
from fh_replication.fh_replication import compute_vix_term_slope
from regressions import (
    build_panel,
    compute_betas, compute_betas_bivariate, compute_betas_trivariate,
    _yhat_univariate, _yhat_bivariate, _yhat_trivariate, _rolling_mu,
    _shade, oos_cumret, stat_start, window_stats, draw_tstat_beta_panel,
    OOS_START, T_THRESH, VVIX_ACT, OOS_GAP, NW_LAGS, RW, THRESHOLDS, LEVELS,
)

# ─── Color constants ──────────────────────────────────────────────────────────
BAH_COLOR = "#d62728"

C_VVIX   = "#3f007d"   # VVIX MA5 — dark purple
C_VRP    = "#08306b"   # VRP      — dark blue
C_BIV    = "#3f007d"   # VRP+VVIX bivariate — purple
C_VRP2   = "#2171b5"   # second t-stat colour in bivariate panel
C_TERM   = "#00441b"   # VRP+Term Slope — dark green
C_TERM2  = "#238b45"   # second t-stat colour for term slope
C_OI     = "#54278f"   # VRP+Open Interest — dark violet
C_TRIV   = "#2e1503"   # VRP+VVIX MA5+Term Slope — dark brown
C_TRIV2  = "#7f3c00"   # second t-stat colour for trivariate
C_TRIV3  = "#c45c00"   # third  t-stat colour for trivariate
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

    _bah_st = window_stats(bah_sim, "BaH", bah_sim.index, start=_stat_start)
    ax.plot(oos_cumret(bah_sim).index,
            oos_cumret(bah_sim).values,
            color=BAH_COLOR, lw=1.5, ls="-.", alpha=0.6,
            label=(f"Buy-and-Hold{_stat_lbl}  "
                   f"[SR={_bah_st['sharpe']:+.2f}  "
                   f"ret={_bah_st['ann_ret']*100:+.1f}%  "
                   f"DD={_bah_st['max_dd']*100:.1f}%]"))

    if _rebased:
        _bah_from   = bah_sim["net_pnl"][bah_sim.index >= _stat_start]
        _bah_act    = (1 + _bah_from).cumprod()
        _bah_act_st = window_stats(bah_sim, "BaH2", bah_sim.index, start=_stat_start)
        ax.plot(_bah_act.index, _bah_act.values,
                color=BAH_COLOR, lw=1.2, ls=":", alpha=0.85,
                label=(f"Buy-and-Hold from {_stat_start.strftime('%Y-%m-%d')}  "
                       f"[SR={_bah_act_st['sharpe']:+.2f}  "
                       f"ret={_bah_act_st['ann_ret']*100:+.1f}%  "
                       f"DD={_bah_act_st['max_dd']*100:.1f}%]"))

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
    return draw_tstat_beta_panel(ax, betas_df, [pred_label], [main_color], nw_lags)


def _draw_tstat_biv(ax, betas_df, c1, c2, pred1_label, pred2_label, nw_lags):
    """Panel 2 — bivariate. Delegates to the shared canonical panel."""
    return draw_tstat_beta_panel(ax, betas_df, [pred1_label, pred2_label],
                                 [c1, c2], nw_lags)


def _draw_tstat_triv(ax, betas_df, c1, c2, c3, pred1_label, pred2_label, pred3_label, nw_lags):
    """Panel 2 — trivariate. Delegates to the shared canonical panel."""
    return draw_tstat_beta_panel(ax, betas_df,
                                 [pred1_label, pred2_label, pred3_label],
                                 [c1, c2, c3], nw_lags)


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


def _finalize(fig, axes, out_path):
    for ax in axes:
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.tick_params(axis="x", which="major", labelbottom=True, labelsize=7, pad=2)
    fig.savefig(out_path, dpi=155, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ─── Unbound: position-sizing helpers ─────────────────────────────────────────

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


# ─── Unbound: univariate ──────────────────────────────────────────────────────

def run_ew_unbound_sym(panel, predictor, fwd_col, oos_gap, nw_lags):
    """Level based on |ŷ|."""
    tag = (f"pos_EW_unbound_sym_{predictor}_{fwd_col}"
           f"_t{int(T_THRESH*100)}_oos{OOS_START}.parquet")
    cache = CACHE_DIR / tag
    if cache.exists():
        return pd.read_parquet(cache).squeeze().rename(f"pos_ubsym_{predictor}")

    sub      = panel.dropna(subset=[predictor, fwd_col]).copy()
    betas_df = compute_betas(panel, predictor, fwd_col, oos_gap, nw_lags)
    y_hat    = _yhat_univariate(panel, predictor, fwd_col, betas_df)
    fire     = betas_df["t_stat"].abs() > T_THRESH

    pos = pd.Series(0.0, index=sub.index, name=f"pos_ubsym_{predictor}")
    idx = y_hat.index[fire.reindex(y_hat.index, fill_value=False)]
    pos.loc[idx] = _signed_levels(y_hat.loc[idx])

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

    sub      = panel.dropna(subset=[predictor, fwd_col]).copy()
    betas_df = compute_betas(panel, predictor, fwd_col, oos_gap, nw_lags)
    y_hat    = _yhat_univariate(panel, predictor, fwd_col, betas_df)
    mu       = _rolling_mu(panel, fwd_col, oos_gap, rolling_window=rolling_window,
                           predictor=predictor)
    fire     = betas_df["t_stat"].abs() > T_THRESH

    pos = pd.Series(0.0, index=sub.index, name=f"pos_ubasym_{predictor}")
    idx = y_hat.index[fire.reindex(y_hat.index, fill_value=False)]
    pos.loc[idx] = _asym_levels(y_hat.loc[idx], mu.reindex(idx))

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

    sub      = panel.dropna(subset=[predictor, fwd_col]).copy()
    betas_df = compute_betas(panel, predictor, fwd_col, oos_gap, nw_lags)
    y_hat    = _yhat_univariate(panel, predictor, fwd_col, betas_df)
    mu       = _rolling_mu(panel, fwd_col, oos_gap, rolling_window=rolling_window,
                           predictor=predictor)
    fire     = betas_df["t_stat"].abs() > T_THRESH

    pos = pd.Series(0.0, index=sub.index, name=f"pos_ubrolmu_{predictor}")
    idx = y_hat.index[fire.reindex(y_hat.index, fill_value=False)]
    pos.loc[idx] = _signed_levels(y_hat.loc[idx] - mu.reindex(idx))

    pos.to_frame().to_parquet(cache)
    return pos


# ─── Unbound: bivariate ───────────────────────────────────────────────────────

def run_ew_biv_unbound_sym(panel, pred1, pred2, fwd_col, oos_gap, nw_lags):
    tag = (f"pos_EW_unbound_sym_{pred1}_{pred2}_{fwd_col}"
           f"_t{int(T_THRESH*100)}_oos{OOS_START}.parquet")
    cache = CACHE_DIR / tag
    if cache.exists():
        return pd.read_parquet(cache).squeeze().rename(f"pos_ubsym_{pred1}_{pred2}")

    sub      = panel.dropna(subset=[pred1, pred2, fwd_col]).copy()
    betas_df = compute_betas_bivariate(panel, pred1, pred2, fwd_col, oos_gap, nw_lags)
    y_hat    = _yhat_bivariate(panel, pred1, pred2, fwd_col, betas_df)
    fire     = ((betas_df["t_stat_1"].abs() > T_THRESH)
                & (betas_df["t_stat_2"].abs() > T_THRESH))

    pos = pd.Series(0.0, index=sub.index, name=f"pos_ubsym_{pred1}_{pred2}")
    idx = y_hat.index[fire.reindex(y_hat.index, fill_value=False)]
    pos.loc[idx] = _signed_levels(y_hat.loc[idx])

    pos.to_frame().to_parquet(cache)
    return pos


def run_ew_biv_unbound_asym(panel, pred1, pred2, fwd_col, oos_gap, nw_lags,
                             rolling_window=500):
    tag = (f"pos_EW_unbound_asym_{pred1}_{pred2}_{fwd_col}"
           f"_t{int(T_THRESH*100)}_rw{rolling_window}_oos{OOS_START}.parquet")
    cache = CACHE_DIR / tag
    if cache.exists():
        return pd.read_parquet(cache).squeeze().rename(f"pos_ubasym_{pred1}_{pred2}")

    sub      = panel.dropna(subset=[pred1, pred2, fwd_col]).copy()
    betas_df = compute_betas_bivariate(panel, pred1, pred2, fwd_col, oos_gap, nw_lags)
    y_hat    = _yhat_bivariate(panel, pred1, pred2, fwd_col, betas_df)
    mu       = _rolling_mu(panel, fwd_col, oos_gap, rolling_window=rolling_window,
                           predictor=[pred1, pred2])
    fire     = ((betas_df["t_stat_1"].abs() > T_THRESH)
                & (betas_df["t_stat_2"].abs() > T_THRESH))

    pos = pd.Series(0.0, index=sub.index, name=f"pos_ubasym_{pred1}_{pred2}")
    idx = y_hat.index[fire.reindex(y_hat.index, fill_value=False)]
    pos.loc[idx] = _asym_levels(y_hat.loc[idx], mu.reindex(idx))

    pos.to_frame().to_parquet(cache)
    return pos


def run_ew_biv_unbound_rolmu(panel, pred1, pred2, fwd_col, oos_gap, nw_lags,
                              rolling_window=500):
    tag = (f"pos_EW_rolmu_unbound_{pred1}_{pred2}_{fwd_col}"
           f"_t{int(T_THRESH*100)}_rw{rolling_window}_oos{OOS_START}.parquet")
    cache = CACHE_DIR / tag
    if cache.exists():
        return pd.read_parquet(cache).squeeze().rename(f"pos_ubrolmu_{pred1}_{pred2}")

    sub      = panel.dropna(subset=[pred1, pred2, fwd_col]).copy()
    betas_df = compute_betas_bivariate(panel, pred1, pred2, fwd_col, oos_gap, nw_lags)
    y_hat    = _yhat_bivariate(panel, pred1, pred2, fwd_col, betas_df)
    mu       = _rolling_mu(panel, fwd_col, oos_gap, rolling_window=rolling_window,
                           predictor=[pred1, pred2])
    fire     = ((betas_df["t_stat_1"].abs() > T_THRESH)
                & (betas_df["t_stat_2"].abs() > T_THRESH))

    pos = pd.Series(0.0, index=sub.index, name=f"pos_ubrolmu_{pred1}_{pred2}")
    idx = y_hat.index[fire.reindex(y_hat.index, fill_value=False)]
    pos.loc[idx] = _signed_levels(y_hat.loc[idx] - mu.reindex(idx))

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

    sub      = panel.dropna(subset=[pred1, pred2, pred3, fwd_col]).copy()
    betas_df = compute_betas_trivariate(panel, pred1, pred2, pred3, fwd_col, oos_gap, nw_lags)
    y_hat    = _yhat_trivariate(panel, pred1, pred2, pred3, fwd_col, betas_df)
    fire     = ((betas_df["t_stat_1"].abs() > T_THRESH)
                & (betas_df["t_stat_2"].abs() > T_THRESH)
                & (betas_df["t_stat_3"].abs() > T_THRESH))

    pos = pd.Series(0.0, index=sub.index, name=f"pos_ubsym_{pred1}_{pred2}_{pred3}")
    idx = y_hat.index[fire.reindex(y_hat.index, fill_value=False)]
    pos.loc[idx] = _signed_levels(y_hat.loc[idx])

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

    sub      = panel.dropna(subset=[pred1, pred2, pred3, fwd_col]).copy()
    betas_df = compute_betas_trivariate(panel, pred1, pred2, pred3, fwd_col, oos_gap, nw_lags)
    y_hat    = _yhat_trivariate(panel, pred1, pred2, pred3, fwd_col, betas_df)
    mu       = _rolling_mu(panel, fwd_col, oos_gap, rolling_window=rolling_window,
                           predictor=[pred1, pred2, pred3])
    fire     = ((betas_df["t_stat_1"].abs() > T_THRESH)
                & (betas_df["t_stat_2"].abs() > T_THRESH)
                & (betas_df["t_stat_3"].abs() > T_THRESH))

    pos = pd.Series(0.0, index=sub.index, name=f"pos_ubasym_{pred1}_{pred2}_{pred3}")
    idx = y_hat.index[fire.reindex(y_hat.index, fill_value=False)]
    pos.loc[idx] = _asym_levels(y_hat.loc[idx], mu.reindex(idx))

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

    sub      = panel.dropna(subset=[pred1, pred2, pred3, fwd_col]).copy()
    betas_df = compute_betas_trivariate(panel, pred1, pred2, pred3, fwd_col, oos_gap, nw_lags)
    y_hat    = _yhat_trivariate(panel, pred1, pred2, pred3, fwd_col, betas_df)
    mu       = _rolling_mu(panel, fwd_col, oos_gap, rolling_window=rolling_window,
                           predictor=[pred1, pred2, pred3])
    fire     = ((betas_df["t_stat_1"].abs() > T_THRESH)
                & (betas_df["t_stat_2"].abs() > T_THRESH)
                & (betas_df["t_stat_3"].abs() > T_THRESH))

    pos = pd.Series(0.0, index=sub.index, name=f"pos_ubrolmu_{pred1}_{pred2}_{pred3}")
    idx = y_hat.index[fire.reindex(y_hat.index, fill_value=False)]
    pos.loc[idx] = _signed_levels(y_hat.loc[idx] - mu.reindex(idx))

    pos.to_frame().to_parquet(cache)
    return pos


def plot_unbound_asymmetric_comparison(sim_vvix, sim_biv, sim_term, bah_sim, out_path,
                                        sim_ma10=None, sim_biv10=None):
    """One-panel comparison: unbound-asymmetric VVIX MA5/MA10 vs VRP+VVIX MA5/MA10 vs VRP+Term Slope."""
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
        f"Unbound Asymmetric: VVIX MA5 / MA10  vs  VRP + VVIX MA5 / MA10  vs  VRP + Term Slope  ·  Expanding Window\n"
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
            label=(f"VVIX MA5 (Unbound Asym)  "
                   f"[SR={vvix_st['sharpe']:+.2f}  "
                   f"ret={vvix_st['ann_ret']*100:+.1f}%  "
                   f"DD={vvix_st['max_dd']*100:.1f}%  "
                   f"L{pL_v:.0f}%/S{pS_v:.0f}%  AvgPos={avg_v:+.2f}]"))

    ax.plot(biv_cum.index, biv_cum.values,
            color=C_BIV, lw=2.2, ls="--", alpha=0.92,
            label=(f"VRP + VVIX MA5 (Unbound Asym)  "
                   f"[SR={biv_st['sharpe']:+.2f}  "
                   f"ret={biv_st['ann_ret']*100:+.1f}%  "
                   f"DD={biv_st['max_dd']*100:.1f}%  "
                   f"L{pL_b:.0f}%/S{pS_b:.0f}%  AvgPos={avg_b:+.2f}]"))

    ax.plot(term_cum.index, term_cum.values,
            color=C_TERM, lw=2.2, ls="-.", alpha=0.92,
            label=(f"VRP + Term Slope (Unbound Asym)  "
                   f"[SR={term_st['sharpe']:+.2f}  "
                   f"ret={term_st['ann_ret']*100:+.1f}%  "
                   f"DD={term_st['max_dd']*100:.1f}%  "
                   f"L{pL_t:.0f}%/S{pS_t:.0f}%  AvgPos={avg_t:+.2f}]"))

    if sim_ma10 is not None:
        ma10_cum = _cum(sim_ma10); ma10_st = _stats(sim_ma10, "VVIX10_asym")
        pL_m, pS_m, avg_m = _pos_pct(sim_ma10)
        ax.plot(ma10_cum.index, ma10_cum.values,
                color=C_MA10, lw=2.2, ls=":", alpha=0.92,
                label=(f"VVIX MA10 (Unbound Asym)  "
                       f"[SR={ma10_st['sharpe']:+.2f}  "
                       f"ret={ma10_st['ann_ret']*100:+.1f}%  "
                       f"DD={ma10_st['max_dd']*100:.1f}%  "
                       f"L{pL_m:.0f}%/S{pS_m:.0f}%  AvgPos={avg_m:+.2f}]"))

    if sim_biv10 is not None:
        biv10_cum = _cum(sim_biv10); biv10_st = _stats(sim_biv10, "BIV10_asym")
        pL_b10, pS_b10, avg_b10 = _pos_pct(sim_biv10)
        ax.plot(biv10_cum.index, biv10_cum.values,
                color=C_BIV10, lw=2.2, ls=(0, (5, 1)), alpha=0.92,
                label=(f"VRP + VVIX MA10 (Unbound Asym)  "
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


def main():
    print("=" * 72)
    print("  Unbound (Leveraged) Strategies")
    print("  VVIX MA5 / VVIX MA10 / VRP / VRP+VVIX MA5 / VRP+VVIX MA10 /")
    print("  VRP+Term Slope / VRP+Open Interest / Trivariate  x  sym / asym / rolmu")
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

    out_vvix  = OUTPUT / "expanding_window" / "VVIX MA5"
    out_ma10  = OUTPUT / "expanding_window" / "VVIX MA10"
    out_vrp   = OUTPUT / "expanding_window" / "VRP"
    out_biv   = OUTPUT / "expanding_window" / "VRP + VVIX MA5"
    out_biv10 = OUTPUT / "expanding_window" / "VRP + VVIX MA10"
    out_term  = OUTPUT / "expanding_window" / "VRP + Term Slope"
    out_oi    = OUTPUT / "expanding_window" / "VRP + Open Interest"
    for d in (out_vvix, out_ma10, out_vrp, out_biv, out_biv10, out_term, out_oi):
        d.mkdir(parents=True, exist_ok=True)

    # ── A: VVIX MA5 symmetric ────────────────────────────────────────────────
    print("\n[A] VVIX MA5 unbound symmetric...")
    pos = run_ew_unbound_sym(panel, "vvix_ma5", FWD, OOS_GAP, NW_LAGS)
    pos = pos.copy(); pos[pos.index < VVIX_ACT] = 0.0
    sim = _report(pos, daily_ret, "ub_sym_VVIX", bah_sim)
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
    sim_vvix_asym = _report(pos, daily_ret, "ub_asym_VVIX", bah_sim)
    plot_unbound_univariate(
        "VVIX MA5", "20-day", "asym", C_VVIX,
        sim_vvix_asym, betas_vvix, bah_sim, yh_vvix, mu_20d,
        out_vvix / "leveraged_asymmetric_VVIX_MA5.png",
        extra_title="\nFlat before VVIX activation (2006-03-06)",
    )

    # ── MA5-rolmu: VVIX MA5 base_return_shift ────────────────────────────────
    print("\n[MA5-rolmu] VVIX MA5 unbound base_return_shift...")
    pos = run_ew_unbound_rolmu(panel, "vvix_ma5", FWD, OOS_GAP, NW_LAGS)
    pos = pos.copy(); pos[pos.index < VVIX_ACT] = 0.0
    sim = _report(pos, daily_ret, "ub_rolmu_VVIX_MA5", bah_sim)
    plot_unbound_univariate(
        "VVIX MA5", "20-day", "rolmu", C_VVIX,
        sim, betas_vvix, bah_sim, yh_vvix, mu_20d,
        out_vvix / "leveraged_base_return_shift_VVIX_MA5.png",
        extra_title="\nFlat before VVIX activation (2006-03-06)",
    )

    # ── R: VVIX MA10 symmetric ───────────────────────────────────────────────
    print("\n[R] VVIX MA10 unbound symmetric...")
    pos = run_ew_unbound_sym(panel, "vvix_ma10", FWD, OOS_GAP, NW_LAGS)
    pos = pos.copy(); pos[pos.index < VVIX_ACT] = 0.0
    sim = _report(pos, daily_ret, "ub_sym_VVIX_MA10", bah_sim)
    plot_unbound_univariate(
        "VVIX MA10", "20-day", "sym", C_MA10,
        sim, betas_ma10, bah_sim, yh_ma10, mu_20d,
        out_ma10 / "leveraged_symmetric_VVIX_MA10.png",
        extra_title="\nFlat before VVIX activation (2006-03-06)",
    )

    # ── S: VVIX MA10 asymmetric ──────────────────────────────────────────────
    print("\n[S] VVIX MA10 unbound asymmetric...")
    pos = run_ew_unbound_asym(panel, "vvix_ma10", FWD, OOS_GAP, NW_LAGS)
    pos = pos.copy(); pos[pos.index < VVIX_ACT] = 0.0
    sim_ma10_asym = _report(pos, daily_ret, "ub_asym_VVIX_MA10", bah_sim)
    plot_unbound_univariate(
        "VVIX MA10", "20-day", "asym", C_MA10,
        sim_ma10_asym, betas_ma10, bah_sim, yh_ma10, mu_20d,
        out_ma10 / "leveraged_asymmetric_VVIX_MA10.png",
        extra_title="\nFlat before VVIX activation (2006-03-06)",
    )

    # ── T: VVIX MA10 base_return_shift ───────────────────────────────────────
    print("\n[T] VVIX MA10 unbound base_return_shift...")
    pos = run_ew_unbound_rolmu(panel, "vvix_ma10", FWD, OOS_GAP, NW_LAGS)
    pos = pos.copy(); pos[pos.index < VVIX_ACT] = 0.0
    sim = _report(pos, daily_ret, "ub_rolmu_VVIX_MA10", bah_sim)
    plot_unbound_univariate(
        "VVIX MA10", "20-day", "rolmu", C_MA10,
        sim, betas_ma10, bah_sim, yh_ma10, mu_20d,
        out_ma10 / "leveraged_base_return_shift_VVIX_MA10.png",
        extra_title="\nFlat before VVIX activation (2006-03-06)",
    )

    # ── C: VRP symmetric ─────────────────────────────────────────────────────
    print("\n[C] VRP unbound symmetric...")
    pos = run_ew_unbound_sym(panel, "VP", FWD, OOS_GAP, NW_LAGS)
    sim = _report(pos, daily_ret, "ub_sym_VRP", bah_sim)
    plot_unbound_univariate(
        "VRP", "20-day", "sym", C_VRP,
        sim, betas_vrp, bah_sim, yh_vrp, mu_20d,
        out_vrp / "leveraged_symmetric_VRP.png",
    )

    # ── D: VRP asymmetric ────────────────────────────────────────────────────
    print("\n[D] VRP unbound asymmetric...")
    pos = run_ew_unbound_asym(panel, "VP", FWD, OOS_GAP, NW_LAGS)
    sim_vrp_asym = _report(pos, daily_ret, "ub_asym_VRP", bah_sim)
    plot_unbound_univariate(
        "VRP", "20-day", "asym", C_VRP,
        sim_vrp_asym, betas_vrp, bah_sim, yh_vrp, mu_20d,
        out_vrp / "leveraged_asymmetric_VRP.png",
    )

    # ── E: VRP base_return_shift ─────────────────────────────────────────────
    print("\n[E] VRP unbound base_return_shift...")
    pos = run_ew_unbound_rolmu(panel, "VP", FWD, OOS_GAP, NW_LAGS)
    sim = _report(pos, daily_ret, "ub_rolmu_VRP", bah_sim)
    plot_unbound_univariate(
        "VRP", "20-day", "rolmu", C_VRP,
        sim, betas_vrp, bah_sim, yh_vrp, mu_20d,
        out_vrp / "leveraged_base_return_shift_VRP.png",
    )

    # ── F: VRP+VVIX MA5 bivariate symmetric ──────────────────────────────────
    print("\n[F] VRP+VVIX MA5 unbound symmetric...")
    pos = run_ew_biv_unbound_sym(panel, "VP", "vvix_ma5", FWD, OOS_GAP, NW_LAGS)
    sim = _report(pos, daily_ret, "ub_sym_biv", bah_sim)
    plot_unbound_bivariate(
        "VRP", "VVIX MA5", "20-day", "sym", C_BIV,
        sim, betas_biv, bah_sim, yh_biv, mu_20d,
        out_biv / "leveraged_symmetric_VRP_+_VVIX_MA5.png",
    )

    # ── G: VRP+VVIX MA5 bivariate asymmetric ─────────────────────────────────
    print("\n[G] VRP+VVIX MA5 unbound asymmetric...")
    pos = run_ew_biv_unbound_asym(panel, "VP", "vvix_ma5", FWD, OOS_GAP, NW_LAGS)
    sim_biv_asym = _report(pos, daily_ret, "ub_asym_biv", bah_sim)
    plot_unbound_bivariate(
        "VRP", "VVIX MA5", "20-day", "asym", C_BIV,
        sim_biv_asym, betas_biv, bah_sim, yh_biv, mu_20d,
        out_biv / "leveraged_asymmetric_VRP_+_VVIX_MA5.png",
    )

    # ── H: VRP+VVIX MA5 bivariate base_return_shift ───────────────────────────
    print("\n[H] VRP+VVIX MA5 unbound base_return_shift...")
    pos = run_ew_biv_unbound_rolmu(panel, "VP", "vvix_ma5", FWD, OOS_GAP, NW_LAGS)
    sim = _report(pos, daily_ret, "ub_rolmu_biv", bah_sim)
    plot_unbound_bivariate(
        "VRP", "VVIX MA5", "20-day", "rolmu", C_BIV,
        sim, betas_biv, bah_sim, yh_biv, mu_20d,
        out_biv / "leveraged_base_return_shift_VRP_+_VVIX_MA5.png",
    )

    # ── U: VRP+VVIX MA10 bivariate symmetric ─────────────────────────────────
    print("\n[U] VRP+VVIX MA10 unbound symmetric...")
    pos = run_ew_biv_unbound_sym(panel, "VP", "vvix_ma10", FWD, OOS_GAP, NW_LAGS)
    sim = _report(pos, daily_ret, "ub_sym_biv10", bah_sim)
    plot_unbound_bivariate(
        "VRP", "VVIX MA10", "20-day", "sym", C_BIV10,
        sim, betas_biv10, bah_sim, yh_biv10, mu_20d,
        out_biv10 / "leveraged_symmetric_VRP_+_VVIX_MA10.png",
    )

    # ── V: VRP+VVIX MA10 bivariate asymmetric ────────────────────────────────
    print("\n[V] VRP+VVIX MA10 unbound asymmetric...")
    pos = run_ew_biv_unbound_asym(panel, "VP", "vvix_ma10", FWD, OOS_GAP, NW_LAGS)
    sim_biv10_asym = _report(pos, daily_ret, "ub_asym_biv10", bah_sim)
    plot_unbound_bivariate(
        "VRP", "VVIX MA10", "20-day", "asym", C_BIV10,
        sim_biv10_asym, betas_biv10, bah_sim, yh_biv10, mu_20d,
        out_biv10 / "leveraged_asymmetric_VRP_+_VVIX_MA10.png",
    )

    # ── W: VRP+VVIX MA10 bivariate base_return_shift ─────────────────────────
    print("\n[W] VRP+VVIX MA10 unbound base_return_shift...")
    pos = run_ew_biv_unbound_rolmu(panel, "VP", "vvix_ma10", FWD, OOS_GAP, NW_LAGS)
    sim = _report(pos, daily_ret, "ub_rolmu_biv10", bah_sim)
    plot_unbound_bivariate(
        "VRP", "VVIX MA10", "20-day", "rolmu", C_BIV10,
        sim, betas_biv10, bah_sim, yh_biv10, mu_20d,
        out_biv10 / "leveraged_base_return_shift_VRP_+_VVIX_MA10.png",
    )

    # ── I: VRP+Term Slope bivariate symmetric ────────────────────────────────
    print("\n[I] VRP+Term Slope unbound symmetric...")
    pos = run_ew_biv_unbound_sym(panel, "VP", "term_slope", FWD, OOS_GAP, NW_LAGS)
    sim = _report(pos, daily_ret, "ub_sym_term", bah_sim)
    plot_unbound_bivariate(
        "VRP", "Term Slope", "20-day", "sym", C_TERM,
        sim, betas_term, bah_sim, yh_term, mu_20d,
        out_term / "leveraged_symmetric_VRP_+_Term_Slope.png",
    )

    # ── J: VRP+Term Slope bivariate asymmetric ───────────────────────────────
    print("\n[J] VRP+Term Slope unbound asymmetric...")
    pos = run_ew_biv_unbound_asym(panel, "VP", "term_slope", FWD, OOS_GAP, NW_LAGS)
    sim_term_asym = _report(pos, daily_ret, "ub_asym_term", bah_sim)
    plot_unbound_bivariate(
        "VRP", "Term Slope", "20-day", "asym", C_TERM,
        sim_term_asym, betas_term, bah_sim, yh_term, mu_20d,
        out_term / "leveraged_asymmetric_VRP_+_Term_Slope.png",
    )

    # ── K: VRP+Term Slope bivariate base_return_shift ────────────────────────
    print("\n[K] VRP+Term Slope unbound base_return_shift...")
    pos = run_ew_biv_unbound_rolmu(panel, "VP", "term_slope", FWD, OOS_GAP, NW_LAGS)
    sim = _report(pos, daily_ret, "ub_rolmu_term", bah_sim)
    plot_unbound_bivariate(
        "VRP", "Term Slope", "20-day", "rolmu", C_TERM,
        sim, betas_term, bah_sim, yh_term, mu_20d,
        out_term / "leveraged_base_return_shift_VRP_+_Term_Slope.png",
    )

    # ── L: VRP+Open Interest bivariate symmetric ──────────────────────────────
    print("\n[L] VRP+Open Interest unbound symmetric...")
    pos = run_ew_biv_unbound_sym(panel, "VP", "open_interest", FWD, OOS_GAP, NW_LAGS)
    sim = _report(pos, daily_ret, "ub_sym_oi", bah_sim)
    plot_unbound_bivariate(
        "VRP", "Open Interest", "20-day", "sym", C_OI,
        sim, betas_oi, bah_sim, yh_oi, mu_20d,
        out_oi / "leveraged_symmetric_VRP_+_Open_Interest.png",
    )

    # ── M: VRP+Open Interest bivariate asymmetric ─────────────────────────────
    print("\n[M] VRP+Open Interest unbound asymmetric...")
    pos = run_ew_biv_unbound_asym(panel, "VP", "open_interest", FWD, OOS_GAP, NW_LAGS)
    sim = _report(pos, daily_ret, "ub_asym_oi", bah_sim)
    plot_unbound_bivariate(
        "VRP", "Open Interest", "20-day", "asym", C_OI,
        sim, betas_oi, bah_sim, yh_oi, mu_20d,
        out_oi / "leveraged_asymmetric_VRP_+_Open_Interest.png",
    )

    # ── N: VRP+Open Interest bivariate base_return_shift ─────────────────────
    print("\n[N] VRP+Open Interest unbound base_return_shift...")
    pos = run_ew_biv_unbound_rolmu(panel, "VP", "open_interest", FWD, OOS_GAP, NW_LAGS)
    sim = _report(pos, daily_ret, "ub_rolmu_oi", bah_sim)
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
    sim = _report(pos, daily_ret, "ub_sym_triv", bah_sim)
    plot_unbound_trivariate(
        "VRP", "VVIX MA5", "Term Slope", "20-day", "sym", C_TRIV,
        sim, betas_triv, bah_sim, yh_triv, mu_20d,
        out_triv / "leveraged_symmetric_VRP_+_VVIX_MA5_+_Term_Slope.png",
        c1=C_TRIV, c2=C_TRIV2, c3=C_TRIV3,
    )

    print("\n[P] VRP+VVIX MA5+Term Slope unbound asymmetric...")
    pos = run_ew_triv_unbound_asym(panel, "VP", "vvix_ma5", "term_slope",
                                   FWD, OOS_GAP, NW_LAGS)
    sim_triv_asym = _report(pos, daily_ret, "ub_asym_triv", bah_sim)
    plot_unbound_trivariate(
        "VRP", "VVIX MA5", "Term Slope", "20-day", "asym", C_TRIV,
        sim_triv_asym, betas_triv, bah_sim, yh_triv, mu_20d,
        out_triv / "leveraged_asymmetric_VRP_+_VVIX_MA5_+_Term_Slope.png",
        c1=C_TRIV, c2=C_TRIV2, c3=C_TRIV3,
    )

    print("\n[Q] VRP+VVIX MA5+Term Slope unbound base_return_shift...")
    pos = run_ew_triv_unbound_rolmu(panel, "VP", "vvix_ma5", "term_slope",
                                    FWD, OOS_GAP, NW_LAGS)
    sim = _report(pos, daily_ret, "ub_rolmu_triv", bah_sim)
    plot_unbound_trivariate(
        "VRP", "VVIX MA5", "Term Slope", "20-day", "rolmu", C_TRIV,
        sim, betas_triv, bah_sim, yh_triv, mu_20d,
        out_triv / "leveraged_base_return_shift_VRP_+_VVIX_MA5_+_Term_Slope.png",
        c1=C_TRIV, c2=C_TRIV2, c3=C_TRIV3,
    )

    # ── Comparison: unbound-asymmetric VVIX MA5/MA10 vs VRP+VVIX MA5/MA10 ────
    print("\n[Comparison] Unbound asymmetric VVIX MA5/MA10 vs VRP+VVIX MA5/MA10...")
    out_cmp = OUTPUT / "expanding_window" / "comparisons"
    plot_unbound_asymmetric_comparison(
        sim_vvix_asym, sim_biv_asym, sim_term_asym, bah_sim,
        out_cmp / "leveraged_asymmetric_vvix_vs_vrp_vvix.png",
        sim_ma10=sim_ma10_asym,
        sim_biv10=sim_biv10_asym,
    )

    print("\nDone.")
    print("=" * 72)


if __name__ == "__main__":
    main()
