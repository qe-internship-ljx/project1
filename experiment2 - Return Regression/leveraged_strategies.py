"""
leveraged_strategies.py
=======================
Leveraged multi-level position sizing variants for all models / three threshold
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

Models (the canonical UNI_MODELS / BIV_MODELS set from regressions.py):
  VRP       -> 20-day (univariate)            -> output/plots/VRP/
  VVIX MA5  -> 20-day (univariate)            -> output/plots/VVIX MA5/
  VVIX MA10 -> 20-day (univariate)            -> output/plots/VVIX MA10/
  VRP + VVIX MA5 -> 20-day (bivariate)       -> output/plots/VRP + VVIX MA5/
  VRP + VVIX MA10 -> 20-day (bivariate)      -> output/plots/VRP + VVIX MA10/
  VRP + Term Slope -> 20-day (bivariate)     -> output/plots/VRP + Term Slope/
  VRP + Open Interest -> 20-day (bivariate)  -> output/plots/VRP + Open Interest/

Level-assignment rules are defined in this file; the cached backtest skeleton,
shared plot panels and CLI live in base_strategies.py, and betas / y_hat helpers
and constants come from regressions.py.
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
    compute_buy_and_hold, simulate_strategy, compute_performance_stats,
)
from regressions import (
    load_standard_panel,
    compute_betas, compute_betas_bivariate,
    _yhat_univariate, _yhat_bivariate, _rolling_mu,
    _shade, oos_cumret, stat_start, window_stats,
    OOS_START, VVIX_ACT, OOS_GAP, NW_LAGS, RW, THRESHOLDS, LEVELS,
    UNI_MODELS, BIV_MODELS,
)
# The cached backtest skeleton, shared panel/label helpers and the --t CLI all
# live in base_strategies.py (the canonical strategy/plotting module).
from base_strategies import (
    BAH_COLOR, _draw_bah, _finalize, draw_tstat_beta_panel,
    _run_gated, _style_cumret_axis, _annotate_position_panel, perf_label,
    run_cli,
)

# |t|-stat gate threshold. Lives here (and in base_strategies.py) — not in
# regressions.py — so the --t-threshold flag can rebind it and have every plot
# title and the t-stat gate band reflect the chosen value.
T_THRESH = 1.65


def _resolve_t(t_thresh):
    """Default to this module's T_THRESH at call time (rebound by --t)."""
    return T_THRESH if t_thresh is None else t_thresh


# ─── Color constants ──────────────────────────────────────────────────────────

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
                   f"{perf_label(st, f'L{pL:.0f}%/S{pS:.0f}%  AvgPos={avg_pos:+.2f}')}"))

    _style_cumret_axis(ax)
    return _stat_start, pL, pS, avg_pos


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
    _annotate_position_panel(ax, sim, main_color, pL, pS, 100.0 - pL - pS, avg_pos)
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


# ─── Leveraged: position builders ───────────────────────────────────────────────
# All six strategies delegate to the shared skeleton in base_strategies.py; the
# wrappers below differ only in the cache key, the stored series name, and the
# level rule (sym / asym / rolmu). Cache keys are kept byte-identical to the
# originals so existing regression_cache/ files hit.

def _run_leveraged(panel, preds, fwd_col, oos_gap, nw_lags, mode, t_thresh,
                   cache_tag, pos_name, rolling_window=None):
    """Multi-level assignment on the shared skeleton. `mode` selects:
        sym   — signed levels of ŷ
        asym  — long levels of (ŷ-µ), short levels of -ŷ (long wins ties)
        rolmu — signed levels of (ŷ-µ)
    """
    def assign(pos, idx, yh, mu):
        if mode == "sym":
            pos.loc[idx] = _signed_levels(yh)
        elif mode == "asym":
            pos.loc[idx] = _asym_levels(yh, mu)
        else:
            pos.loc[idx] = _signed_levels(yh - mu)

    return _run_gated(panel, preds, fwd_col, oos_gap, nw_lags, t_thresh,
                      CACHE_DIR / cache_tag, pos_name, assign,
                      use_mu=(mode != "sym"), rolling_window=rolling_window)


def run_ew_leveraged_sym(panel, predictor, fwd_col, oos_gap, nw_lags, t_thresh=None):
    """Level based on |ŷ|."""
    t_thresh = _resolve_t(t_thresh)
    tag = (f"pos_EW_leveraged_sym_{predictor}_{fwd_col}"
           f"_t{int(t_thresh*100)}_oos{OOS_START}.parquet")
    return _run_leveraged(panel, [predictor], fwd_col, oos_gap, nw_lags, "sym",
                          t_thresh, tag, f"pos_ubsym_{predictor}")


def run_ew_leveraged_asym(panel, predictor, fwd_col, oos_gap, nw_lags,
                        rolling_window=500, t_thresh=None):
    """Long: level by (ŷ-µ); short: level by |ŷ| when ŷ<0."""
    t_thresh = _resolve_t(t_thresh)
    tag = (f"pos_EW_leveraged_asym_{predictor}_{fwd_col}"
           f"_t{int(t_thresh*100)}_rw{rolling_window}_oos{OOS_START}.parquet")
    return _run_leveraged(panel, [predictor], fwd_col, oos_gap, nw_lags, "asym",
                          t_thresh, tag, f"pos_ubasym_{predictor}",
                          rolling_window=rolling_window)


def run_ew_leveraged_rolmu(panel, predictor, fwd_col, oos_gap, nw_lags,
                         rolling_window=500, t_thresh=None):
    """Level based on |ŷ - µ|."""
    t_thresh = _resolve_t(t_thresh)
    tag = (f"pos_EW_rolmu_leveraged_{predictor}_{fwd_col}"
           f"_t{int(t_thresh*100)}_rw{rolling_window}_oos{OOS_START}.parquet")
    return _run_leveraged(panel, [predictor], fwd_col, oos_gap, nw_lags, "rolmu",
                          t_thresh, tag, f"pos_ubrolmu_{predictor}",
                          rolling_window=rolling_window)


def run_ew_biv_leveraged_sym(panel, pred1, pred2, fwd_col, oos_gap, nw_lags, t_thresh=None):
    t_thresh = _resolve_t(t_thresh)
    tag = (f"pos_EW_leveraged_sym_{pred1}_{pred2}_{fwd_col}"
           f"_t{int(t_thresh*100)}_oos{OOS_START}.parquet")
    return _run_leveraged(panel, [pred1, pred2], fwd_col, oos_gap, nw_lags, "sym",
                          t_thresh, tag, f"pos_ubsym_{pred1}_{pred2}")


def run_ew_biv_leveraged_asym(panel, pred1, pred2, fwd_col, oos_gap, nw_lags,
                             rolling_window=500, t_thresh=None):
    t_thresh = _resolve_t(t_thresh)
    tag = (f"pos_EW_leveraged_asym_{pred1}_{pred2}_{fwd_col}"
           f"_t{int(t_thresh*100)}_rw{rolling_window}_oos{OOS_START}.parquet")
    return _run_leveraged(panel, [pred1, pred2], fwd_col, oos_gap, nw_lags, "asym",
                          t_thresh, tag, f"pos_ubasym_{pred1}_{pred2}",
                          rolling_window=rolling_window)


def run_ew_biv_leveraged_rolmu(panel, pred1, pred2, fwd_col, oos_gap, nw_lags,
                              rolling_window=500, t_thresh=None):
    t_thresh = _resolve_t(t_thresh)
    tag = (f"pos_EW_rolmu_leveraged_{pred1}_{pred2}_{fwd_col}"
           f"_t{int(t_thresh*100)}_rw{rolling_window}_oos{OOS_START}.parquet")
    return _run_leveraged(panel, [pred1, pred2], fwd_col, oos_gap, nw_lags, "rolmu",
                          t_thresh, tag, f"pos_ubrolmu_{pred1}_{pred2}",
                          rolling_window=rolling_window)


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


def _plot_leveraged(labels, tstat_colors, horizon_label, threshold_type,
                    main_color, sim, betas_df, bah_sim,
                    y_hat_ser, mu_ser, out_path,
                    extra_title="", nw_lags=NW_LAGS):
    """4-panel leveraged figure shared by the univariate / bivariate wrappers.
    `labels` / `tstat_colors` are the 1-2 predictor names and t-stat panel
    colours (parallel lists)."""
    oos_dt = pd.Timestamp(OOS_START)
    e_dt   = betas_df.index[-1]

    fig, axes = plt.subplots(
        4, 1, figsize=(14, 15.4), sharex=True,
        gridspec_kw={"height_ratios": [2.5, 1.2, 1.0, 1.2], "hspace": 0.35},
    )
    ax_ret, ax_t, ax_p, ax_yh = axes

    gate = f"|t| > {T_THRESH:.2f} gate" + (" (both)" if len(labels) > 1 else "")
    fig.suptitle(
        f"{' + '.join(labels)} -> {horizon_label} Forward Return  "
        f"(Expanding Window, OOS from {OOS_START})  "
        f"{_THRESH_LBL[threshold_type]}{extra_title}\n"
        f"{_THRESH_DESC[threshold_type]}  |  "
        f"NW-HAC {nw_lags} lags; {gate}; 0.05% slippage",
        fontsize=10, y=0.998,
    )
    fig.subplots_adjust(top=0.955, bottom=0.03, left=0.10, right=0.93)
    for ax in axes:
        ax.set_xlim(oos_dt, e_dt)
        _shade(ax, oos_dt, e_dt)

    _, pL, pS, avg_pos = _draw_cumret(ax_ret, sim, bah_sim, main_color, "Leveraged +/-1..4")
    draw_tstat_beta_panel(ax_t, betas_df, labels, tstat_colors, nw_lags,
                          t_thresh=T_THRESH)
    _draw_position(ax_p, sim, main_color, pL, pS, avg_pos)
    _draw_predicted(ax_yh, y_hat_ser, mu_ser, main_color, threshold_type)
    _finalize(fig, axes, out_path)


def plot_leveraged_univariate(pred_label, horizon_label, threshold_type,
                            main_color, sim, betas_df, bah_sim,
                            y_hat_ser, mu_ser, out_path,
                            extra_title="", nw_lags=NW_LAGS):
    _plot_leveraged([pred_label], [main_color], horizon_label, threshold_type,
                    main_color, sim, betas_df, bah_sim, y_hat_ser, mu_ser,
                    out_path, extra_title=extra_title, nw_lags=nw_lags)


def plot_leveraged_bivariate(pred1_label, pred2_label, horizon_label, threshold_type,
                           main_color, sim, betas_df, bah_sim,
                           y_hat_ser, mu_ser, out_path,
                           extra_title="", nw_lags=NW_LAGS):
    _plot_leveraged([pred1_label, pred2_label], [C_VRP, C_VRP2], horizon_label,
                    threshold_type, main_color, sim, betas_df, bah_sim,
                    y_hat_ser, mu_ser, out_path,
                    extra_title=extra_title, nw_lags=nw_lags)


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
    panel = load_standard_panel()
    FWD   = "fwd_20d"
    print(f"    {len(panel):,} obs  "
          f"[{panel.index.min().date()} - {panel.index.max().date()}]")

    daily_ret = panel["daily_ret"].dropna()
    bah_sim   = simulate_strategy(compute_buy_and_hold(daily_ret), daily_ret)
    mu_20d    = _rolling_mu(panel, FWD, OOS_GAP, RW)

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

    # Per-model plot colours (uni keyed by predictor, biv by second predictor).
    uni_colors = {"VP": C_VRP, "vvix_ma5": C_VVIX, "vvix_ma10": C_MA10}
    biv_colors = {"vvix_ma5": C_BIV, "vvix_ma10": C_BIV10,
                  "term_slope": C_TERM, "open_interest": C_OI}
    # VVIX-based signals stay flat before the VVIX activation date.
    flat_before_vvix = {"vvix_ma5", "vvix_ma10"}

    sims = {}   # (model label, threshold_type) → simulated DataFrame

    for col, label in UNI_MODELS:
        betas   = compute_betas(panel, col, FWD, OOS_GAP, NW_LAGS)
        yhat    = _yhat_univariate(panel, col, FWD, betas)
        out_dir = OUTPUT / "plots" / label
        out_dir.mkdir(parents=True, exist_ok=True)
        for thr, prefix, run_fn in uni_strats:
            print(f"\n{label} leveraged {prefix}...")
            pos   = run_fn(panel, col, FWD, OOS_GAP, NW_LAGS)
            extra = ""
            if col in flat_before_vvix:
                pos = pos.copy(); pos[pos.index < VVIX_ACT] = 0.0
                extra = "\nFlat before VVIX activation (2006-03-06)"
            sim = _report(pos, daily_ret, f"ub_{thr}_{label.replace(' ', '_')}", bah_sim)
            sims[(label, thr)] = sim
            plot_leveraged_univariate(
                label, "20-day", thr, uni_colors[col],
                sim, betas, bah_sim, yhat, mu_20d,
                out_dir / f"leveraged_{prefix}_{label.replace(' ', '_')}.png",
                extra_title=extra,
            )

    for col2, label2 in BIV_MODELS:
        betas   = compute_betas_bivariate(panel, "VP", col2, FWD, OOS_GAP, NW_LAGS)
        yhat    = _yhat_bivariate(panel, "VP", col2, FWD, betas)
        out_dir = OUTPUT / "plots" / f"VRP + {label2}"
        out_dir.mkdir(parents=True, exist_ok=True)
        for thr, prefix, run_fn in biv_strats:
            print(f"\nVRP+{label2} leveraged {prefix}...")
            pos = run_fn(panel, "VP", col2, FWD, OOS_GAP, NW_LAGS)
            sim = _report(pos, daily_ret, f"ub_{thr}_VRP_{col2}", bah_sim)
            sims[(f"VRP + {label2}", thr)] = sim
            fname = f"leveraged_{prefix}_VRP_+_{label2.replace(' ', '_')}.png"
            plot_leveraged_bivariate(
                "VRP", label2, "20-day", thr, biv_colors[col2],
                sim, betas, bah_sim, yhat, mu_20d,
                out_dir / fname,
            )

    print("\nDone.")
    print("=" * 72)


if __name__ == "__main__":
    run_cli(main, "Leveraged multi-level position-sizing strategy evaluation.",
            T_THRESH)
