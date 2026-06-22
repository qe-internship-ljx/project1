"""
base_strategies.py
==================
Base (unit-position) strategy evaluation and plotting for all models.

Strategies:
  symmetric       — ±1 when |ŷ| > delta (×4 deltas) and t-stat gate passes
  asymmetric      — Long if ŷ > µ₅₀₀, Short if ŷ < 0
  base-return-shift — Long if ŷ > µ₅₀₀, Short if ŷ < µ₅₀₀

Models (ignoring poor-correlation baselines):
  Univariate:  VRP · VVIX MA5 · VVIX MA10
  Bivariate:   VRP+VVIX MA5 · VRP+VVIX MA10 · VRP+Term Slope · VRP+Open Interest

Running main() regenerates all 21 output PNGs.
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
import matplotlib.ticker as mticker

ROOT      = Path(__file__).parent
OUTPUT    = ROOT / "output"
CACHE_DIR = OUTPUT / "regression_cache"

sys.path.insert(0, str(ROOT.parent / "bh_replication"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from helpers import (
    load_vrp_series, load_es_front_month, load_vvix,
    compute_vvix_ma5, compute_vvix_ma10,
    load_vix_spot, load_vix_futures_term_structure,
    load_es_open_interest, load_vix_basis, compute_trend_quotient,
    compute_buy_and_hold, simulate_strategy, compute_performance_stats,
)
from fh_replication.fh_replication import compute_vix_term_slope
from regressions import (
    build_panel,
    compute_betas, compute_betas_bivariate,
    _yhat_univariate, _yhat_bivariate, _rolling_mu,
    _shade, oos_cumret, stat_start, window_stats, draw_tstat_beta_panel,
    OOS_START, T_THRESH, DELTAS, DELTA_LBL, OOS_GAP, NW_LAGS,
)
# Panel-1 Buy-and-Hold drawing and figure finalisation are shared with the
# leveraged variants so both produce identical legends / axes.
from leveraged_strategies import _draw_bah, _finalize


# ─── Plotting helpers ─────────────────────────────────────────────────────────

def _stat_window(sim_dict_or_sim, bah_sim):
    """Plot legend window via the shared stat_start rule.

    Returns (rebase_start, stat_start, stat_lbl); rebase_start is None when the
    window is the full OOS span (no post-2020 activation rebase)."""
    sims = ([sim for _, sim in sim_dict_or_sim.values()]
            if isinstance(sim_dict_or_sim, dict) else sim_dict_or_sim)
    start        = stat_start(sims, bah_sim.index)
    rebase_start = start if start > pd.Timestamp(OOS_START) else None
    stat_lbl     = (f" · stats from {start.strftime('%Y-%m-%d')}"
                    if rebase_start is not None else "")
    return rebase_start, start, stat_lbl


def _print_delta_stats(sim_dict, bah_sim):
    """Print per-delta SR/return/drawdown over the same window the plot legend uses
    (shared stat_start across the delta grid)."""
    sims  = [sim for _, sim in sim_dict.values()]
    start = stat_start(sims, bah_sim.index)
    if start > pd.Timestamp(OOS_START):
        print(f"    stats from {start.strftime('%Y-%m-%d')} (first activation)")
    for di in sorted(sim_dict):
        _, sim = sim_dict[di]
        st = window_stats(sim, f"d{di}", bah_sim.index, start=start)
        print(f"    delta={DELTA_LBL[di]}  SR={st['sharpe']:+.2f}  "
              f"ret={st['ann_ret']*100:+.1f}%  DD={st['max_dd']*100:.1f}%")


# ─── Shared per-delta figure (symmetric / asymmetric / base-return-shift) ─────

# Per-mode title fragments: (suffix after "OOS from {date}", description prefix).
# A None description falls back to the symmetric "training grows daily" line.
_MODE_TITLE = {
    "sym":   ("", None),
    "asym":  (", Asymmetric Threshold",
              "Long: (ŷ-µ₅₀₀) > delta  |  Short: (-ŷ) > delta  |  "),
    "rolmu": (", Base-Return-Shift",
              "Long: (ŷ-µ₅₀₀) > delta  |  Short: (µ₅₀₀-ŷ) > delta  |  "),
}


def _plot_delta_grid(labels, mode, horizon_label, oos_gap, nw_lags,
                     color_palette, sim_dict, betas_df, bah_sim, out_path,
                     r2_oos=None, oos_start=None, extra_title=""):
    """Cumulative-return + t-stat + per-delta-position figure shared by every base
    (unit-position) strategy.

    `labels` is the 1-2 predictor names (joined for the title and fed to the
    shared t-stat panel). `mode` (sym / asym / rolmu) only selects the title text
    — the panels themselves are identical across all three thresholds.
    """
    _oos_start = oos_start if oos_start is not None else OOS_START
    oos_dt   = pd.Timestamp(_oos_start)
    e_dt     = betas_df.index[-1]
    xlim     = (oos_dt, e_dt)
    n_deltas = len(DELTAS)

    _rebase_start, _stat_start, _stat_lbl = _stat_window(sim_dict, bah_sim)

    r2_str = f"  R²_OOS = {r2_oos:+.4f}" if r2_oos is not None else ""

    mode_tag, desc = _MODE_TITLE[mode]
    if desc is None:
        desc = f"Training grows daily; OOS gap = {oos_gap} days; "
    gate_clause = (f"|t| > {T_THRESH:.2f} gate"
                   + (" (both betas)" if len(labels) > 1 else ""))

    h_ratios = [2.5, 1.2] + [1.0] * n_deltas
    fig, axes = plt.subplots(
        2 + n_deltas, 1, figsize=(14, 11 + 2.2 * n_deltas), sharex=True,
        gridspec_kw={"height_ratios": h_ratios, "hspace": 0.35},
    )
    ax_ret = axes[0]
    ax_t   = axes[1]
    ax_pos = axes[2:]

    fig.suptitle(
        f"{' + '.join(labels)} -> {horizon_label} Forward Return  "
        f"(Expanding Window, OOS from {_oos_start}{mode_tag}){r2_str}"
        f"{extra_title}\n"
        f"{desc}"
        f"NW-HAC {nw_lags} lags; {gate_clause}; 0.05% slippage",
        fontsize=10, y=0.998,
    )
    fig.subplots_adjust(top=0.955, bottom=0.03, left=0.10, right=0.93)

    ax_ret.set_xlim(*xlim)
    _shade(ax_ret, oos_dt, e_dt)

    _draw_bah(ax_ret, bah_sim, _stat_start, _stat_lbl, _rebase_start,
              oos_start=_oos_start)

    for di, (delta, lbl) in enumerate(zip(DELTAS, DELTA_LBL)):
        _, sim = sim_dict[di]
        cum     = oos_cumret(sim, start=_oos_start)
        st_plot = compute_performance_stats(
            sim[sim.index >= _stat_start], f"EW_{mode}_{di}")
        pos_stat = sim["position"][sim.index >= _stat_start]
        pL = float((pos_stat == 1).mean() * 100)
        pS = float((pos_stat == -1).mean() * 100)
        ax_ret.plot(cum.index, cum.values,
                    color=color_palette[di], lw=1.8, alpha=0.9,
                    label=(f"{lbl}  "
                           f"[SR={st_plot['sharpe']:+.2f}  "
                           f"ret={st_plot['ann_ret']*100:+.1f}%  "
                           f"DD={st_plot['max_dd']*100:.1f}%  "
                           f"L{pL:.0f}%/S{pS:.0f}%]"))

    ax_ret.axhline(1, color="black", lw=0.4, ls=":")
    ax_ret.set_yscale("log")
    ax_ret.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2f}x"))
    ax_ret.set_ylabel("Cumulative Net Return (log)", fontsize=9)
    ax_ret.legend(fontsize=8, loc="upper left", framealpha=0.92)
    ax_ret.grid(axis="y", alpha=0.2, lw=0.6)
    ax_ret.spines[["top", "right"]].set_visible(False)

    ax_t.set_xlim(*xlim)
    _shade(ax_t, oos_dt, e_dt)

    draw_tstat_beta_panel(ax_t, betas_df, labels,
                          [color_palette[0]] * len(labels), nw_lags)

    for di, (delta, lbl, ax_p) in enumerate(zip(DELTAS, DELTA_LBL, ax_pos)):
        _, sim = sim_dict[di]
        pos = sim["position"][sim.index >= _oos_start]
        ax_p.set_xlim(*xlim)
        _shade(ax_p, oos_dt, e_dt)
        ax_p.fill_between(pos.index, pos.where(pos ==  1, 0), 0,
                          color=color_palette[di], alpha=0.75, label="Long")
        ax_p.fill_between(pos.index, pos.where(pos == -1, 0), 0,
                          color=color_palette[di], alpha=0.30, hatch="///", label="Short")
        ax_p.axhline(0, color="black", lw=0.4)
        ax_p.set_ylim(-1.5, 1.5)
        ax_p.set_yticks([-1, 0, 1])
        ax_p.set_yticklabels(["Short", "Flat", "Long"], fontsize=8)
        ax_p.set_ylabel(lbl, fontsize=9, rotation=0,
                        ha="right", va="center", labelpad=56, color=color_palette[di])
        pL = float((pos == 1).mean() * 100)
        pS = float((pos == -1).mean() * 100)
        pF = float((pos == 0).mean() * 100)
        ax_p.text(0.01, 0.97,
                  f"Long {pL:.1f}%  Short {pS:.1f}%  Flat {pF:.1f}%  "
                  f"AvgPos={float(pos.mean()):+.3f}",
                  transform=ax_p.transAxes, fontsize=7.5, va="top",
                  color=color_palette[di])
        sim_oos   = sim[sim.index >= _oos_start]
        prev_pos  = sim_oos["position"].shift(1)
        long_mask  = prev_pos > 0
        short_mask = prev_pos < 0
        long_acc   = float((sim_oos.loc[long_mask,  "gross_pnl"] > 0).mean()) if long_mask.any()  else float("nan")
        short_acc  = float((sim_oos.loc[short_mask, "gross_pnl"] > 0).mean()) if short_mask.any() else float("nan")
        long_acc_str  = f"{long_acc  * 100:.1f}%" if not np.isnan(long_acc)  else "N/A"
        short_acc_str = f"{short_acc * 100:.1f}%" if not np.isnan(short_acc) else "N/A"
        ax_p.text(0.01, 0.83,
                  f"Long accuracy={long_acc_str}  Short accuracy={short_acc_str}",
                  transform=ax_p.transAxes, fontsize=7.5, va="top",
                  color=color_palette[di])
        ax_p.spines[["top", "right"]].set_visible(False)

    _finalize(fig, [ax_ret, ax_t] + list(ax_pos), out_path)


# Thin wrappers preserving the per-strategy public API (used by main() and
# cross_market.py); all delegate to _plot_delta_grid.

def plot_2panel(pred_label, horizon_label, oos_gap, nw_lags,
                color_palette, sim_dict, betas_df, bah_sim, out_path,
                r2_oos=None, oos_start=None):
    _plot_delta_grid([pred_label], "sym", horizon_label, oos_gap, nw_lags,
                     color_palette, sim_dict, betas_df, bah_sim, out_path,
                     r2_oos=r2_oos, oos_start=oos_start)


def plot_2panel_bivariate(pred1_label, pred2_label, horizon_label, oos_gap, nw_lags,
                          color_palette, sim_dict, betas_df, bah_sim, out_path,
                          r2_oos=None, oos_start=None):
    _plot_delta_grid([pred1_label, pred2_label], "sym", horizon_label, oos_gap,
                     nw_lags, color_palette, sim_dict, betas_df, bah_sim, out_path,
                     r2_oos=r2_oos, oos_start=oos_start)


def plot_asym(pred_label, horizon_label, oos_gap, nw_lags,
              color_palette, sim_dict, betas_df, bah_sim, out_path,
              r2_oos=None, oos_start=None, extra_title=""):
    _plot_delta_grid([pred_label], "asym", horizon_label, oos_gap, nw_lags,
                     color_palette, sim_dict, betas_df, bah_sim, out_path,
                     r2_oos=r2_oos, oos_start=oos_start, extra_title=extra_title)


def plot_asym_bivariate(pred1_label, pred2_label, horizon_label, oos_gap, nw_lags,
                        color_palette, sim_dict, betas_df, bah_sim, out_path,
                        r2_oos=None, oos_start=None, extra_title=""):
    _plot_delta_grid([pred1_label, pred2_label], "asym", horizon_label, oos_gap,
                     nw_lags, color_palette, sim_dict, betas_df, bah_sim, out_path,
                     r2_oos=r2_oos, oos_start=oos_start, extra_title=extra_title)


def plot_rolmu(pred_label, horizon_label, oos_gap, nw_lags,
               color_palette, sim_dict, betas_df, bah_sim, out_path,
               r2_oos=None, oos_start=None, extra_title=""):
    _plot_delta_grid([pred_label], "rolmu", horizon_label, oos_gap, nw_lags,
                     color_palette, sim_dict, betas_df, bah_sim, out_path,
                     r2_oos=r2_oos, oos_start=oos_start, extra_title=extra_title)


def plot_rolmu_bivariate(pred1_label, pred2_label, horizon_label, oos_gap, nw_lags,
                         color_palette, sim_dict, betas_df, bah_sim, out_path,
                         r2_oos=None, oos_start=None, extra_title=""):
    _plot_delta_grid([pred1_label, pred2_label], "rolmu", horizon_label, oos_gap,
                     nw_lags, color_palette, sim_dict, betas_df, bah_sim, out_path,
                     r2_oos=r2_oos, oos_start=oos_start, extra_title=extra_title)


# ─── Unit-position builders ───────────────────────────────────────────────────
# All six strategies share one skeleton: load-from-cache → expanding-window betas
# + ŷ + |t|-gate → mode-specific long/short assignment → cache. The wrappers below
# differ only in the cache key, the stored series name, and the threshold rule
# (sym / asym / rolmu); they delegate the skeleton to _run_unit. Cache keys are
# kept byte-identical to the originals so existing regression_cache/ files hit.

def _run_unit(panel, preds, fwd_col, oos_gap, nw_lags, mode, delta, t_thresh,
              cache_tag, pos_name):
    """Shared unit-position backtest. `preds` is the 1-2 predictor columns; `mode`
    selects the assignment rule:
        sym   — long ŷ > delta, short ŷ < -delta
        asym  — long (ŷ-µ) > delta, short (-ŷ) > delta   (long wins ties)
        rolmu — long (ŷ-µ) > delta, short (µ-ŷ) > delta   (long wins ties)
    """
    cache = CACHE_DIR / cache_tag
    if cache.exists():
        return pd.read_parquet(cache).squeeze().rename(pos_name)

    sub = panel.dropna(subset=[*preds, fwd_col]).copy()
    if len(preds) == 1:
        betas_df = compute_betas(panel, preds[0], fwd_col, oos_gap, nw_lags)
        y_hat    = _yhat_univariate(panel, preds[0], fwd_col, betas_df)
        fire     = betas_df["t_stat"].abs() > t_thresh
    else:
        betas_df = compute_betas_bivariate(panel, preds[0], preds[1], fwd_col, oos_gap, nw_lags)
        y_hat    = _yhat_bivariate(panel, preds[0], preds[1], fwd_col, betas_df)
        fire     = ((betas_df["t_stat_1"].abs() > t_thresh)
                    & (betas_df["t_stat_2"].abs() > t_thresh))

    pos = pd.Series(0.0, index=sub.index, name=pos_name)
    idx = y_hat.index[fire.reindex(y_hat.index, fill_value=False)]
    yh  = y_hat.loc[idx]

    if mode == "sym":
        pos.loc[idx[yh >  delta]] =  1.0
        pos.loc[idx[yh < -delta]] = -1.0
    else:
        mu         = _rolling_mu(panel, fwd_col, oos_gap,
                                 predictor=(preds[0] if len(preds) == 1 else list(preds)))
        excess     = yh - mu.reindex(idx)
        long_mask  = excess > delta
        short_mask = (-yh > delta) if mode == "asym" else (-excess > delta)
        pos.loc[idx[long_mask]]               =  1.0
        pos.loc[idx[short_mask & ~long_mask]] = -1.0

    pos.to_frame().to_parquet(cache)
    return pos


def run_ew(panel, predictor, fwd_col, oos_gap, nw_lags, delta, t_thresh=T_THRESH):
    """±1 when |ŷ| > delta and t-stat gate passes."""
    tag = (f"pos_EW_{predictor}_{fwd_col}_d{int(delta*10000)}bps"
           f"_t{int(t_thresh*100)}_oos{OOS_START}.parquet")
    return _run_unit(panel, [predictor], fwd_col, oos_gap, nw_lags, "sym",
                     delta, t_thresh, tag, f"pos_{predictor}_{delta}")


def run_ew_bivariate(panel, pred1, pred2, fwd_col, oos_gap, nw_lags, delta, t_thresh=T_THRESH):
    """Both betas must pass the |t| > t_thresh gate."""
    tag = (f"pos_EWbiv_{pred1}_{pred2}_{fwd_col}_d{int(delta*10000)}bps"
           f"_t{int(t_thresh*100)}_oos{OOS_START}.parquet")
    return _run_unit(panel, [pred1, pred2], fwd_col, oos_gap, nw_lags, "sym",
                     delta, t_thresh, tag, f"pos_{pred1}_{pred2}_{delta}")


def run_ew_asym(panel, predictor, fwd_col, oos_gap, nw_lags, delta=0.0, t_thresh=T_THRESH):
    """Long if (ŷ-µ₅₀₀) > delta, short if -ŷ > delta."""
    d_sfx = f"_d{int(round(delta * 10000))}" if delta > 0 else ""
    tag = (f"pos_EWasym_{predictor}_{fwd_col}"
           f"_t{int(t_thresh*100)}{d_sfx}_oos{OOS_START}.parquet")
    return _run_unit(panel, [predictor], fwd_col, oos_gap, nw_lags, "asym",
                     delta, t_thresh, tag, f"pos_asym_{predictor}")


def run_ew_asym_bivariate(panel, pred1, pred2, fwd_col, oos_gap, nw_lags, delta=0.0, t_thresh=T_THRESH):
    """Both betas must pass gate. Long if (ŷ-µ₅₀₀) > delta, short if -ŷ > delta."""
    d_sfx = f"_d{int(round(delta * 10000))}" if delta > 0 else ""
    tag = (f"pos_EWasym_{pred1}_{pred2}_{fwd_col}"
           f"_t{int(t_thresh*100)}{d_sfx}_oos{OOS_START}.parquet")
    return _run_unit(panel, [pred1, pred2], fwd_col, oos_gap, nw_lags, "asym",
                     delta, t_thresh, tag, f"pos_asym_{pred1}_{pred2}")


def run_ew_rolmu(panel, predictor, fwd_col, oos_gap, nw_lags, delta=0.0, t_thresh=T_THRESH):
    """Long if (ŷ-µ₅₀₀) > delta, short if (µ₅₀₀-ŷ) > delta."""
    d_sfx = f"_d{int(round(delta * 10000))}" if delta > 0 else ""
    tag = (f"pos_EWrolmu_{predictor}_{fwd_col}"
           f"_t{int(t_thresh*100)}{d_sfx}_oos{OOS_START}.parquet")
    return _run_unit(panel, [predictor], fwd_col, oos_gap, nw_lags, "rolmu",
                     delta, t_thresh, tag, f"pos_rolmu_{predictor}")


def run_ew_rolmu_bivariate(panel, pred1, pred2, fwd_col, oos_gap, nw_lags, delta=0.0, t_thresh=T_THRESH):
    """Both betas must pass gate. Long if (ŷ-µ₅₀₀) > delta, short if (µ₅₀₀-ŷ) > delta."""
    d_sfx = f"_d{int(round(delta * 10000))}" if delta > 0 else ""
    tag = (f"pos_EWrolmu_{pred1}_{pred2}_{fwd_col}"
           f"_t{int(t_thresh*100)}{d_sfx}_oos{OOS_START}.parquet")
    return _run_unit(panel, [pred1, pred2], fwd_col, oos_gap, nw_lags, "rolmu",
                     delta, t_thresh, tag, f"pos_rolmu_{pred1}_{pred2}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("  base_strategies.py — base (unit-position) strategy evaluation")
    print("  7 models × 3 strategies = 21 plots")
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

    daily_ret = panel["daily_ret"].dropna()
    bah_pos   = compute_buy_and_hold(daily_ret)
    bah_sim   = simulate_strategy(bah_pos, daily_ret)
    FWD       = "fwd_20d"

    # ── Output directories ──
    out_vrp   = OUTPUT / "expanding_window" / "VRP"
    out_vvix5 = OUTPUT / "expanding_window" / "VVIX MA5"
    out_vvix10= OUTPUT / "expanding_window" / "VVIX MA10"
    out_biv5  = OUTPUT / "expanding_window" / "VRP + VVIX MA5"
    out_biv10 = OUTPUT / "expanding_window" / "VRP + VVIX MA10"
    out_ts    = OUTPUT / "expanding_window" / "VRP + Term Slope"
    out_oi    = OUTPUT / "expanding_window" / "VRP + Open Interest"
    for d in [out_vrp, out_vvix5, out_vvix10, out_biv5, out_biv10, out_ts, out_oi]:
        d.mkdir(parents=True, exist_ok=True)

    # ── Shared palettes ──
    pal_vrp   = ["#08306b", "#2171b5", "#4292c6", "#6baed6"]
    pal_vvix5 = ["#3f007d", "#6a51a3", "#807dba", "#9e9ac8"]
    pal_vvix10= ["#7a0177", "#c51b8a", "#f768a1", "#fbb4b9"]
    pal_biv5  = ["#3f007d", "#6a51a3", "#9e9ac8", "#dadaeb"]
    pal_biv10 = ["#ae017e", "#dd3497", "#f768a1", "#fbb4b9"]
    pal_ts    = ["#00441b", "#006d2c", "#31a354", "#74c476"]
    pal_oi    = ["#54278f", "#756bb1", "#9e9ac8", "#cbc9e2"]

    # ── Pre-compute betas ──
    print("\n[2] Computing betas...")
    vrp_betas   = compute_betas(panel, "VP",         FWD, OOS_GAP, NW_LAGS)
    vvix5_betas = compute_betas(panel, "vvix_ma5",   FWD, OOS_GAP, NW_LAGS)
    vvix10_betas= compute_betas(panel, "vvix_ma10",  FWD, OOS_GAP, NW_LAGS)
    biv5_betas  = compute_betas_bivariate(panel, "VP", "vvix_ma5",      FWD, OOS_GAP, NW_LAGS)
    biv10_betas = compute_betas_bivariate(panel, "VP", "vvix_ma10",     FWD, OOS_GAP, NW_LAGS)
    ts_betas    = compute_betas_bivariate(panel, "VP", "term_slope",    FWD, OOS_GAP, NW_LAGS)
    oi_betas    = compute_betas_bivariate(panel, "VP", "open_interest", FWD, OOS_GAP, NW_LAGS)

    # ══════════════════════════════════════════════════════════════════════════
    # 1. VRP
    # ══════════════════════════════════════════════════════════════════════════

    print("\n[1.1] VRP symmetric...")
    sim_dict_vrp = {}
    for di, delta in enumerate(DELTAS):
        pos = run_ew(panel, "VP", FWD, OOS_GAP, NW_LAGS, delta)
        sim_dict_vrp[di] = (delta, simulate_strategy(pos, daily_ret))
    _print_delta_stats(sim_dict_vrp, bah_sim)
    plot_2panel(
        pred_label="VRP", horizon_label="20-day",
        oos_gap=OOS_GAP, nw_lags=NW_LAGS,
        color_palette=pal_vrp,
        sim_dict=sim_dict_vrp, betas_df=vrp_betas, bah_sim=bah_sim,
        out_path=out_vrp / "symmetric_VRP.png",
    )

    print("\n[1.2] VRP asymmetric...")
    asym_sims_vrp = {}
    for di, delta in enumerate(DELTAS):
        pos = run_ew_asym(panel, "VP", FWD, OOS_GAP, NW_LAGS, delta=delta)
        asym_sims_vrp[di] = (delta, simulate_strategy(pos, daily_ret))
    _print_delta_stats(asym_sims_vrp, bah_sim)
    plot_asym(
        pred_label="VRP", horizon_label="20-day",
        oos_gap=OOS_GAP, nw_lags=NW_LAGS,
        color_palette=pal_vrp,
        sim_dict=asym_sims_vrp, betas_df=vrp_betas, bah_sim=bah_sim,
        out_path=out_vrp / "asymmetric_VRP.png",
    )

    print("\n[1.3] VRP base-return-shift...")
    rolmu_sims_vrp = {}
    for di, delta in enumerate(DELTAS):
        pos = run_ew_rolmu(panel, "VP", FWD, OOS_GAP, NW_LAGS, delta=delta)
        rolmu_sims_vrp[di] = (delta, simulate_strategy(pos, daily_ret))
    _print_delta_stats(rolmu_sims_vrp, bah_sim)
    plot_rolmu(
        pred_label="VRP", horizon_label="20-day",
        oos_gap=OOS_GAP, nw_lags=NW_LAGS,
        color_palette=pal_vrp,
        sim_dict=rolmu_sims_vrp, betas_df=vrp_betas, bah_sim=bah_sim,
        out_path=out_vrp / "base_return_shift_VRP.png",
    )

    # ══════════════════════════════════════════════════════════════════════════
    # 2. VVIX MA5
    # ══════════════════════════════════════════════════════════════════════════

    print("\n[2.1] VVIX MA5 symmetric...")
    sim_dict_vvix5 = {}
    for di, delta in enumerate(DELTAS):
        pos = run_ew(panel, "vvix_ma5", FWD, OOS_GAP, NW_LAGS, delta)
        sim_dict_vvix5[di] = (delta, simulate_strategy(pos, daily_ret))
    _print_delta_stats(sim_dict_vvix5, bah_sim)
    plot_2panel(
        pred_label="VVIX MA5", horizon_label="20-day",
        oos_gap=OOS_GAP, nw_lags=NW_LAGS,
        color_palette=pal_vvix5,
        sim_dict=sim_dict_vvix5, betas_df=vvix5_betas, bah_sim=bah_sim,
        out_path=out_vvix5 / "symmetric_VVIX_MA5.png",
    )

    print("\n[2.2] VVIX MA5 asymmetric...")
    asym_sims_vvix5 = {}
    for di, delta in enumerate(DELTAS):
        pos = run_ew_asym(panel, "vvix_ma5", FWD, OOS_GAP, NW_LAGS, delta=delta)
        asym_sims_vvix5[di] = (delta, simulate_strategy(pos, daily_ret))
    _print_delta_stats(asym_sims_vvix5, bah_sim)
    plot_asym(
        pred_label="VVIX MA5", horizon_label="20-day",
        oos_gap=OOS_GAP, nw_lags=NW_LAGS,
        color_palette=pal_vvix5,
        sim_dict=asym_sims_vvix5, betas_df=vvix5_betas, bah_sim=bah_sim,
        out_path=out_vvix5 / "asymmetric_VVIX_MA5.png",
    )

    print("\n[2.3] VVIX MA5 base-return-shift...")
    rolmu_sims_vvix5 = {}
    for di, delta in enumerate(DELTAS):
        pos = run_ew_rolmu(panel, "vvix_ma5", FWD, OOS_GAP, NW_LAGS, delta=delta)
        rolmu_sims_vvix5[di] = (delta, simulate_strategy(pos, daily_ret))
    _print_delta_stats(rolmu_sims_vvix5, bah_sim)
    plot_rolmu(
        pred_label="VVIX MA5", horizon_label="20-day",
        oos_gap=OOS_GAP, nw_lags=NW_LAGS,
        color_palette=pal_vvix5,
        sim_dict=rolmu_sims_vvix5, betas_df=vvix5_betas, bah_sim=bah_sim,
        out_path=out_vvix5 / "base_return_shift_VVIX_MA5.png",
    )

    # ══════════════════════════════════════════════════════════════════════════
    # 3. VVIX MA10
    # ══════════════════════════════════════════════════════════════════════════

    print("\n[3.1] VVIX MA10 symmetric...")
    sim_dict_vvix10 = {}
    for di, delta in enumerate(DELTAS):
        pos = run_ew(panel, "vvix_ma10", FWD, OOS_GAP, NW_LAGS, delta)
        sim_dict_vvix10[di] = (delta, simulate_strategy(pos, daily_ret))
    _print_delta_stats(sim_dict_vvix10, bah_sim)
    plot_2panel(
        pred_label="VVIX MA10", horizon_label="20-day",
        oos_gap=OOS_GAP, nw_lags=NW_LAGS,
        color_palette=pal_vvix10,
        sim_dict=sim_dict_vvix10, betas_df=vvix10_betas, bah_sim=bah_sim,
        out_path=out_vvix10 / "symmetric_VVIX_MA10.png",
    )

    print("\n[3.2] VVIX MA10 asymmetric...")
    asym_sims_vvix10 = {}
    for di, delta in enumerate(DELTAS):
        pos = run_ew_asym(panel, "vvix_ma10", FWD, OOS_GAP, NW_LAGS, delta=delta)
        asym_sims_vvix10[di] = (delta, simulate_strategy(pos, daily_ret))
    _print_delta_stats(asym_sims_vvix10, bah_sim)
    plot_asym(
        pred_label="VVIX MA10", horizon_label="20-day",
        oos_gap=OOS_GAP, nw_lags=NW_LAGS,
        color_palette=pal_vvix10,
        sim_dict=asym_sims_vvix10, betas_df=vvix10_betas, bah_sim=bah_sim,
        out_path=out_vvix10 / "asymmetric_VVIX_MA10.png",
    )

    print("\n[3.3] VVIX MA10 base-return-shift...")
    rolmu_sims_vvix10 = {}
    for di, delta in enumerate(DELTAS):
        pos = run_ew_rolmu(panel, "vvix_ma10", FWD, OOS_GAP, NW_LAGS, delta=delta)
        rolmu_sims_vvix10[di] = (delta, simulate_strategy(pos, daily_ret))
    _print_delta_stats(rolmu_sims_vvix10, bah_sim)
    plot_rolmu(
        pred_label="VVIX MA10", horizon_label="20-day",
        oos_gap=OOS_GAP, nw_lags=NW_LAGS,
        color_palette=pal_vvix10,
        sim_dict=rolmu_sims_vvix10, betas_df=vvix10_betas, bah_sim=bah_sim,
        out_path=out_vvix10 / "base_return_shift_VVIX_MA10.png",
    )

    # ══════════════════════════════════════════════════════════════════════════
    # 4. VRP + VVIX MA5
    # ══════════════════════════════════════════════════════════════════════════

    print("\n[4.1] VRP + VVIX MA5 symmetric...")
    sim_dict_biv5 = {}
    for di, delta in enumerate(DELTAS):
        pos = run_ew_bivariate(panel, "VP", "vvix_ma5", FWD, OOS_GAP, NW_LAGS, delta)
        sim_dict_biv5[di] = (delta, simulate_strategy(pos, daily_ret))
    _print_delta_stats(sim_dict_biv5, bah_sim)
    plot_2panel_bivariate(
        pred1_label="VRP", pred2_label="VVIX MA5", horizon_label="20-day",
        oos_gap=OOS_GAP, nw_lags=NW_LAGS,
        color_palette=pal_biv5,
        sim_dict=sim_dict_biv5, betas_df=biv5_betas, bah_sim=bah_sim,
        out_path=out_biv5 / "symmetric_VRP_+_VVIX_MA5.png",
    )

    print("\n[4.2] VRP + VVIX MA5 asymmetric...")
    asym_sims_biv5 = {}
    for di, delta in enumerate(DELTAS):
        pos = run_ew_asym_bivariate(panel, "VP", "vvix_ma5", FWD, OOS_GAP, NW_LAGS, delta=delta)
        asym_sims_biv5[di] = (delta, simulate_strategy(pos, daily_ret))
    _print_delta_stats(asym_sims_biv5, bah_sim)
    plot_asym_bivariate(
        pred1_label="VRP", pred2_label="VVIX MA5", horizon_label="20-day",
        oos_gap=OOS_GAP, nw_lags=NW_LAGS,
        color_palette=pal_biv5,
        sim_dict=asym_sims_biv5, betas_df=biv5_betas, bah_sim=bah_sim,
        out_path=out_biv5 / "asymmetric_VRP_+_VVIX_MA5.png",
    )

    print("\n[4.3] VRP + VVIX MA5 base-return-shift...")
    rolmu_sims_biv5 = {}
    for di, delta in enumerate(DELTAS):
        pos = run_ew_rolmu_bivariate(panel, "VP", "vvix_ma5", FWD, OOS_GAP, NW_LAGS, delta=delta)
        rolmu_sims_biv5[di] = (delta, simulate_strategy(pos, daily_ret))
    _print_delta_stats(rolmu_sims_biv5, bah_sim)
    plot_rolmu_bivariate(
        pred1_label="VRP", pred2_label="VVIX MA5", horizon_label="20-day",
        oos_gap=OOS_GAP, nw_lags=NW_LAGS,
        color_palette=pal_biv5,
        sim_dict=rolmu_sims_biv5, betas_df=biv5_betas, bah_sim=bah_sim,
        out_path=out_biv5 / "base_return_shift_VRP_+_VVIX_MA5.png",
    )

    # ══════════════════════════════════════════════════════════════════════════
    # 5. VRP + VVIX MA10
    # ══════════════════════════════════════════════════════════════════════════

    print("\n[5.1] VRP + VVIX MA10 symmetric...")
    sim_dict_biv10 = {}
    for di, delta in enumerate(DELTAS):
        pos = run_ew_bivariate(panel, "VP", "vvix_ma10", FWD, OOS_GAP, NW_LAGS, delta)
        sim_dict_biv10[di] = (delta, simulate_strategy(pos, daily_ret))
    _print_delta_stats(sim_dict_biv10, bah_sim)
    plot_2panel_bivariate(
        pred1_label="VRP", pred2_label="VVIX MA10", horizon_label="20-day",
        oos_gap=OOS_GAP, nw_lags=NW_LAGS,
        color_palette=pal_biv10,
        sim_dict=sim_dict_biv10, betas_df=biv10_betas, bah_sim=bah_sim,
        out_path=out_biv10 / "symmetric_VRP_+_VVIX_MA10.png",
    )

    print("\n[5.2] VRP + VVIX MA10 asymmetric...")
    asym_sims_biv10 = {}
    for di, delta in enumerate(DELTAS):
        pos = run_ew_asym_bivariate(panel, "VP", "vvix_ma10", FWD, OOS_GAP, NW_LAGS, delta=delta)
        asym_sims_biv10[di] = (delta, simulate_strategy(pos, daily_ret))
    _print_delta_stats(asym_sims_biv10, bah_sim)
    plot_asym_bivariate(
        pred1_label="VRP", pred2_label="VVIX MA10", horizon_label="20-day",
        oos_gap=OOS_GAP, nw_lags=NW_LAGS,
        color_palette=pal_biv10,
        sim_dict=asym_sims_biv10, betas_df=biv10_betas, bah_sim=bah_sim,
        out_path=out_biv10 / "asymmetric_VRP_+_VVIX_MA10.png",
    )

    print("\n[5.3] VRP + VVIX MA10 base-return-shift...")
    rolmu_sims_biv10 = {}
    for di, delta in enumerate(DELTAS):
        pos = run_ew_rolmu_bivariate(panel, "VP", "vvix_ma10", FWD, OOS_GAP, NW_LAGS, delta=delta)
        rolmu_sims_biv10[di] = (delta, simulate_strategy(pos, daily_ret))
    _print_delta_stats(rolmu_sims_biv10, bah_sim)
    plot_rolmu_bivariate(
        pred1_label="VRP", pred2_label="VVIX MA10", horizon_label="20-day",
        oos_gap=OOS_GAP, nw_lags=NW_LAGS,
        color_palette=pal_biv10,
        sim_dict=rolmu_sims_biv10, betas_df=biv10_betas, bah_sim=bah_sim,
        out_path=out_biv10 / "base_return_shift_VRP_+_VVIX_MA10.png",
    )

    # ══════════════════════════════════════════════════════════════════════════
    # 6. VRP + Term Slope
    # ══════════════════════════════════════════════════════════════════════════

    print("\n[6.1] VRP + Term Slope symmetric...")
    sim_dict_ts = {}
    for di, delta in enumerate(DELTAS):
        pos = run_ew_bivariate(panel, "VP", "term_slope", FWD, OOS_GAP, NW_LAGS, delta)
        sim_dict_ts[di] = (delta, simulate_strategy(pos, daily_ret))
    _print_delta_stats(sim_dict_ts, bah_sim)
    plot_2panel_bivariate(
        pred1_label="VRP", pred2_label="Term Slope", horizon_label="20-day",
        oos_gap=OOS_GAP, nw_lags=NW_LAGS,
        color_palette=pal_ts,
        sim_dict=sim_dict_ts, betas_df=ts_betas, bah_sim=bah_sim,
        out_path=out_ts / "symmetric_VRP_+_Term_Slope.png",
    )

    print("\n[6.2] VRP + Term Slope asymmetric...")
    asym_sims_ts = {}
    for di, delta in enumerate(DELTAS):
        pos = run_ew_asym_bivariate(panel, "VP", "term_slope", FWD, OOS_GAP, NW_LAGS, delta=delta)
        asym_sims_ts[di] = (delta, simulate_strategy(pos, daily_ret))
    _print_delta_stats(asym_sims_ts, bah_sim)
    plot_asym_bivariate(
        pred1_label="VRP", pred2_label="Term Slope", horizon_label="20-day",
        oos_gap=OOS_GAP, nw_lags=NW_LAGS,
        color_palette=pal_ts,
        sim_dict=asym_sims_ts, betas_df=ts_betas, bah_sim=bah_sim,
        out_path=out_ts / "asymmetric_VRP_+_Term_Slope.png",
    )

    print("\n[6.3] VRP + Term Slope base-return-shift...")
    rolmu_sims_ts = {}
    for di, delta in enumerate(DELTAS):
        pos = run_ew_rolmu_bivariate(panel, "VP", "term_slope", FWD, OOS_GAP, NW_LAGS, delta=delta)
        rolmu_sims_ts[di] = (delta, simulate_strategy(pos, daily_ret))
    _print_delta_stats(rolmu_sims_ts, bah_sim)
    plot_rolmu_bivariate(
        pred1_label="VRP", pred2_label="Term Slope", horizon_label="20-day",
        oos_gap=OOS_GAP, nw_lags=NW_LAGS,
        color_palette=pal_ts,
        sim_dict=rolmu_sims_ts, betas_df=ts_betas, bah_sim=bah_sim,
        out_path=out_ts / "base_return_shift_VRP_+_Term_Slope.png",
    )

    # ══════════════════════════════════════════════════════════════════════════
    # 7. VRP + Open Interest
    # ══════════════════════════════════════════════════════════════════════════

    print("\n[7.1] VRP + Open Interest symmetric...")
    sim_dict_oi = {}
    for di, delta in enumerate(DELTAS):
        pos = run_ew_bivariate(panel, "VP", "open_interest", FWD, OOS_GAP, NW_LAGS, delta)
        sim_dict_oi[di] = (delta, simulate_strategy(pos, daily_ret))
    _print_delta_stats(sim_dict_oi, bah_sim)
    plot_2panel_bivariate(
        pred1_label="VRP", pred2_label="Open Interest", horizon_label="20-day",
        oos_gap=OOS_GAP, nw_lags=NW_LAGS,
        color_palette=pal_oi,
        sim_dict=sim_dict_oi, betas_df=oi_betas, bah_sim=bah_sim,
        out_path=out_oi / "symmetric_VRP_+_Open_Interest.png",
    )

    print("\n[7.2] VRP + Open Interest asymmetric...")
    asym_sims_oi = {}
    for di, delta in enumerate(DELTAS):
        pos = run_ew_asym_bivariate(panel, "VP", "open_interest", FWD, OOS_GAP, NW_LAGS, delta=delta)
        asym_sims_oi[di] = (delta, simulate_strategy(pos, daily_ret))
    _print_delta_stats(asym_sims_oi, bah_sim)
    plot_asym_bivariate(
        pred1_label="VRP", pred2_label="Open Interest", horizon_label="20-day",
        oos_gap=OOS_GAP, nw_lags=NW_LAGS,
        color_palette=pal_oi,
        sim_dict=asym_sims_oi, betas_df=oi_betas, bah_sim=bah_sim,
        out_path=out_oi / "asymmetric_VRP_+_Open_Interest.png",
    )

    print("\n[7.3] VRP + Open Interest base-return-shift...")
    rolmu_sims_oi = {}
    for di, delta in enumerate(DELTAS):
        pos = run_ew_rolmu_bivariate(panel, "VP", "open_interest", FWD, OOS_GAP, NW_LAGS, delta=delta)
        rolmu_sims_oi[di] = (delta, simulate_strategy(pos, daily_ret))
    _print_delta_stats(rolmu_sims_oi, bah_sim)
    plot_rolmu_bivariate(
        pred1_label="VRP", pred2_label="Open Interest", horizon_label="20-day",
        oos_gap=OOS_GAP, nw_lags=NW_LAGS,
        color_palette=pal_oi,
        sim_dict=rolmu_sims_oi, betas_df=oi_betas, bah_sim=bah_sim,
        out_path=out_oi / "base_return_shift_VRP_+_Open_Interest.png",
    )

    print("\n" + "=" * 72)
    print("  Done — 21 base strategy plots saved.")
    print("=" * 72)


if __name__ == "__main__":
    main()
