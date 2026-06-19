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
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

ROOT      = Path(__file__).parent
OUTPUT    = ROOT / "output"
CACHE_DIR = OUTPUT / "regression_cache"

sys.path.insert(0, str(ROOT.parent / "bh_replication"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from statsmodels.api import OLS, add_constant
from har_model import _nw_se

from experiment2 import (
    load_vrp_series, load_es_front_month, load_vvix,
    compute_vvix_ma5, compute_vvix_ma10,
    load_vix_spot, load_vix_futures_term_structure,
    load_es_open_interest,
    compute_buy_and_hold, simulate_strategy, compute_performance_stats,
)
from fh_replication.fh_replication import compute_vix_term_slope
from regressions import (
    build_panel,
    compute_betas, compute_betas_bivariate,
    compute_oos_r2, compute_oos_r2_bivariate,
    _yhat_univariate, _yhat_bivariate, _rolling_mu,
    _shade, oos_cumret,
    OOS_START, MIN_WIN, T_THRESH, DELTAS, DELTA_LBL, OOS_GAP, NW_LAGS,
)

BAH_COLOR = "#d62728"


# ─── Plotting helpers ─────────────────────────────────────────────────────────

def _align_zero(ax_left, ax_right):
    lo1, hi1 = ax_left.get_ylim()
    lo2, hi2 = ax_right.get_ylim()
    half1 = max(abs(lo1), abs(hi1)) or 1.0
    half2 = max(abs(lo2), abs(hi2)) or 1.0
    ax_left.set_ylim(-half1, half1)
    ax_right.set_ylim(-half2, half2)


def _stat_window(sim_dict_or_sim, bah_sim):
    """Find earliest activation; return (rebase_start, stat_start, stat_lbl)."""
    if isinstance(sim_dict_or_sim, dict):
        candidates = []
        for _, sim in sim_dict_or_sim.values():
            p = sim["position"][sim.index >= OOS_START]
            a = p[p != 0]
            if len(a):
                candidates.append(a.index.min())
        activation = min(candidates) if candidates else None
    else:
        sim = sim_dict_or_sim
        p = sim["position"][sim.index >= OOS_START]
        a = p[p != 0]
        activation = a.index.min() if len(a) else None

    rebase_start = None
    if activation is not None and activation > pd.Timestamp("2020-01-01"):
        idx          = bah_sim.index
        act_iloc     = idx.searchsorted(activation)
        rebase_start = idx[min(act_iloc + 1, len(idx) - 1)]

    stat_start = rebase_start if rebase_start is not None else pd.Timestamp(OOS_START)
    stat_lbl   = (f" · stats from {stat_start.strftime('%Y-%m-%d')}"
                  if rebase_start is not None else "")
    return rebase_start, stat_start, stat_lbl


# ─── plot_2panel: symmetric multi-delta (univariate) ─────────────────────────

def plot_2panel(pred_label, horizon_label, oos_gap, nw_lags,
                color_palette, sim_dict, betas_df, bah_sim, out_path,
                r2_oos=None, oos_start=None):
    _oos_start = oos_start if oos_start is not None else OOS_START
    oos_dt   = pd.Timestamp(_oos_start)
    e_dt     = betas_df.index[-1]
    xlim     = (oos_dt, e_dt)
    n_deltas = len(DELTAS)

    _rebase_start, _stat_start, _stat_lbl = _stat_window(sim_dict, bah_sim)

    r2_str = f"  R²_OOS = {r2_oos:+.4f}" if r2_oos is not None else ""

    h_ratios = [2.5, 1.2] + [1.0] * n_deltas
    fig, axes = plt.subplots(
        2 + n_deltas, 1, figsize=(14, 11 + 2.2 * n_deltas), sharex=True,
        gridspec_kw={"height_ratios": h_ratios, "hspace": 0.35},
    )
    ax_ret = axes[0]
    ax_t   = axes[1]
    ax_pos = axes[2:]

    fig.suptitle(
        f"{pred_label} -> {horizon_label} Forward Return  "
        f"(Expanding Window, OOS from {_oos_start}){r2_str}\n"
        f"Training grows daily; OOS gap = {oos_gap} days; "
        f"NW-HAC {nw_lags} lags; |t| > {T_THRESH:.2f} gate; 0.05% slippage",
        fontsize=10, y=0.998,
    )
    fig.subplots_adjust(top=0.955, bottom=0.03, left=0.10, right=0.93)

    ax_ret.set_xlim(*xlim)
    _shade(ax_ret, oos_dt, e_dt)

    _bah_st_plot = compute_performance_stats(
        bah_sim[bah_sim.index >= _stat_start], "BaH_plot")
    bah_oos = oos_cumret(bah_sim, start=_oos_start)
    ax_ret.plot(bah_oos.index, bah_oos.values,
                color=BAH_COLOR, lw=1.5, ls="-.", alpha=0.6,
                label=(f"Buy-and-Hold{_stat_lbl}  "
                       f"[SR={_bah_st_plot['sharpe']:+.2f}  "
                       f"ret={_bah_st_plot['ann_ret']*100:+.1f}%  "
                       f"DD={_bah_st_plot['max_dd']*100:.1f}%]"))

    if _rebase_start is not None:
        _bah_from   = bah_sim["net_pnl"][bah_sim.index >= _rebase_start]
        _bah_act    = (1 + _bah_from).cumprod()
        _bah_act_st = compute_performance_stats(
            bah_sim[bah_sim.index >= _rebase_start], "BaH_act")
        ax_ret.plot(
            _bah_act.index, _bah_act.values,
            color=BAH_COLOR, lw=1.2, ls=":", alpha=0.85,
            label=(f"Buy-and-Hold from {_rebase_start.strftime('%Y-%m-%d')}  "
                   f"[SR={_bah_act_st['sharpe']:+.2f}  "
                   f"ret={_bah_act_st['ann_ret']*100:+.1f}%  "
                   f"DD={_bah_act_st['max_dd']*100:.1f}%]"),
        )

    for di, (delta, lbl) in enumerate(zip(DELTAS, DELTA_LBL)):
        _, sim = sim_dict[di]
        cum     = oos_cumret(sim, start=_oos_start)
        st_plot = compute_performance_stats(
            sim[sim.index >= _stat_start], f"EW_{di}_plot")
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

    t_series = betas_df["t_stat"]
    b_series = betas_df["beta"]

    ax_t.plot(t_series.index, t_series.values,
              color=color_palette[0], lw=1.0, alpha=0.85,
              label=f"NW t-stat of {pred_label} ({nw_lags}-lag HAC)")
    ax_t.fill_between(t_series.index, -T_THRESH, T_THRESH,
                      color="firebrick", alpha=0.05, label="Below gate (flat zone)")
    ax_t.axhline( T_THRESH, color="firebrick", lw=1.2, ls="--",
                 label=f"|t| = {T_THRESH:.2f} gate")
    ax_t.axhline(-T_THRESH, color="firebrick", lw=1.2, ls="--")
    ax_t.axhline(0, color="black", lw=0.5, ls=":")
    ax_t.set_ylabel(f"NW t-stat ({pred_label})", fontsize=9)
    ax_t.grid(axis="y", alpha=0.2, lw=0.6)
    ax_t.spines["top"].set_visible(False)

    ax_t2 = ax_t.twinx()
    ax_t2.plot(b_series.index, b_series.values,
               color="dimgrey", lw=1.0, ls="--", alpha=0.60,
               label=f"Beta ({pred_label})")
    ax_t2.axhline(0, color="dimgrey", lw=0.4, ls=":")
    ax_t2.set_ylabel(f"Beta ({pred_label})", fontsize=8, color="dimgrey")
    ax_t2.tick_params(axis="y", labelcolor="dimgrey", labelsize=7)
    ax_t2.spines["top"].set_visible(False)
    _align_zero(ax_t, ax_t2)

    lines1, labs1 = ax_t.get_legend_handles_labels()
    lines2, labs2 = ax_t2.get_legend_handles_labels()
    ax_t.legend(lines1 + lines2, labs1 + labs2, fontsize=8, loc="upper left")

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

    for ax in [ax_ret, ax_t] + list(ax_pos):
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.tick_params(axis="x", which="major", labelbottom=True, labelsize=7, pad=2)

    fig.savefig(out_path, dpi=155, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ─── plot_2panel_bivariate: symmetric multi-delta (bivariate) ────────────────

def plot_2panel_bivariate(pred1_label, pred2_label, horizon_label, oos_gap, nw_lags,
                          color_palette, sim_dict, betas_df, bah_sim, out_path,
                          r2_oos=None, oos_start=None):
    _oos_start = oos_start if oos_start is not None else OOS_START
    oos_dt   = pd.Timestamp(_oos_start)
    e_dt     = betas_df.index[-1]
    xlim     = (oos_dt, e_dt)
    n_deltas = len(DELTAS)

    _rebase_start, _stat_start, _stat_lbl = _stat_window(sim_dict, bah_sim)

    r2_str = f"  R²_OOS = {r2_oos:+.4f}" if r2_oos is not None else ""

    h_ratios = [2.5, 1.2] + [1.0] * n_deltas
    fig, axes = plt.subplots(
        2 + n_deltas, 1, figsize=(14, 11 + 2.2 * n_deltas), sharex=True,
        gridspec_kw={"height_ratios": h_ratios, "hspace": 0.35},
    )
    ax_ret = axes[0]
    ax_t   = axes[1]
    ax_pos = axes[2:]

    fig.suptitle(
        f"{pred1_label} + {pred2_label} -> {horizon_label} Forward Return  "
        f"(Expanding Window, OOS from {_oos_start}){r2_str}\n"
        f"Training grows daily; OOS gap = {oos_gap} days; "
        f"NW-HAC {nw_lags} lags; |t| > {T_THRESH:.2f} gate (both betas); 0.05% slippage",
        fontsize=10, y=0.998,
    )
    fig.subplots_adjust(top=0.955, bottom=0.03, left=0.10, right=0.93)

    ax_ret.set_xlim(*xlim)
    _shade(ax_ret, oos_dt, e_dt)

    _bah_st_plot = compute_performance_stats(
        bah_sim[bah_sim.index >= _stat_start], "BaH_plot")
    bah_oos = oos_cumret(bah_sim, start=_oos_start)
    ax_ret.plot(bah_oos.index, bah_oos.values,
                color=BAH_COLOR, lw=1.5, ls="-.", alpha=0.6,
                label=(f"Buy-and-Hold{_stat_lbl}  "
                       f"[SR={_bah_st_plot['sharpe']:+.2f}  "
                       f"ret={_bah_st_plot['ann_ret']*100:+.1f}%  "
                       f"DD={_bah_st_plot['max_dd']*100:.1f}%]"))

    if _rebase_start is not None:
        _bah_from   = bah_sim["net_pnl"][bah_sim.index >= _rebase_start]
        _bah_act    = (1 + _bah_from).cumprod()
        _bah_act_st = compute_performance_stats(
            bah_sim[bah_sim.index >= _rebase_start], "BaH_act")
        ax_ret.plot(
            _bah_act.index, _bah_act.values,
            color=BAH_COLOR, lw=1.2, ls=":", alpha=0.85,
            label=(f"Buy-and-Hold from {_rebase_start.strftime('%Y-%m-%d')}  "
                   f"[SR={_bah_act_st['sharpe']:+.2f}  "
                   f"ret={_bah_act_st['ann_ret']*100:+.1f}%  "
                   f"DD={_bah_act_st['max_dd']*100:.1f}%]"),
        )

    for di, (delta, lbl) in enumerate(zip(DELTAS, DELTA_LBL)):
        _, sim  = sim_dict[di]
        cum     = oos_cumret(sim, start=_oos_start)
        st_plot = compute_performance_stats(
            sim[sim.index >= _stat_start], f"EWbiv_{di}_plot")
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

    t1 = betas_df["t_stat_1"]
    t2 = betas_df["t_stat_2"]
    b1 = betas_df["beta_1"]
    b2 = betas_df["beta_2"]

    ax_t.plot(t1.index, t1.values, color=color_palette[0], lw=1.0, alpha=0.85,
              label=f"NW t-stat: {pred1_label} ({nw_lags}-lag HAC)")
    ax_t.plot(t2.index, t2.values, color=color_palette[0], lw=1.0, alpha=0.60, ls="--",
              label=f"NW t-stat: {pred2_label} ({nw_lags}-lag HAC)")
    ax_t.fill_between(t1.index, -T_THRESH, T_THRESH,
                      color="firebrick", alpha=0.05, label="Below gate (flat zone)")
    ax_t.axhline( T_THRESH, color="firebrick", lw=1.2, ls="--",
                 label=f"|t| = {T_THRESH:.2f} gate")
    ax_t.axhline(-T_THRESH, color="firebrick", lw=1.2, ls="--")
    ax_t.axhline(0, color="black", lw=0.5, ls=":")
    ax_t.set_ylabel("NW t-stat", fontsize=9)
    ax_t.grid(axis="y", alpha=0.2, lw=0.6)
    ax_t.spines["top"].set_visible(False)

    ax_t2 = ax_t.twinx()
    b1_peak = b1.abs().max() or 1.0
    b2_peak = b2.abs().max() or 1.0
    ax_t2.plot(b1.index, b1.values / b1_peak, color="dimgrey", lw=1.0, ls="-", alpha=0.60,
               label=f"Beta: {pred1_label} (peak={b1_peak:.3g})")
    ax_t2.plot(b2.index, b2.values / b2_peak, color="dimgrey", lw=1.0, ls=":", alpha=0.60,
               label=f"Beta: {pred2_label} (peak={b2_peak:.3g})")
    ax_t2.axhline(0, color="dimgrey", lw=0.4, ls=":")
    ax_t2.set_ylabel("Beta (own scale)", fontsize=8, color="dimgrey")
    ax_t2.tick_params(axis="y", labelcolor="dimgrey", labelsize=7)
    ax_t2.spines["top"].set_visible(False)
    _align_zero(ax_t, ax_t2)

    lines1, labs1 = ax_t.get_legend_handles_labels()
    lines2, labs2 = ax_t2.get_legend_handles_labels()
    ax_t.legend(lines1 + lines2, labs1 + labs2, fontsize=8, loc="upper left")

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

    for ax in [ax_ret, ax_t] + list(ax_pos):
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.tick_params(axis="x", which="major", labelbottom=True, labelsize=7, pad=2)

    fig.savefig(out_path, dpi=155, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ─── plot_asym: asymmetric threshold, 4 deltas (univariate) ──────────────────

def plot_asym(pred_label, horizon_label, oos_gap, nw_lags,
              color_palette, sim_dict, betas_df, bah_sim, out_path,
              r2_oos=None, oos_start=None, extra_title=""):
    _oos_start = oos_start if oos_start is not None else OOS_START
    oos_dt   = pd.Timestamp(_oos_start)
    e_dt     = betas_df.index[-1]
    xlim     = (oos_dt, e_dt)
    n_deltas = len(DELTAS)

    _rebase_start, _stat_start, _stat_lbl = _stat_window(sim_dict, bah_sim)

    r2_str = f"  R²_OOS = {r2_oos:+.4f}" if r2_oos is not None else ""

    h_ratios = [2.5, 1.2] + [1.0] * n_deltas
    fig, axes = plt.subplots(
        2 + n_deltas, 1, figsize=(14, 11 + 2.2 * n_deltas), sharex=True,
        gridspec_kw={"height_ratios": h_ratios, "hspace": 0.35},
    )
    ax_ret = axes[0]
    ax_t   = axes[1]
    ax_pos = axes[2:]

    fig.suptitle(
        f"{pred_label} -> {horizon_label} Forward Return  "
        f"(Expanding Window, OOS from {_oos_start}, Asymmetric Threshold){r2_str}"
        f"{extra_title}\n"
        f"Long: (ŷ-µ₅₀₀) > delta  |  Short: (-ŷ) > delta  |  "
        f"NW-HAC {nw_lags} lags; |t| > {T_THRESH:.2f} gate; 0.05% slippage",
        fontsize=10, y=0.998,
    )
    fig.subplots_adjust(top=0.955, bottom=0.03, left=0.10, right=0.93)

    ax_ret.set_xlim(*xlim)
    _shade(ax_ret, oos_dt, e_dt)

    _bah_st_plot = compute_performance_stats(bah_sim[bah_sim.index >= _stat_start], "BaH_plot")
    bah_oos = oos_cumret(bah_sim, start=_oos_start)
    ax_ret.plot(bah_oos.index, bah_oos.values,
                color=BAH_COLOR, lw=1.5, ls="-.", alpha=0.6,
                label=(f"Buy-and-Hold{_stat_lbl}  "
                       f"[SR={_bah_st_plot['sharpe']:+.2f}  "
                       f"ret={_bah_st_plot['ann_ret']*100:+.1f}%  "
                       f"DD={_bah_st_plot['max_dd']*100:.1f}%]"))

    if _rebase_start is not None:
        _bah_from   = bah_sim["net_pnl"][bah_sim.index >= _rebase_start]
        _bah_act    = (1 + _bah_from).cumprod()
        _bah_act_st = compute_performance_stats(
            bah_sim[bah_sim.index >= _rebase_start], "BaH_act")
        ax_ret.plot(
            _bah_act.index, _bah_act.values,
            color=BAH_COLOR, lw=1.2, ls=":", alpha=0.85,
            label=(f"Buy-and-Hold from {_rebase_start.strftime('%Y-%m-%d')}  "
                   f"[SR={_bah_act_st['sharpe']:+.2f}  "
                   f"ret={_bah_act_st['ann_ret']*100:+.1f}%  "
                   f"DD={_bah_act_st['max_dd']*100:.1f}%]"),
        )

    for di, (delta, lbl) in enumerate(zip(DELTAS, DELTA_LBL)):
        _, sim = sim_dict[di]
        cum      = oos_cumret(sim, start=_oos_start)
        st_plot  = compute_performance_stats(sim[sim.index >= _stat_start], f"asym_{di}")
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

    t_series = betas_df["t_stat"]
    b_series = betas_df["beta"]

    ax_t.plot(t_series.index, t_series.values,
              color=color_palette[0], lw=1.0, alpha=0.85,
              label=f"NW t-stat of {pred_label} ({nw_lags}-lag HAC)")
    ax_t.fill_between(t_series.index, -T_THRESH, T_THRESH,
                      color="firebrick", alpha=0.05, label="Below gate (flat zone)")
    ax_t.axhline( T_THRESH, color="firebrick", lw=1.2, ls="--",
                 label=f"|t| = {T_THRESH:.2f} gate")
    ax_t.axhline(-T_THRESH, color="firebrick", lw=1.2, ls="--")
    ax_t.axhline(0, color="black", lw=0.5, ls=":")
    ax_t.set_ylabel(f"NW t-stat ({pred_label})", fontsize=9)
    ax_t.grid(axis="y", alpha=0.2, lw=0.6)
    ax_t.spines["top"].set_visible(False)

    ax_t2 = ax_t.twinx()
    ax_t2.plot(b_series.index, b_series.values,
               color="dimgrey", lw=1.0, ls="--", alpha=0.60,
               label=f"Beta ({pred_label})")
    ax_t2.axhline(0, color="dimgrey", lw=0.4, ls=":")
    ax_t2.set_ylabel(f"Beta ({pred_label})", fontsize=8, color="dimgrey")
    ax_t2.tick_params(axis="y", labelcolor="dimgrey", labelsize=7)
    ax_t2.spines["top"].set_visible(False)
    _align_zero(ax_t, ax_t2)

    lines1, labs1 = ax_t.get_legend_handles_labels()
    lines2, labs2 = ax_t2.get_legend_handles_labels()
    ax_t.legend(lines1 + lines2, labs1 + labs2, fontsize=8, loc="upper left")

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

    for ax in [ax_ret, ax_t] + list(ax_pos):
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.tick_params(axis="x", which="major", labelbottom=True, labelsize=7, pad=2)

    fig.savefig(out_path, dpi=155, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ─── plot_asym_bivariate: asymmetric threshold (bivariate) ────────────────────

def plot_asym_bivariate(pred1_label, pred2_label, horizon_label, oos_gap, nw_lags,
                        color_palette, sim_dict, betas_df, bah_sim, out_path,
                        r2_oos=None, oos_start=None, extra_title=""):
    _oos_start = oos_start if oos_start is not None else OOS_START
    oos_dt   = pd.Timestamp(_oos_start)
    e_dt     = betas_df.index[-1]
    xlim     = (oos_dt, e_dt)
    n_deltas = len(DELTAS)

    _rebase_start, _stat_start, _stat_lbl = _stat_window(sim_dict, bah_sim)

    r2_str = f"  R²_OOS = {r2_oos:+.4f}" if r2_oos is not None else ""

    h_ratios = [2.5, 1.2] + [1.0] * n_deltas
    fig, axes = plt.subplots(
        2 + n_deltas, 1, figsize=(14, 11 + 2.2 * n_deltas), sharex=True,
        gridspec_kw={"height_ratios": h_ratios, "hspace": 0.35},
    )
    ax_ret = axes[0]
    ax_t   = axes[1]
    ax_pos = axes[2:]

    fig.suptitle(
        f"{pred1_label} + {pred2_label} -> {horizon_label} Forward Return  "
        f"(Expanding Window, OOS from {_oos_start}, Asymmetric Threshold){r2_str}"
        f"{extra_title}\n"
        f"Long: (ŷ-µ₅₀₀) > delta  |  Short: (-ŷ) > delta  |  "
        f"NW-HAC {nw_lags} lags; |t| > {T_THRESH:.2f} gate (both betas); 0.05% slippage",
        fontsize=10, y=0.998,
    )
    fig.subplots_adjust(top=0.955, bottom=0.03, left=0.10, right=0.93)

    ax_ret.set_xlim(*xlim)
    _shade(ax_ret, oos_dt, e_dt)

    _bah_st_plot = compute_performance_stats(bah_sim[bah_sim.index >= _stat_start], "BaH_plot")
    bah_oos = oos_cumret(bah_sim, start=_oos_start)
    ax_ret.plot(bah_oos.index, bah_oos.values,
                color=BAH_COLOR, lw=1.5, ls="-.", alpha=0.6,
                label=(f"Buy-and-Hold{_stat_lbl}  "
                       f"[SR={_bah_st_plot['sharpe']:+.2f}  "
                       f"ret={_bah_st_plot['ann_ret']*100:+.1f}%  "
                       f"DD={_bah_st_plot['max_dd']*100:.1f}%]"))

    if _rebase_start is not None:
        _bah_from   = bah_sim["net_pnl"][bah_sim.index >= _rebase_start]
        _bah_act    = (1 + _bah_from).cumprod()
        _bah_act_st = compute_performance_stats(
            bah_sim[bah_sim.index >= _rebase_start], "BaH_act")
        ax_ret.plot(
            _bah_act.index, _bah_act.values,
            color=BAH_COLOR, lw=1.2, ls=":", alpha=0.85,
            label=(f"Buy-and-Hold from {_rebase_start.strftime('%Y-%m-%d')}  "
                   f"[SR={_bah_act_st['sharpe']:+.2f}  "
                   f"ret={_bah_act_st['ann_ret']*100:+.1f}%  "
                   f"DD={_bah_act_st['max_dd']*100:.1f}%]"),
        )

    for di, (delta, lbl) in enumerate(zip(DELTAS, DELTA_LBL)):
        _, sim  = sim_dict[di]
        cum     = oos_cumret(sim, start=_oos_start)
        st_plot = compute_performance_stats(
            sim[sim.index >= _stat_start], f"asym_biv_{di}")
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

    t1 = betas_df["t_stat_1"]
    t2 = betas_df["t_stat_2"]
    b1 = betas_df["beta_1"]
    b2 = betas_df["beta_2"]

    ax_t.plot(t1.index, t1.values, color=color_palette[0], lw=1.0, alpha=0.85,
              label=f"NW t-stat: {pred1_label} ({nw_lags}-lag HAC)")
    ax_t.plot(t2.index, t2.values, color=color_palette[0], lw=1.0, alpha=0.60, ls="--",
              label=f"NW t-stat: {pred2_label} ({nw_lags}-lag HAC)")
    ax_t.fill_between(t1.index, -T_THRESH, T_THRESH,
                      color="firebrick", alpha=0.05, label="Below gate (flat zone)")
    ax_t.axhline( T_THRESH, color="firebrick", lw=1.2, ls="--",
                 label=f"|t| = {T_THRESH:.2f} gate")
    ax_t.axhline(-T_THRESH, color="firebrick", lw=1.2, ls="--")
    ax_t.axhline(0, color="black", lw=0.5, ls=":")
    ax_t.set_ylabel("NW t-stat", fontsize=9)
    ax_t.grid(axis="y", alpha=0.2, lw=0.6)
    ax_t.spines["top"].set_visible(False)

    ax_t2 = ax_t.twinx()
    b1_peak = b1.abs().max() or 1.0
    b2_peak = b2.abs().max() or 1.0
    ax_t2.plot(b1.index, b1.values / b1_peak, color="dimgrey", lw=1.0, ls="-", alpha=0.60,
               label=f"Beta: {pred1_label} (peak={b1_peak:.3g})")
    ax_t2.plot(b2.index, b2.values / b2_peak, color="dimgrey", lw=1.0, ls=":", alpha=0.60,
               label=f"Beta: {pred2_label} (peak={b2_peak:.3g})")
    ax_t2.axhline(0, color="dimgrey", lw=0.4, ls=":")
    ax_t2.set_ylabel("Beta (own scale)", fontsize=8, color="dimgrey")
    ax_t2.tick_params(axis="y", labelcolor="dimgrey", labelsize=7)
    ax_t2.spines["top"].set_visible(False)
    _align_zero(ax_t, ax_t2)

    lines1, labs1 = ax_t.get_legend_handles_labels()
    lines2, labs2 = ax_t2.get_legend_handles_labels()
    ax_t.legend(lines1 + lines2, labs1 + labs2, fontsize=8, loc="upper left")

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

    for ax in [ax_ret, ax_t] + list(ax_pos):
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.tick_params(axis="x", which="major", labelbottom=True, labelsize=7, pad=2)

    fig.savefig(out_path, dpi=155, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ─── plot_rolmu: base-return-shift (univariate) ───────────────────────────────

def plot_rolmu(pred_label, horizon_label, oos_gap, nw_lags,
               color_palette, sim_dict, betas_df, bah_sim, out_path,
               r2_oos=None, oos_start=None, extra_title=""):
    _oos_start = oos_start if oos_start is not None else OOS_START
    oos_dt   = pd.Timestamp(_oos_start)
    e_dt     = betas_df.index[-1]
    xlim     = (oos_dt, e_dt)
    n_deltas = len(DELTAS)

    _rebase_start, _stat_start, _stat_lbl = _stat_window(sim_dict, bah_sim)

    r2_str = f"  R²_OOS = {r2_oos:+.4f}" if r2_oos is not None else ""

    h_ratios = [2.5, 1.2] + [1.0] * n_deltas
    fig, axes = plt.subplots(
        2 + n_deltas, 1, figsize=(14, 11 + 2.2 * n_deltas), sharex=True,
        gridspec_kw={"height_ratios": h_ratios, "hspace": 0.35},
    )
    ax_ret = axes[0]
    ax_t   = axes[1]
    ax_pos = axes[2:]

    fig.suptitle(
        f"{pred_label} -> {horizon_label} Forward Return  "
        f"(Expanding Window, OOS from {_oos_start}, Base-Return-Shift){r2_str}"
        f"{extra_title}\n"
        f"Long: (ŷ-µ₅₀₀) > delta  |  Short: (µ₅₀₀-ŷ) > delta  |  "
        f"NW-HAC {nw_lags} lags; |t| > {T_THRESH:.2f} gate; 0.05% slippage",
        fontsize=10, y=0.998,
    )
    fig.subplots_adjust(top=0.955, bottom=0.03, left=0.10, right=0.93)

    ax_ret.set_xlim(*xlim)
    _shade(ax_ret, oos_dt, e_dt)

    _bah_st_plot = compute_performance_stats(bah_sim[bah_sim.index >= _stat_start], "BaH_plot")
    bah_oos = oos_cumret(bah_sim, start=_oos_start)
    ax_ret.plot(bah_oos.index, bah_oos.values,
                color=BAH_COLOR, lw=1.5, ls="-.", alpha=0.6,
                label=(f"Buy-and-Hold{_stat_lbl}  "
                       f"[SR={_bah_st_plot['sharpe']:+.2f}  "
                       f"ret={_bah_st_plot['ann_ret']*100:+.1f}%  "
                       f"DD={_bah_st_plot['max_dd']*100:.1f}%]"))

    if _rebase_start is not None:
        _bah_from   = bah_sim["net_pnl"][bah_sim.index >= _rebase_start]
        _bah_act    = (1 + _bah_from).cumprod()
        _bah_act_st = compute_performance_stats(
            bah_sim[bah_sim.index >= _rebase_start], "BaH_act")
        ax_ret.plot(
            _bah_act.index, _bah_act.values,
            color=BAH_COLOR, lw=1.2, ls=":", alpha=0.85,
            label=(f"Buy-and-Hold from {_rebase_start.strftime('%Y-%m-%d')}  "
                   f"[SR={_bah_act_st['sharpe']:+.2f}  "
                   f"ret={_bah_act_st['ann_ret']*100:+.1f}%  "
                   f"DD={_bah_act_st['max_dd']*100:.1f}%]"),
        )

    for di, (delta, lbl) in enumerate(zip(DELTAS, DELTA_LBL)):
        _, sim = sim_dict[di]
        cum      = oos_cumret(sim, start=_oos_start)
        st_plot  = compute_performance_stats(sim[sim.index >= _stat_start], f"rolmu_{di}")
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

    t_series = betas_df["t_stat"]
    b_series = betas_df["beta"]

    ax_t.plot(t_series.index, t_series.values,
              color=color_palette[0], lw=1.0, alpha=0.85,
              label=f"NW t-stat of {pred_label} ({nw_lags}-lag HAC)")
    ax_t.fill_between(t_series.index, -T_THRESH, T_THRESH,
                      color="firebrick", alpha=0.05, label="Below gate (flat zone)")
    ax_t.axhline( T_THRESH, color="firebrick", lw=1.2, ls="--",
                 label=f"|t| = {T_THRESH:.2f} gate")
    ax_t.axhline(-T_THRESH, color="firebrick", lw=1.2, ls="--")
    ax_t.axhline(0, color="black", lw=0.5, ls=":")
    ax_t.set_ylabel(f"NW t-stat ({pred_label})", fontsize=9)
    ax_t.grid(axis="y", alpha=0.2, lw=0.6)
    ax_t.spines["top"].set_visible(False)

    ax_t2 = ax_t.twinx()
    ax_t2.plot(b_series.index, b_series.values,
               color="dimgrey", lw=1.0, ls="--", alpha=0.60,
               label=f"Beta ({pred_label})")
    ax_t2.axhline(0, color="dimgrey", lw=0.4, ls=":")
    ax_t2.set_ylabel(f"Beta ({pred_label})", fontsize=8, color="dimgrey")
    ax_t2.tick_params(axis="y", labelcolor="dimgrey", labelsize=7)
    ax_t2.spines["top"].set_visible(False)
    _align_zero(ax_t, ax_t2)

    lines1, labs1 = ax_t.get_legend_handles_labels()
    lines2, labs2 = ax_t2.get_legend_handles_labels()
    ax_t.legend(lines1 + lines2, labs1 + labs2, fontsize=8, loc="upper left")

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

    for ax in [ax_ret, ax_t] + list(ax_pos):
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.tick_params(axis="x", which="major", labelbottom=True, labelsize=7, pad=2)

    fig.savefig(out_path, dpi=155, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ─── plot_rolmu_bivariate: base-return-shift (bivariate) ─────────────────────

def plot_rolmu_bivariate(pred1_label, pred2_label, horizon_label, oos_gap, nw_lags,
                         color_palette, sim_dict, betas_df, bah_sim, out_path,
                         r2_oos=None, oos_start=None, extra_title=""):
    _oos_start = oos_start if oos_start is not None else OOS_START
    oos_dt   = pd.Timestamp(_oos_start)
    e_dt     = betas_df.index[-1]
    xlim     = (oos_dt, e_dt)
    n_deltas = len(DELTAS)

    _rebase_start, _stat_start, _stat_lbl = _stat_window(sim_dict, bah_sim)

    r2_str = f"  R²_OOS = {r2_oos:+.4f}" if r2_oos is not None else ""

    h_ratios = [2.5, 1.2] + [1.0] * n_deltas
    fig, axes = plt.subplots(
        2 + n_deltas, 1, figsize=(14, 11 + 2.2 * n_deltas), sharex=True,
        gridspec_kw={"height_ratios": h_ratios, "hspace": 0.35},
    )
    ax_ret = axes[0]
    ax_t   = axes[1]
    ax_pos = axes[2:]

    fig.suptitle(
        f"{pred1_label} + {pred2_label} -> {horizon_label} Forward Return  "
        f"(Expanding Window, OOS from {_oos_start}, Base-Return-Shift){r2_str}"
        f"{extra_title}\n"
        f"Long: (ŷ-µ₅₀₀) > delta  |  Short: (µ₅₀₀-ŷ) > delta  |  "
        f"NW-HAC {nw_lags} lags; |t| > {T_THRESH:.2f} gate (both betas); 0.05% slippage",
        fontsize=10, y=0.998,
    )
    fig.subplots_adjust(top=0.955, bottom=0.03, left=0.10, right=0.93)

    ax_ret.set_xlim(*xlim)
    _shade(ax_ret, oos_dt, e_dt)

    _bah_st_plot = compute_performance_stats(bah_sim[bah_sim.index >= _stat_start], "BaH_plot")
    bah_oos = oos_cumret(bah_sim, start=_oos_start)
    ax_ret.plot(bah_oos.index, bah_oos.values,
                color=BAH_COLOR, lw=1.5, ls="-.", alpha=0.6,
                label=(f"Buy-and-Hold{_stat_lbl}  "
                       f"[SR={_bah_st_plot['sharpe']:+.2f}  "
                       f"ret={_bah_st_plot['ann_ret']*100:+.1f}%  "
                       f"DD={_bah_st_plot['max_dd']*100:.1f}%]"))

    if _rebase_start is not None:
        _bah_from   = bah_sim["net_pnl"][bah_sim.index >= _rebase_start]
        _bah_act    = (1 + _bah_from).cumprod()
        _bah_act_st = compute_performance_stats(
            bah_sim[bah_sim.index >= _rebase_start], "BaH_act")
        ax_ret.plot(
            _bah_act.index, _bah_act.values,
            color=BAH_COLOR, lw=1.2, ls=":", alpha=0.85,
            label=(f"Buy-and-Hold from {_rebase_start.strftime('%Y-%m-%d')}  "
                   f"[SR={_bah_act_st['sharpe']:+.2f}  "
                   f"ret={_bah_act_st['ann_ret']*100:+.1f}%  "
                   f"DD={_bah_act_st['max_dd']*100:.1f}%]"),
        )

    for di, (delta, lbl) in enumerate(zip(DELTAS, DELTA_LBL)):
        _, sim  = sim_dict[di]
        cum     = oos_cumret(sim, start=_oos_start)
        st_plot = compute_performance_stats(
            sim[sim.index >= _stat_start], f"rolmu_biv_{di}")
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

    t1 = betas_df["t_stat_1"]
    t2 = betas_df["t_stat_2"]
    b1 = betas_df["beta_1"]
    b2 = betas_df["beta_2"]

    ax_t.plot(t1.index, t1.values, color=color_palette[0], lw=1.0, alpha=0.85,
              label=f"NW t-stat: {pred1_label} ({nw_lags}-lag HAC)")
    ax_t.plot(t2.index, t2.values, color=color_palette[0], lw=1.0, alpha=0.60, ls="--",
              label=f"NW t-stat: {pred2_label} ({nw_lags}-lag HAC)")
    ax_t.fill_between(t1.index, -T_THRESH, T_THRESH,
                      color="firebrick", alpha=0.05, label="Below gate (flat zone)")
    ax_t.axhline( T_THRESH, color="firebrick", lw=1.2, ls="--",
                 label=f"|t| = {T_THRESH:.2f} gate")
    ax_t.axhline(-T_THRESH, color="firebrick", lw=1.2, ls="--")
    ax_t.axhline(0, color="black", lw=0.5, ls=":")
    ax_t.set_ylabel("NW t-stat", fontsize=9)
    ax_t.grid(axis="y", alpha=0.2, lw=0.6)
    ax_t.spines["top"].set_visible(False)

    ax_t2 = ax_t.twinx()
    b1_peak = b1.abs().max() or 1.0
    b2_peak = b2.abs().max() or 1.0
    ax_t2.plot(b1.index, b1.values / b1_peak, color="dimgrey", lw=1.0, ls="-", alpha=0.60,
               label=f"Beta: {pred1_label} (peak={b1_peak:.3g})")
    ax_t2.plot(b2.index, b2.values / b2_peak, color="dimgrey", lw=1.0, ls=":", alpha=0.60,
               label=f"Beta: {pred2_label} (peak={b2_peak:.3g})")
    ax_t2.axhline(0, color="dimgrey", lw=0.4, ls=":")
    ax_t2.set_ylabel("Beta (own scale)", fontsize=8, color="dimgrey")
    ax_t2.tick_params(axis="y", labelcolor="dimgrey", labelsize=7)
    ax_t2.spines["top"].set_visible(False)
    _align_zero(ax_t, ax_t2)

    lines1, labs1 = ax_t.get_legend_handles_labels()
    lines2, labs2 = ax_t2.get_legend_handles_labels()
    ax_t.legend(lines1 + lines2, labs1 + labs2, fontsize=8, loc="upper left")

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

    for ax in [ax_ret, ax_t] + list(ax_pos):
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.tick_params(axis="x", which="major", labelbottom=True, labelsize=7, pad=2)

    fig.savefig(out_path, dpi=155, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ─── Unit-position: symmetric ─────────────────────────────────────────────────

def run_ew(panel, predictor, fwd_col, oos_gap, nw_lags, delta):
    """±1 when |ŷ| > delta and t-stat gate passes."""
    tag = (f"pos_EW_{predictor}_{fwd_col}_d{int(delta*10000)}bps"
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
        if abs(float(res.params.iloc[1]) / float(nw[1])) <= T_THRESH:
            continue
        test  = sub.iloc[[i]][[predictor]].copy(); test.insert(0, "const", 1.0)
        y_hat = float(res.predict(test).iloc[0])
        if   y_hat >  delta: pos.iloc[i] =  1.0
        elif y_hat < -delta: pos.iloc[i] = -1.0

    pos.to_frame().to_parquet(cache)
    return pos


def run_ew_bivariate(panel, pred1, pred2, fwd_col, oos_gap, nw_lags, delta):
    """Both betas must pass the |t| > T_THRESH gate."""
    tag = (f"pos_EWbiv_{pred1}_{pred2}_{fwd_col}_d{int(delta*10000)}bps"
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
        test  = sub.iloc[[i]][[pred1, pred2]].copy(); test.insert(0, "const", 1.0)
        y_hat = float(res.predict(test).iloc[0])
        if   y_hat >  delta: pos.iloc[i] =  1.0
        elif y_hat < -delta: pos.iloc[i] = -1.0

    pos.to_frame().to_parquet(cache)
    return pos


# ─── Unit-position: asymmetric ────────────────────────────────────────────────

def run_ew_asym(panel, predictor, fwd_col, oos_gap, nw_lags, delta=0.0):
    """Long if (ŷ-µ₅₀₀) > delta, short if -ŷ > delta."""
    d_sfx = f"_d{int(round(delta * 10000))}" if delta > 0 else ""
    tag = (f"pos_EWasym_{predictor}_{fwd_col}"
           f"_t{int(T_THRESH*100)}{d_sfx}_oos{OOS_START}.parquet")
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
        if abs(float(res.params.iloc[1]) / float(nw[1])) <= T_THRESH:
            continue
        test  = sub.iloc[[i]][[predictor]].copy(); test.insert(0, "const", 1.0)
        y_hat = float(res.predict(test).iloc[0])
        mu500 = float(sub[fwd_col].iloc[max(0, i - oos_gap - 500) : i - oos_gap].mean())
        if   (y_hat - mu500) > delta: pos.iloc[i] =  1.0
        elif       (-y_hat)  > delta: pos.iloc[i] = -1.0

    pos.to_frame().to_parquet(cache)
    return pos


def run_ew_asym_bivariate(panel, pred1, pred2, fwd_col, oos_gap, nw_lags, delta=0.0):
    """Both betas must pass gate. Long if (ŷ-µ₅₀₀) > delta, short if -ŷ > delta."""
    d_sfx = f"_d{int(round(delta * 10000))}" if delta > 0 else ""
    tag = (f"pos_EWasym_{pred1}_{pred2}_{fwd_col}"
           f"_t{int(T_THRESH*100)}{d_sfx}_oos{OOS_START}.parquet")
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
        test  = sub.iloc[[i]][[pred1, pred2]].copy(); test.insert(0, "const", 1.0)
        y_hat = float(res.predict(test).iloc[0])
        mu500 = float(sub[fwd_col].iloc[max(0, i - oos_gap - 500) : i - oos_gap].mean())
        if   (y_hat - mu500) > delta: pos.iloc[i] =  1.0
        elif       (-y_hat)  > delta: pos.iloc[i] = -1.0

    pos.to_frame().to_parquet(cache)
    return pos


# ─── Unit-position: base-return-shift ────────────────────────────────────────

def run_ew_rolmu(panel, predictor, fwd_col, oos_gap, nw_lags, delta=0.0):
    """Long if (ŷ-µ₅₀₀) > delta, short if (µ₅₀₀-ŷ) > delta."""
    d_sfx = f"_d{int(round(delta * 10000))}" if delta > 0 else ""
    tag = (f"pos_EWrolmu_{predictor}_{fwd_col}"
           f"_t{int(T_THRESH*100)}{d_sfx}_oos{OOS_START}.parquet")
    cache = CACHE_DIR / tag
    if cache.exists():
        return pd.read_parquet(cache).squeeze().rename(f"pos_rolmu_{predictor}")

    sub = panel.dropna(subset=[predictor, fwd_col]).copy()
    N   = len(sub)
    pos = pd.Series(0.0, index=sub.index, name=f"pos_rolmu_{predictor}")
    oos_idx = sub.index.searchsorted(pd.Timestamp(OOS_START))
    start_i = max(MIN_WIN + oos_gap, oos_idx)

    for i in range(start_i, N):
        train = sub.iloc[0 : i - oos_gap]
        X_tr  = add_constant(train[[predictor]], has_constant="skip")
        res   = OLS(train[fwd_col], X_tr).fit()
        nw    = _nw_se(res, nlags=nw_lags)
        if abs(float(res.params.iloc[1]) / float(nw[1])) <= T_THRESH:
            continue
        mu500 = float(sub[fwd_col].iloc[max(0, i - oos_gap - 500) : i - oos_gap].mean())
        test  = sub.iloc[[i]][[predictor]].copy(); test.insert(0, "const", 1.0)
        y_hat = float(res.predict(test).iloc[0])
        if   (y_hat - mu500) >  delta: pos.iloc[i] =  1.0
        elif (mu500 - y_hat) >  delta: pos.iloc[i] = -1.0

    pos.to_frame().to_parquet(cache)
    return pos


def run_ew_rolmu_bivariate(panel, pred1, pred2, fwd_col, oos_gap, nw_lags, delta=0.0):
    """Both betas must pass gate. Long if (ŷ-µ₅₀₀) > delta, short if (µ₅₀₀-ŷ) > delta."""
    d_sfx = f"_d{int(round(delta * 10000))}" if delta > 0 else ""
    tag = (f"pos_EWrolmu_{pred1}_{pred2}_{fwd_col}"
           f"_t{int(T_THRESH*100)}{d_sfx}_oos{OOS_START}.parquet")
    cache = CACHE_DIR / tag
    if cache.exists():
        return pd.read_parquet(cache).squeeze().rename(f"pos_rolmu_{pred1}_{pred2}")

    sub = panel.dropna(subset=[pred1, pred2, fwd_col]).copy()
    N   = len(sub)
    pos = pd.Series(0.0, index=sub.index, name=f"pos_rolmu_{pred1}_{pred2}")
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
        mu500 = float(sub[fwd_col].iloc[max(0, i - oos_gap - 500) : i - oos_gap].mean())
        test  = sub.iloc[[i]][[pred1, pred2]].copy(); test.insert(0, "const", 1.0)
        y_hat = float(res.predict(test).iloc[0])
        if   (y_hat - mu500) >  delta: pos.iloc[i] =  1.0
        elif (mu500 - y_hat) >  delta: pos.iloc[i] = -1.0

    pos.to_frame().to_parquet(cache)
    return pos


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
    term_slope = compute_vix_term_slope(load_vix_futures_term_structure())
    oi         = load_es_open_interest()

    panel = build_panel(vrp, es, vvix_ma5, vix_spot, term_slope, oi)
    panel["vvix_ma10"] = vvix_ma10.reindex(panel.index)
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

    # ── Pre-compute betas and OOS R² ──
    print("\n[2] Computing betas and OOS R²...")
    vrp_betas   = compute_betas(panel, "VP",         FWD, OOS_GAP, NW_LAGS)
    vvix5_betas = compute_betas(panel, "vvix_ma5",   FWD, OOS_GAP, NW_LAGS)
    vvix10_betas= compute_betas(panel, "vvix_ma10",  FWD, OOS_GAP, NW_LAGS)
    biv5_betas  = compute_betas_bivariate(panel, "VP", "vvix_ma5",      FWD, OOS_GAP, NW_LAGS)
    biv10_betas = compute_betas_bivariate(panel, "VP", "vvix_ma10",     FWD, OOS_GAP, NW_LAGS)
    ts_betas    = compute_betas_bivariate(panel, "VP", "term_slope",    FWD, OOS_GAP, NW_LAGS)
    oi_betas    = compute_betas_bivariate(panel, "VP", "open_interest", FWD, OOS_GAP, NW_LAGS)

    vrp_r2    = compute_oos_r2(panel, "VP",         FWD, OOS_GAP, NW_LAGS)
    vvix5_r2  = compute_oos_r2(panel, "vvix_ma5",   FWD, OOS_GAP, NW_LAGS)
    vvix10_r2 = compute_oos_r2(panel, "vvix_ma10",  FWD, OOS_GAP, NW_LAGS)
    biv5_r2   = compute_oos_r2_bivariate(panel, "VP", "vvix_ma5",      FWD, OOS_GAP, NW_LAGS)
    biv10_r2  = compute_oos_r2_bivariate(panel, "VP", "vvix_ma10",     FWD, OOS_GAP, NW_LAGS)
    ts_r2     = compute_oos_r2_bivariate(panel, "VP", "term_slope",    FWD, OOS_GAP, NW_LAGS)
    oi_r2     = compute_oos_r2_bivariate(panel, "VP", "open_interest", FWD, OOS_GAP, NW_LAGS)

    # ══════════════════════════════════════════════════════════════════════════
    # 1. VRP
    # ══════════════════════════════════════════════════════════════════════════

    print("\n[1.1] VRP symmetric...")
    sim_dict_vrp = {}
    for di, delta in enumerate(DELTAS):
        pos = run_ew(panel, "VP", FWD, OOS_GAP, NW_LAGS, delta)
        sim_dict_vrp[di] = (delta, simulate_strategy(pos, daily_ret))
    plot_2panel(
        pred_label="VRP", horizon_label="20-day",
        oos_gap=OOS_GAP, nw_lags=NW_LAGS,
        color_palette=pal_vrp,
        sim_dict=sim_dict_vrp, betas_df=vrp_betas, bah_sim=bah_sim,
        out_path=out_vrp / "symmetric_VRP.png",
        r2_oos=vrp_r2,
    )

    print("\n[1.2] VRP asymmetric...")
    asym_sims_vrp = {}
    for di, delta in enumerate(DELTAS):
        pos = run_ew_asym(panel, "VP", FWD, OOS_GAP, NW_LAGS, delta=delta)
        asym_sims_vrp[di] = (delta, simulate_strategy(pos, daily_ret))
        st = compute_performance_stats(
            asym_sims_vrp[di][1][asym_sims_vrp[di][1].index >= OOS_START],
            f"asym_vrp_{di}")
        print(f"    delta={DELTA_LBL[di]}  SR={st['sharpe']:+.2f}  "
              f"ret={st['ann_ret']*100:+.1f}%  DD={st['max_dd']*100:.1f}%")
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
        st = compute_performance_stats(
            rolmu_sims_vrp[di][1][rolmu_sims_vrp[di][1].index >= OOS_START],
            f"rolmu_vrp_{di}")
        print(f"    delta={DELTA_LBL[di]}  SR={st['sharpe']:+.2f}  "
              f"ret={st['ann_ret']*100:+.1f}%  DD={st['max_dd']*100:.1f}%")
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
    plot_2panel(
        pred_label="VVIX MA5", horizon_label="20-day",
        oos_gap=OOS_GAP, nw_lags=NW_LAGS,
        color_palette=pal_vvix5,
        sim_dict=sim_dict_vvix5, betas_df=vvix5_betas, bah_sim=bah_sim,
        out_path=out_vvix5 / "symmetric_VVIX_MA5.png",
        r2_oos=vvix5_r2,
    )

    print("\n[2.2] VVIX MA5 asymmetric...")
    asym_sims_vvix5 = {}
    for di, delta in enumerate(DELTAS):
        pos = run_ew_asym(panel, "vvix_ma5", FWD, OOS_GAP, NW_LAGS, delta=delta)
        asym_sims_vvix5[di] = (delta, simulate_strategy(pos, daily_ret))
        st = compute_performance_stats(
            asym_sims_vvix5[di][1][asym_sims_vvix5[di][1].index >= OOS_START],
            f"asym_vvix5_{di}")
        print(f"    delta={DELTA_LBL[di]}  SR={st['sharpe']:+.2f}  "
              f"ret={st['ann_ret']*100:+.1f}%  DD={st['max_dd']*100:.1f}%")
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
        st = compute_performance_stats(
            rolmu_sims_vvix5[di][1][rolmu_sims_vvix5[di][1].index >= OOS_START],
            f"rolmu_vvix5_{di}")
        print(f"    delta={DELTA_LBL[di]}  SR={st['sharpe']:+.2f}  "
              f"ret={st['ann_ret']*100:+.1f}%  DD={st['max_dd']*100:.1f}%")
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
    plot_2panel(
        pred_label="VVIX MA10", horizon_label="20-day",
        oos_gap=OOS_GAP, nw_lags=NW_LAGS,
        color_palette=pal_vvix10,
        sim_dict=sim_dict_vvix10, betas_df=vvix10_betas, bah_sim=bah_sim,
        out_path=out_vvix10 / "symmetric_VVIX_MA10.png",
        r2_oos=vvix10_r2,
    )

    print("\n[3.2] VVIX MA10 asymmetric...")
    asym_sims_vvix10 = {}
    for di, delta in enumerate(DELTAS):
        pos = run_ew_asym(panel, "vvix_ma10", FWD, OOS_GAP, NW_LAGS, delta=delta)
        asym_sims_vvix10[di] = (delta, simulate_strategy(pos, daily_ret))
        st = compute_performance_stats(
            asym_sims_vvix10[di][1][asym_sims_vvix10[di][1].index >= OOS_START],
            f"asym_vvix10_{di}")
        print(f"    delta={DELTA_LBL[di]}  SR={st['sharpe']:+.2f}  "
              f"ret={st['ann_ret']*100:+.1f}%  DD={st['max_dd']*100:.1f}%")
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
        st = compute_performance_stats(
            rolmu_sims_vvix10[di][1][rolmu_sims_vvix10[di][1].index >= OOS_START],
            f"rolmu_vvix10_{di}")
        print(f"    delta={DELTA_LBL[di]}  SR={st['sharpe']:+.2f}  "
              f"ret={st['ann_ret']*100:+.1f}%  DD={st['max_dd']*100:.1f}%")
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
    plot_2panel_bivariate(
        pred1_label="VRP", pred2_label="VVIX MA5", horizon_label="20-day",
        oos_gap=OOS_GAP, nw_lags=NW_LAGS,
        color_palette=pal_biv5,
        sim_dict=sim_dict_biv5, betas_df=biv5_betas, bah_sim=bah_sim,
        out_path=out_biv5 / "symmetric_VRP_+_VVIX_MA5.png",
        r2_oos=biv5_r2,
    )

    print("\n[4.2] VRP + VVIX MA5 asymmetric...")
    asym_sims_biv5 = {}
    for di, delta in enumerate(DELTAS):
        pos = run_ew_asym_bivariate(panel, "VP", "vvix_ma5", FWD, OOS_GAP, NW_LAGS, delta=delta)
        asym_sims_biv5[di] = (delta, simulate_strategy(pos, daily_ret))
        st = compute_performance_stats(
            asym_sims_biv5[di][1][asym_sims_biv5[di][1].index >= OOS_START],
            f"asym_biv5_{di}")
        print(f"    delta={DELTA_LBL[di]}  SR={st['sharpe']:+.2f}  "
              f"ret={st['ann_ret']*100:+.1f}%  DD={st['max_dd']*100:.1f}%")
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
        st = compute_performance_stats(
            rolmu_sims_biv5[di][1][rolmu_sims_biv5[di][1].index >= OOS_START],
            f"rolmu_biv5_{di}")
        print(f"    delta={DELTA_LBL[di]}  SR={st['sharpe']:+.2f}  "
              f"ret={st['ann_ret']*100:+.1f}%  DD={st['max_dd']*100:.1f}%")
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
    plot_2panel_bivariate(
        pred1_label="VRP", pred2_label="VVIX MA10", horizon_label="20-day",
        oos_gap=OOS_GAP, nw_lags=NW_LAGS,
        color_palette=pal_biv10,
        sim_dict=sim_dict_biv10, betas_df=biv10_betas, bah_sim=bah_sim,
        out_path=out_biv10 / "symmetric_VRP_+_VVIX_MA10.png",
        r2_oos=biv10_r2,
    )

    print("\n[5.2] VRP + VVIX MA10 asymmetric...")
    asym_sims_biv10 = {}
    for di, delta in enumerate(DELTAS):
        pos = run_ew_asym_bivariate(panel, "VP", "vvix_ma10", FWD, OOS_GAP, NW_LAGS, delta=delta)
        asym_sims_biv10[di] = (delta, simulate_strategy(pos, daily_ret))
        st = compute_performance_stats(
            asym_sims_biv10[di][1][asym_sims_biv10[di][1].index >= OOS_START],
            f"asym_biv10_{di}")
        print(f"    delta={DELTA_LBL[di]}  SR={st['sharpe']:+.2f}  "
              f"ret={st['ann_ret']*100:+.1f}%  DD={st['max_dd']*100:.1f}%")
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
        st = compute_performance_stats(
            rolmu_sims_biv10[di][1][rolmu_sims_biv10[di][1].index >= OOS_START],
            f"rolmu_biv10_{di}")
        print(f"    delta={DELTA_LBL[di]}  SR={st['sharpe']:+.2f}  "
              f"ret={st['ann_ret']*100:+.1f}%  DD={st['max_dd']*100:.1f}%")
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
    plot_2panel_bivariate(
        pred1_label="VRP", pred2_label="Term Slope", horizon_label="20-day",
        oos_gap=OOS_GAP, nw_lags=NW_LAGS,
        color_palette=pal_ts,
        sim_dict=sim_dict_ts, betas_df=ts_betas, bah_sim=bah_sim,
        out_path=out_ts / "symmetric_VRP_+_Term_Slope.png",
        r2_oos=ts_r2,
    )

    print("\n[6.2] VRP + Term Slope asymmetric...")
    asym_sims_ts = {}
    for di, delta in enumerate(DELTAS):
        pos = run_ew_asym_bivariate(panel, "VP", "term_slope", FWD, OOS_GAP, NW_LAGS, delta=delta)
        asym_sims_ts[di] = (delta, simulate_strategy(pos, daily_ret))
        st = compute_performance_stats(
            asym_sims_ts[di][1][asym_sims_ts[di][1].index >= OOS_START],
            f"asym_ts_{di}")
        print(f"    delta={DELTA_LBL[di]}  SR={st['sharpe']:+.2f}  "
              f"ret={st['ann_ret']*100:+.1f}%  DD={st['max_dd']*100:.1f}%")
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
        st = compute_performance_stats(
            rolmu_sims_ts[di][1][rolmu_sims_ts[di][1].index >= OOS_START],
            f"rolmu_ts_{di}")
        print(f"    delta={DELTA_LBL[di]}  SR={st['sharpe']:+.2f}  "
              f"ret={st['ann_ret']*100:+.1f}%  DD={st['max_dd']*100:.1f}%")
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
    plot_2panel_bivariate(
        pred1_label="VRP", pred2_label="Open Interest", horizon_label="20-day",
        oos_gap=OOS_GAP, nw_lags=NW_LAGS,
        color_palette=pal_oi,
        sim_dict=sim_dict_oi, betas_df=oi_betas, bah_sim=bah_sim,
        out_path=out_oi / "symmetric_VRP_+_Open_Interest.png",
        r2_oos=oi_r2,
    )

    print("\n[7.2] VRP + Open Interest asymmetric...")
    asym_sims_oi = {}
    for di, delta in enumerate(DELTAS):
        pos = run_ew_asym_bivariate(panel, "VP", "open_interest", FWD, OOS_GAP, NW_LAGS, delta=delta)
        asym_sims_oi[di] = (delta, simulate_strategy(pos, daily_ret))
        st = compute_performance_stats(
            asym_sims_oi[di][1][asym_sims_oi[di][1].index >= OOS_START],
            f"asym_oi_{di}")
        print(f"    delta={DELTA_LBL[di]}  SR={st['sharpe']:+.2f}  "
              f"ret={st['ann_ret']*100:+.1f}%  DD={st['max_dd']*100:.1f}%")
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
        st = compute_performance_stats(
            rolmu_sims_oi[di][1][rolmu_sims_oi[di][1].index >= OOS_START],
            f"rolmu_oi_{di}")
        print(f"    delta={DELTA_LBL[di]}  SR={st['sharpe']:+.2f}  "
              f"ret={st['ann_ret']*100:+.1f}%  DD={st['max_dd']*100:.1f}%")
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
