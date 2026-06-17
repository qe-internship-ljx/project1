"""
leveraged_vvix_ma5.py
=====================
Unbound base-return-shift strategy for VVIX MA5 -> 20-day forward return.

Multi-level position sizing based on (y_hat - rolling_mu):
  > 1.0%  ->  +4   |  < -1.0%  ->  -4
  > 0.75% ->  +3   |  < -0.75% ->  -3
  > 0.5%  ->  +2   |  < -0.5%  ->  -2
  > 0.2%  ->  +1   |  < -0.2%  ->  -1
  otherwise ->  0  (also 0 when |t| <= 1.28)

Trading cost: flat 0.05% per trade regardless of position-size change.

Output:
  output/expanding_window/VVIX MA5/leveraged_base_return_shift_VVIX_MA5.png
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

from statsmodels.api import OLS, add_constant

ROOT      = Path(__file__).parent
OUTPUT    = ROOT / "output"
CACHE_DIR = OUTPUT / "regression_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT.parent / "bh_replication"))
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT))

from har_model import _nw_se
from experiment2 import (
    load_vrp_series, load_es_front_month, load_vvix, compute_vvix_ma5,
    load_vix_spot, load_vix_futures_term_structure,
    compute_buy_and_hold, simulate_strategy, compute_performance_stats,
)
from fh_replication.fh_replication import compute_vix_term_slope
from horizon_regression import build_panel, compute_betas, _shade

OOS_START    = "2012-01-01"
MIN_WIN      = 500
T_THRESH     = 1.28
OOS_GAP      = 20
NW_LAGS      = 20
ROLLING_WIN  = 500
VVIX_ACT     = pd.Timestamp("2006-03-06")
BAH_COLOR    = "#d62728"
MAIN_COLOR   = "#3f007d"

# Thresholds sorted high to low; matching DELTAS in horizon_regression
THRESHOLDS = [0.010, 0.0075, 0.005, 0.002]
LEVELS     = [4, 3, 2, 1]


# ─── Position computation ────────────────────────────────────────────────────

def run_ew_rolmu_unbound(panel, predictor, fwd_col, oos_gap, nw_lags,
                         rolling_window=500):
    """Expanding-window rolling-mu threshold with unbound multi-level positions.

    Excess = y_hat - mu  (mu = trailing rolling_window-day mean of fwd returns).
    |t| gate: flat when |t| <= T_THRESH.
    Level assignment (symmetric):
        |excess| > 1.0%  -> ±4
        |excess| > 0.75% -> ±3
        |excess| > 0.5%  -> ±2
        |excess| > 0.2%  -> ±1
        otherwise        ->  0
    """
    tag = (f"pos_EW_rolmu_unbound_{predictor}_{fwd_col}"
           f"_t{int(T_THRESH * 100)}_rw{rolling_window}_oos{OOS_START}.parquet")
    cache = CACHE_DIR / tag
    if cache.exists():
        return pd.read_parquet(cache).squeeze().rename(
            f"pos_rolmu_unbound_{predictor}")

    sub = panel.dropna(subset=[predictor, fwd_col]).copy()
    N   = len(sub)
    pos = pd.Series(0.0, index=sub.index,
                    name=f"pos_rolmu_unbound_{predictor}")
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
        lo    = max(0, i - oos_gap - rolling_window)
        mu    = float(np.mean(fwd[lo : i - oos_gap]))
        test  = sub.iloc[[i]][[predictor]].copy()
        test.insert(0, "const", 1.0)
        y_hat  = float(res.predict(test).iloc[0])
        excess = y_hat - mu
        # Assign level
        level = 0
        for thresh, lv in zip(THRESHOLDS, LEVELS):
            if abs(excess) >= thresh:
                level = lv
                break
        if level > 0:
            pos.iloc[i] = float(level) if excess > 0 else float(-level)

    pos.to_frame().to_parquet(cache)
    return pos


# ─── Derived series for the predicted-return panel ──────────────────────────

def compute_yhat_series(panel, predictor, fwd_col, betas_df):
    """Reconstruct y_hat = alpha + beta * predictor at each prediction date."""
    sub = panel.dropna(subset=[predictor, fwd_col]).copy()
    idx = betas_df.index.intersection(sub.index)
    return (betas_df.loc[idx, "alpha"]
            + betas_df.loc[idx, "beta"] * sub.loc[idx, predictor]).rename("y_hat")


def compute_rolling_mu_series(panel, fwd_col, oos_gap, rolling_window=500):
    """Trailing at-most rolling_window-day mean of fwd_col, lagged by oos_gap."""
    sub = panel.dropna(subset=["vvix_ma5", fwd_col]).copy()
    return (sub[fwd_col]
            .rolling(rolling_window, min_periods=1)
            .mean()
            .shift(oos_gap)
            .rename("rolling_mu"))


# ─── Utilities ───────────────────────────────────────────────────────────────

def oos_cumret(sim, start=OOS_START):
    net = sim["net_pnl"]
    s   = net[net.index >= start]
    return (1 + s).cumprod()


# ─── Plot ────────────────────────────────────────────────────────────────────

def plot_unbound(pred_label, horizon_label, oos_gap, nw_lags,
                 main_color, sim, betas_df, bah_sim,
                 y_hat_series, mu_series, out_path, extra_title=""):
    """4-panel plot: cumulative return, t-stat/beta, position, predicted return."""
    oos_dt = pd.Timestamp(OOS_START)
    e_dt   = betas_df.index[-1]
    xlim   = (oos_dt, e_dt)

    h_ratios = [2.5, 1.2, 1.0, 1.2]
    fig, axes = plt.subplots(
        4, 1, figsize=(14, 15.4), sharex=True,
        gridspec_kw={"height_ratios": h_ratios, "hspace": 0.35},
    )
    ax_ret, ax_t, ax_p, ax_yhat = axes

    fig.suptitle(
        f"{pred_label} -> {horizon_label} Forward Return  "
        f"(Expanding Window, OOS from {OOS_START})  Unbound Position Sizing"
        f"{extra_title}\n"
        f"Long/Short ±1..4 at rolling-μ ± 0.2/0.5/0.75/1.0%  |  "
        f"NW-HAC {nw_lags} lags; |t| > {T_THRESH:.2f} gate; 0.05% slippage",
        fontsize=10, y=0.998,
    )
    fig.subplots_adjust(top=0.955, bottom=0.03, left=0.10, right=0.93)

    # ── Panel 1: Cumulative Net Return ────────────────────────────────────────
    ax_ret.set_xlim(*xlim)
    _shade(ax_ret, oos_dt, e_dt)

    pos_oos    = sim["position"][sim.index >= OOS_START]
    active     = pos_oos[pos_oos != 0]
    _activation = active.index.min() if len(active) else None

    _rebase_start = None
    if _activation is not None and _activation > pd.Timestamp("2020-01-01"):
        _bah_idx      = bah_sim.index
        _act_iloc     = _bah_idx.searchsorted(_activation)
        _rebase_start = _bah_idx[min(_act_iloc + 1, len(_bah_idx) - 1)]

    _stat_start = _rebase_start if _rebase_start is not None else pd.Timestamp(OOS_START)
    _stat_lbl   = (f" · stats from {_stat_start.strftime('%Y-%m-%d')}"
                   if _rebase_start is not None else "")

    _bah_st = compute_performance_stats(
        bah_sim[bah_sim.index >= _stat_start], "BaH_plot")
    bah_oos = oos_cumret(bah_sim)
    ax_ret.plot(bah_oos.index, bah_oos.values,
                color=BAH_COLOR, lw=1.5, ls="-.", alpha=0.6,
                label=(f"Buy-and-Hold{_stat_lbl}  "
                       f"[SR={_bah_st['sharpe']:+.2f}  "
                       f"ret={_bah_st['ann_ret']*100:+.1f}%  "
                       f"DD={_bah_st['max_dd']*100:.1f}%]"))

    if _rebase_start is not None:
        _bah_from   = bah_sim["net_pnl"][bah_sim.index >= _rebase_start]
        _bah_act    = (1 + _bah_from).cumprod()
        _bah_act_st = compute_performance_stats(
            bah_sim[bah_sim.index >= _rebase_start], "BaH_act")
        ax_ret.plot(_bah_act.index, _bah_act.values,
                    color=BAH_COLOR, lw=1.2, ls=":", alpha=0.85,
                    label=(f"Buy-and-Hold from {_rebase_start.strftime('%Y-%m-%d')}  "
                           f"[SR={_bah_act_st['sharpe']:+.2f}  "
                           f"ret={_bah_act_st['ann_ret']*100:+.1f}%  "
                           f"DD={_bah_act_st['max_dd']*100:.1f}%]"))

    st_plot  = compute_performance_stats(sim[sim.index >= _stat_start], "unbound")
    cum      = oos_cumret(sim)
    pos_stat = sim["position"][sim.index >= _stat_start]
    pL = float((pos_stat > 0).mean() * 100)
    pS = float((pos_stat < 0).mean() * 100)
    avg_pos = float(pos_stat.mean())
    ax_ret.plot(cum.index, cum.values,
                color=main_color, lw=1.8, alpha=0.9,
                label=(f"Unbound ±1..4  "
                       f"[SR={st_plot['sharpe']:+.2f}  "
                       f"ret={st_plot['ann_ret']*100:+.1f}%  "
                       f"DD={st_plot['max_dd']*100:.1f}%  "
                       f"L{pL:.0f}%/S{pS:.0f}%  AvgPos={avg_pos:+.2f}]"))

    ax_ret.axhline(1, color="black", lw=0.4, ls=":")
    ax_ret.set_yscale("log")
    ax_ret.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{v:.2f}x"))
    ax_ret.set_ylabel("Cumulative Net Return (log)", fontsize=9)
    ax_ret.legend(fontsize=8, loc="upper left", framealpha=0.92)
    ax_ret.grid(axis="y", alpha=0.2, lw=0.6)
    ax_ret.spines[["top", "right"]].set_visible(False)

    # ── Panel 2: NW t-stat + beta ─────────────────────────────────────────────
    ax_t.set_xlim(*xlim)
    _shade(ax_t, oos_dt, e_dt)

    t_ser = betas_df["t_stat"]
    b_ser = betas_df["beta"]

    ax_t.plot(t_ser.index, t_ser.values,
              color=main_color, lw=1.0, alpha=0.85,
              label=f"NW t-stat of {pred_label} ({nw_lags}-lag HAC)")
    ax_t.fill_between(t_ser.index, -T_THRESH, T_THRESH,
                      color="firebrick", alpha=0.05,
                      label="Below gate (flat zone)")
    ax_t.axhline( T_THRESH, color="firebrick", lw=1.2, ls="--",
                 label=f"|t| = {T_THRESH:.2f} gate")
    ax_t.axhline(-T_THRESH, color="firebrick", lw=1.2, ls="--")
    ax_t.axhline(0, color="black", lw=0.5, ls=":")
    ax_t.set_ylabel(f"NW t-stat ({pred_label})", fontsize=9)
    ax_t.grid(axis="y", alpha=0.2, lw=0.6)
    ax_t.spines["top"].set_visible(False)

    ax_t2 = ax_t.twinx()
    ax_t2.plot(b_ser.index, b_ser.values,
               color="dimgrey", lw=1.0, ls="--", alpha=0.60,
               label=f"Beta ({pred_label})")
    ax_t2.axhline(0, color="dimgrey", lw=0.4, ls=":")
    ax_t2.set_ylabel(f"Beta ({pred_label})", fontsize=8, color="dimgrey")
    ax_t2.tick_params(axis="y", labelcolor="dimgrey", labelsize=7)
    ax_t2.spines["top"].set_visible(False)

    if "r2_insample" in betas_df.columns:
        r2_s = betas_df["r2_insample"]
        ax_t2.plot(r2_s.index, r2_s.values,
                   color="forestgreen", lw=0.9, ls=":", alpha=0.75,
                   label="In-sample R²")

    lines1, labs1 = ax_t.get_legend_handles_labels()
    lines2, labs2 = ax_t2.get_legend_handles_labels()
    ax_t.legend(lines1 + lines2, labs1 + labs2, fontsize=8, loc="upper left")

    # ── Panel 3: Position over time (±4 scale) ────────────────────────────────
    pos_plot = sim["position"][sim.index >= OOS_START]
    ax_p.set_xlim(*xlim)
    _shade(ax_p, oos_dt, e_dt)
    ax_p.fill_between(pos_plot.index, pos_plot.clip(lower=0), 0,
                      color=main_color, alpha=0.70, label="Long (+1..+4)")
    ax_p.fill_between(pos_plot.index, pos_plot.clip(upper=0), 0,
                      color=main_color, alpha=0.30, hatch="///",
                      label="Short (-1..-4)")
    ax_p.axhline(0, color="black", lw=0.4)
    ax_p.set_ylim(-4.5, 4.5)
    ax_p.set_yticks([-4, -3, -2, -1, 0, 1, 2, 3, 4])
    ax_p.tick_params(axis="y", labelsize=7)
    ax_p.set_ylabel("Position", fontsize=9, rotation=0,
                    ha="right", va="center", labelpad=56, color=main_color)
    pF = float((pos_stat == 0).mean() * 100)
    ax_p.text(0.01, 0.97,
              f"Long {pL:.1f}%  Short {pS:.1f}%  Flat {pF:.1f}%  "
              f"AvgPos={avg_pos:+.3f}",
              transform=ax_p.transAxes, fontsize=7.5, va="top", color=main_color)
    ax_p.spines[["top", "right"]].set_visible(False)

    # ── Panel 4: Predicted return over time ──────────────────────────────────
    y_hat_oos = y_hat_series[y_hat_series.index >= OOS_START]
    mu_oos    = mu_series[mu_series.index >= OOS_START]
    idx_c     = y_hat_oos.index.intersection(mu_oos.index)
    yh        = y_hat_oos.loc[idx_c]
    mu_       = mu_oos.loc[idx_c]

    ax_yhat.set_xlim(*xlim)
    _shade(ax_yhat, oos_dt, e_dt)

    # Threshold bands around rolling mu (outermost first for correct layering)
    band_colors = ["#cbc9e2", "#9e9ac8", "#807dba", "#6a51a3"]
    for tv, bc in zip(THRESHOLDS, band_colors):
        ax_yhat.fill_between(
            mu_.index,
            (mu_ - tv).values * 100,
            (mu_ + tv).values * 100,
            alpha=0.18, color=bc, linewidth=0,
        )

    ax_yhat.plot(mu_.index, mu_.values * 100,
                 color="black", lw=1.2, ls="--", alpha=0.75,
                 label="Rolling μ (500d fwd return)")
    ax_yhat.plot(yh.index, yh.values * 100,
                 color=main_color, lw=0.8, alpha=0.85,
                 label="ŷ (predicted 20d return)")
    ax_yhat.axhline(0, color="black", lw=0.4, ls=":")

    band_patch = mpatches.Patch(
        facecolor="#6a51a3", alpha=0.4,
        label="Threshold bands μ ± 0.2/0.5/0.75/1.0%")
    handles, labels = ax_yhat.get_legend_handles_labels()
    ax_yhat.legend(handles + [band_patch], labels + [band_patch.get_label()],
                   fontsize=8, loc="upper left")

    ax_yhat.set_ylabel("Pred. Return (%)", fontsize=9)
    ax_yhat.grid(axis="y", alpha=0.2, lw=0.6)
    ax_yhat.spines[["top", "right"]].set_visible(False)

    for ax in axes:
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.tick_params(axis="x", which="major",
                       labelbottom=True, labelsize=7, pad=2)

    fig.savefig(out_path, dpi=155, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("  Unbound Base-Return-Shift Strategy  —  VVIX MA5 -> 20-day return")
    print("  Positions +/-1/2/3/4 at rolling-mu +/- 0.2/0.5/0.75/1.0%")
    print("=" * 72)

    print("\n[1] Loading data...")
    vrp        = load_vrp_series()
    es         = load_es_front_month()
    vvix_raw   = load_vvix()
    vvix_ma5   = compute_vvix_ma5(vvix_raw)
    vix_spot   = load_vix_spot()
    term_slope = compute_vix_term_slope(load_vix_futures_term_structure())
    panel      = build_panel(vrp, es, vvix_raw, vvix_ma5, vix_spot, term_slope)
    print(f"    {len(panel):,} obs  "
          f"[{panel.index.min().date()} - {panel.index.max().date()}]")

    daily_ret = panel["daily_ret"].dropna()
    bah_pos   = compute_buy_and_hold(daily_ret)
    bah_sim   = simulate_strategy(bah_pos, daily_ret)

    print("\n[2] Computing unbound multi-level positions (may take a while)...")
    pos = run_ew_rolmu_unbound(panel, "vvix_ma5", "fwd_20d",
                               oos_gap=OOS_GAP, nw_lags=NW_LAGS,
                               rolling_window=ROLLING_WIN)
    pos = pos.copy()
    pos[pos.index < VVIX_ACT] = 0.0
    sim = simulate_strategy(pos, daily_ret)
    st  = compute_performance_stats(sim[sim.index >= OOS_START], "unbound_VVIX_MA5")
    p   = pos[pos.index >= OOS_START]
    print(f"    SR={st['sharpe']:+.3f}  "
          f"L%={(p > 0).mean()*100:.1f}%  "
          f"S%={(p < 0).mean()*100:.1f}%  "
          f"F%={(p == 0).mean()*100:.1f}%  "
          f"AvgPos={p.mean():+.3f}")
    print(f"    Position distribution:")
    for lv in [4, 3, 2, 1, 0, -1, -2, -3, -4]:
        pct = float((p == lv).mean() * 100)
        if pct > 0.1:
            print(f"      {lv:+d}: {pct:.1f}%")

    print("\n[3] Loading betas (from cache or recomputing)...")
    vvix_betas = compute_betas(panel, "vvix_ma5", "fwd_20d",
                               oos_gap=OOS_GAP, nw_lags=NW_LAGS)

    print("\n[4] Computing predicted-return and rolling-mu series...")
    y_hat_ser = compute_yhat_series(panel, "vvix_ma5", "fwd_20d", vvix_betas)
    mu_ser    = compute_rolling_mu_series(panel, "fwd_20d", OOS_GAP, ROLLING_WIN)

    print("\n[5] Plotting...")
    out_dir = OUTPUT / "expanding_window" / "VVIX MA5"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_unbound(
        pred_label="VVIX MA5", horizon_label="20-day",
        oos_gap=OOS_GAP, nw_lags=NW_LAGS,
        main_color=MAIN_COLOR,
        sim=sim, betas_df=vvix_betas,
        bah_sim=bah_sim,
        y_hat_series=y_hat_ser,
        mu_series=mu_ser,
        out_path=out_dir / "leveraged_base_return_shift_VVIX_MA5.png",
        extra_title="\nPositions flat before VVIX activation (2006-03-06)",
    )

    print("\nDone.")
    print("  output/expanding_window/VVIX MA5/leveraged_base_return_shift_VVIX_MA5.png")
    print("=" * 72)


if __name__ == "__main__":
    main()
