"""
plot_vvix_ma10.py
=================
Expanding-window OOS evaluation: univariate VVIX MA10 -> 20-day forward return.
Produces two plots per plot.md spec:
  1. Fixed threshold (y_hat > delta)
  2. Shifted/rolling-mu threshold (y_hat > mu + delta)
Saved to output/expanding_window/VVIX MA10/
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import importlib.util
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT.parent / "bh_replication"))

from statsmodels.api import OLS, add_constant
from har_model import _nw_se


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


fh_mod = load_module("fh_replication", str(ROOT.parent / "fh_replication" / "fh_replication.py"))
exp2   = load_module("experiment2_mod", str(ROOT / "experiment2.py"))

# ── Constants ──────────────────────────────────────────────────────────────────
OOS_START = "2012-01-01"
MIN_WIN   = 500
T_THRESH  = 1.28
NW_LAGS   = 20
ROLL_WIN  = 500
DELTAS    = [0.002, 0.005, 0.0075, 0.010]
DELTA_LBL = ["d=0.2%", "d=0.5%", "d=0.75%", "d=1.0%"]
PALETTE   = ["#3f007d", "#6a51a3", "#807dba", "#9e9ac8"]
BAH_COLOR = "#d62728"

OUT_DIR = ROOT / "output" / "expanding_window" / "VVIX MA10"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading data...")
vrp      = exp2.load_vrp_series()
es       = exp2.load_es_front_month()
vx_df    = exp2.load_vix_futures_term_structure()
slope    = fh_mod.compute_vix_term_slope(vx_df)
trend_q  = exp2.compute_trend_quotient(es)
vvix     = exp2.load_vvix()
vvix_ma5 = exp2.compute_vvix_ma5(vvix)

vvix_ma10 = vvix.rolling(10).mean()
vvix_ma10.name = "vvix_ma10"

panel = exp2.build_master_panel(vrp, es, slope, trend_q, vvix_ma5)
panel = panel[panel.index >= "2006-03-06"].copy()
panel = panel.join(vvix_ma10, how="left")

daily_ret = panel["daily_ret"].dropna()

# ── Expanding window ───────────────────────────────────────────────────────────
sub = panel.dropna(subset=["vvix_ma10", "fwd_ret_20"]).copy()
N   = len(sub)
fwd = sub["fwd_ret_20"].values

oos_idx = sub.index.searchsorted(pd.Timestamp(OOS_START))
start_i = max(MIN_WIN + 20, oos_idx)

pos_fixed = {d: pd.Series(0.0, index=sub.index) for d in DELTAS}
pos_rolmu = {d: pd.Series(0.0, index=sub.index) for d in DELTAS}
beta_records = []
# Accumulate OOS predictions for Goyal-Welch OOS R² computation
oos_r2_y_hat   = []   # predicted forward return at each OOS step
oos_r2_y_act   = []   # actual forward return at each OOS step
oos_r2_y_bar   = []   # prevailing mean (expanding mean of training y) at each step

print(f"Running expanding window ({N - start_i} OOS steps)...")
for i in range(start_i, N):
    train = sub.iloc[0 : i - 20]
    X_tr  = add_constant(train[["vvix_ma10"]], has_constant="skip")
    res   = OLS(train["fwd_ret_20"], X_tr).fit()
    nw    = _nw_se(res, nlags=NW_LAGS)
    b     = float(res.params.iloc[1])
    se    = float(nw[1])
    t_b   = b / se if se > 0 else 0.0
    beta_records.append({
        "date":        sub.index[i],
        "beta":        b,
        "se":          se,
        "t_stat":      t_b,
        "r2_insample": float(res.rsquared),
    })
    # Accumulate for OOS R² regardless of t-stat gate
    test_row_r2 = sub.iloc[[i]][["vvix_ma10"]].copy()
    test_row_r2.insert(0, "const", 1.0)
    y_hat_r2 = float(res.predict(test_row_r2).iloc[0])
    oos_r2_y_hat.append(y_hat_r2)
    oos_r2_y_act.append(float(sub["fwd_ret_20"].iloc[i]))
    oos_r2_y_bar.append(float(train["fwd_ret_20"].mean()))

    if abs(t_b) <= T_THRESH:
        continue
    test_row = sub.iloc[[i]][["vvix_ma10"]].copy()
    test_row.insert(0, "const", 1.0)
    y_hat = float(res.predict(test_row).iloc[0])
    lo = max(0, i - 20 - ROLL_WIN)
    mu = float(np.mean(fwd[lo : i - 20]))
    for d in DELTAS:
        if   y_hat >  d:      pos_fixed[d].iloc[i] =  1.0
        elif y_hat < -d:      pos_fixed[d].iloc[i] = -1.0
        if   y_hat > mu + d:  pos_rolmu[d].iloc[i] =  1.0
        elif y_hat < mu - d:  pos_rolmu[d].iloc[i] = -1.0

betas_df = pd.DataFrame(beta_records).set_index("date")

# Compute Goyal-Welch OOS R²
if len(oos_r2_y_hat) > 0:
    _ya  = np.array(oos_r2_y_act)
    _yh  = np.array(oos_r2_y_hat)
    _yb  = np.array(oos_r2_y_bar)
    _ss_res = float(np.sum((_ya - _yh) ** 2))
    _ss_tot = float(np.sum((_ya - _yb) ** 2))
    OOS_R2 = (1.0 - _ss_res / _ss_tot) if _ss_tot > 0 else float("nan")
else:
    OOS_R2 = float("nan")
print(f"OOS R² (Goyal-Welch) = {OOS_R2:.4f}")
OOS_DT = pd.Timestamp(OOS_START)
E_DT   = betas_df.index[-1]
print("Done.")

# ── Buy-and-hold benchmark ─────────────────────────────────────────────────────
bah_pos = exp2.compute_buy_and_hold(daily_ret)
bah_sim = exp2.simulate_strategy(bah_pos, daily_ret)
bah_st  = exp2.compute_performance_stats(bah_sim[bah_sim.index >= OOS_START], "Buy-and-Hold")


def oos_cumret(sim):
    net = sim["net_pnl"]
    s   = net[net.index >= OOS_START]
    return (1 + s).cumprod()


def _shade(ax):
    for a, b in [("2020-02-01", "2020-06-01"), ("2022-01-01", "2022-12-31")]:
        a, b = pd.Timestamp(a), pd.Timestamp(b)
        if b > OOS_DT and a < E_DT:
            ax.axvspan(max(a, OOS_DT), min(b, E_DT), alpha=0.08, color="grey", lw=0)


def make_plot(pos_dict, title_suffix, out_path):
    sim_dict = {}
    for di, d in enumerate(DELTAS):
        pos     = pos_dict[d].reindex(daily_ret.index).fillna(0)
        sim     = exp2.simulate_strategy(pos, daily_ret)
        sim_oos = sim[sim.index >= OOS_START]
        st      = exp2.compute_performance_stats(sim_oos, DELTA_LBL[di])
        sim_dict[di] = (st, sim, pos)

    n_d      = len(DELTAS)
    h_ratios = [2.5, 1.2] + [1.0] * n_d
    fig, axes = plt.subplots(
        2 + n_d, 1,
        figsize=(14, 11 + 2.2 * n_d),
        sharex=True,
        gridspec_kw={"height_ratios": h_ratios, "hspace": 0.35},
    )

    # OOS R² label in title
    _oos_r2_str = f"\nOOS R² = {OOS_R2:.4f}" if not np.isnan(OOS_R2) else ""

    fig.suptitle(
        f"VVIX MA10 -> 20-day Forward Return  (Expanding Window, OOS from {OOS_START})"
        f"{title_suffix}{_oos_r2_str}\n"
        f"Training grows daily; OOS gap = 20 days; NW-HAC 20 lags; "
        f"|t| > {T_THRESH:.2f} gate; 0.05% slippage",
        fontsize=10, y=0.998,
    )
    fig.subplots_adjust(top=0.955, bottom=0.03, left=0.10, right=0.93)

    ax_ret = axes[0]
    ax_t   = axes[1]
    ax_pos = axes[2:]

    # Panel 1 — cumulative return
    ax_ret.set_xlim(OOS_DT, E_DT)
    _shade(ax_ret)

    # Find earliest activation across all deltas (for stat-window consistency)
    _activation = None
    for di in range(len(DELTAS)):
        _, _, pos = sim_dict[di]
        p_oos = pos[pos.index >= OOS_START]
        active = p_oos[p_oos != 0]
        if len(active):
            _cand = active.index.min()
            if _activation is None or _cand < _activation:
                _activation = _cand

    _rebase_start = None
    if _activation is not None and _activation > pd.Timestamp("2020-01-01"):
        _bah_net      = bah_sim["net_pnl"]
        _bah_idx      = _bah_net.index
        _act_iloc     = _bah_idx.searchsorted(_activation)
        _rebase_start = _bah_idx[min(_act_iloc + 1, len(_bah_idx) - 1)]

    _stat_start = _rebase_start if _rebase_start is not None else pd.Timestamp(OOS_START)
    _stat_lbl   = (f" · stats from {_stat_start.strftime('%Y-%m-%d')}"
                   if _rebase_start is not None else "")

    # Main B&H — stats from _stat_start
    _bah_st_plot = exp2.compute_performance_stats(
        bah_sim[bah_sim.index >= _stat_start], "BaH_plot")
    bah_oos = oos_cumret(bah_sim)
    ax_ret.plot(
        bah_oos.index, bah_oos.values,
        color=BAH_COLOR, lw=1.5, ls="-.", alpha=0.6,
        label=(f"Buy-and-Hold{_stat_lbl}  "
               f"[SR={_bah_st_plot['sharpe']:+.2f}  "
               f"ret={_bah_st_plot['ann_ret']*100:+.1f}%  "
               f"DD={_bah_st_plot['max_dd']*100:.1f}%]"),
    )
    # B&H rebased to 1x at first signal activation (if post-2020)
    if _rebase_start is not None:
        _bah_from     = bah_sim["net_pnl"][bah_sim.index >= _rebase_start]
        _bah_act      = (1 + _bah_from).cumprod()
        _bah_act_st   = exp2.compute_performance_stats(
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
        _, sim, pos = sim_dict[di]
        st_plot = exp2.compute_performance_stats(
            sim[sim.index >= _stat_start], f"ma10_plot_{di}")
        cum   = oos_cumret(sim)
        p_oos = pos[pos.index >= _stat_start]
        pL = float((p_oos == 1).mean() * 100)
        pS = float((p_oos == -1).mean() * 100)
        ax_ret.plot(
            cum.index, cum.values,
            color=PALETTE[di], lw=1.8, alpha=0.9,
            label=(f"{lbl}  "
                   f"[SR={st_plot['sharpe']:+.2f}  "
                   f"ret={st_plot['ann_ret']*100:+.1f}%  "
                   f"DD={st_plot['max_dd']*100:.1f}%  "
                   f"L{pL:.0f}%/S{pS:.0f}%]"),
        )
    ax_ret.axhline(1, color="black", lw=0.4, ls=":")
    ax_ret.set_yscale("log")
    ax_ret.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2f}x"))
    ax_ret.set_ylabel("Cumulative Net Return (log)", fontsize=9)
    ax_ret.legend(fontsize=8, loc="upper left", framealpha=0.92)
    ax_ret.grid(axis="y", alpha=0.2, lw=0.6)
    ax_ret.spines[["top", "right"]].set_visible(False)

    # Panel 2 — t-stat + beta + in-sample R²
    ax_t.set_xlim(OOS_DT, E_DT)
    _shade(ax_t)
    t_s = betas_df["t_stat"]
    b_s = betas_df["beta"]
    ax_t.plot(t_s.index, t_s.values, color=PALETTE[0], lw=1.0, alpha=0.85,
              label="NW t-stat: VVIX MA10 (20-lag HAC)")
    ax_t.fill_between(t_s.index, -T_THRESH, T_THRESH,
                      color="firebrick", alpha=0.05, label="Below gate (flat zone)")
    ax_t.axhline( T_THRESH, color="firebrick", lw=1.2, ls="--",
                 label=f"|t| = {T_THRESH:.2f} gate")
    ax_t.axhline(-T_THRESH, color="firebrick", lw=1.2, ls="--")
    ax_t.axhline(0, color="black", lw=0.5, ls=":")
    ax_t.set_ylabel("NW t-stat (VVIX MA10)", fontsize=9)
    ax_t.grid(axis="y", alpha=0.2, lw=0.6)
    ax_t.spines["top"].set_visible(False)

    ax_t2 = ax_t.twinx()
    ax_t2.plot(b_s.index, b_s.values, color="dimgrey", lw=1.0, ls="--", alpha=0.60,
               label="Beta (VVIX MA10)")
    ax_t2.axhline(0, color="dimgrey", lw=0.4, ls=":")
    ax_t2.set_ylabel("Beta (VVIX MA10)", fontsize=8, color="dimgrey")
    ax_t2.tick_params(axis="y", labelcolor="dimgrey", labelsize=7)
    ax_t2.spines["top"].set_visible(False)

    # In-sample R² on the right axis (shared with beta axis = ax_t2)
    if "r2_insample" in betas_df.columns:
        r2_s = betas_df["r2_insample"]
        ax_t2.plot(r2_s.index, r2_s.values,
                   color="forestgreen", lw=0.9, ls=":", alpha=0.75,
                   label="In-sample R²")

    l1, lb1 = ax_t.get_legend_handles_labels()
    l2, lb2 = ax_t2.get_legend_handles_labels()
    ax_t.legend(l1 + l2, lb1 + lb2, fontsize=8, loc="upper left")

    # Panels 3-6 — positions
    for di, (delta, lbl, ax_p) in enumerate(zip(DELTAS, DELTA_LBL, ax_pos)):
        st, sim, pos = sim_dict[di]
        p_oos = pos[pos.index >= OOS_START]
        ax_p.set_xlim(OOS_DT, E_DT)
        _shade(ax_p)
        ax_p.fill_between(p_oos.index, p_oos.where(p_oos ==  1, 0), 0,
                          color=PALETTE[di], alpha=0.75)
        ax_p.fill_between(p_oos.index, p_oos.where(p_oos == -1, 0), 0,
                          color=PALETTE[di], alpha=0.30, hatch="///")
        ax_p.axhline(0, color="black", lw=0.4)
        ax_p.set_ylim(-1.5, 1.5)
        ax_p.set_yticks([-1, 0, 1])
        ax_p.set_yticklabels(["Short", "Flat", "Long"], fontsize=8)
        ax_p.set_ylabel(lbl, fontsize=9, rotation=0, ha="right", va="center",
                        labelpad=56, color=PALETTE[di])
        pL = float((p_oos ==  1).mean() * 100)
        pS = float((p_oos == -1).mean() * 100)
        pF = float((p_oos ==  0).mean() * 100)
        ax_p.text(
            0.01, 0.97,
            f"Long {pL:.1f}%  Short {pS:.1f}%  Flat {pF:.1f}%  AvgPos={float(p_oos.mean()):+.3f}",
            transform=ax_p.transAxes, fontsize=7.5, va="top", color=PALETTE[di],
        )
        ax_p.spines[["top", "right"]].set_visible(False)

    for ax in [ax_ret, ax_t] + list(ax_pos):
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.tick_params(axis="x", which="major", labelbottom=True, labelsize=7, pad=2)

    fig.savefig(out_path, dpi=155, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path.name}")


make_plot(pos_fixed, "", OUT_DIR / "symmetric_VVIX_MA10.png")
make_plot(pos_rolmu, "  · Threshold = rolling-avg(20d return) ± δ", OUT_DIR / "base_return_shift_VVIX_MA10.png")
print("All done.")
