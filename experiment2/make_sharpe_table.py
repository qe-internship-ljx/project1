"""One-time script: 3x6 Sharpe ratio table, stat windows matching plot logic exactly."""
import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT.parent / "bh_replication"))

from experiment2 import (
    load_vrp_series, load_es_front_month, load_vvix, compute_vvix_ma5,
    load_vix_futures_term_structure, load_es_open_interest,
    compute_buy_and_hold, simulate_strategy,
    compute_trend_quotient, build_master_panel,
)
from fh_replication.fh_replication import compute_vix_term_slope
from regressions import (
    run_ew, run_ew_bivariate,
    run_ew_asym, run_ew_asym_bivariate,
    run_ew_rolmu, run_ew_rolmu_bivariate,
    run_ew_unbound_sym, run_ew_unbound_asym, run_ew_unbound_rolmu,
    run_ew_biv_unbound_sym, run_ew_biv_unbound_asym, run_ew_biv_unbound_rolmu,
    OOS_START, OOS_GAP, NW_LAGS, DELTAS,
)

print("Loading data...")
vrp        = load_vrp_series()
es         = load_es_front_month()
vvix_raw   = load_vvix()
vvix_ma5   = compute_vvix_ma5(vvix_raw)
term_slope = compute_vix_term_slope(load_vix_futures_term_structure())
trend_q    = compute_trend_quotient(es)
oi         = load_es_open_interest()
panel      = build_master_panel(vrp, es, term_slope, trend_q, vvix_ma5)
panel      = panel[panel.index >= "2006-03-06"].copy()
panel["open_interest"] = oi.reindex(panel.index)
FWD       = "fwd_20d"
DELTA_020 = 0.002   # DELTAS[0]

daily_ret = panel["daily_ret"].dropna()
bah_sim   = simulate_strategy(compute_buy_and_hold(daily_ret), daily_ret)


# ── Stat-window helpers (mirror plot logic exactly) ───────────────────────────

def _activation_to_stat_start(activation):
    """Given a first-activation date, return the stat_start used by the plots."""
    if activation is not None and activation > pd.Timestamp("2020-01-01"):
        idx   = bah_sim.index
        act_i = idx.searchsorted(activation)
        return idx[min(act_i + 1, len(idx) - 1)]
    return pd.Timestamp(OOS_START)


def stat_start_single(sim):
    """Mirror _draw_cumret / _stat_window(single sim): first non-zero OOS position."""
    pos_oos = sim["position"][sim.index >= OOS_START]
    active  = pos_oos[pos_oos != 0]
    return _activation_to_stat_start(active.index.min() if len(active) else None)


def stat_start_dict(sims):
    """Mirror _stat_window(dict): min activation across all sims in the list."""
    candidates = []
    for sim in sims:
        pos_oos = sim["position"][sim.index >= OOS_START]
        active  = pos_oos[pos_oos != 0]
        if len(active):
            candidates.append(active.index.min())
    return _activation_to_stat_start(min(candidates) if candidates else None)


def sharpe(sim, stat_start):
    daily = sim.loc[sim.index >= stat_start, "net_pnl"].dropna()
    n     = len(daily)
    ann   = float((1 + daily).prod() ** (252 / n) - 1) if n > 0 else np.nan
    vol   = float(daily.std() * np.sqrt(252))
    ann_excess = ann - 0.03
    return ann_excess / vol if vol > 0 else np.nan


def sim_from_pos(pos):
    return simulate_strategy(pos, daily_ret)


# ── Non-leveraged strategies (symmetric / asymmetric / rolmu):
#    All three plot functions build a dict of 4 delta sims and call
#    _stat_window(dict, bah_sim) to get min activation across all 4.
#    We replicate that exactly, then report d=0.2% (DELTAS[0]) SR.
def multi_delta_sharpe(pos_list):
    """4-delta sims → min-activation stat_start → SR at d=0.2% (pos_list[0])."""
    all_sims = [sim_from_pos(pos) for pos in pos_list]
    stat = stat_start_dict(all_sims)
    return sharpe(all_sims[0], stat), stat


def base_sharpe(pred1, pred2=None):
    """Symmetric: run_ew / run_ew_bivariate across all 4 deltas."""
    pos_list = []
    for d in DELTAS:
        if pred2 is None:
            pos_list.append(run_ew(panel, pred1, FWD, OOS_GAP, NW_LAGS, delta=d))
        else:
            pos_list.append(run_ew_bivariate(panel, pred1, pred2, FWD, OOS_GAP, NW_LAGS, delta=d))
    return multi_delta_sharpe(pos_list)


def asym_sharpe(pred1, pred2=None):
    """Asymmetric: run_ew_asym / run_ew_asym_bivariate across all 4 deltas."""
    pos_list = []
    for d in DELTAS:
        if pred2 is None:
            pos_list.append(run_ew_asym(panel, pred1, FWD, OOS_GAP, NW_LAGS, delta=d))
        else:
            pos_list.append(run_ew_asym_bivariate(panel, pred1, pred2, FWD, OOS_GAP, NW_LAGS, delta=d))
    return multi_delta_sharpe(pos_list)


def rolmu_sharpe(pred1, pred2=None):
    """Base-return-shift: run_ew_rolmu / run_ew_rolmu_bivariate across all 4 deltas."""
    pos_list = []
    for d in DELTAS:
        if pred2 is None:
            pos_list.append(run_ew_rolmu(panel, pred1, FWD, OOS_GAP, NW_LAGS, delta=d))
        else:
            pos_list.append(run_ew_rolmu_bivariate(panel, pred1, pred2, FWD, OOS_GAP, NW_LAGS, delta=d))
    return multi_delta_sharpe(pos_list)


# ── Leveraged strategies: single sim (no delta variants) → single stat_start.
def single_sharpe(pos):
    sim  = sim_from_pos(pos)
    stat = stat_start_single(sim)
    return sharpe(sim, stat), stat


# ── Compute all 18 cells ─────────────────────────────────────────────────────
MODELS    = ["VRP", "VRP + Term Slope", "VRP + VVIX MA5"]
COL_KEYS  = ["symmetric", "asymmetric", "base_return_shift",
             "lev_symmetric", "lev_asymmetric", "lev_base_return_shift"]
COL_LABELS = [
    "Symmetric\n(d=0.2%)", "Asymmetric", "Base-Return\nShift",
    "Lev. Symmetric", "Lev. Asymmetric", "Lev. Base-Return\nShift",
]

data    = np.full((3, 6), np.nan)
windows = {}   # (i,j) -> stat_start date string for printing

def fill_row(row_idx, pred1, pred2=None):
    tag = f"{pred1}" if pred2 is None else f"{pred1}+{pred2}"
    print(f"\n  {tag}")

    sr, ss = base_sharpe(pred1, pred2)
    data[row_idx, 0] = sr
    windows[(row_idx, 0)] = ss.strftime("%Y-%m-%d")
    print(f"    symmetric     SR={sr:+.3f}  stats_from={ss.date()}")

    sr, ss = asym_sharpe(pred1, pred2)
    data[row_idx, 1] = sr
    windows[(row_idx, 1)] = ss.strftime("%Y-%m-%d")
    print(f"    asymmetric    SR={sr:+.3f}  stats_from={ss.date()}")

    sr, ss = rolmu_sharpe(pred1, pred2)
    data[row_idx, 2] = sr
    windows[(row_idx, 2)] = ss.strftime("%Y-%m-%d")
    print(f"    base_ret_shift SR={sr:+.3f}  stats_from={ss.date()}")

    if pred2 is None:
        pos = run_ew_unbound_sym(panel, pred1, FWD, OOS_GAP, NW_LAGS)
    else:
        pos = run_ew_biv_unbound_sym(panel, pred1, pred2, FWD, OOS_GAP, NW_LAGS)
    sr, ss = single_sharpe(pos)
    data[row_idx, 3] = sr
    windows[(row_idx, 3)] = ss.strftime("%Y-%m-%d")
    print(f"    lev_symmetric  SR={sr:+.3f}  stats_from={ss.date()}")

    if pred2 is None:
        pos = run_ew_unbound_asym(panel, pred1, FWD, OOS_GAP, NW_LAGS)
    else:
        pos = run_ew_biv_unbound_asym(panel, pred1, pred2, FWD, OOS_GAP, NW_LAGS)
    sr, ss = single_sharpe(pos)
    data[row_idx, 4] = sr
    windows[(row_idx, 4)] = ss.strftime("%Y-%m-%d")
    print(f"    lev_asymmetric SR={sr:+.3f}  stats_from={ss.date()}")

    if pred2 is None:
        pos = run_ew_unbound_rolmu(panel, pred1, FWD, OOS_GAP, NW_LAGS)
    else:
        pos = run_ew_biv_unbound_rolmu(panel, pred1, pred2, FWD, OOS_GAP, NW_LAGS)
    sr, ss = single_sharpe(pos)
    data[row_idx, 5] = sr
    windows[(row_idx, 5)] = ss.strftime("%Y-%m-%d")
    print(f"    lev_base_ret   SR={sr:+.3f}  stats_from={ss.date()}")


print("Computing Sharpe ratios (hitting cache)...")
fill_row(0, "VP")
fill_row(1, "VP", "term_slope")
fill_row(2, "VP", "vvix_ma5")

# ── Build table image ─────────────────────────────────────────────────────────
max_idx = np.unravel_index(np.nanargmax(data), data.shape)
pos_max = data[data > 0].max() if (data > 0).any() else 1.0

fig, ax = plt.subplots(figsize=(17, 4.5))
ax.axis("off")

cell_text = [[f"{v:+.3f}" for v in row] for row in data]

tbl = ax.table(
    cellText=cell_text,
    rowLabels=MODELS,
    colLabels=COL_LABELS,
    loc="center",
    cellLoc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(12)
tbl.scale(1.25, 2.6)

HEADER_BG  = "#1a2a3a"
HEADER_FG  = "white"
NEG_COLOR  = "#fde8e8"
GOLD_COLOR = "#f5a623"

for j in range(len(COL_LABELS)):
    cell = tbl[0, j]
    cell.set_facecolor(HEADER_BG)
    cell.get_text().set_color(HEADER_FG)
    cell.get_text().set_fontweight("bold")

for i in range(len(MODELS)):
    cell = tbl[i + 1, -1]
    cell.set_facecolor(HEADER_BG)
    cell.get_text().set_color(HEADER_FG)
    cell.get_text().set_fontweight("bold")

for i in range(len(MODELS)):
    for j in range(len(COL_LABELS)):
        v    = data[i, j]
        cell = tbl[i + 1, j]
        if v > 0:
            intensity = min(v / pos_max, 1.0)
            r = 1.0 - 0.55 * intensity
            g = 1.0 - 0.10 * intensity
            b = 1.0 - 0.55 * intensity
            cell.set_facecolor((r, g, b))
        else:
            cell.set_facecolor(NEG_COLOR)

r, c = max_idx
tbl[r + 1, c].set_facecolor(GOLD_COLOR)
tbl[r + 1, c].get_text().set_fontweight("bold")
tbl[r + 1, c].get_text().set_fontsize(13)

ax.set_title(
    f"Sharpe Ratio  (annualised, 3% risk-free  ·  stat window from signal activation, matching plot logic  ·  "
    f"symmetric base at d=0.2%  ·  gold = global max)",
    fontsize=11, fontweight="bold", pad=18,
)

out_path = ROOT / "output" / "expanding_window" / "collage" / "sharpe_table.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: {out_path}")
