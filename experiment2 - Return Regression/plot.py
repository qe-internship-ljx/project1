"""
plot.py
=======
Regenerates all non-poor-correlation plots under output/expanding_window/ in one shot.

Poor-correlation baselines (Vol Trend, VIX, Term Slope univariate, Expanding VRP)
are intentionally excluded.  Run horizon_regression.py directly to regenerate those.

Generated outputs
-----------------
  === Non-leveraged (unit-position) strategies ===
  output/expanding_window/VRP/symmetric_VRP.png
  output/expanding_window/VRP/asymmetric_VRP.png
  output/expanding_window/VRP/base_return_shift_VRP.png
  output/expanding_window/VVIX MA5/symmetric_VVIX_MA5.png
  output/expanding_window/VVIX MA5/asymmetric_VVIX_MA5.png
  output/expanding_window/VVIX MA5/base_return_shift_VVIX_MA5.png
  output/expanding_window/VVIX MA10/symmetric_VVIX_MA10.png
  output/expanding_window/VVIX MA10/asymmetric_VVIX_MA10.png
  output/expanding_window/VVIX MA10/base_return_shift_VVIX_MA10.png
  output/expanding_window/VRP + VVIX MA5/symmetric_VRP_+_VVIX_MA5.png
  output/expanding_window/VRP + VVIX MA5/asymmetric_VRP_+_VVIX_MA5.png
  output/expanding_window/VRP + VVIX MA5/base_return_shift_VRP_+_VVIX_MA5.png
  output/expanding_window/VRP + VVIX MA10/symmetric_VRP_+_VVIX_MA10.png
  output/expanding_window/VRP + VVIX MA10/asymmetric_VRP_+_VVIX_MA10.png
  output/expanding_window/VRP + VVIX MA10/base_return_shift_VRP_+_VVIX_MA10.png
  output/expanding_window/VRP + Term Slope/symmetric_VRP_+_Term_Slope.png
  output/expanding_window/VRP + Term Slope/asymmetric_VRP_+_Term_Slope.png
  output/expanding_window/VRP + Term Slope/base_return_shift_VRP_+_Term_Slope.png
  output/expanding_window/VRP + Open Interest/symmetric_VRP_+_Open_Interest.png
  output/expanding_window/VRP + Open Interest/asymmetric_VRP_+_Open_Interest.png
  output/expanding_window/VRP + Open Interest/base_return_shift_VRP_+_Open_Interest.png

  === Leveraged (leveraged) strategies ===
  output/expanding_window/VVIX MA5/leveraged_{symmetric,asymmetric,base_return_shift}_VVIX_MA5.png
  output/expanding_window/VVIX MA10/leveraged_{symmetric,asymmetric,base_return_shift}_VVIX_MA10.png
  output/expanding_window/VRP/leveraged_{symmetric,asymmetric,base_return_shift}_VRP.png
  output/expanding_window/VRP + VVIX MA5/leveraged_{symmetric,asymmetric,base_return_shift}_VRP_+_VVIX_MA5.png
  output/expanding_window/VRP + VVIX MA10/leveraged_{symmetric,asymmetric,base_return_shift}_VRP_+_VVIX_MA10.png
  output/expanding_window/VRP + Term Slope/leveraged_{symmetric,asymmetric,base_return_shift}_VRP_+_Term_Slope.png
  output/expanding_window/VRP + Open Interest/leveraged_{symmetric,asymmetric,base_return_shift}_VRP_+_Open_Interest.png
  output/expanding_window/trivariate/leveraged_{symmetric,asymmetric,base_return_shift}_VRP_+_VVIX_MA5_+_Term_Slope.png
  output/expanding_window/comparisons/leveraged_asymmetric_vvix_vs_vrp_vvix.png

  === Summary table ===
  output/expanding_window/sharpe_table_extended.png   (7 models x 6 strategies)

All regression positions and betas are cached in output/regression_cache/, so
re-runs only redo the plot rendering.
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

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT.parent / "bh_replication"))

import base_strategies
import leveraged_strategies


def make_vrp_vvix_ma5_scatter():
    """Scatter plot of VRP against VVIX MA5 on their common dates.

    VRP is the 'VP' column from Experiment 1's production-loop output; VVIX MA5
    is the 5-day moving average of (monthly-unit) VVIX.  Annotated with the
    Pearson correlation and saved to
    output/expanding_window/scatter_vrp_vs_vvix_ma5.png.
    """
    from helpers import load_vrp_series, load_vvix, compute_vvix_ma5

    print("\nBuilding VRP vs VVIX MA5 scatter — loading data...")
    vrp      = load_vrp_series()["VP"]
    vvix_ma5 = compute_vvix_ma5(load_vvix())

    df = pd.concat([vrp, vvix_ma5], axis=1, keys=["VRP", "vvix_ma5"]).dropna()
    corr = df["VRP"].corr(df["vvix_ma5"])

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(df["vvix_ma5"], df["VRP"], s=10, alpha=0.4, color="#1a2a3a",
               edgecolors="none")
    ax.set_xlabel("VVIX MA5")
    ax.set_ylabel("VRP")
    ax.set_title(f"VRP vs VVIX MA5  (n={len(df)}, Pearson r = {corr:+.3f})",
                 fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)

    out_path = ROOT / "output" / "expanding_window" / "scatter_vrp_vs_vvix_ma5.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def make_sharpe_table():
    """Render the 7x6 extended Sharpe ratio table (all main models x 6 strategies).

    Stat-window logic mirrors the plot logic exactly (stats measured from signal
    activation when activation is post-2020), base strategies use d=0.2%, and
    Sharpe ratios are annualised against a 3% risk-free rate.  Saved to
    output/expanding_window/sharpe_table_extended.png.
    """
    from helpers import (
        load_vrp_series, load_es_front_month, load_vvix, compute_vvix_ma5, compute_vvix_ma10,
        load_vix_futures_term_structure, load_es_open_interest,
        compute_buy_and_hold, simulate_strategy,
        compute_trend_quotient, build_master_panel,
    )
    from fh_replication.fh_replication import compute_vix_term_slope
    from regressions import (
        OOS_START, OOS_GAP, NW_LAGS, DELTAS,
    )
    from base_strategies import (
        run_ew, run_ew_bivariate,
        run_ew_asym, run_ew_asym_bivariate,
        run_ew_rolmu, run_ew_rolmu_bivariate,
    )
    from leveraged_strategies import (
        run_ew_leveraged_sym, run_ew_leveraged_asym, run_ew_leveraged_rolmu,
        run_ew_biv_leveraged_sym, run_ew_biv_leveraged_asym, run_ew_biv_leveraged_rolmu,
    )

    print("\nBuilding extended Sharpe table — loading data...")
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
    panel["vvix_ma10"]     = vvix_ma10.reindex(panel.index)
    FWD = "fwd_20d"

    daily_ret = panel["daily_ret"].dropna()
    bah_sim   = simulate_strategy(compute_buy_and_hold(daily_ret), daily_ret)

    # ── Stat-window helpers (mirror plot logic exactly) ───────────────────────
    def _activation_to_stat_start(activation):
        if activation is not None and activation > pd.Timestamp("2020-01-01"):
            idx   = bah_sim.index
            act_i = idx.searchsorted(activation)
            return idx[min(act_i + 1, len(idx) - 1)]
        return pd.Timestamp(OOS_START)

    def stat_start_single(sim):
        pos_oos = sim["position"][sim.index >= OOS_START]
        active  = pos_oos[pos_oos != 0]
        return _activation_to_stat_start(active.index.min() if len(active) else None)

    def stat_start_dict(sims):
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

    def multi_delta_sharpe(pos_list):
        all_sims = [sim_from_pos(pos) for pos in pos_list]
        stat = stat_start_dict(all_sims)
        return sharpe(all_sims[0], stat), stat

    def base_sharpe(pred1, pred2=None):
        pos_list = [
            run_ew(panel, pred1, FWD, OOS_GAP, NW_LAGS, delta=d) if pred2 is None
            else run_ew_bivariate(panel, pred1, pred2, FWD, OOS_GAP, NW_LAGS, delta=d)
            for d in DELTAS
        ]
        return multi_delta_sharpe(pos_list)

    def asym_sharpe(pred1, pred2=None):
        pos_list = [
            run_ew_asym(panel, pred1, FWD, OOS_GAP, NW_LAGS, delta=d) if pred2 is None
            else run_ew_asym_bivariate(panel, pred1, pred2, FWD, OOS_GAP, NW_LAGS, delta=d)
            for d in DELTAS
        ]
        return multi_delta_sharpe(pos_list)

    def rolmu_sharpe(pred1, pred2=None):
        pos_list = [
            run_ew_rolmu(panel, pred1, FWD, OOS_GAP, NW_LAGS, delta=d) if pred2 is None
            else run_ew_rolmu_bivariate(panel, pred1, pred2, FWD, OOS_GAP, NW_LAGS, delta=d)
            for d in DELTAS
        ]
        return multi_delta_sharpe(pos_list)

    def single_sharpe(pos):
        sim  = sim_from_pos(pos)
        stat = stat_start_single(sim)
        return sharpe(sim, stat), stat

    # ── Model definitions ──────────────────────────────────────────────────────
    MODELS = [
        ("VRP",               "VP",  None),
        ("VRP + Term Slope",  "VP",  "term_slope"),
        ("VRP + VVIX MA5",    "VP",  "vvix_ma5"),
        ("VVIX MA5",          "vvix_ma5", None),
        ("VVIX MA10",         "vvix_ma10", None),
        ("VRP + VVIX MA10",   "VP",  "vvix_ma10"),
        ("VRP + Open Int.",   "VP",  "open_interest"),
    ]

    COL_LABELS = [
        "Symmetric\n(d=0.2%)", "Asymmetric", "Base-Return\nShift",
        "Lev. Symmetric", "Lev. Asymmetric", "Lev. Base-Return\nShift",
    ]

    N_MODELS = len(MODELS)
    data    = np.full((N_MODELS, 6), np.nan)
    windows = {}

    def fill_row(row_idx, pred1, pred2):
        tag = pred1 if pred2 is None else f"{pred1}+{pred2}"
        print(f"\n  {tag}")

        sr, ss = base_sharpe(pred1, pred2)
        data[row_idx, 0] = sr
        windows[(row_idx, 0)] = ss.strftime("%Y-%m-%d")
        print(f"    symmetric      SR={sr:+.3f}  stats_from={ss.date()}")

        sr, ss = asym_sharpe(pred1, pred2)
        data[row_idx, 1] = sr
        windows[(row_idx, 1)] = ss.strftime("%Y-%m-%d")
        print(f"    asymmetric     SR={sr:+.3f}  stats_from={ss.date()}")

        sr, ss = rolmu_sharpe(pred1, pred2)
        data[row_idx, 2] = sr
        windows[(row_idx, 2)] = ss.strftime("%Y-%m-%d")
        print(f"    base_ret_shift SR={sr:+.3f}  stats_from={ss.date()}")

        pos = (run_ew_leveraged_sym(panel, pred1, FWD, OOS_GAP, NW_LAGS) if pred2 is None
               else run_ew_biv_leveraged_sym(panel, pred1, pred2, FWD, OOS_GAP, NW_LAGS))
        sr, ss = single_sharpe(pos)
        data[row_idx, 3] = sr
        windows[(row_idx, 3)] = ss.strftime("%Y-%m-%d")
        print(f"    lev_symmetric  SR={sr:+.3f}  stats_from={ss.date()}")

        pos = (run_ew_leveraged_asym(panel, pred1, FWD, OOS_GAP, NW_LAGS) if pred2 is None
               else run_ew_biv_leveraged_asym(panel, pred1, pred2, FWD, OOS_GAP, NW_LAGS))
        sr, ss = single_sharpe(pos)
        data[row_idx, 4] = sr
        windows[(row_idx, 4)] = ss.strftime("%Y-%m-%d")
        print(f"    lev_asymmetric SR={sr:+.3f}  stats_from={ss.date()}")

        pos = (run_ew_leveraged_rolmu(panel, pred1, FWD, OOS_GAP, NW_LAGS) if pred2 is None
               else run_ew_biv_leveraged_rolmu(panel, pred1, pred2, FWD, OOS_GAP, NW_LAGS))
        sr, ss = single_sharpe(pos)
        data[row_idx, 5] = sr
        windows[(row_idx, 5)] = ss.strftime("%Y-%m-%d")
        print(f"    lev_base_ret   SR={sr:+.3f}  stats_from={ss.date()}")

    print("Computing Sharpe ratios (hitting cache)...")
    for i, (_, pred1, pred2) in enumerate(MODELS):
        fill_row(i, pred1, pred2)

    # ── Build table image ──────────────────────────────────────────────────────
    ROW_LABELS = [m[0] for m in MODELS]
    max_idx    = np.unravel_index(np.nanargmax(data), data.shape)
    pos_max    = data[data > 0].max() if (data > 0).any() else 1.0

    fig, ax = plt.subplots(figsize=(17, 9))
    ax.axis("off")

    cell_text = [[f"{v:+.3f}" for v in row] for row in data]

    tbl = ax.table(
        cellText=cell_text,
        rowLabels=ROW_LABELS,
        colLabels=COL_LABELS,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    tbl.scale(1.25, 2.4)

    HEADER_BG  = "#1a2a3a"
    HEADER_FG  = "white"
    NEG_COLOR  = "#fde8e8"
    GOLD_COLOR = "#f5a623"

    for j in range(len(COL_LABELS)):
        cell = tbl[0, j]
        cell.set_facecolor(HEADER_BG)
        cell.get_text().set_color(HEADER_FG)
        cell.get_text().set_fontweight("bold")

    for i in range(N_MODELS):
        cell = tbl[i + 1, -1]
        cell.set_facecolor(HEADER_BG)
        cell.get_text().set_color(HEADER_FG)
        cell.get_text().set_fontweight("bold")

    for i in range(N_MODELS):
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
        "Sharpe Ratio  (annualised, 3% risk-free  ·  stat window from signal activation, matching plot logic  ·  "
        "base strategies at d=0.2%  ·  gold = global max)",
        fontsize=11, fontweight="bold", pad=18,
    )

    out_path = ROOT / "output" / "expanding_window" / "sharpe_table_extended.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_path}")


def main():
    base_strategies.main()
    leveraged_strategies.main()
    make_sharpe_table()
    make_vrp_vvix_ma5_scatter()


if __name__ == "__main__":
    main()
