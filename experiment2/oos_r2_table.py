"""
oos_r2_table.py
===============
In-sample and out-of-sample R² / RMSE for every standard expanding-window
20-day-return regression.

In-sample  : full-sample OLS R² and RMSE.
OOS R²     : Campbell-Thompson (2008) — 1 - SS_res / SS_tot, where the
             benchmark forecast is the prevailing historical mean at each date.
OOS RMSE   : sqrt(mean squared prediction error) over the OOS period.
No t-stat gate is applied — pure predictive accuracy, not strategy P&L.
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.api import OLS, add_constant

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "bh_replication"))

from har_model import _nw_se
from experiment2 import (
    load_vrp_series, load_es_front_month, load_vvix, compute_vvix_ma5,
    load_vix_spot, load_vix_futures_term_structure,
    compute_trend_quotient, load_vix_basis,
    build_master_panel,
)
from fh_replication.fh_replication import compute_vix_term_slope

OOS_START = "2012-01-01"
MIN_WIN   = 500
OOS_GAP   = 20   # label-leakage buffer for 20-day overlapping returns
FWD_COL   = "fwd_ret_20"

MODELS = {
    "Base — VRP":                   ["VP"],
    "Model A — VRP + Term Slope":   ["VP", "term_slope"],
    "Model B — VRP + Trend Q":      ["VP", "trend_q"],
    "Model C — VRP + VVIX MA5":     ["VP", "vvix_ma5"],
    "Model G — VRP x VVIX MA5":     ["vrp_vvix"],
    "Model H — VRP+/VRP-":          ["vrp_pos", "vrp_neg"],
    "Model VS — VVIX MA5 + TS":     ["vvix_ma5", "term_slope"],
    "VVIX MA5 (univariate)":        ["vvix_ma5"],
    "VVIX raw (univariate)":        ["vvix_raw"],
    "VIX Basis":                    ["vix_basis"],
    "VIX Spot":                     ["vix"],
    "Vol Trend":                    ["vol_trend"],
    "Term Slope":                   ["term_slope"],
}


def build_panel():
    vrp      = load_vrp_series()
    es       = load_es_front_month()
    vvix_raw = load_vvix()
    vvix_ma5 = compute_vvix_ma5(vvix_raw)
    vix_spot = load_vix_spot()
    slope    = compute_vix_term_slope(load_vix_futures_term_structure())
    trend_q  = compute_trend_quotient(es)
    vix_basis = load_vix_basis()

    panel = build_master_panel(vrp, es, slope, trend_q, vvix_ma5)

    # Add extra signals not in build_master_panel
    panel = panel.join(vix_spot.rename("vix"), how="left")
    panel = panel.join(vix_basis.rename("vix_basis"), how="left")
    panel = panel.join(vvix_raw.rename("vvix_raw"), how="left")

    ret   = es["returns"]
    r2    = ret ** 2
    rv5   = np.sqrt(r2.rolling(5).mean()  * 252)
    rv22  = np.sqrt(r2.rolling(22).mean() * 252)
    panel["vol_trend"] = np.log(rv5 / rv22)
    panel["vrp_vvix"]  = panel["VP"] * panel["vvix_ma5"]
    panel["vrp_pos"]   = panel["VP"].clip(lower=0)
    panel["vrp_neg"]   = panel["VP"].clip(upper=0)

    return panel


def compute_is(panel, predictors):
    sub = panel.dropna(subset=predictors + [FWD_COL])
    X   = add_constant(sub[predictors], has_constant="skip")
    res = OLS(sub[FWD_COL], X).fit()
    resid = sub[FWD_COL] - res.fittedvalues
    rmse  = float(np.sqrt((resid ** 2).mean()))
    return res.rsquared, rmse


def compute_oos(panel, predictors):
    sub     = panel.dropna(subset=predictors + [FWD_COL]).copy()
    N       = len(sub)
    fwd     = sub[FWD_COL].values
    oos_idx = sub.index.searchsorted(pd.Timestamp(OOS_START))
    start_i = max(MIN_WIN + OOS_GAP, oos_idx)

    y_hat_list  = []
    y_mean_list = []
    y_act_list  = []

    for i in range(start_i, N):
        train = sub.iloc[0 : i - OOS_GAP]
        X_tr  = add_constant(train[predictors], has_constant="skip")
        res   = OLS(train[FWD_COL], X_tr).fit()

        test  = sub.iloc[[i]][predictors].copy()
        test.insert(0, "const", 1.0)
        y_hat = float(res.predict(test).iloc[0])

        y_bar = float(np.mean(fwd[0 : i - OOS_GAP]))   # prevailing historical mean

        y_hat_list.append(y_hat)
        y_mean_list.append(y_bar)
        y_act_list.append(fwd[i])

    y_hat  = np.array(y_hat_list)
    y_mean = np.array(y_mean_list)
    y_act  = np.array(y_act_list)

    ss_res  = np.sum((y_act - y_hat)  ** 2)
    ss_tot  = np.sum((y_act - y_mean) ** 2)
    oos_r2  = 1.0 - ss_res / ss_tot
    oos_rmse = float(np.sqrt(np.mean((y_act - y_hat) ** 2)))

    return oos_r2, oos_rmse


def main():
    print("Loading data...")
    panel = build_panel()
    print(f"  {len(panel):,} obs  [{panel.index.min().date()} — {panel.index.max().date()}]")
    print()

    rows = []
    for name, preds in MODELS.items():
        missing = [p for p in preds if p not in panel.columns]
        if missing:
            print(f"  SKIP {name}: missing columns {missing}")
            continue
        print(f"  {name}...", flush=True)
        is_r2,  is_rmse  = compute_is(panel, preds)
        oos_r2, oos_rmse = compute_oos(panel, preds)
        rows.append({
            "Model":    name,
            "IS R²":    is_r2,
            "IS RMSE":  is_rmse * 100,   # in %
            "OOS R²":   oos_r2,
            "OOS RMSE": oos_rmse * 100,  # in %
        })

    df = pd.DataFrame(rows).set_index("Model")

    out_dir = ROOT / "output" / "expanding_window"
    out_csv = out_dir / "oos_r2_table.csv"
    df.to_csv(out_csv, float_format="%.6f")
    print(f"  Saved: {out_csv}")

    # ── Table visualization: positive-OOS-R² models only ─────────────────────
    PRED_LABELS = {
        "Base — VRP":                 "VRP",
        "Model A — VRP + Term Slope": "VRP + Term Slope",
        "Model B — VRP + Trend Q":    "VRP + Trend Quotient",
        "Model C — VRP + VVIX MA5":   "VRP + VVIX MA5",
        "Model G — VRP x VVIX MA5":   "VRP x VVIX MA5",
        "Model H — VRP+/VRP-":        "VRP+ / VRP-",
        "Model VS — VVIX MA5 + TS":   "VVIX MA5 + Term Slope",
        "VVIX MA5 (univariate)":      "VVIX MA5",
        "VVIX raw (univariate)":      "VVIX (raw)",
        "VIX Basis":                  "VIX Basis",
        "VIX Spot":                   "VIX",
        "Vol Trend":                  "Vol Trend",
        "Term Slope":                 "Term Slope",
    }

    pos = df[df["OOS R²"] > 0].copy()
    pos = pos.sort_values("OOS R²", ascending=False)
    pos.index = [PRED_LABELS.get(m, m) for m in pos.index]

    col_headers = ["IS R²", "IS RMSE", "OOS R²", "OOS RMSE"]
    table_data = []
    for _, row in pos.iterrows():
        table_data.append([
            f"{row['IS R²']*100:.2f}%",
            f"{row['IS RMSE']:.4f}%",
            f"{row['OOS R²']*100:+.2f}%",
            f"{row['OOS RMSE']:.4f}%",
        ])

    n_rows = len(pos)
    fig_h  = 0.55 + n_rows * 0.42
    fig, ax = plt.subplots(figsize=(8.5, fig_h))
    ax.axis("off")

    tbl = ax.table(
        cellText=table_data,
        rowLabels=list(pos.index),
        colLabels=col_headers,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.55)

    # Style header row
    for col in range(len(col_headers)):
        cell = tbl[0, col]
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold")

    # Style row labels and data cells
    row_colors = ["#f0f4f8", "#ffffff"]
    for row_i in range(1, n_rows + 1):
        bg = row_colors[(row_i - 1) % 2]
        # row label
        tbl[row_i, -1].set_facecolor(bg)
        tbl[row_i, -1].set_text_props(ha="right", fontweight="semibold")
        for col_i in range(len(col_headers)):
            cell = tbl[row_i, col_i]
            cell.set_facecolor(bg)
            # Highlight OOS R² column (col index 2)
            if col_i == 2:
                cell.set_facecolor("#d4edda" if (row_i - 1) % 2 == 0 else "#c3e6cb")
                cell.set_text_props(fontweight="bold", color="#155724")

    ax.set_title(
        "In-Sample vs Out-of-Sample Fit  —  20-Day Expanding Window Regression\n"
        "OOS from 2012-01-01 | Campbell-Thompson R² | No t-stat gate",
        fontsize=10, pad=14, fontweight="bold",
    )

    out_png = out_dir / "oos_r2_table.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_png}")
    print()


if __name__ == "__main__":
    main()
