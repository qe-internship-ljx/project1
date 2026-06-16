"""
oos_r2_table.py
===============
In-sample and out-of-sample R² for every standard 20-day-return regression.

In-sample  : static OLS on the full available sample (2006–2026).
OOS R²     : static OLS trained through 2021-12-31, evaluated on 2022–2026.
             A 20-day gap is left between the last training day and the first
             OOS evaluation day to avoid label leakage from overlapping returns.
             OOS R² = 1 - SS_res / SS_tot (benchmark is the OOS sample mean).
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

TRAIN_END = "2021-12-31"   # last date included in the static training window
OOS_GAP   = 20             # trading-day gap to avoid 20-day label leakage
FWD_COL   = "fwd_20d"

MODELS = {
    "VRP":                  ["VP"],
    "VVIX MA5":             ["vvix_ma5"],
    "Term Slope":           ["term_slope"],
    "VRP + VVIX MA5":       ["VP", "vvix_ma5"],
    "VVIX MA5 + Term Slope":["vvix_ma5", "term_slope"],
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
    panel["vrp_pos"]   = panel["VP"].clip(lower=0)
    panel["vrp_neg"]   = panel["VP"].clip(upper=0)

    return panel


def compute_is(panel, predictors):
    """Full-sample static OLS R² (2006–2026, or from first available data)."""
    sub = panel.dropna(subset=predictors + [FWD_COL])
    X   = add_constant(sub[predictors], has_constant="skip")
    res = OLS(sub[FWD_COL], X).fit()
    return res.rsquared


def compute_oos(panel, predictors):
    """
    Static OLS trained through TRAIN_END, evaluated on the window that
    starts OOS_GAP observations after the last training row.
    OOS R² = 1 - SS_res / SS_tot  (benchmark: OOS sample mean).
    """
    sub = panel.dropna(subset=predictors + [FWD_COL]).copy()

    train = sub[sub.index <= pd.Timestamp(TRAIN_END)]
    if len(train) < 2:
        return np.nan

    X_tr = add_constant(train[predictors], has_constant="skip")
    res  = OLS(train[FWD_COL], X_tr).fit()

    # First OOS row is OOS_GAP positions after the last training row
    n_train = len(train)
    oos     = sub.iloc[n_train + OOS_GAP :]
    if len(oos) == 0:
        return np.nan

    X_oos = add_constant(oos[predictors], has_constant="skip")
    y_hat = res.predict(X_oos).values
    y_act = oos[FWD_COL].values

    ss_res = np.sum((y_act - y_hat)     ** 2)
    ss_tot = np.sum((y_act - y_act.mean()) ** 2)
    return 1.0 - ss_res / ss_tot


def get_oos_residuals(panel, predictors):
    """Return a DataFrame with columns [date, model_resid, mean_resid] for the OOS window."""
    sub = panel.dropna(subset=predictors + [FWD_COL]).copy()

    train = sub[sub.index <= pd.Timestamp(TRAIN_END)]
    X_tr  = add_constant(train[predictors], has_constant="skip")
    res   = OLS(train[FWD_COL], X_tr).fit()

    n_train = len(train)
    oos     = sub.iloc[n_train + OOS_GAP :]

    X_oos   = add_constant(oos[predictors], has_constant="skip")
    y_hat   = res.predict(X_oos).values
    y_act   = oos[FWD_COL].values
    y_bar   = y_act.mean()   # OOS sample mean (consistent with R² denominator)

    return pd.DataFrame({
        "model_resid": y_act - y_hat,
        "mean_resid":  y_act - y_bar,
        "y_hat":       y_hat,
    }, index=oos.index)


def plot_vrp_oos_residuals(panel, out_dir):
    resid = get_oos_residuals(panel, ["VP"])

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax2 = ax.twinx()

    ax.plot(resid.index, resid["model_resid"] * 100, color="#1a6faf", lw=1.0,
            label="VRP model residual  (y − ŷ)", zorder=2)
    ax.plot(resid.index, resid["mean_resid"]  * 100, color="#e05c2a", lw=1.0,
            ls="--", label="Mean baseline residual  (y − ȳ_OOS)", zorder=3)
    ax.axhline(0, color="black", lw=0.7, ls="--")

    ax2.plot(resid.index, resid["y_hat"] * 100, color="#2ca02c", lw=1.2,
             ls="-", label="Predicted return  ŷ", zorder=1)
    ax2.axhline(0, color="#2ca02c", lw=0.4, ls=":")

    ax.set_title(
        "OOS Prediction Residuals & Predicted Return — Univariate VRP  (static model trained ≤ 2021-12-31)\n"
        f"OOS window: {resid.index.min().date()} – {resid.index.max().date()}  |  20-day gap applied",
        fontsize=10, fontweight="bold",
    )
    ax.set_ylabel("Residual (%)")
    ax2.set_ylabel("Predicted return (%)", color="#2ca02c")
    ax2.tick_params(axis="y", labelcolor="#2ca02c")
    ax.set_xlabel("Date")
    ax.grid(axis="y", ls=":", alpha=0.5)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9)

    out_png = out_dir / "vrp_oos_residuals.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_png}")


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
        is_r2  = compute_is(panel, preds)
        oos_r2 = compute_oos(panel, preds)
        rows.append({
            "Model":  name,
            "IS R²":  is_r2,
            "OOS R²": oos_r2,
        })

    df = pd.DataFrame(rows).set_index("Model")

    out_dir = ROOT / "output" / "expanding_window"

    # ── Table visualization: positive-OOS-R² models only ─────────────────────
    PRED_LABELS = {
        "VRP":                   "VRP",
        "VVIX MA5":              "VVIX MA5",
        "Term Slope":            "Term Slope",
        "VRP + VVIX MA5":        "VRP + VVIX MA5",
        "VVIX MA5 + Term Slope": "VVIX MA5 + Term Slope",
    }

    pos = df.copy()
    pos = pos.sort_values("OOS R²", ascending=False)
    pos.index = [PRED_LABELS.get(m, m) for m in pos.index]

    col_headers = ["IS R²", "OOS R²"]
    table_data = []
    for _, row in pos.iterrows():
        table_data.append([
            f"{row['IS R²']*100:.2f}%",
            f"{row['OOS R²']*100:+.2f}%",
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
        oos_val = pos.iloc[row_i - 1]["OOS R²"]
        for col_i in range(len(col_headers)):
            cell = tbl[row_i, col_i]
            cell.set_facecolor(bg)
            if col_i == 1:
                if oos_val >= 0:
                    cell.set_facecolor("#d4edda" if (row_i - 1) % 2 == 0 else "#c3e6cb")
                    cell.set_text_props(fontweight="bold", color="#155724")
                else:
                    cell.set_facecolor("#f8d7da" if (row_i - 1) % 2 == 0 else "#f5c6cb")
                    cell.set_text_props(fontweight="bold", color="#721c24")

    ax.set_title(
        "In-Sample vs Out-of-Sample Fit  —  20-Day Static Regression\n"
        "IS: full sample (2006–2026)  |  OOS: trained ≤2021, evaluated 2022–2026 (20-day gap)",
        fontsize=10, pad=14, fontweight="bold",
    )

    out_png = out_dir / "oos_r2_table.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_png}")
    print()

    plot_vrp_oos_residuals(panel, out_dir)


if __name__ == "__main__":
    main()
