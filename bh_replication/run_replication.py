"""
run_replication.py
==================
Master script for the Bekaert & Hoerova (2014) replication.

Produces two combined output images (one per formulation):

  bh_vix_summary.png
    Top:    Predicted VRP time series with long-run mean ± 1σ
    Bottom: Table of IS adj-R², OOS MZ-R², and per-variable beta / NW-SE / t-stat

  bh_vs_summary.png
    Same layout but using the pure VS²/12 formulation.
    VS regression starts only from when VS data is first available.
    VIX data is NEVER mixed into the VS model.

Paper benchmarks (Table 3, Model 8):
  OOS  RMSE = 46.077  MAE = 16.856  MAPE = 0.347  R² = 0.555
  IS   RMSE = 10.508
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
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.stats.sandwich_covariance import cov_hac

sys.path.insert(0, str(Path(__file__).parent))
from data_prep import build_panel, load_sp500_returns, load_vix, compute_rv_components
from har_model import estimate_har, out_of_sample_forecast, NW_LAGS

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT   = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

PAPER_START = "1990-01-02"
PAPER_END   = "2010-10-01"
PAPER_SPLIT = "2005-07-15"   # 75% split matching B&H

PAPER_COEFS = {
    "const":    3.730,
    "VIX2_lag": 0.108,
    "RV22_lag": 0.199,
    "RV5_lag":  0.330,
    "RV1_lag":  0.107,
}
PAPER_OOS = {"rmse": 46.077, "mae": 16.856, "mape": 0.347, "mz_r2": 0.555}


# ─────────────────────────────────────────────────────────────────────────────
# Data loaders
# ─────────────────────────────────────────────────────────────────────────────

def _build_vs_panel() -> pd.DataFrame:
    """
    Build panel using VS²/12 as the implied-variance predictor.
    Only rows where VS data exists are included; VIX is never used.
    """
    # SP500 returns and RV components
    sp_ret = load_sp500_returns()
    rv     = compute_rv_components(sp_ret)

    # Variance swap 1m SPX (pure VS, no fallback to VIX)
    vs_raw = pd.read_csv(DATA_DIR / "EquityIndexVarianceSwapData.csv",
                         parse_dates=["DATE"])
    vs = (vs_raw[(vs_raw["UNDERLYING"] == "SPX") & (vs_raw["TENOR_MONTHS"] == 1.0)]
          .sort_values("DATE")
          .set_index("DATE")["IMPLIED_VOLATILITY"])
    vs.index.name = "date"

    vs2 = (vs ** 2 / 12.0).rename("VS2")

    panel = rv.join(vs2, how="inner").dropna()
    panel["RV22_fwd"] = panel["RV22"].shift(-22)
    # Name the implied-variance lag 'VIX2_lag' so estimate_har() works directly.
    # The column contains VS²/12 data — the name is a structural label only.
    panel["VIX2_lag"] = panel["VS2"].shift(1)
    panel["RV22_lag"] = panel["RV22"].shift(1)
    panel["RV5_lag"]  = panel["RV5"].shift(1)
    panel["RV1_lag"]  = panel["RV1"].shift(1)
    return panel.dropna()


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def _label_crises(ax, start, end):
    crises = [
        ("Gulf War",      "1990-08-01", "1991-03-01"),
        ("Asian/LTCM",    "1997-07-01", "1999-01-01"),
        ("9/11",          "2001-09-01", "2001-12-01"),
        ("GFC",           "2007-06-01", "2009-06-01"),
        ("COVID",         "2020-02-01", "2020-06-01"),
    ]
    for lbl, s, e in crises:
        s, e = pd.Timestamp(s), pd.Timestamp(e)
        if e < start or s > end:
            continue
        ax.axvspan(max(s, start), min(e, end), alpha=0.09,
                   color="grey", linewidth=0)


def plot_bh_formulation_summary(
    panel: pd.DataFrame,
    res_is: dict,
    res_oos: dict,
    formulation: str,          # "VIX" or "VS"
    ivar_col: str,             # column in panel holding IVar (e.g. "VIX2" or "VS2")
    ivar_lag_label: str,       # display label for lag predictor
    output_path: Path,
) -> Path:
    """
    Two-panel figure:
      Top:    Predicted VRP over time with long-run mean ± 1σ
      Bottom: Results table (IS adj-R², OOS MZ-R², betas, t-stats)
    """
    # Compute VRP = IVar - CV (using full-sample IS fitted values)
    panel = panel.copy()
    panel["CV"] = res_is["fitted"]
    panel["VP"] = panel[ivar_col] - panel["CV"]

    vrp_mean = float(panel["VP"].mean())
    vrp_std  = float(panel["VP"].std())
    s, e     = panel.index[0], panel.index[-1]

    fig = plt.figure(figsize=(14, 13))
    gs  = fig.add_gridspec(2, 1, height_ratios=[1.6, 1.0], hspace=0.35)
    ax_vrp = fig.add_subplot(gs[0])
    ax_tbl = fig.add_subplot(gs[1])

    # ── Top: VRP time series ─────────────────────────────────────────────────
    _label_crises(ax_vrp, s, e)
    ax_vrp.fill_between(panel.index, panel["VP"], 0,
                        where=(panel["VP"] >= 0), color="steelblue",
                        alpha=0.55, label="VRP > 0")
    ax_vrp.fill_between(panel.index, panel["VP"], 0,
                        where=(panel["VP"] < 0), color="salmon",
                        alpha=0.55, label="VRP < 0")
    ax_vrp.axhline(0, color="black", lw=0.5)
    ax_vrp.axhline(vrp_mean, color="navy", lw=1.6, ls="--",
                   label=f"Long-run mean = {vrp_mean:.2f}")
    ax_vrp.axhline(vrp_mean + vrp_std, color="steelblue", lw=1.1, ls=":",
                   label=f"+1σ  ({vrp_mean + vrp_std:.2f})")
    ax_vrp.axhline(vrp_mean - vrp_std, color="salmon", lw=1.1, ls=":",
                   label=f"−1σ  ({vrp_mean - vrp_std:.2f})")
    ax_vrp.set_ylabel("VRP = IVar − CV  (%² monthly)", fontsize=9)
    ax_vrp.set_title(
        f"Predicted Variance Risk Premium — HAR-RV-{formulation}  "
        f"[{s.date()} – {e.date()}]",
        fontsize=10,
    )
    ax_vrp.legend(fontsize=8, loc="upper right", ncol=2)
    ax_vrp.set_ylim(-280, 520)
    step = 5 if (e - s).days > 5000 else 2
    ax_vrp.xaxis.set_major_locator(mdates.YearLocator(step))
    ax_vrp.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # ── Bottom: results table ────────────────────────────────────────────────
    ax_tbl.axis("off")

    var_names   = ["const", "VIX2_lag", "RV22_lag", "RV5_lag", "RV1_lag"]
    var_display = ["const", ivar_lag_label, "RV(22) β^m", "RV(5) β^w", "RV(1) β^d"]

    NCOLS = 5
    paper_col_hdr = "Paper coef" if formulation == "VIX" else "—"

    # Summary rows (padded to NCOLS)
    summary_data = [
        ["Metric",    "In-Sample (full)",           "Out-of-Sample (75% split)", "", ""],
        ["Adj-R²",    f"{res_is['adj_r2']:.4f}",    "—",                         "", ""],
        ["RMSE",      f"{res_is['rmse_is']:.3f}",   f"{res_oos['oos_rmse']:.3f}","", ""],
        ["MZ-R²",     "—",                           f"{res_oos['oos_mz_r2']:.4f}","",""],
        ["n obs",     f"{res_is['n']:,}",
                      f"train {res_oos['n_train']:,} / test {res_oos['n_test']:,}", "", ""],
    ]

    coef_header = ["Variable", "Beta (IS)", "NW-SE", "t-stat (NW)", paper_col_hdr]
    coef_rows = [coef_header]
    for vn, vd in zip(var_names, var_display):
        beta  = res_is["params"].get(vn, np.nan)
        nwse  = res_is["nw_se"].get(vn, np.nan)
        tstat = res_is["t_stats"].get(vn, np.nan)
        paper = PAPER_COEFS.get(vn, "—") if formulation == "VIX" else "—"
        paper_str = f"{paper:.3f}" if isinstance(paper, float) else paper
        coef_rows.append([vd,
                          f"{beta:.4f}",
                          f"{nwse:.4f}",
                          f"{tstat:.3f}",
                          paper_str])

    blank_row  = [[""] * NCOLS]
    all_rows   = summary_data + blank_row + coef_rows

    tbl = ax_tbl.table(
        cellText=all_rows,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.15, 1.55)

    # Style header rows
    for col in range(NCOLS):
        tbl[0, col].set_facecolor("#d0d8e8")
        tbl[0, col].set_text_props(fontweight="bold")
    coef_header_row = len(summary_data) + 1  # +1 for blank row
    for col in range(NCOLS):
        tbl[coef_header_row, col].set_facecolor("#d0d8e8")
        tbl[coef_header_row, col].set_text_props(fontweight="bold")

    ax_tbl.set_title(
        f"IS/OOS Statistics — HAR-RV-{formulation}   "
        f"(Newey-West SEs, {NW_LAGS} lags)",
        fontsize=9, pad=4,
    )

    plt.suptitle(
        f"BH Replication — HAR-RV-{formulation} Model",
        fontsize=11, fontweight="bold",
    )
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved {output_path}")
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("  Bekaert & Hoerova (2014) — HAR-RV-VIX and HAR-RV-VS")
    print("=" * 72)

    # ── VIX formulation ───────────────────────────────────────────────────────
    print("\n[VIX 1] Building VIX panel (full sample, 1990-present)…")
    panel_vix = build_panel()     # uses VIX²/12 as implied-variance predictor

    print(f"    VIX panel: {panel_vix.shape[0]:,} obs  "
          f"({panel_vix.index.min().date()} – {panel_vix.index.max().date()})")

    print("[VIX 2] In-sample HAR-RV-VIX estimate…")
    res_vix_is  = estimate_har(panel_vix, "VIX full-sample IS")

    print("[VIX 3] OOS forecast (75% split = 2005-07-15)…")
    res_vix_oos = out_of_sample_forecast(panel_vix, PAPER_SPLIT, "VIX OOS")

    print(f"    IS  Adj-R²={res_vix_is['adj_r2']:.4f}  "
          f"IS  RMSE={res_vix_is['rmse_is']:.3f}")
    print(f"    OOS MZ-R²={res_vix_oos['oos_mz_r2']:.4f}  "
          f"OOS RMSE={res_vix_oos['oos_rmse']:.3f}  "
          f"[paper: MZ-R²=0.555  RMSE=46.077]")

    print("[VIX 4] Generating VIX summary image…")
    plot_bh_formulation_summary(
        panel       = panel_vix,
        res_is      = res_vix_is,
        res_oos     = res_vix_oos,
        formulation = "VIX",
        ivar_col    = "VIX2",
        ivar_lag_label = "VIX²/12 (α)",
        output_path = OUTPUT / "bh_vix_summary.png",
    )

    # ── VS formulation ────────────────────────────────────────────────────────
    # VS panel uses only VS²/12; starts from first VS data point.
    # No VIX data is used in training or in the implied-variance feature.
    print("\n[VS 1] Building VS panel (VS-available period only, no VIX)…")
    panel_vs = _build_vs_panel()

    print(f"    VS panel: {panel_vs.shape[0]:,} obs  "
          f"({panel_vs.index.min().date()} – {panel_vs.index.max().date()})")

    print("[VS 2] In-sample HAR-RV-VS estimate…")
    res_vs_is = estimate_har(panel_vs, "VS full-sample IS")

    # 75% split of the VS-available sample
    vs_split_idx = panel_vs.index[int(0.75 * len(panel_vs))]
    print(f"    VS OOS split: {vs_split_idx.date()}  (75% of {len(panel_vs)} obs)")
    res_vs_oos = out_of_sample_forecast(panel_vs, str(vs_split_idx.date()), "VS OOS")

    print(f"    IS  Adj-R²={res_vs_is['adj_r2']:.4f}  "
          f"IS  RMSE={res_vs_is['rmse_is']:.3f}")
    print(f"    OOS MZ-R²={res_vs_oos['oos_mz_r2']:.4f}  "
          f"OOS RMSE={res_vs_oos['oos_rmse']:.3f}")

    # The panel column holding the actual VS²/12 series (needed for VRP plot)
    # In _build_vs_panel(), VS2 is the column.
    print("[VS 3] Generating VS summary image…")
    plot_bh_formulation_summary(
        panel       = panel_vs,
        res_is      = res_vs_is,
        res_oos     = res_vs_oos,
        formulation = "VS",
        ivar_col    = "VS2",
        ivar_lag_label = "VS²/12 (α)",
        output_path = OUTPUT / "bh_vs_summary.png",
    )

    print(f"\nOutputs saved to {OUTPUT}/")
    print("=" * 72)


if __name__ == "__main__":
    main()
