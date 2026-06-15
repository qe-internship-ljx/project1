"""
run_bh_intraday.py
==================
Replicates bh_replication/run_replication.py exactly, but uses intraday
5-min E-mini S&P 500 realized variance in place of squared daily returns.

Produces two output images (one per formulation):

  bh_intraday_vix_summary.png
    Top:    Predicted VRP time series with long-run mean ± 1σ
    Bottom: Table of IS adj-R², OOS MZ-R², and per-variable beta / NW-SE / t-stat

  bh_intraday_vs_summary.png
    Same layout but using VS²/12 as the implied-variance predictor.
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

ROOT   = Path(__file__).parent
DATA   = ROOT.parent / "data"
OUTPUT = ROOT / "output"
OUTPUT.mkdir(exist_ok=True)

# Import intraday RV loader from experiment.py and HAR model from bh_replication
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "bh_replication"))

from experiment import load_intraday_rv, load_vix_ivar
from har_model import estimate_har, out_of_sample_forecast, NW_LAGS


# ─────────────────────────────────────────────────────────────────────────────
# Panel builders
# ─────────────────────────────────────────────────────────────────────────────

def build_vix_panel() -> pd.DataFrame:
    """Intraday 5-min RV + VIX²/12 as implied variance."""
    rv   = load_intraday_rv()
    ivar = load_vix_ivar()          # Series named "IVar" = VIX²/12

    panel = rv.join(ivar, how="inner").dropna()
    panel["VIX2"] = panel["IVar"]   # rename to match plotting convention

    panel["RV22_fwd"] = panel["RV22"].shift(-22)
    panel["VIX2_lag"] = panel["VIX2"].shift(1)
    panel["RV22_lag"] = panel["RV22"].shift(1)
    panel["RV5_lag"]  = panel["RV5"].shift(1)
    panel["RV1_lag"]  = panel["RV1"].shift(1)
    return panel.dropna()


def build_vs_panel() -> pd.DataFrame:
    """Intraday 5-min RV + VS²/12 as implied variance (VS-available dates only)."""
    rv = load_intraday_rv()

    vs_raw = pd.read_csv(DATA / "EquityIndexVarianceSwapData.csv", parse_dates=["DATE"])
    vs = (vs_raw[(vs_raw["UNDERLYING"] == "SPX") & (vs_raw["TENOR_MONTHS"] == 1.0)]
          .sort_values("DATE")
          .set_index("DATE")["IMPLIED_VOLATILITY"])
    vs.index.name = "date"
    vs2 = (vs ** 2 / 12.0).rename("VS2")

    panel = rv.join(vs2, how="inner").dropna()

    panel["RV22_fwd"] = panel["RV22"].shift(-22)
    # Column is named VIX2_lag so estimate_har() works without modification;
    # it contains VS²/12 data — the name is a structural label only.
    panel["VIX2_lag"] = panel["VS2"].shift(1)
    panel["RV22_lag"] = panel["RV22"].shift(1)
    panel["RV5_lag"]  = panel["RV5"].shift(1)
    panel["RV1_lag"]  = panel["RV1"].shift(1)
    return panel.dropna()


# ─────────────────────────────────────────────────────────────────────────────
# Plotting — identical layout to bh_replication/run_replication.py
# ─────────────────────────────────────────────────────────────────────────────

def _label_crises(ax, start, end):
    crises = [
        ("Gulf War",   "1990-08-01", "1991-03-01"),
        ("Asian/LTCM", "1997-07-01", "1999-01-01"),
        ("9/11",       "2001-09-01", "2001-12-01"),
        ("GFC",        "2007-06-01", "2009-06-01"),
        ("COVID",      "2020-02-01", "2020-06-01"),
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
    formulation: str,
    ivar_col: str,
    ivar_lag_label: str,
    output_path: Path,
) -> Path:
    """
    Two-panel figure (identical layout to bh_replication):
      Top:    Predicted VRP over time with long-run mean ± 1σ
      Bottom: Results table (IS adj-R², OOS MZ-R², betas, t-stats)
    """
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
        f"(Intraday 5-min RV)   [{s.date()} – {e.date()}]",
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

    summary_data = [
        ["Metric",  "In-Sample (full)",           "Out-of-Sample (75% split)", "", ""],
        ["Adj-R²",  f"{res_is['adj_r2']:.4f}",    "—",                         "", ""],
        ["RMSE",    f"{res_is['rmse_is']:.3f}",   f"{res_oos['oos_rmse']:.3f}","", ""],
        ["MZ-R²",   "—",                           f"{res_oos['oos_mz_r2']:.4f}","",""],
        ["n obs",   f"{res_is['n']:,}",
                    f"train {res_oos['n_train']:,} / test {res_oos['n_test']:,}", "", ""],
    ]

    coef_header = ["Variable", "Beta (IS)", "NW-SE", "t-stat (NW)", "—"]
    coef_rows   = [coef_header]
    for vn, vd in zip(var_names, var_display):
        beta  = res_is["params"].get(vn, np.nan)
        nwse  = res_is["nw_se"].get(vn, np.nan)
        tstat = res_is["t_stats"].get(vn, np.nan)
        coef_rows.append([vd,
                          f"{beta:.4f}",
                          f"{nwse:.4f}",
                          f"{tstat:.3f}",
                          "—"])

    blank_row = [[""] * NCOLS]
    all_rows  = summary_data + blank_row + coef_rows

    tbl = ax_tbl.table(cellText=all_rows, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.15, 1.55)

    for col in range(NCOLS):
        tbl[0, col].set_facecolor("#d0d8e8")
        tbl[0, col].set_text_props(fontweight="bold")
    coef_header_row = len(summary_data) + 1
    for col in range(NCOLS):
        tbl[coef_header_row, col].set_facecolor("#d0d8e8")
        tbl[coef_header_row, col].set_text_props(fontweight="bold")

    ax_tbl.set_title(
        f"IS/OOS Statistics — HAR-RV-{formulation} (Intraday 5-min RV)   "
        f"(Newey-West SEs, {NW_LAGS} lags)",
        fontsize=9, pad=4,
    )

    plt.suptitle(
        f"BH Replication (Intraday RV) — HAR-RV-{formulation} Model",
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
    print("  BH Replication — Intraday 5-min RV (VIX and VS formulations)")
    print("=" * 72)

    # ── VIX formulation ───────────────────────────────────────────────────────
    print("\n[VIX 1] Building VIX panel (intraday RV + VIX²/12)…")
    panel_vix = build_vix_panel()
    print(f"    VIX panel: {panel_vix.shape[0]:,} obs  "
          f"({panel_vix.index.min().date()} – {panel_vix.index.max().date()})")

    print("[VIX 2] In-sample HAR-RV-VIX estimate…")
    res_vix_is = estimate_har(panel_vix, "VIX full-sample IS")

    vix_split = panel_vix.index[int(0.75 * len(panel_vix))]
    print(f"[VIX 3] OOS forecast (75% split = {vix_split.date()})…")
    res_vix_oos = out_of_sample_forecast(panel_vix, str(vix_split.date()), "VIX OOS")

    print(f"    IS  Adj-R²={res_vix_is['adj_r2']:.4f}  "
          f"IS  RMSE={res_vix_is['rmse_is']:.3f}")
    print(f"    OOS MZ-R²={res_vix_oos['oos_mz_r2']:.4f}  "
          f"OOS RMSE={res_vix_oos['oos_rmse']:.3f}")

    print("[VIX 4] Generating VIX summary image…")
    plot_bh_formulation_summary(
        panel          = panel_vix,
        res_is         = res_vix_is,
        res_oos        = res_vix_oos,
        formulation    = "VIX",
        ivar_col       = "VIX2",
        ivar_lag_label = "VIX²/12 (α)",
        output_path    = OUTPUT / "bh_intraday_vix_summary.png",
    )

    # ── VS formulation ────────────────────────────────────────────────────────
    print("\n[VS 1] Building VS panel (intraday RV + VS²/12, VS-available dates only)…")
    panel_vs = build_vs_panel()
    print(f"    VS panel: {panel_vs.shape[0]:,} obs  "
          f"({panel_vs.index.min().date()} – {panel_vs.index.max().date()})")

    print("[VS 2] In-sample HAR-RV-VS estimate…")
    res_vs_is = estimate_har(panel_vs, "VS full-sample IS")

    vs_split = panel_vs.index[int(0.75 * len(panel_vs))]
    print(f"[VS 3] OOS forecast (75% split = {vs_split.date()})…")
    res_vs_oos = out_of_sample_forecast(panel_vs, str(vs_split.date()), "VS OOS")

    print(f"    IS  Adj-R²={res_vs_is['adj_r2']:.4f}  "
          f"IS  RMSE={res_vs_is['rmse_is']:.3f}")
    print(f"    OOS MZ-R²={res_vs_oos['oos_mz_r2']:.4f}  "
          f"OOS RMSE={res_vs_oos['oos_rmse']:.3f}")

    print("[VS 4] Generating VS summary image…")
    plot_bh_formulation_summary(
        panel          = panel_vs,
        res_is         = res_vs_is,
        res_oos        = res_vs_oos,
        formulation    = "VS",
        ivar_col       = "VS2",
        ivar_lag_label = "VS²/12 (α)",
        output_path    = OUTPUT / "bh_intraday_vs_summary.png",
    )

    print(f"\nOutputs saved to {OUTPUT}/")
    print("=" * 72)


if __name__ == "__main__":
    main()
