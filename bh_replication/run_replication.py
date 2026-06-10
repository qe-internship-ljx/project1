"""
run_replication.py
==================
Master script for the Bekaert & Hoerova (2014) replication.

Steps:
  1. Load panel of daily SP500 returns and VIX
  2. Estimate Model 8 (HAR-RV-VIX) over:
       (a) Paper sample  1990-01-02 – 2010-10-01
       (b) Full sample   1990-01-02 – 2026-02-28
  3. Out-of-sample (OOS) Mincer-Zarnowitz analysis
       Paper splits: 75% (train end = 2005-07-15) — matching B&H exactly
  4. Variance risk premium series: VP = VIX²/12 − fitted CV
  5. Predictive regressions for SP500 excess returns (as in Table 4)
  6. Rolling in-sample R² over time
  7. Save all outputs to /output/

Paper benchmarks (Table 3, Model 8):
  OOS  RMSE = 46.077  MAE = 16.856  MAPE = 0.347  R² = 0.555
  IS   RMSE = 10.508
"""

import warnings
warnings.filterwarnings("ignore")

import sys, os
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
from har_model import estimate_har, out_of_sample_forecast, rolling_r2, NW_LAGS

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

# ── Paper benchmarks ──────────────────────────────────────────────────────────
PAPER_COEFS = {
    "const":    3.730,
    "VIX2_lag": 0.108,
    "RV22_lag": 0.199,
    "RV5_lag":  0.330,
    "RV1_lag":  0.107,
}
PAPER_NW_SE = {
    "const":    1.903,
    "VIX2_lag": 0.072,
    "RV22_lag": 0.096,
    "RV5_lag":  0.117,
    "RV1_lag":  0.026,
}
PAPER_OOS = {"rmse": 46.077, "mae": 16.856, "mape": 0.347, "mz_r2": 0.555}
PAPER_IS_RMSE = 10.508

# ── Date windows ─────────────────────────────────────────────────────────────
PAPER_START  = "1990-01-02"
PAPER_END    = "2010-10-01"
PAPER_SPLIT  = "2005-07-15"   # 75% split
FULL_END     = None            # all available data

# ─────────────────────────────────────────────────────────────────────────────
def fmt_coef_table(res: dict) -> str:
    """Format coefficient table as a string."""
    lines = [
        f"\n{'Variable':<14} {'Coef':>10} {'NW-SE':>10} {'t-stat':>9}  "
        f"{'Paper Coef':>11} {'Paper SE':>10}",
        "-" * 68,
    ]
    for v in ["const", "VIX2_lag", "RV22_lag", "RV5_lag", "RV1_lag"]:
        pc  = PAPER_COEFS.get(v, float("nan"))
        pse = PAPER_NW_SE.get(v, float("nan"))
        lines.append(
            f"  {v:<12} {res['params'][v]:>10.4f} {res['nw_se'][v]:>10.4f}"
            f" {res['t_stats'][v]:>9.3f}  {pc:>11.3f} {pse:>10.3f}"
        )
    return "\n".join(lines)


def run_predictive_regression(
    panel: pd.DataFrame,
    sp500_ret: pd.Series,
    horizon: int,
) -> dict:
    """
    Regress horizon-average excess returns on VP and CV (Table 4 replication).
    horizon in months (1, 3, 12), matching B&H Table 4 convention.
    Uses end-of-month observations with Newey-West SEs (max(3, 2*h) lags).
    """
    # Build monthly VP/CV panel (end-of-month)
    monthly = panel.resample("ME").last()[["VP", "CV"]].dropna()

    # Monthly compounded SP500 return (decimal), then h-month forward average
    sp_m = sp500_ret.resample("ME").agg(lambda x: (1 + x).prod() - 1)
    log_sp = np.log(1 + sp_m)
    # Annualised forward return: sum of next h log-returns, then annualised
    fwd_log = log_sp.rolling(horizon).sum().shift(-horizon)
    fwd_ann = (np.exp(fwd_log) - 1) * (12.0 / horizon) * 100.0

    monthly["ret_fwd"] = fwd_ann
    monthly = monthly.dropna()

    if len(monthly) < 20:
        return {"horizon": horizon, "n": 0}

    y  = monthly["ret_fwd"]
    X  = add_constant(monthly[["VP"]])
    X2 = add_constant(monthly[["VP", "CV"]])

    nw_lags = max(3, 2 * horizon)
    res  = OLS(y, X).fit()
    nw   = np.sqrt(np.diag(cov_hac(res,  nlags=nw_lags)))
    res2 = OLS(y, X2).fit()
    nw2  = np.sqrt(np.diag(cov_hac(res2, nlags=nw_lags)))

    return {
        "horizon": horizon,
        "n":       len(monthly),
        "univariate": {
            "params": dict(zip(X.columns,  res.params)),
            "nw_se":  dict(zip(X.columns,  nw)),
            "t_stat": dict(zip(X.columns,  res.params / nw)),
            "adj_r2": res.rsquared_adj,
        },
        "bivariate": {
            "params": dict(zip(X2.columns, res2.params)),
            "nw_se":  dict(zip(X2.columns, nw2)),
            "t_stat": dict(zip(X2.columns, res2.params / nw2)),
            "adj_r2": res2.rsquared_adj,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 72)
    print("  Bekaert & Hoerova (2014) Replication — HAR-RV-VIX (Model 8)")
    print("  Data-constrained: daily squared returns proxy for intraday RV")
    print("=" * 72)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("\n[1] Building daily panel …")
    panel_full = build_panel()
    sp_ret     = load_sp500_returns()
    vix_raw    = load_vix()

    # ── Paper sample ──────────────────────────────────────────────────────────
    panel_paper = panel_full[
        (panel_full.index >= PAPER_START) & (panel_full.index <= PAPER_END)
    ].copy()

    print(f"    Full  panel: {panel_full.shape[0]:,} obs  "
          f"({panel_full.index.min().date()} – {panel_full.index.max().date()})")
    print(f"    Paper panel: {panel_paper.shape[0]:,} obs  "
          f"({panel_paper.index.min().date()} – {panel_paper.index.max().date()})")

    # ── 2. Full-sample IS regressions ─────────────────────────────────────────
    print("\n[2] In-sample regressions …")
    res_paper = estimate_har(panel_paper, label="1990–2010 (paper sample)")
    res_full  = estimate_har(panel_full,  label="1990–2026 (full sample)")

    for res in [res_paper, res_full]:
        print(f"\n  === {res['label']}  (n={res['n']:,}) ===")
        print(f"  Adj-R²  : {res['adj_r2']:.4f}  (paper IS benchmark n/a; "
              f"OOS MZ-R² benchmark = 0.555)")
        print(f"  RMSE    : {res['rmse_is']:.3f}  (paper IS RMSE = {PAPER_IS_RMSE:.3f})")
        print(fmt_coef_table(res))

    # ── 3. OOS forecasting — paper's 75% split ────────────────────────────────
    print("\n[3] Out-of-sample analysis (75% split ending 2005-07-15) …")
    oos_paper = out_of_sample_forecast(panel_paper, PAPER_SPLIT,
                                       label="Paper OOS 2005–2010")
    oos_full  = out_of_sample_forecast(panel_full, PAPER_SPLIT,
                                       label="Full OOS 2005–2026")

    for oos in [oos_paper, oos_full]:
        print(f"\n  {oos['label']}  (train={oos['n_train']:,}, test={oos['n_test']:,})")
        print(f"  IS  Adj-R²  : {oos['is_adj_r2']:.4f}")
        print(f"  OOS MZ-R²  : {oos['oos_mz_r2']:.4f}  "
              f"(paper = {PAPER_OOS['mz_r2']:.3f})")
        print(f"  OOS RMSE   : {oos['oos_rmse']:.3f}  "
              f"(paper = {PAPER_OOS['rmse']:.3f})")
        print(f"  OOS MAE    : {oos['oos_mae']:.3f}  "
              f"(paper = {PAPER_OOS['mae']:.3f})")
        print(f"  OOS MAPE   : {oos['oos_mape']:.4f}  "
              f"(paper = {PAPER_OOS['mape']:.3f})")

    # ── 4. VP series & predictive regressions ─────────────────────────────────
    print("\n[4] Variance risk premium series …")
    # CV = fitted conditional variance from Model 8 (in-sample)
    panel_full["CV"]  = res_full["fitted"]
    panel_paper["CV"] = res_paper["fitted"]

    # VP = VIX²/12 - CV  (both in monthly %² units)
    panel_full["VP"]  = panel_full["VIX2"]  - panel_full["CV"]
    panel_paper["VP"] = panel_paper["VIX2"] - panel_paper["CV"]

    print(f"  VP mean (paper sample): {panel_paper['VP'].mean():.3f}")
    print(f"  CV mean (paper sample): {panel_paper['CV'].mean():.3f}")
    print(f"  VP mean (full  sample): {panel_full['VP'].mean():.3f}")
    print(f"  CV mean (full  sample): {panel_full['CV'].mean():.3f}")

    # Stock return predictability (Table 4 replica)
    print("\n[5] Return predictability (VP & CV -> excess returns) ...")
    # Pass decimal daily returns (not ×100); annualisation done inside
    pred_results = {}
    for h in [1, 3, 12]:
        pred_results[h] = run_predictive_regression(
            panel_paper.copy(), sp_ret, horizon=h
        )
        if "univariate" not in pred_results[h]:
            print(f"\n  Horizon {h:>2}m: insufficient data (n={pred_results[h].get('n',0)})")
            continue
        uni  = pred_results[h]["univariate"]
        biv  = pred_results[h]["bivariate"]
        vp_t = uni["params"]["VP"] / uni["nw_se"]["VP"]
        print(
            f"\n  Horizon {h:>2}m (n={pred_results[h]['n']}): "
            f"VP coef={uni['params']['VP']:.4f} "
            f"(NW-SE={uni['nw_se']['VP']:.4f}, t={vp_t:.2f}, "
            f"adj-R2={uni['adj_r2']:.3f}) | "
            f"bivariate adj-R2={biv['adj_r2']:.3f}"
        )

    # ── 5. Rolling R² ─────────────────────────────────────────────────────────
    print("\n[6] Computing rolling R² …")
    roll_paper = rolling_r2(panel_paper, window=252, step=22)
    roll_full  = rolling_r2(panel_full,  window=252, step=22)

    # ── 6. Plots ──────────────────────────────────────────────────────────────
    print("\n[7] Generating plots …")
    _plot_rv_fit(panel_paper, oos_paper, "paper_sample")
    _plot_rv_fit(panel_full,  oos_full,  "full_sample")
    _plot_rolling_r2(roll_paper, roll_full)
    _plot_vp_cv(panel_paper, "paper_sample")
    _plot_vp_cv(panel_full,  "full_sample")
    _plot_forecast_scatter(oos_paper, "paper")
    _plot_forecast_scatter(oos_full,  "full")

    # ── 7. Save summary CSV ───────────────────────────────────────────────────
    print("\n[8] Saving summary CSVs …")
    _save_summaries(res_paper, res_full, oos_paper, oos_full)
    _save_vp_cv_series(panel_paper, panel_full)

    print(f"\nAll outputs saved to {OUTPUT}/")
    return {
        "res_paper": res_paper, "res_full": res_full,
        "oos_paper": oos_paper, "oos_full": oos_full,
        "panel_paper": panel_paper, "panel_full": panel_full,
        "pred_results": pred_results,
        "roll_paper": roll_paper, "roll_full": roll_full,
    }


# ── Plot helpers ──────────────────────────────────────────────────────────────

def _plot_rv_fit(panel, oos, tag):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)

    # Top: actual vs fitted (IS)
    ax = axes[0]
    ax.plot(panel.index, panel["RV22_fwd"], color="steelblue",
            alpha=0.6, linewidth=0.8, label="Actual RV (next 22d)")
    ax.plot(panel.index, panel["CV"], color="firebrick",
            alpha=0.8, linewidth=0.8, label="HAR-VIX fitted CV")
    ax.set_ylabel("Monthly RV (daily-sq proxy, %²)")
    ax.set_title(f"Model 8 – In-sample fit  [{panel.index.min().date()}–{panel.index.max().date()}]")
    ax.legend(fontsize=8)
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_ylim(-20, 300)

    # Bottom: OOS actual vs forecast
    ax = axes[1]
    y_te  = oos["y_test"]
    y_hat = oos["y_hat"]
    ax.plot(y_te.index,  y_te.values,  color="steelblue",
            alpha=0.6, linewidth=0.8, label="Actual RV")
    ax.plot(y_hat.index, y_hat.values, color="darkorange",
            alpha=0.8, linewidth=0.8, label="HAR-VIX OOS forecast")
    ax.axvline(pd.Timestamp(oos["train_end"]), color="grey",
               linestyle="--", linewidth=1, label="Train/test split")
    ax.set_ylabel("Monthly RV (daily-sq proxy, %²)")
    ax.set_title(
        f"Out-of-sample forecast  "
        f"MZ-R²={oos['oos_mz_r2']:.3f}  RMSE={oos['oos_rmse']:.1f}  "
        f"[Paper MZ-R²=0.555  RMSE=46.1]"
    )
    ax.legend(fontsize=8)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_ylim(-20, 500)

    plt.tight_layout()
    plt.savefig(OUTPUT / f"rv_fit_{tag}.png", dpi=150)
    plt.close()


def _plot_rolling_r2(roll_paper, roll_full):
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=False)
    for ax, roll, title, colour in zip(
        axes,
        [roll_paper, roll_full],
        ["Paper sample (1990–2010)", "Full sample (1990–2026)"],
        ["steelblue", "darkorange"],
    ):
        ax.plot(roll.index, roll["adj_r2"], color=colour, linewidth=1)
        ax.axhline(0.555, color="firebrick", linestyle="--",
                   linewidth=1, label="Paper OOS benchmark (0.555)")
        ax.axhline(roll["adj_r2"].mean(), color="grey", linestyle=":",
                   linewidth=1, label=f"Mean = {roll['adj_r2'].mean():.3f}")
        ax.set_ylim(-0.1, 0.85)
        ax.set_ylabel("Adj R²")
        ax.set_title(f"Rolling 252-day in-sample Adj-R²  — {title}")
        ax.legend(fontsize=8)
        ax.xaxis.set_major_locator(mdates.YearLocator(5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    plt.savefig(OUTPUT / "rolling_r2.png", dpi=150)
    plt.close()


def _plot_vp_cv(panel, tag):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    ax = axes[0]
    ax.fill_between(panel.index, panel["VP"], 0,
                    where=panel["VP"] >= 0, color="steelblue",
                    alpha=0.5, label="VP > 0")
    ax.fill_between(panel.index, panel["VP"], 0,
                    where=panel["VP"] < 0, color="salmon",
                    alpha=0.5, label="VP < 0")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("VP = VIX²/12 − CV  (%²)")
    ax.set_title(
        f"Variance Risk Premium (VP)  [{panel.index.min().date()}–{panel.index.max().date()}]"
    )
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(panel.index, panel["CV"], color="darkorange",
            linewidth=0.8, label="CV (fitted conditional variance)")
    ax.plot(panel.index, panel["VIX2"], color="steelblue",
            alpha=0.4, linewidth=0.6, label="VIX²/12")
    ax.set_ylabel("Variance (%²)")
    ax.set_title("Conditional Variance (CV) vs VIX²/12")
    ax.legend(fontsize=8)
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    plt.savefig(OUTPUT / f"vp_cv_{tag}.png", dpi=150)
    plt.close()


def _plot_forecast_scatter(oos, tag):
    y_te  = oos["y_test"]
    y_hat = oos["y_hat"]
    fig, ax = plt.subplots(figsize=(7, 7))
    # Cap display at 99th pct for readability
    cap = np.percentile(y_te, 99)
    mask = (y_te <= cap) & (y_hat <= cap)
    ax.scatter(y_hat[mask], y_te[mask], alpha=0.3, s=4, color="steelblue")
    m = max(y_hat[mask].max(), y_te[mask].max())
    ax.plot([0, m], [0, m], "r--", linewidth=1)
    ax.set_xlabel("Forecast CV")
    ax.set_ylabel("Actual RV")
    ax.set_title(
        f"OOS forecast vs actual  ({tag})\n"
        f"MZ-R²={oos['oos_mz_r2']:.3f}  RMSE={oos['oos_rmse']:.1f}  "
        f"[Paper: MZ-R²=0.555  RMSE=46.1]"
    )
    plt.tight_layout()
    plt.savefig(OUTPUT / f"scatter_{tag}.png", dpi=150)
    plt.close()


def _save_summaries(res_paper, res_full, oos_paper, oos_full):
    rows = []
    for label, res, oos in [
        ("1990–2010 (paper)", res_paper, oos_paper),
        ("1990–2026 (full)",  res_full,  oos_full),
    ]:
        rows.append({
            "sample":        label,
            "n_obs":         res["n"],
            "IS_adj_r2":     round(res["adj_r2"], 4),
            "IS_rmse":       round(res["rmse_is"], 3),
            "OOS_mz_r2":     round(oos["oos_mz_r2"], 4),
            "OOS_rmse":      round(oos["oos_rmse"], 3),
            "OOS_mae":       round(oos["oos_mae"], 3),
            "OOS_mape":      round(oos["oos_mape"], 4),
            "paper_OOS_mz_r2": 0.555,
            "paper_OOS_rmse":  46.077,
            "r2_penalty":    round(0.555 - oos["oos_mz_r2"], 4),
        })
        for v in ["const", "VIX2_lag", "RV22_lag", "RV5_lag", "RV1_lag"]:
            rows[-1][f"coef_{v}"]   = round(res["params"][v], 4)
            rows[-1][f"nwse_{v}"]   = round(res["nw_se"][v], 4)
            rows[-1][f"tstat_{v}"]  = round(res["t_stats"][v], 3)
            rows[-1][f"paper_{v}"]  = PAPER_COEFS.get(v, np.nan)

    pd.DataFrame(rows).to_csv(OUTPUT / "model8_summary.csv", index=False)


def _save_vp_cv_series(panel_paper, panel_full):
    panel_paper[["VIX2", "CV", "VP", "RV22_fwd"]].to_csv(
        OUTPUT / "vp_cv_paper_sample.csv"
    )
    panel_full[["VIX2", "CV", "VP", "RV22_fwd"]].to_csv(
        OUTPUT / "vp_cv_full_sample.csv"
    )


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    results = main()
