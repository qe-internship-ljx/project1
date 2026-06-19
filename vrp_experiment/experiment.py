"""
experiment.py
=============
Bekaert & Hoerova (2014) — Full 8-Step Experiment
"The VIX, the Variance Premium and Stock Market Volatility"

Steps
-----
1.  Implied Variance from VIX: IVar = VIX²/12 throughout (monthly %²-units)
2.  Physical Realized Variance: rolling 22-day sum of squared daily returns
3.  HAR panel fitting (Model 8) — strictly out-of-sample coefficients
4.  VRP = ImpliedVariance − CV (fitted conditional variance)
5.  500-day rolling-window OLS production loop (day-by-day, no look-ahead)
6.  RMSE, MAE, MAPE evaluation
7.  Comparison vs. martingale (BTZ Model 30) and univariate VRP return regression

Modules shared with bh_replication (called directly, not duplicated)
---------------------------------------------------------------------
  data_prep : load_sp500_returns, load_vix, compute_rv_components
  har_model : estimate_har, out_of_sample_forecast, NW_LAGS, _nw_se

Outputs (all in ./output/)
--------------------------
  Plots: vrp_experiment_summary_full.png
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
from scipy import stats

from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

# ── Shared modules from bh_replication ────────────────────────────────────────
BH_DIR = Path(__file__).parent.parent / "bh_replication"
sys.path.insert(0, str(BH_DIR))
from data_prep import load_sp500_returns, load_vix, compute_rv_components
from har_model import estimate_har, out_of_sample_forecast, NW_LAGS, _nw_se

ROOT   = Path(__file__).parent
OUTPUT = ROOT / "output"
OUTPUT.mkdir(exist_ok=True)
DATA   = ROOT.parent / "data"

PAPER_START      = "1990-01-02"
PAPER_END        = "2010-10-01"
PAPER_SPLIT      = "2005-07-15"   # 75% train split (matching B&H)
ROLL_WIN         = 500            # production-loop rolling window (trading days)
EXP_TRAIN_START  = "1990-01-02"  # expanding-window anchor (full history from paper start)
EXP_OOS_START    = "2006-01-01"  # first OOS prediction (after 1990-2005 initial training)

# ── Paper benchmarks (Table 3, Model 8) ──────────────────────────────────────
PAPER_COEFS = {"const": 3.730, "VIX2_lag": 0.108, "RV22_lag": 0.199,
               "RV5_lag": 0.330, "RV1_lag": 0.107}
PAPER_NW_SE = {"const": 1.903, "VIX2_lag": 0.072, "RV22_lag": 0.096,
               "RV5_lag": 0.117, "RV1_lag": 0.026}
PAPER_OOS   = {"rmse": 46.077, "mae": 16.856, "mape": 0.347, "mz_r2": 0.555}
PAPER_IS_RMSE = 10.508   # from paper Table 3


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Implied Variance
# ══════════════════════════════════════════════════════════════════════════════
def load_implied_variance() -> pd.Series:
    """
    Implied variance from VIX: IVar = VIX²/12  (monthly %²-units).
    VIX is the CBOE model-free risk-neutral expected variance proxy (annualised %).
    """
    vix = load_vix()                              # from bh_replication.data_prep
    ivar = vix ** 2 / 12.0
    ivar.name = "IVar"
    return ivar


def load_implied_variance_vs() -> pd.Series:
    """
    Implied variance from SPX 1-month variance swap: IVar = VS²/12  (monthly %²-units).
    Pure VS series — no VIX fallback. Available from November 2008 onwards only.
    """
    swap  = pd.read_csv(DATA / "EquityIndexVarianceSwapData.csv", parse_dates=["DATE"])
    spx1m = (swap[(swap["UNDERLYING"] == "SPX") & (swap["TENOR_MONTHS"] == 1.0)]
             .sort_values("DATE")
             .set_index("DATE")["IMPLIED_VOLATILITY"])
    spx1m.index.name = "date"
    ivar = spx1m ** 2 / 12.0
    ivar.name = "IVar"
    return ivar


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2+3 — Panel Assembly
# ══════════════════════════════════════════════════════════════════════════════
def _build_panel_from_ivar(ivar: pd.Series) -> pd.DataFrame:
    """Shared panel builder: join IVar with RV components and create lag features."""
    ret   = load_sp500_returns()
    rv    = compute_rv_components(ret)
    panel = rv.join(ivar, how="inner").dropna()
    panel["RV22_fwd"] = panel["RV22"].shift(-22)
    # VIX2_lag is the lagged implied variance predictor.
    # Named VIX2_lag so har_model.estimate_har / out_of_sample_forecast work directly.
    panel["VIX2_lag"] = panel["IVar"].shift(1)
    panel["RV22_lag"] = panel["RV22"].shift(1)
    panel["RV5_lag"]  = panel["RV5"].shift(1)
    panel["RV1_lag"]  = panel["RV1"].shift(1)
    return panel.dropna()


def build_full_panel() -> pd.DataFrame:
    """Panel using VIX²/12 as implied variance throughout (full history from 1990)."""
    return _build_panel_from_ivar(load_implied_variance())


def build_panel_vs() -> pd.DataFrame:
    """Panel using pure VS²/12 as implied variance (restricted to VS availability, ~2008+)."""
    return _build_panel_from_ivar(load_implied_variance_vs())


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — VRP Extraction
# ══════════════════════════════════════════════════════════════════════════════
def extract_vrp(panel: pd.DataFrame, cv_series: pd.Series) -> pd.DataFrame:
    """VP = IVar − CV  (both in monthly %² units)."""
    out       = panel.copy()
    out["CV"] = cv_series
    out["VP"] = out["IVar"] - out["CV"]
    return out


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Production Loop (500-day rolling OLS)
# ══════════════════════════════════════════════════════════════════════════════
def production_loop(panel: pd.DataFrame, window: int = ROLL_WIN,
                    return_stats: bool = False):
    """
    For each day t ≥ window, fit HAR on [t-window, t-1], forecast RV_t+22.
    Returns DataFrame: date, y_actual, y_hat, error, CV, IVar, VP.
    If return_stats=True, also returns a second DataFrame with IS adj_r2,
    betas, and NW t-stats at each step.
    Strictly no look-ahead (parameters estimated only on past data).

    Uses VIX2_lag as the implied-variance feature (matching har_model convention).
    """
    rows       = []
    stats_rows = []
    idx        = panel.index
    N          = len(idx)
    feats      = ["VIX2_lag", "RV22_lag", "RV5_lag", "RV1_lag"]

    print(f"    Running {N - window - 22} daily production steps (window={window}, oos_gap=22)…",
          flush=True)

    for i in range(window + 22, N - 22):
        train_sl = panel.iloc[i - window - 22 : i - 22]
        if len(train_sl) < 100:
            continue
        y_tr = train_sl["RV22_fwd"]
        X_tr = add_constant(train_sl[feats])
        if X_tr.shape[0] < 50:
            continue

        res_tr = OLS(y_tr, X_tr).fit()

        test_row = panel.iloc[[i]]
        X_te     = add_constant(test_row[feats], has_constant="add")
        y_actual = panel["RV22_fwd"].iloc[i]
        y_hat    = float(res_tr.predict(X_te).iloc[0])
        cv_hat   = y_hat                           # CV = conditional variance forecast
        ivar_t   = float(panel["IVar"].iloc[i])
        vp_t     = ivar_t - cv_hat

        rows.append({
            "date":     idx[i],
            "y_actual": y_actual,
            "y_hat":    y_hat,
            "error":    y_actual - y_hat,
            "CV":       cv_hat,
            "IVar":     ivar_t,
            "VP":       vp_t,
        })

        if return_stats:
            col_names = X_tr.columns.tolist()
            try:
                nw_ses = _nw_se(res_tr, nlags=NW_LAGS)
            except Exception:
                nw_ses = np.full(len(col_names), np.nan)
            stat_row = {"date": idx[i], "adj_r2": float(res_tr.rsquared_adj)}
            for j, col in enumerate(col_names):
                stat_row[col] = float(res_tr.params[col])
                nw_j = nw_ses[j] if not np.isnan(nw_ses[j]) else np.nan
                stat_row[f"t_{col}"] = (float(res_tr.params[col] / nw_j)
                                        if (nw_j and not np.isnan(nw_j) and nw_j != 0)
                                        else np.nan)
            stats_rows.append(stat_row)

    df = pd.DataFrame(rows).set_index("date")
    if return_stats:
        stats_df = pd.DataFrame(stats_rows).set_index("date")
        return df, stats_df
    return df


def production_loop_expanding(panel: pd.DataFrame,
                               train_start: str = EXP_TRAIN_START,
                               oos_start: str = EXP_OOS_START,
                               return_stats: bool = False):
    """
    Expanding-window production loop.
    The training window is anchored at train_start and grows by one day each step.
    Predictions begin only from oos_start (after the 2006-2012 initial training period).
    A strict 22-day gap between the last training label and the prediction row is maintained
    (matching the rolling-window design: train slice uses panel.iloc[anchor : i-22]).

    Returns DataFrame: date, y_actual, y_hat, error, CV, IVar, VP.
    If return_stats=True, also returns (df, stats_df) with IS adj_r2, betas, NW t-stats.
    """
    rows       = []
    stats_rows = []
    idx        = panel.index
    N          = len(idx)
    feats      = ["VIX2_lag", "RV22_lag", "RV5_lag", "RV1_lag"]

    anchor_i  = int(idx.searchsorted(pd.Timestamp(train_start)))
    oos_i     = int(idx.searchsorted(pd.Timestamp(oos_start)))

    print(f"    Running {max(0, N - oos_i - 22)} expanding-window steps "
          f"(anchor={train_start}, OOS from {oos_start}, oos_gap=22)…", flush=True)

    for i in range(oos_i, N - 22):
        # Training slice: anchor up to i-22 (exclusive), matching rolling-window gap
        train_sl = panel.iloc[anchor_i : i - 22]
        if len(train_sl) < 100:
            continue
        y_tr = train_sl["RV22_fwd"]
        X_tr = add_constant(train_sl[feats])
        if X_tr.shape[0] < 50:
            continue

        res_tr = OLS(y_tr, X_tr).fit()

        test_row = panel.iloc[[i]]
        X_te     = add_constant(test_row[feats], has_constant="add")
        y_actual = panel["RV22_fwd"].iloc[i]
        y_hat    = float(res_tr.predict(X_te).iloc[0])
        cv_hat   = y_hat
        ivar_t   = float(panel["IVar"].iloc[i])
        vp_t     = ivar_t - cv_hat

        rows.append({
            "date":     idx[i],
            "y_actual": y_actual,
            "y_hat":    y_hat,
            "error":    y_actual - y_hat,
            "CV":       cv_hat,
            "IVar":     ivar_t,
            "VP":       vp_t,
        })

        if return_stats:
            col_names = X_tr.columns.tolist()
            try:
                nw_ses = _nw_se(res_tr, nlags=NW_LAGS)
            except Exception:
                nw_ses = np.full(len(col_names), np.nan)
            stat_row = {"date": idx[i], "adj_r2": float(res_tr.rsquared_adj)}
            for j, col in enumerate(col_names):
                stat_row[col] = float(res_tr.params[col])
                nw_j = nw_ses[j] if not np.isnan(nw_ses[j]) else np.nan
                stat_row[f"t_{col}"] = (float(res_tr.params[col] / nw_j)
                                        if (nw_j and not np.isnan(nw_j) and nw_j != 0)
                                        else np.nan)
            stats_rows.append(stat_row)

    df = pd.DataFrame(rows).set_index("date")
    if return_stats:
        stats_df = pd.DataFrame(stats_rows).set_index("date")
        return df, stats_df
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Statistical Accuracy Evaluation
# ══════════════════════════════════════════════════════════════════════════════
def compute_metrics(y_actual: np.ndarray, y_hat: np.ndarray,
                    label: str = "") -> dict:
    """RMSE, MAE, MAPE, MZ-R²."""
    valid = ~(np.isnan(y_actual) | np.isnan(y_hat))
    ya    = y_actual[valid]
    yh    = y_hat[valid]
    err   = ya - yh
    rmse  = float(np.sqrt((err**2).mean()))
    mae   = float(np.abs(err).mean())
    mape  = float((np.abs(err) / np.clip(ya, 0.01, None)).mean())
    mz    = OLS(ya, add_constant(yh)).fit()
    return {
        "label":  label,
        "n":      int(valid.sum()),
        "rmse":   round(rmse, 3),
        "mae":    round(mae,  3),
        "mape":   round(mape, 4),
        "mz_r2":  round(float(mz.rsquared), 4),
    }


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — Baseline Comparisons
# ══════════════════════════════════════════════════════════════════════════════
def martingale_forecast(panel: pd.DataFrame, train_end: str) -> pd.Series:
    """BTZ martingale: E[RV_{t+1}] = RV_t (lagged RV22)."""
    test = panel[panel.index > train_end]
    return test["RV22_lag"]


# ══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════════
def _label_crises(ax, start, end):
    """Shade known crisis periods on a time-series axis."""
    crises = [
        ("Gulf War",      "1990-08-01", "1991-03-01"),
        ("Mexican",       "1994-12-01", "1995-03-01"),
        ("Asian/LTCM",    "1997-07-01", "1999-01-01"),
        ("9/11",          "2001-09-01", "2001-12-01"),
        ("Corp Scandals", "2002-01-01", "2003-03-01"),
        ("GFC",           "2007-06-01", "2009-06-01"),
        ("COVID",         "2020-02-01", "2020-06-01"),
    ]
    for lbl, s, e in crises:
        s, e = pd.Timestamp(s), pd.Timestamp(e)
        if e < start or s > end:
            continue
        ax.axvspan(max(s, start), min(e, end), alpha=0.10,
                   color="grey", linewidth=0)


def plot_vp_cv(panel, tag, title_extra=""):
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    s, e = panel.index[0], panel.index[-1]

    ax = axes[0]
    _label_crises(ax, s, e)
    ax.fill_between(panel.index, panel["VP"], 0,
                    where=(panel["VP"] >= 0), color="steelblue",
                    alpha=0.55, label="VP > 0")
    ax.fill_between(panel.index, panel["VP"], 0,
                    where=(panel["VP"] < 0), color="salmon",
                    alpha=0.55, label="VP < 0")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylabel("VP = IVar − CV  (%² monthly)")
    ax.set_title(f"Variance Risk Premium (VP) {title_extra}")
    ax.legend(fontsize=8); ax.set_ylim(-200, 350)

    ax = axes[1]
    _label_crises(ax, s, e)
    ax.plot(panel.index, panel["CV"],   color="darkorange",
            linewidth=0.7, label="CV — HAR fitted")
    ax.plot(panel.index, panel["IVar"], color="steelblue",
            linewidth=0.5, alpha=0.5, label="Implied Var (VIX²/12)")
    ax.set_ylabel("Variance (%² monthly)")
    ax.set_title("Conditional Variance (CV) vs Implied Variance")
    ax.legend(fontsize=8)
    ax.xaxis.set_major_locator(mdates.YearLocator(5 if (e-s).days > 5000 else 2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_ylim(-20, 600)

    plt.tight_layout()
    path = OUTPUT / f"vp_cv_{tag}.png"
    plt.savefig(path, dpi=150); plt.close()
    return path


def plot_oos_forecast(y_te, y_hat, oos_metrics, mart_hat, mart_metrics, tag):
    fig, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=False)

    ax = axes[0]
    ax.plot(y_te.index,  y_te.values,   color="steelblue",  lw=0.8,
            alpha=0.8, label="Actual RV")
    ax.plot(y_hat.index, y_hat.values,  color="darkorange", lw=0.8,
            alpha=0.9, label=f"HAR-VIX  MZ-R²={oos_metrics['mz_r2']:.3f}")
    if mart_hat is not None:
        ax.plot(mart_hat.index, mart_hat.values, color="green", lw=0.7,
                alpha=0.7, linestyle="--",
                label=f"Martingale  MZ-R²={mart_metrics['mz_r2']:.3f}")
    ax.set_ylabel("Monthly RV (%²)"); ax.set_title(f"OOS Forecast — {tag}")
    ax.legend(fontsize=8); ax.set_ylim(-20, 600)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    ax = axes[1]
    err_har = y_te - y_hat
    ax.bar(err_har.index, err_har.values, width=1, color="steelblue",
           alpha=0.5, label="HAR error")
    if mart_hat is not None:
        err_m = y_te - mart_hat.reindex(y_te.index)
        ax.bar(err_m.index, err_m.values, width=1, color="green",
               alpha=0.3, label="Martingale error")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("Forecast error"); ax.set_title("Forecast Errors")
    ax.legend(fontsize=8)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    ax = axes[2]
    cap  = np.percentile(y_te.dropna(), 98)
    mask = (y_te <= cap) & (y_hat <= cap) & (~np.isnan(y_te)) & (~np.isnan(y_hat))
    ax.scatter(y_hat[mask], y_te[mask], alpha=0.25, s=3, color="steelblue")
    m = max(float(y_hat[mask].max()), float(y_te[mask].max()))
    ax.plot([0, m], [0, m], "r--", lw=1)
    ax.set_xlabel("HAR Forecast"); ax.set_ylabel("Actual RV")
    ax.set_title(f"Forecast vs Actual  RMSE={oos_metrics['rmse']:.1f}  "
                 f"MAE={oos_metrics['mae']:.1f}  MAPE={oos_metrics['mape']:.3f}  "
                 f"[Paper: RMSE=46.1 MAPE=0.347]")

    plt.tight_layout()
    path = OUTPUT / f"oos_forecast_{tag}.png"
    plt.savefig(path, dpi=150); plt.close()
    return path


def plot_production_loop(prod_df: pd.DataFrame, tag: str):
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    s, e = prod_df.index[0], prod_df.index[-1]

    ax = axes[0]
    _label_crises(ax, s, e)
    ax.plot(prod_df.index, prod_df["y_actual"], color="steelblue",
            lw=0.7, alpha=0.8, label="Actual RV")
    ax.plot(prod_df.index, prod_df["y_hat"],    color="darkorange",
            lw=0.7, alpha=0.9, label="Prod-loop HAR forecast")
    ax.set_ylabel("Monthly RV (%²)")
    ax.set_title(f"Production Loop ({ROLL_WIN}-day rolling OLS) — {tag}")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(prod_df.index, prod_df["VP"], color="steelblue",
            lw=0.7, label="VP (production)")
    ax.plot(prod_df.index, prod_df["CV"], color="darkorange",
            lw=0.7, label="CV (production)")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("Variance (%² monthly)")
    ax.set_title("Real-time VP and CV from Production Loop")
    ax.legend(fontsize=8)

    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()
    path = OUTPUT / f"production_loop_{tag}.png"
    plt.savefig(path, dpi=150); plt.close()
    return path


def plot_return_pred(results: dict, tag: str):
    horizons = sorted(results.keys())
    vp_coefs = [results[h]["univariate"]["params"].get("VP", np.nan)
                for h in horizons]
    vp_tstat = [results[h]["univariate"]["t_stat"].get("VP", np.nan)
                for h in horizons]
    vp_r2    = [results[h]["univariate"]["adj_r2"] for h in horizons]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    xlbl = [f"{h}m" for h in horizons]

    axes[0].bar(xlbl, vp_coefs, color="steelblue", alpha=0.7)
    axes[0].axhline(0, color="black", lw=0.5)
    axes[0].set_title("VP Coefficient (univariate)")
    axes[0].set_ylabel("Coefficient")

    axes[1].bar(xlbl, vp_tstat,
                color=["red" if abs(t) < 1.96 else "steelblue" for t in vp_tstat],
                alpha=0.7)
    axes[1].axhline( 1.96, color="grey", ls="--", lw=0.8)
    axes[1].axhline(-1.96, color="grey", ls="--", lw=0.8)
    axes[1].set_title("VP t-statistic")
    axes[1].set_ylabel("t-stat")

    axes[2].bar(xlbl, [max(0, r) for r in vp_r2], color="steelblue", alpha=0.7)
    axes[2].set_title("Adj. R² (VP predicting excess returns)")
    axes[2].set_ylabel("Adj. R²")

    plt.suptitle(f"Return Predictability — VP univariate ({tag})")
    plt.tight_layout()
    path = OUTPUT / f"return_pred_{tag}.png"
    plt.savefig(path, dpi=150); plt.close()
    return path


def plot_vrp_comparison(prod_vix: pd.DataFrame, prod_vs: pd.DataFrame,
                        diag_vix: pd.DataFrame, diag_vs: pd.DataFrame):
    """
    Three-panel comparison of VIX-based vs VS-based VRP over the overlap period.
      Row 1: VP time series (VIX vs VS)
      Row 2: IVar time series (VIX²/12 vs VS²/12)
      Row 3: Monthly MZ-R² from each production loop
    """
    overlap_start = prod_vs.index.min()
    overlap_end   = min(prod_vix.index.max(), prod_vs.index.max())

    vix_ol = prod_vix.loc[overlap_start:overlap_end]
    vs_ol  = prod_vs.loc[overlap_start:overlap_end]

    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=False)
    s, e = overlap_start, overlap_end

    # ── Row 1: VP time series ─────────────────────────────────────────────────
    ax = axes[0]
    _label_crises(ax, s, e)
    ax.plot(vix_ol.index, vix_ol["VP"], color="steelblue",
            lw=0.7, alpha=0.85, label="VRP (VIX²/12)")
    ax.plot(vs_ol.index,  vs_ol["VP"],  color="darkorange",
            lw=0.7, alpha=0.85, label="VRP (VS²/12)")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("VP = IVar - CV  (%² monthly)")
    ax.set_title(
        f"Variance Risk Premium: VIX²/12 vs VS²/12  "
        f"[{overlap_start.date()} – {overlap_end.date()}]\n"
        f"VIX mean={vix_ol['VP'].mean():.2f}  VS mean={vs_ol['VP'].mean():.2f}  "
        f"Corr={vix_ol['VP'].corr(vs_ol['VP'].reindex(vix_ol.index)):.3f}"
    )
    ax.legend(fontsize=9)
    ax.set_ylim(-250, 400)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # ── Row 2: IVar time series ───────────────────────────────────────────────
    ax = axes[1]
    _label_crises(ax, s, e)
    ax.plot(vix_ol.index, vix_ol["IVar"], color="steelblue",
            lw=0.7, alpha=0.85, label="IVar (VIX²/12)")
    ax.plot(vs_ol.index,  vs_ol["IVar"],  color="darkorange",
            lw=0.7, alpha=0.85, label="IVar (VS²/12)")
    ax.set_ylabel("Implied Variance (%² monthly)")
    ax.set_title(
        f"Implied Variance: VIX²/12 vs VS²/12  "
        f"[VIX mean={vix_ol['IVar'].mean():.2f}  VS mean={vs_ol['IVar'].mean():.2f}  "
        f"Ratio={vix_ol['IVar'].mean()/vs_ol['IVar'].mean():.3f}]"
    )
    ax.legend(fontsize=9)
    ax.set_ylim(-10, 600)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # ── Row 3: Monthly MZ-R² from each production loop ───────────────────────
    ax = axes[2]
    diag_vix_ol = diag_vix[diag_vix.index >= overlap_start]
    diag_vs_ol  = diag_vs[diag_vs.index  >= overlap_start]

    ax.plot(diag_vix_ol.index, diag_vix_ol["mz_r2"], color="steelblue",
            lw=1.2, marker="o", markersize=2, label="MZ-R² (VIX-based HAR)")
    ax.plot(diag_vs_ol.index,  diag_vs_ol["mz_r2"],  color="darkorange",
            lw=1.2, marker="o", markersize=2, label="MZ-R² (VS-based HAR)")
    ax.axhline(PAPER_OOS["mz_r2"], color="firebrick", lw=1, ls="--",
               label=f"Paper OOS R²={PAPER_OOS['mz_r2']:.3f}")
    ax.axhline(0, color="black", lw=0.4)
    ax.set_ylabel("Monthly MZ-R²  (12-month trailing window)")
    ax.set_title(
        f"Forecast R² over time: VIX-based HAR vs VS-based HAR  "
        f"[mean VIX={diag_vix_ol['mz_r2'].mean():.3f}  "
        f"mean VS={diag_vs_ol['mz_r2'].mean():.3f}]"
    )
    ax.legend(fontsize=9)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    path = OUTPUT / "vrp_comparison_vix_vs.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return path


# ══════════════════════════════════════════════════════════════════════════════
# COMBINED SUMMARY PLOT
# ══════════════════════════════════════════════════════════════════════════════
def _trailing_oos_mz_r2(prod_df: pd.DataFrame, window: int = 252) -> pd.Series:
    """Trailing-window OOS Mincer-Zarnowitz R² from production loop forecasts."""
    rows    = []
    arr_act = prod_df["y_actual"].values
    arr_hat = prod_df["y_hat"].values
    dates   = prod_df.index
    for i in range(window, len(dates)):
        ya    = arr_act[i - window : i]
        yh    = arr_hat[i - window : i]
        valid = ~(np.isnan(ya) | np.isnan(yh))
        if valid.sum() < 20:
            continue
        mz = OLS(ya[valid], add_constant(yh[valid])).fit()
        rows.append({"date": dates[i], "oos_mz_r2": float(mz.rsquared)})
    if not rows:
        return pd.Series(dtype=float, name="oos_mz_r2")
    return pd.DataFrame(rows).set_index("date")["oos_mz_r2"]


def plot_combined_vrp_summary(prod_df: pd.DataFrame, stats_df: pd.DataFrame,
                               tag: str = "full",
                               window_label: str = "500-day rolling OLS") -> Path:
    """
    Single combined image (4 panels):
      Row 1: Predicted VRP with long-run mean ± 1σ labelled
      Row 2: IS Adj-R² and trailing-252d OOS MZ-R²
      Row 3: HAR betas over time
      Row 4: NW t-statistics for all variables
    window_label is embedded in titles (e.g. "500-day rolling OLS" or
    "expanding window OLS (initial train 2006-2012)").
    """
    vrp_mean = float(prod_df["VP"].mean())
    vrp_std  = float(prod_df["VP"].std())
    oos_r2   = _trailing_oos_mz_r2(prod_df, window=252)

    FEAT_COLS   = ["VIX2_lag", "RV22_lag", "RV5_lag", "RV1_lag"]
    FEAT_LABELS = ["VIX²/12 (α)", "RV(22) (β^m)", "RV(5) (β^w)", "RV(1) (β^d)"]
    FEAT_COLORS = ["steelblue", "darkorange", "green", "firebrick"]

    fig, axes = plt.subplots(
        4, 1, figsize=(16, 19),
        gridspec_kw={"height_ratios": [1.8, 1.2, 1.3, 1.3]},
    )
    s, e = prod_df.index[0], prod_df.index[-1]

    # ── Row 1: VRP time series ────────────────────────────────────────────────
    ax = axes[0]
    _label_crises(ax, s, e)
    ax.fill_between(prod_df.index, prod_df["VP"], 0,
                    where=(prod_df["VP"] >= 0), color="steelblue",
                    alpha=0.55, label="VRP > 0")
    ax.fill_between(prod_df.index, prod_df["VP"], 0,
                    where=(prod_df["VP"] < 0), color="salmon",
                    alpha=0.55, label="VRP < 0")
    ax.axhline(0, color="black", lw=0.5)
    ax.axhline(vrp_mean, color="navy", lw=1.6, ls="--",
               label=f"Long-run mean = {vrp_mean:.2f}")
    ax.axhline(vrp_mean + vrp_std, color="steelblue", lw=1.1, ls=":",
               label=f"+1σ  ({vrp_mean + vrp_std:.2f})")
    ax.axhline(vrp_mean - vrp_std, color="salmon",    lw=1.1, ls=":",
               label=f"−1σ  ({vrp_mean - vrp_std:.2f})")
    ax.set_ylabel("VRP = IVar − CV  (%² monthly)", fontsize=9)
    ax.set_title(
        f"Predicted Variance Risk Premium — {window_label}  "
        f"(VIX formulation, no VS)   [{s.date()} – {e.date()}]",
        fontsize=10,
    )
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.set_ylim(-280, 520)
    ax.xaxis.set_major_locator(mdates.YearLocator(4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # ── Row 2: IS Adj-R² and OOS MZ-R² ───────────────────────────────────────
    ax = axes[1]
    ax.plot(stats_df.index, stats_df["adj_r2"], color="steelblue", lw=0.8,
            alpha=0.9, label=f"IS Adj-R² ({window_label}, mean={stats_df['adj_r2'].mean():.3f})")
    if len(oos_r2) > 0:
        ax.plot(oos_r2.index, oos_r2, color="darkorange", lw=0.8, alpha=0.9,
                label=f"Trailing 252d OOS MZ-R² (mean={oos_r2.mean():.3f})")
    ax.axhline(0, color="black", lw=0.4)
    ax.set_ylabel("R²", fontsize=9)
    ax.set_title("IS Adj-R² and Trailing OOS MZ-R² Over Time", fontsize=9)
    ax.legend(fontsize=8, loc="upper right")
    ax.xaxis.set_major_locator(mdates.YearLocator(4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # ── Row 3: Beta coefficients ───────────────────────────────────────────────
    ax = axes[2]
    for feat, lbl, col in zip(FEAT_COLS, FEAT_LABELS, FEAT_COLORS):
        if feat in stats_df.columns:
            ax.plot(stats_df.index, stats_df[feat], color=col, lw=0.8,
                    alpha=0.9, label=lbl)
    if "const" in stats_df.columns:
        ax2 = ax.twinx()
        ax2.plot(stats_df.index, stats_df["const"], color="purple", lw=0.7,
                 alpha=0.7, ls="--", label="const (right)")
        ax2.set_ylabel("const", fontsize=8, color="purple")
        ax2.tick_params(axis="y", labelcolor="purple", labelsize=7)
        ax2.legend(fontsize=7, loc="lower right")
    ax.axhline(0, color="black", lw=0.4)
    ax.set_ylabel("Beta", fontsize=9)
    ax.set_title(f"HAR Beta Coefficients Over Time ({window_label})", fontsize=9)
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.xaxis.set_major_locator(mdates.YearLocator(4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # ── Row 4: t-statistics ───────────────────────────────────────────────────
    ax = axes[3]
    for feat, lbl, col in zip(FEAT_COLS, FEAT_LABELS, FEAT_COLORS):
        tcol = f"t_{feat}"
        if tcol in stats_df.columns:
            ax.plot(stats_df.index, stats_df[tcol], color=col, lw=0.8,
                    alpha=0.9, label=lbl)
    if "t_const" in stats_df.columns:
        ax.plot(stats_df.index, stats_df["t_const"], color="purple", lw=0.7,
                alpha=0.7, ls="--", label="const")
    ax.axhline( 1.96, color="grey", lw=0.9, ls="--", label="±1.96")
    ax.axhline(-1.96, color="grey", lw=0.9, ls="--")
    ax.axhline(0, color="black", lw=0.4)
    ax.set_ylabel("NW t-statistic", fontsize=9)
    ax.set_title("Newey-West t-Statistics Over Time (44 lags)", fontsize=9)
    ax.legend(fontsize=8, loc="upper right", ncol=3)
    ax.xaxis.set_major_locator(mdates.YearLocator(4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.suptitle(
        f"VRP Experiment — {window_label} (VIX only)",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    plt.subplots_adjust(top=0.94)
    path = OUTPUT / f"vrp_experiment_summary_{tag}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 72)
    print("  Bekaert & Hoerova (2014) — Full 8-Step Experiment")
    print("=" * 72)

    # ── Load & build panel ────────────────────────────────────────────────────
    print("\n[1-2] Building daily panel (returns + implied variance + RV)…")
    panel_full = build_full_panel()
    sp_ret     = load_sp500_returns()

    panel_paper = panel_full[
        (panel_full.index >= PAPER_START) &
        (panel_full.index <= PAPER_END)
    ].copy()

    print(f"    Paper panel: {panel_paper.shape[0]:,} obs  "
          f"({panel_paper.index.min().date()} – {panel_paper.index.max().date()})")
    print(f"    Full  panel: {panel_full.shape[0]:,} obs  "
          f"({panel_full.index.min().date()} – {panel_full.index.max().date()})")

    # ── STEP 3: HAR in-sample & OOS (via bh_replication.har_model) ───────────
    print("\n[3] HAR Model 8 — in-sample and OOS estimates…")
    res_paper = estimate_har(panel_paper, "1990–2010 paper")
    res_full  = estimate_har(panel_full,  "1990–present full")

    oos_paper = out_of_sample_forecast(panel_paper, PAPER_SPLIT, "Paper OOS")
    oos_full  = out_of_sample_forecast(panel_full,  PAPER_SPLIT, "Full  OOS")

    for tag, res, oos in [("Paper", res_paper, oos_paper),
                           ("Full",  res_full,  oos_full)]:
        print(f"\n  [{tag}] IS Adj-R²={res['adj_r2']:.4f}  "
              f"IS-RMSE={res['rmse_is']:.3f}  "
              f"OOS MZ-R²={oos['oos_mz_r2']:.4f}  "
              f"OOS RMSE={oos['oos_rmse']:.3f}")
        for v in ["const", "VIX2_lag", "RV22_lag", "RV5_lag", "RV1_lag"]:
            p  = res["params"][v]
            se = res["nw_se"][v]
            t  = res["t_stats"][v]
            pc = PAPER_COEFS.get(v, np.nan)
            print(f"    {v:<12} {p:>9.4f}  SE={se:.4f}  t={t:>7.3f}"
                  f"  [paper: {pc:.3f}]")

    # ── STEP 4: VP extraction ─────────────────────────────────────────────────
    print("\n[4] Extracting VP and CV series…")
    panel_paper = extract_vrp(panel_paper, res_paper["fitted"])
    panel_full  = extract_vrp(panel_full,  res_full["fitted"])

    print(f"    VP mean (paper): {panel_paper['VP'].mean():.3f}")
    print(f"    CV mean (paper): {panel_paper['CV'].mean():.3f}")
    print(f"    VP mean (full):  {panel_full['VP'].mean():.3f}")

    # ── STEP 5: Production loop ───────────────────────────────────────────────
    print("\n[5] Production loop (500-day rolling OLS)…")
    print("  Paper sample:")
    prod_paper = production_loop(panel_paper, ROLL_WIN)
    print("  Full sample (with IS stats for summary plot):")
    prod_full, stats_full = production_loop(panel_full, ROLL_WIN, return_stats=True)


    # ── STEP 7: Metrics ───────────────────────────────────────────────────────
    print("\n[7] Computing accuracy metrics…")
    mart_paper = martingale_forecast(panel_paper, PAPER_SPLIT)
    mart_full  = martingale_forecast(panel_full,  PAPER_SPLIT)

    y_te_p = oos_paper["y_test"]
    y_te_f = oos_full["y_test"]

    metrics_p_har  = compute_metrics(y_te_p.values, oos_paper["y_hat"].values,
                                     "HAR paper OOS")
    metrics_f_har  = compute_metrics(y_te_f.values, oos_full["y_hat"].values,
                                     "HAR full  OOS")
    metrics_p_mart = compute_metrics(
        y_te_p.values,
        mart_paper.reindex(y_te_p.index).values,
        "Martingale paper OOS")
    metrics_f_mart = compute_metrics(
        y_te_f.values,
        mart_full.reindex(y_te_f.index).values,
        "Martingale full  OOS")

    for m in [metrics_p_har, metrics_f_har, metrics_p_mart, metrics_f_mart]:
        print(f"    {m['label']:<28} RMSE={m['rmse']:.3f}  MAE={m['mae']:.3f}"
              f"  MAPE={m['mape']:.4f}  MZ-R²={m['mz_r2']:.4f}")


    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\n[9] Generating combined VRP summary plot (VIX only)…")
    plot_combined_vrp_summary(prod_full, stats_full, tag="full",
                              window_label="500-day Rolling OLS")

    # ── Expanding-window production loop (initial train 2006-2012) ──────────
    print(f"\n[EW] Expanding-window production loop "
          f"(anchor={EXP_TRAIN_START}, OOS from {EXP_OOS_START})…")
    prod_ew, stats_ew = production_loop_expanding(
        panel_full, train_start=EXP_TRAIN_START, oos_start=EXP_OOS_START,
        return_stats=True
    )
    print(f"    EW loop: {len(prod_ew):,} OOS steps  "
          f"({prod_ew.index.min().date()} – {prod_ew.index.max().date()})")
    print(f"    EW VRP mean={prod_ew['VP'].mean():.3f}  "
          f"EW OOS RMSE={np.sqrt((prod_ew['error']**2).mean()):.3f}")
    plot_combined_vrp_summary(
        prod_ew, stats_ew, tag="expanding",
        window_label="Expanding Window OLS (initial train 1990–2005)"
    )
    prod_ew.to_csv(OUTPUT / "production_loop_expanding.csv")

    # ── VS-based parallel run (pure VS²/12, ~2008 onwards) ───────────────────
    print("\n[VS] Building VS-based panel (pure VS²/12, no VIX fallback)…")
    panel_vs = build_panel_vs()
    print(f"    VS panel: {panel_vs.shape[0]:,} obs  "
          f"({panel_vs.index.min().date()} – {panel_vs.index.max().date()})")

    print("  Running VS production loop…")
    prod_vs = production_loop(panel_vs, ROLL_WIN)

    overlap_start = prod_vs.index.min()
    vix_ol = prod_full.loc[overlap_start:]
    vs_ol  = prod_vs
    print(f"    VS VRP mean={vs_ol['VP'].mean():.3f}  "
          f"VIX VRP mean (overlap)={vix_ol['VP'].mean():.3f}  "
          f"Corr={vix_ol['VP'].corr(vs_ol['VP'].reindex(vix_ol.index)):.4f}")

    print(f"\nAll outputs saved to {OUTPUT}/")
    print("=" * 72)


if __name__ == "__main__":
    main()
