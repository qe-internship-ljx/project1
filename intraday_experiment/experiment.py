"""
intraday_experiment/experiment.py
==================================
VRP Experiment — Intraday-Based Realized Variance

Key differences from vrp_experiment/experiment.py:
  - RV is computed from intraday 5-minute E-mini S&P 500 (ES) futures returns
    (sum of squared 5-min log returns per day) instead of squared daily returns
  - Intraday data is only available from 2016-01-04 onwards
  - Identical pipeline otherwise: VIX²/12 implied variance, 500-day rolling
    CORSI HAR (with VIX²) regression, 22-day training-prediction gap
  - VIX data read directly from data/VolatilityIndexData.csv

Steps
-----
1.  Load intraday ES data → front-month selection → daily RV per day
2.  Compute multi-frequency RV aggregates (RV1, RV5, RV22) in monthly %² units
3.  Implied Variance: IVar = VIX²/12  (monthly %²-units)
4.  Build panel: join RV aggregates with IVar, create lag features
5.  500-day rolling-window OLS production loop (22-day gap, no look-ahead)
6.  VRP = IVar − CV  (implied variance minus fitted conditional variance)
7.  Plot: vrp_experiment_summary_intraday.png  (same 4-panel layout as
         vrp_experiment_summary_full.png)

Outputs (all in ./output/)
--------------------------
  vrp_experiment_summary_intraday.png
  production_loop_intraday.csv
"""

import warnings
warnings.filterwarnings("ignore")

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

ROOT   = Path(__file__).parent
OUTPUT = ROOT / "output"
OUTPUT.mkdir(exist_ok=True)
DATA   = ROOT.parent / "data"

ROLL_WIN = 500  # rolling window in trading days
NW_LAGS  = 44   # Newey-West lags (matching vrp_experiment)


def _nw_se(res, nlags: int = NW_LAGS) -> np.ndarray:
    cov = cov_hac(res, nlags=nlags)
    return np.sqrt(np.diag(cov))


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1–2: Intraday RV from 5-min ES futures bins
# ══════════════════════════════════════════════════════════════════════════════
def load_intraday_rv() -> pd.DataFrame:
    """
    Compute daily realized variance from 5-minute intraday ES futures.

    Method: for each date, select the front-month ES contract (nearest expiry),
    compute log returns across the 79 regular-session bins (08:30–15:00),
    and sum squared returns to get daily RV.

    Output columns (monthly %²-units, same convention as bh_replication):
        RV1   daily RV × 22          (annualised-to-monthly scaling)
        RV5   5-day rolling avg × 22
        RV22  22-day rolling sum
    """
    print("  Loading intraday ES data…", flush=True)
    df = pd.read_csv(DATA / "es_intraday_sorted.csv")
    es = df[df["QCODE"] == "ES"].copy()
    es["date"] = pd.to_datetime(es["PUBLICATION_DATE"])

    # Expiry dates for front-month selection
    meta    = pd.read_parquet(DATA / "EquityFuture_security_meta.parquet")
    es_meta = meta[meta["curve_group"] == "ES"][["security", "expiry_yearmonth"]].copy()
    es_meta["expiry_date"] = pd.to_datetime(es_meta["expiry_yearmonth"], format="%Y-%m")

    es = es.merge(
        es_meta[["security", "expiry_date"]],
        left_on="SECURITY", right_on="security", how="left",
    )

    # Keep only front-month contract bins on each date
    es = es.sort_values(["date", "expiry_date", "BIN_START_TIME"])
    es["_min_expiry"] = es.groupby("date")["expiry_date"].transform("min")
    es = es[es["expiry_date"] == es["_min_expiry"]].copy()
    es.drop(columns=["_min_expiry"], inplace=True)

    print(f"  Front-month bins: {len(es):,}  "
          f"({es['date'].min().date()} – {es['date'].max().date()})", flush=True)

    # Drop bins with missing CLOSE prices
    es = es.dropna(subset=["CLOSE"]).copy()

    # Within-day log returns: diff of log(CLOSE) grouped by date
    es = es.sort_values(["date", "BIN_START_TIME"])
    es["log_ret"] = (np.log(es["CLOSE"]).groupby(es["date"]).diff() * 100)

    # Daily intraday RV = sum of squared demeaned within-day log returns
    daily_rv = (
        es.dropna(subset=["log_ret"])
        .groupby("date")["log_ret"]
        .apply(lambda x: float((x ** 2).sum()))
    )
    daily_rv.index.name = "date"
    daily_rv.name = "RV_daily"

    # RV aggregates in monthly %²-units (same scaling as compute_rv_components)
    rv = pd.DataFrame(index=daily_rv.index)
    rv.index.name = "date"
    rv["RV1"]  = daily_rv * 22
    rv["RV5"]  = daily_rv.rolling(5).mean() * 22
    rv["RV22"] = daily_rv.rolling(22).sum()

    print(f"  Daily RV: mean={daily_rv.mean():.4f} %²  "
          f"| Monthly RV22: mean={rv['RV22'].dropna().mean():.4f} %²", flush=True)
    return rv


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: Implied Variance from VIX
# ══════════════════════════════════════════════════════════════════════════════
def load_vix_ivar() -> pd.Series:
    """IVar = VIX²/12  (monthly %²-units) from data/VolatilityIndexData.csv."""
    vix_df = pd.read_csv(DATA / "VolatilityIndexData.csv", parse_dates=["DATE"])
    vix = (
        vix_df[vix_df["SECURITY"] == "VIX Index"]
        .sort_values("DATE")
        .set_index("DATE")["INDEX_VALUE"]
    )
    vix.index.name = "date"
    ivar = vix ** 2 / 12.0
    ivar.name = "IVar"
    return ivar


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: Panel Assembly
# ══════════════════════════════════════════════════════════════════════════════
def build_panel() -> pd.DataFrame:
    """
    Join intraday RV components with VIX-based implied variance.
    Creates forward target (RV22_fwd) and lagged predictors.
    """
    rv   = load_intraday_rv()
    ivar = load_vix_ivar()

    panel = rv.join(ivar, how="inner").dropna()

    # Forward-looking target: RV22 at t+22 (same shift convention as vrp_experiment)
    panel["RV22_fwd"] = panel["RV22"].shift(-22)

    # Lagged predictors (strictly predetermined)
    panel["VIX2_lag"] = panel["IVar"].shift(1)
    panel["RV22_lag"] = panel["RV22"].shift(1)
    panel["RV5_lag"]  = panel["RV5"].shift(1)
    panel["RV1_lag"]  = panel["RV1"].shift(1)

    return panel.dropna()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: Production Loop (500-day rolling OLS, 22-day gap)
# ══════════════════════════════════════════════════════════════════════════════
def production_loop(panel: pd.DataFrame, window: int = ROLL_WIN):
    """
    For each day t, train on [t-window-22 : t-22], forecast RV_{t+22}.
    Returns (prod_df, stats_df) with no look-ahead.

    Columns in prod_df:  y_actual, y_hat, error, CV, IVar, VP
    Columns in stats_df: adj_r2, betas and NW t-stats per feature
    """
    rows       = []
    stats_rows = []
    idx        = panel.index
    N          = len(idx)
    feats      = ["VIX2_lag", "RV22_lag", "RV5_lag", "RV1_lag"]

    n_steps = N - window - 22 - 22
    print(f"  Running {n_steps} daily production steps "
          f"(window={window}, oos_gap=22)…", flush=True)

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
        y_actual = float(panel["RV22_fwd"].iloc[i])
        y_hat    = float(res_tr.predict(X_te).iloc[0])
        ivar_t   = float(panel["IVar"].iloc[i])

        rows.append({
            "date":     idx[i],
            "y_actual": y_actual,
            "y_hat":    y_hat,
            "error":    y_actual - y_hat,
            "CV":       y_hat,
            "IVar":     ivar_t,
            "VP":       ivar_t - y_hat,
        })

        col_names = X_tr.columns.tolist()
        try:
            nw_ses = _nw_se(res_tr, nlags=NW_LAGS)
        except Exception:
            nw_ses = np.full(len(col_names), np.nan)
        stat_row = {"date": idx[i], "adj_r2": float(res_tr.rsquared_adj)}
        for j, col in enumerate(col_names):
            stat_row[col] = float(res_tr.params[col])
            nw_j = nw_ses[j]
            stat_row[f"t_{col}"] = (
                float(res_tr.params[col] / nw_j)
                if (np.isfinite(nw_j) and nw_j != 0) else np.nan
            )
        stats_rows.append(stat_row)

    prod_df  = pd.DataFrame(rows).set_index("date")
    stats_df = pd.DataFrame(stats_rows).set_index("date")
    return prod_df, stats_df


# ══════════════════════════════════════════════════════════════════════════════
# PLOTS — identical layout to vrp_experiment_summary_full.png
# ══════════════════════════════════════════════════════════════════════════════
def _label_crises(ax, start, end):
    crises = [
        ("GFC",    "2007-06-01", "2009-06-01"),
        ("COVID",  "2020-02-01", "2020-06-01"),
        ("Hiking", "2022-01-01", "2023-01-01"),
    ]
    for lbl, s, e in crises:
        s, e = pd.Timestamp(s), pd.Timestamp(e)
        if e < start or s > end:
            continue
        ax.axvspan(max(s, start), min(e, end), alpha=0.10,
                   color="grey", linewidth=0)


def _trailing_oos_mz_r2(prod_df: pd.DataFrame, window: int = 252) -> pd.Series:
    """Trailing-window OOS Mincer-Zarnowitz R²."""
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
                               tag: str = "intraday") -> Path:
    """
    Four-panel summary (identical layout to vrp_experiment_summary_full.png):
      Row 1: VRP time series with long-run mean ± 1σ bands
      Row 2: IS Adj-R² (500d window) and trailing-252d OOS MZ-R²
      Row 3: Rolling HAR beta coefficients
      Row 4: Newey-West t-statistics
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
    # Use 2-year tick spacing (shorter period than the full vrp_experiment)
    span_years = (e - s).days / 365.25
    year_step  = 4 if span_years > 15 else 2

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
    ax.axhline(vrp_mean - vrp_std, color="salmon", lw=1.1, ls=":",
               label=f"−1σ  ({vrp_mean - vrp_std:.2f})")
    ax.set_ylabel("VRP = IVar − CV  (%² monthly)", fontsize=9)
    ax.set_title(
        f"Predicted Variance Risk Premium — {ROLL_WIN}-day rolling OLS  "
        f"(Intraday 5-min RV, VIX²/12 IVar)   [{s.date()} – {e.date()}]",
        fontsize=10,
    )
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.set_ylim(-280, 520)
    ax.xaxis.set_major_locator(mdates.YearLocator(year_step))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # ── Row 2: IS Adj-R² and trailing OOS MZ-R² ──────────────────────────────
    ax = axes[1]
    ax.plot(stats_df.index, stats_df["adj_r2"], color="steelblue", lw=0.8,
            alpha=0.9, label=f"IS Adj-R² (500d, mean={stats_df['adj_r2'].mean():.3f})")
    if len(oos_r2) > 0:
        ax.plot(oos_r2.index, oos_r2, color="darkorange", lw=0.8, alpha=0.9,
                label=f"Trailing 252d OOS MZ-R² (mean={oos_r2.mean():.3f})")
    ax.axhline(0, color="black", lw=0.4)
    ax.set_ylabel("R²", fontsize=9)
    ax.set_title("IS Adj-R² and Trailing OOS MZ-R² Over Time", fontsize=9)
    ax.legend(fontsize=8, loc="upper right")
    ax.xaxis.set_major_locator(mdates.YearLocator(year_step))
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
    ax.set_title(
        f"HAR Beta Coefficients Over Time ({ROLL_WIN}-day rolling window)", fontsize=9)
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.xaxis.set_major_locator(mdates.YearLocator(year_step))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # ── Row 4: NW t-statistics ────────────────────────────────────────────────
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
    ax.set_title(f"Newey-West t-Statistics Over Time ({NW_LAGS} lags)", fontsize=9)
    ax.legend(fontsize=8, loc="upper right", ncol=3)
    ax.xaxis.set_major_locator(mdates.YearLocator(year_step))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.suptitle(
        "VRP Experiment (Intraday 5-min RV) — 500-day Rolling Production Loop",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    plt.subplots_adjust(top=0.94)

    path = OUTPUT / f"vrp_experiment_summary_{tag}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}", flush=True)
    return path


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 72)
    print("  VRP Experiment — Intraday 5-min Realized Variance")
    print("=" * 72)

    print("\n[1–4] Building panel (intraday RV + VIX IVar)…")
    panel = build_panel()
    print(f"  Panel: {panel.shape[0]:,} obs  "
          f"({panel.index.min().date()} – {panel.index.max().date()})")

    print("\n[5] Production loop (500-day rolling OLS, 22-day gap)…")
    prod_df, stats_df = production_loop(panel, ROLL_WIN)
    print(f"  Steps: {len(prod_df):,}  "
          f"({prod_df.index.min().date()} – {prod_df.index.max().date()})")
    print(f"  VRP — mean={prod_df['VP'].mean():.3f}  "
          f"std={prod_df['VP'].std():.3f}  "
          f"% > 0: {(prod_df['VP'] > 0).mean() * 100:.1f}%")

    # Save production loop CSV
    csv_path = OUTPUT / "production_loop_intraday.csv"
    prod_df.to_csv(csv_path)
    print(f"  Saved {csv_path}")

    print("\n[6] Generating summary plot…")
    plot_combined_vrp_summary(prod_df, stats_df, tag="intraday")

    print(f"\nAll outputs saved to {OUTPUT}/")
    print("=" * 72)


if __name__ == "__main__":
    main()
