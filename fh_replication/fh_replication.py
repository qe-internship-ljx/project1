"""
fh_replication.py
=================
Replication of the linear cross-sectional curve geometry model from:

  Fassas, A.P. & Hourvouliades, N. (2019).  "VIX Futures as a Market Timing
  Indicator."  Journal of International Financial Markets, Institutions and
  Money, 59, 21–36.  https://doi.org/10.1016/j.intfin.2018.11.001

The model (Eq. 1 in the paper):

    price_{i,t} = α_{0,t} + β_{0,t} · TtM_{i,t} + ε_{i,t}

where:
  price_{i,t}   = settlement price of VIX futures contract i on day t
  TtM_{i,t}     = time-to-maturity of contract i on day t (in years)
  α_{0,t}       = intercept; proxy for the short-end volatility level
  β_{0,t}       = slope; VIX points per year of TtM
                    > 0 → contango  (longer-dated contracts at premium)
                    < 0 → backwardation (longer-dated contracts at discount)

For each trading day the OLS is re-estimated using all available listed VIX
futures (typically 5–9 contracts).  Full regression diagnostics — including
R², t-statistics, and the number of contracts — are stored alongside the slope.

Usage
-----
  python fh_replication.py          run full replication, save all outputs
  from fh_replication.fh_replication import compute_vix_term_slope
                                    import the slope function for external use

Outputs (saved to ./output/)
------------------------------
  fh_daily_results.csv              daily α, β, R², t-stats, n_contracts
  fh_main_results.png               3-panel: β ± 2SE, R², t-stat(β) over time
  fh_sample_fits.png                cross-sectional fits at four key dates
  fh_summary_stats.png              slope distribution and regime summary
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
from scipy import stats as scipy_stats
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

ROOT   = Path(__file__).parent
OUTPUT = ROOT / "output"
OUTPUT.mkdir(exist_ok=True)
DATA   = ROOT.parent / "data"

# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

def load_vix_futures_term_structure(data_dir: Path = DATA) -> pd.DataFrame:
    """
    Load VIX futures (VX curve_group) with time-to-maturity in years.

    Returns DataFrame with columns [date, security, price, ttm_years].
    Negative / zero TtM rows (expired contracts) are dropped.
    """
    sec_meta = pd.read_parquet(data_dir / "VolatilityIndexFuture_security_meta.parquet")
    hist     = pd.read_parquet(data_dir / "VolatilityIndexFuture_historical.parquet")

    vx_secs = sec_meta[sec_meta["curve_group"] == "VX"][
        ["security", "last_trade_date"]].copy()
    vx_secs["last_trade_date"] = pd.to_datetime(vx_secs["last_trade_date"])

    vx_hist = hist[hist["security"].isin(set(vx_secs["security"]))].copy()
    vx_hist["date"] = pd.to_datetime(vx_hist["date"])

    vx = vx_hist.merge(vx_secs, on="security")
    vx["ttm_years"] = (vx["last_trade_date"] - vx["date"]).dt.days / 365.25
    vx = vx[vx["ttm_years"] > 0].dropna(subset=["price", "ttm_years"])
    return vx[["date", "security", "price", "ttm_years"]].sort_values("date")


# ═══════════════════════════════════════════════════════════════════════════
# CORE MODEL — daily cross-sectional OLS
# ═══════════════════════════════════════════════════════════════════════════

def fit_daily_cross_section(vx_df: pd.DataFrame,
                             min_contracts: int = 3) -> pd.DataFrame:
    """
    Fassas & Hourvouliades (2019) Eq. (1):
        price_{i,t} = α_{0,t} + β_{0,t} · TtM_{i,t} + ε_{i,t}

    For each trading day t, fit a cross-sectional OLS across all available
    listed VIX futures and record the full regression diagnostics.

    Parameters
    ----------
    vx_df          : DataFrame with columns [date, price, ttm_years]
    min_contracts  : Minimum contracts required per day (default 3)

    Returns
    -------
    DataFrame indexed by date with columns:
        alpha        : intercept α_{0,t}  — short-end vol proxy
        beta         : slope β_{0,t}      — curve geometry (VIX pts / yr)
        se_alpha     : OLS SE of alpha
        se_beta      : OLS SE of beta
        t_alpha      : t-statistic of alpha
        t_beta       : t-statistic of beta (key output for significance)
        r2           : R² of the cross-sectional fit
        r2_adj       : Adjusted R²
        n_contracts  : number of contracts in the cross-section
        residual_std : residual standard deviation
    """
    records = {}
    for date, grp in vx_df.groupby("date"):
        grp = grp.dropna(subset=["price", "ttm_years"])
        if len(grp) < min_contracts:
            continue
        y = grp["price"].values
        X = add_constant(grp["ttm_years"].values)
        try:
            res = OLS(y, X).fit()
            records[date] = {
                "alpha":        float(res.params[0]),
                "beta":         float(res.params[1]),
                "se_alpha":     float(res.bse[0]),
                "se_beta":      float(res.bse[1]),
                "t_alpha":      float(res.tvalues[0]),
                "t_beta":       float(res.tvalues[1]),
                "r2":           float(res.rsquared),
                "r2_adj":       float(res.rsquared_adj),
                "n_contracts":  int(len(grp)),
                "residual_std": float(np.sqrt(res.mse_resid)) if res.mse_resid > 0 else np.nan,
            }
        except Exception:
            continue

    df = pd.DataFrame.from_dict(records, orient="index")
    df.index.name = "date"
    return df.sort_index()


def compute_vix_term_slope(vx_df: pd.DataFrame,
                            min_contracts: int = 3) -> pd.Series:
    """
    Return the daily VIX term structure slope β_{0,t} from the
    Fassas & Hourvouliades (2019) cross-sectional model.

    Positive value = contango; negative = backwardation.
    This is a thin wrapper around fit_daily_cross_section() for use in
    downstream modules (e.g., experiment2).

    Parameters
    ----------
    vx_df         : DataFrame with columns [date, price, ttm_years]
    min_contracts : Minimum contracts per day (default 3)

    Returns
    -------
    pd.Series named 'term_slope', indexed by date
    """
    results = fit_daily_cross_section(vx_df, min_contracts=min_contracts)
    return results["beta"].rename("term_slope")


# ═══════════════════════════════════════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════════════════════════════════════

def _shade_crises(ax, start, end):
    crises = [
        ("GFC",   "2007-06-01", "2009-06-01"),
        ("COVID", "2020-02-01", "2020-06-01"),
        ("2022",  "2022-01-01", "2022-12-31"),
    ]
    for lbl, s, e in crises:
        s_ts, e_ts = pd.Timestamp(s), pd.Timestamp(e)
        if e_ts < start or s_ts > end:
            continue
        ax.axvspan(max(s_ts, start), min(e_ts, end), alpha=0.08, color="grey", lw=0)
        mid = max(s_ts, start) + (min(e_ts, end) - max(s_ts, start)) / 2
        ax.text(mid, ax.get_ylim()[1] * 0.97, lbl,
                ha="center", va="top", fontsize=6.5, color="grey", style="italic")


def plot_main_results(results: pd.DataFrame, out_path: Path) -> None:
    """
    Three-panel time series:
      Panel 1: β (slope) ± 2 OLS SE bands
      Panel 2: R² of the cross-sectional fit
      Panel 3: t-statistic of β with significance thresholds
    """
    s, e = results.index[0], results.index[-1]

    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    fig.suptitle(
        "Fassas & Hourvouliades (2019) — Daily Cross-Sectional VIX Curve Geometry\n"
        r"$\mathrm{price}_{i,t} = \alpha_{0,t} + \beta_{0,t} \cdot \mathrm{TtM}_{i,t} + \varepsilon_{i,t}$"
        "   [all available VIX futures, OLS per day]",
        fontsize=11,
    )

    # ── Panel 1: Beta ────────────────────────────────────────────────────────
    ax = axes[0]
    _shade_crises(ax, s, e)
    beta = results["beta"]
    se   = results["se_beta"]
    ax.fill_between(beta.index, beta - 2*se, beta + 2*se,
                    color="steelblue", alpha=0.18, label="± 2 OLS SE")
    ax.plot(beta.index, beta.values, color="steelblue", lw=0.8, label="β (slope)")
    ax.axhline(0, color="black", lw=0.6, ls="--")
    ax.set_ylabel("β (VIX pts / yr)", fontsize=9)
    ax.set_title("Slope β₀,t — contango (+) vs backwardation (−)", fontsize=9, loc="left")

    pct_contango = (beta > 0).mean() * 100
    ax.text(0.01, 0.95,
            f"Mean: {beta.mean():.2f}  Median: {beta.median():.2f}  "
            f"Contango: {pct_contango:.1f}% of days",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(fc="white", ec="#cccccc", lw=0.6, pad=3))
    ax.legend(fontsize=8, loc="upper right")

    # ── Panel 2: R² ─────────────────────────────────────────────────────────
    ax = axes[1]
    _shade_crises(ax, s, e)
    r2 = results["r2"]
    ax.fill_between(r2.index, r2.values, 0, color="darkorange", alpha=0.4, label="R²")
    ax.plot(r2.index, r2.values, color="darkorange", lw=0.6)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("R²", fontsize=9)
    ax.set_title("R² of Cross-Sectional Fit", fontsize=9, loc="left")
    ax.text(0.01, 0.95,
            f"Mean R²: {r2.mean():.3f}  Median R²: {r2.median():.3f}  "
            f"% days R² > 0.90: {(r2 > 0.90).mean()*100:.1f}%",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(fc="white", ec="#cccccc", lw=0.6, pad=3))
    ax.legend(fontsize=8, loc="upper right")

    # ── Panel 3: t-stat of β ────────────────────────────────────────────────
    ax = axes[2]
    _shade_crises(ax, s, e)
    t_beta = results["t_beta"]
    ax.plot(t_beta.index, t_beta.values, color="purple", lw=0.7, label="t(β)")
    ax.axhline( 1.96, color="green",     lw=0.9, ls="--", label="±1.96 (5%)")
    ax.axhline(-1.96, color="green",     lw=0.9, ls="--")
    ax.axhline( 2.576, color="firebrick", lw=0.9, ls=":",  label="±2.576 (1%)")
    ax.axhline(-2.576, color="firebrick", lw=0.9, ls=":")
    ax.axhline(0, color="black", lw=0.5)
    ax.fill_between(t_beta.index, -1.96, 1.96, color="firebrick", alpha=0.04,
                    label="Insignificant band")
    ax.set_ylabel("t-stat(β)", fontsize=9)
    ax.set_title("t-statistic of Slope β₀,t", fontsize=9, loc="left")
    pct_sig = (t_beta.abs() > 1.96).mean() * 100
    ax.text(0.01, 0.95,
            f"Mean |t|: {t_beta.abs().mean():.2f}  "
            f"% days |t| > 1.96: {pct_sig:.1f}%",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(fc="white", ec="#cccccc", lw=0.6, pad=3))
    ax.legend(fontsize=8, loc="upper right", ncol=2)

    axes[2].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_sample_fits(vx_df: pd.DataFrame, results: pd.DataFrame,
                     out_path: Path) -> None:
    """
    Four-panel scatter of the cross-sectional fit on selected dates showing
    the VIX futures price curve (price vs. TtM) with the fitted OLS line.
    """
    sample_dates = [
        pd.Timestamp("2008-10-15"),   # peak GFC
        pd.Timestamp("2014-06-02"),   # low-vol era
        pd.Timestamp("2020-03-16"),   # COVID spike
        pd.Timestamp("2022-06-13"),   # 2022 drawdown
    ]
    # Snap to nearest available date in results index
    avail = results.index
    sample_dates = [avail[avail.get_indexer([d], method="nearest")[0]] for d in sample_dates]

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(
        "Fassas & Hourvouliades (2019) — Cross-Sectional Fits at Selected Dates\n"
        "Each point = one VIX futures contract.  Line = fitted OLS.",
        fontsize=11,
    )
    axes = axes.flatten()

    for ax, date in zip(axes, sample_dates):
        grp = vx_df[vx_df["date"] == date].sort_values("ttm_years")
        if len(grp) == 0:
            ax.set_title(f"{date.date()} — no data", fontsize=9)
            continue

        ax.scatter(grp["ttm_years"], grp["price"], color="steelblue",
                   s=50, zorder=5, label="VIX futures")

        row = results.loc[date]
        x_line = np.linspace(0, grp["ttm_years"].max() * 1.05, 100)
        y_line = row["alpha"] + row["beta"] * x_line
        ax.plot(x_line, y_line, color="firebrick", lw=1.8, label="OLS fit")

        label = (f"α = {row['alpha']:.2f}   β = {row['beta']:.2f}\n"
                 f"t(β) = {row['t_beta']:.2f}   R² = {row['r2']:.3f}\n"
                 f"n = {int(row['n_contracts'])} contracts")
        ax.text(0.04, 0.96, label, transform=ax.transAxes,
                fontsize=8.5, va="top",
                bbox=dict(fc="white", ec="#cccccc", lw=0.6, pad=3))

        ax.set_xlabel("TtM (years)", fontsize=8)
        ax.set_ylabel("Price (VIX points)", fontsize=8)
        ax.set_title(date.strftime("%Y-%m-%d"), fontsize=9)
        ax.legend(fontsize=8)
        ax.set_xlim(left=0)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_summary_stats(results: pd.DataFrame, out_path: Path) -> None:
    """
    Two-panel: (left) histogram of daily β slopes, (right) regime timeline.
    """
    beta = results["beta"]
    r2   = results["r2"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Fassas & Hourvouliades (2019) — Summary Statistics",
        fontsize=11,
    )

    # ── Left: slope histogram ─────────────────────────────────────────────
    ax = axes[0]
    ax.hist(beta.values, bins=80, color="steelblue", alpha=0.7,
            edgecolor="white", lw=0.3, density=True)

    # Overlay KDE
    x_grid = np.linspace(beta.min() - 1, beta.max() + 1, 300)
    kde = scipy_stats.gaussian_kde(beta.dropna())(x_grid)
    ax.plot(x_grid, kde, color="navy", lw=1.8, label="KDE")

    ax.axvline(0,             color="firebrick", lw=1.4, ls="--", label="β = 0")
    ax.axvline(beta.mean(),   color="darkorange", lw=1.2, ls="-",
               label=f"Mean = {beta.mean():.2f}")
    ax.axvline(beta.median(), color="green", lw=1.2, ls=":",
               label=f"Median = {beta.median():.2f}")
    ax.set_xlabel("β (slope, VIX pts/yr)", fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.set_title("Distribution of Daily Slope β₀,t", fontsize=9)
    ax.legend(fontsize=8)

    # ── Right: R² distribution ────────────────────────────────────────────
    ax2 = axes[1]
    ax2.hist(r2.values, bins=60, color="darkorange", alpha=0.7,
             edgecolor="white", lw=0.3, density=True)
    x_grid2 = np.linspace(0, 1, 200)
    kde2 = scipy_stats.gaussian_kde(r2.dropna())(x_grid2)
    ax2.plot(x_grid2, kde2, color="saddlebrown", lw=1.8, label="KDE")
    ax2.axvline(r2.mean(), color="navy", lw=1.2, ls="-",
                label=f"Mean R² = {r2.mean():.3f}")
    ax2.axvline(0.90, color="firebrick", lw=1.2, ls="--", label="R² = 0.90")
    ax2.set_xlabel("R²", fontsize=9)
    ax2.set_ylabel("Density", fontsize=9)
    ax2.set_title("Distribution of Daily R²", fontsize=9)
    ax2.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def run_replication(data_dir: Path = DATA, output_dir: Path = OUTPUT) -> pd.DataFrame:
    """
    Run the full FH replication pipeline:
      1. Load VIX futures data
      2. Fit daily cross-sectional OLS
      3. Save results CSV
      4. Generate and save plots
      5. Return the results DataFrame

    Parameters
    ----------
    data_dir   : Path to the data directory (default: ../data/)
    output_dir : Path to save outputs (default: ./output/)
    """
    output_dir.mkdir(exist_ok=True)

    print("=" * 68)
    print("  Fassas & Hourvouliades (2019) -- Cross-Sectional VIX Curve Model")
    print("=" * 68)

    # ── Load data ─────────────────────────────────────────────────────────
    print("\n[1] Loading VIX futures data...")
    vx_df = load_vix_futures_term_structure(data_dir)
    n_dates = vx_df["date"].nunique()
    print(f"    Date range : {vx_df['date'].min().date()} – {vx_df['date'].max().date()}")
    print(f"    Total rows : {len(vx_df):,}  ({n_dates:,} unique trading days)")
    print(f"    Contracts  : {vx_df['security'].nunique()} unique VIX futures")
    avg_n = vx_df.groupby("date").size().mean()
    print(f"    Avg contracts/day : {avg_n:.1f}")

    # ── Fit model ─────────────────────────────────────────────────────────
    print("\n[2] Fitting daily cross-sectional OLS (FH Eq. 1)...")
    results = fit_daily_cross_section(vx_df, min_contracts=3)
    print(f"    Days fitted: {len(results):,}")

    beta = results["beta"]
    r2   = results["r2"]
    t_b  = results["t_beta"]

    print(f"\n    -- Slope beta_t ----------------------------------------")
    print(f"    Mean:    {beta.mean():+.3f} VIX pts/yr")
    print(f"    Median:  {beta.median():+.3f} VIX pts/yr")
    print(f"    Std dev: {beta.std():.3f} VIX pts/yr")
    print(f"    Min:     {beta.min():+.3f}   Max: {beta.max():+.3f}")
    print(f"    % days contango (beta > 0):      {(beta > 0).mean()*100:.1f}%")
    print(f"    % days backwardation (beta < 0): {(beta < 0).mean()*100:.1f}%")

    print(f"\n    -- R-squared -------------------------------------------")
    print(f"    Mean R2:           {r2.mean():.4f}")
    print(f"    Median R2:         {r2.median():.4f}")
    print(f"    % days R2 > 0.90:  {(r2 > 0.90).mean()*100:.1f}%")
    print(f"    % days R2 > 0.95:  {(r2 > 0.95).mean()*100:.1f}%")

    print(f"\n    -- t-statistic of beta ---------------------------------")
    print(f"    Mean |t(beta)|:         {t_b.abs().mean():.2f}")
    print(f"    % days |t| > 1.96:      {(t_b.abs() > 1.96).mean()*100:.1f}%")
    print(f"    % days |t| > 2.576:     {(t_b.abs() > 2.576).mean()*100:.1f}%")

    # ── Save CSV ──────────────────────────────────────────────────────────
    print("\n[3] Saving results CSV...")
    csv_path = output_dir / "fh_daily_results.csv"
    results.to_csv(csv_path)
    print(f"    Saved: {csv_path.name}  ({len(results):,} rows)")

    # ── Generate plots ────────────────────────────────────────────────────
    print("\n[4] Generating plots...")
    plot_main_results(results, output_dir / "fh_main_results.png")
    plot_sample_fits(vx_df, results, output_dir / "fh_sample_fits.png")
    plot_summary_stats(results, output_dir / "fh_summary_stats.png")

    print(f"\nAll outputs saved to {output_dir}/")
    print("=" * 68)

    return results


if __name__ == "__main__":
    run_replication()
