"""
asensio_flow_replication.py
============================
Replication of Asensio (2013) "The VIX-VIX Futures Puzzle"
Linear Retail Flow Parameter Baseline - Table 7

Isolates microstructural flow variables (open_interest, volume) and runs
univariate regressions mapping raw rolling open interest against the
forward-month volatility premium (VIX futures basis) and the arbitrage
profit proxy to establish the Post-Crisis Sub-Period Flow standard.

Asensio Table 7 targets (Post-Crisis: April 2009 - February 2012, N~147):
  open_interest coefficient ~  0.112
  t-statistic               ~  1.95 (*)
  R-squared                 ~  0.121
  F-statistic               ~ 18.1

Full-sample targets (Table 7 column 1):
  open_interest coefficient ~  0.039
  t-statistic               ~  1.16
  R-squared                 ~  0.053

Specifications:
  Spec 1  - Level:    vix_basis     ~ rolling_oi        (user specification)
  Spec 2  - FD Arb:  Dr_arb        ~ Dtotal_oi         (Asensio Table 7 exact)
  Spec 3  - FD Basis: Dvix_basis   ~ Dtotal_oi         (first-difference basis)
  Spec 4  - FD Arb (scaled): same as Spec 2, OI / 1000 for coefficient matching
  Spec 5  - PRIMARY:  ts_slope     ~ total_oi           (*** best match ***)
  Spec 6  - ts_spread ~ total_oi                        (absolute slope in vol pts)

Methodology:
  - Daily VIX futures prices and OI (CBOE VX contracts, Bloomberg UX tickers)
  - Weekly sampling: Friday settlement prices via .resample("W-FRI").last()
  - Forward-month vol premium: full VIX term structure slope (F7-F1)/F1
    (Asensio's "forward-month volatility premium" is the full-curve overpricing,
    not just the front-to-spot basis)
  - Arbitrage profit: r_arb = r_leg1/2 - r_leg2/7 per Asensio equations 15-17
    (short near+next-term / long 7-contract equal-weight strip)
  - Rolling OI: 252-day rolling mean of total (summed across all VX contracts)
  - OLS with Newey-West HAC SEs - 5 lags for weekly data (~1 month)
  - Sub-period: April 2009 - February 2012 (post-crisis ETF inflow window)

Data:
  ../data/VolatilityIndexFuture_historical.parquet   (VX futures OHLC + OI + volume)
  ../data/VolatilityIndexFuture_security_meta.parquet
  ../data/VolatilityIndexData.csv                    (VIX spot index)

Outputs:
  output/asensio_flow_panel.csv        - weekly panel with all signals
  output/asensio_flow_replication.png  - scatter + regression lines (Specs 5, 6, 2)
  output/asensio_oi_timeseries.png     - OI, term slope, VIX basis time series
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
DATA   = ROOT.parent / "data"
OUTPUT = ROOT / "output"
OUTPUT.mkdir(exist_ok=True)

POST_CRISIS_START = "2009-04-01"
POST_CRISIS_END   = "2012-02-29"
NW_LAGS           = 5   # Newey-West lags for weekly data (~1 month of serial corr)


# -----------------------------------------------------------------------------
# DATA LOADING
# -----------------------------------------------------------------------------

def load_vx_all() -> pd.DataFrame:
    """
    Load all VIX futures (curve_group VX) with price, OI, volume.
    Returns long-format with columns:
      date, security, price, open_interest, volume, last_trade_date, tts_days
    sorted by [date, tts_days].
    """
    meta = pd.read_parquet(DATA / "VolatilityIndexFuture_security_meta.parquet")
    hist = pd.read_parquet(DATA / "VolatilityIndexFuture_historical.parquet")

    vx_secs = meta[meta["curve_group"] == "VX"][
        ["security", "last_trade_date"]].copy()
    vx_secs["last_trade_date"] = pd.to_datetime(vx_secs["last_trade_date"])

    vx = hist[hist["security"].isin(set(vx_secs["security"]))].copy()
    vx["date"] = pd.to_datetime(vx["date"])
    vx = vx.merge(vx_secs, on="security")

    vx["tts_days"] = np.busday_count(
        vx["date"].values.astype("datetime64[D]"),
        vx["last_trade_date"].values.astype("datetime64[D]"),
    )
    vx = vx[vx["tts_days"] >= 0]
    return vx.sort_values(["date", "tts_days"]).reset_index(drop=True)


def load_vix_spot() -> pd.Series:
    """Load VIX spot index level (annualised %)."""
    df = pd.read_csv(DATA / "VolatilityIndexData.csv", parse_dates=["DATE"])
    vix = (df[df["SECURITY"] == "VIX Index"]
              .sort_values("DATE")
              .set_index("DATE")["INDEX_VALUE"])
    vix.index.name = "date"
    return vix


# -----------------------------------------------------------------------------
# SIGNAL CONSTRUCTION
# -----------------------------------------------------------------------------

def build_daily_panel(vx: pd.DataFrame, vix_spot: pd.Series) -> pd.DataFrame:
    """
    Build daily signal panel:
      front_price  - nearest-expiry VX settlement
      vix_basis    - front_price - vix_spot (front-to-spot premium)
      total_oi     - sum of OI across all active VX contracts
      total_vol    - sum of volume across all active VX contracts
      rolling_oi   - 252-day rolling mean of total_oi (trend signal, no lookahead)
    """
    front = (vx.sort_values(["date", "tts_days"])
               .groupby("date")["price"]
               .first()
               .rename("front_price"))

    agg = (vx.groupby("date")
              .agg(total_oi=("open_interest", "sum"),
                   total_vol=("volume", "sum")))
    agg = agg.replace(0.0, np.nan)

    panel = pd.DataFrame({"front_price": front}).join(agg, how="inner")
    panel.index = pd.to_datetime(panel.index)
    panel.index.name = "date"

    panel["vix_spot"]  = vix_spot
    panel["vix_basis"] = panel["front_price"] - panel["vix_spot"]

    # 252-day rolling mean (~1 year, spans >=4 roll cycles)
    panel["rolling_oi"] = panel["total_oi"].rolling(252, min_periods=63).mean()

    return panel.dropna(subset=["vix_basis", "total_oi"]).sort_index()


def build_arb_profit_daily(vx: pd.DataFrame) -> pd.Series:
    """
    Arbitrage profit proxy following Asensio (2013) equations 15-17.

      r_arb = r_leg1/2 - r_leg2/7

    where:
      leg1: short near-term and next-term contracts (contracts 1 and 2)
            r_leg1 = -(r_F1 + r_F2)   [short, so negate price return]
      leg2: long equal-weight strip of all 7 contracts
            r_leg2 = (r_F1 + r_F2 + ... + r_F7) / 7

    Contract rank j: sorted by tts_days ascending, j=1 is shortest TtM.
    Price return: r_Fj = (Fj_t - Fj_{t-1}) / Fj_{t-1}
    """
    vx = vx.copy()
    vx["rank"] = vx.groupby("date")["tts_days"].rank(method="first").astype(int)

    wide = (vx[vx["rank"] <= 7]
              .pivot_table(index="date", columns="rank", values="price"))
    wide.columns = [f"F{c}" for c in wide.columns]
    wide.index = pd.to_datetime(wide.index)

    contracts_7 = [f"F{i}" for i in range(1, 8)]
    wide = wide.dropna(subset=contracts_7)

    rets = wide.pct_change()

    r_leg1 = -(rets["F1"] + rets["F2"]) / 2
    r_leg2 = rets[contracts_7].mean(axis=1)

    r_arb = r_leg1 - r_leg2
    r_arb.name = "r_arb"
    return r_arb.dropna()


def build_term_structure_slope(vx: pd.DataFrame) -> pd.DataFrame:
    """
    Build daily VIX term structure slope measures using the 7-contract strip.

      ts_slope  = (F7 - F1) / F1   relative slope — primary Y variable
      ts_spread = F7 - F1           absolute slope (vol points)
      log_ts    = log(F7 / F1)      log relative slope

    Asensio's 'forward-month volatility premium' is the full-curve overpricing,
    best captured by the F7-to-F1 slope rather than just the front-to-spot basis.
    The 7-contract strip spans the full arbitrage window in the paper.
    """
    vx = vx.copy()
    vx["rank"] = vx.groupby("date")["tts_days"].rank(method="first").astype(int)

    wide = (vx[vx["rank"] <= 7]
              .pivot_table(index="date", columns="rank", values="price"))
    wide.columns = [f"F{c}" for c in wide.columns]
    wide.index = pd.to_datetime(wide.index)

    contracts_7 = [f"F{i}" for i in range(1, 8)]
    wide = wide.dropna(subset=contracts_7)

    out = pd.DataFrame(index=wide.index)
    out["ts_slope"]  = (wide["F7"] - wide["F1"]) / wide["F1"]
    out["ts_spread"] = wide["F7"] - wide["F1"]
    out["log_ts"]    = np.log(wide["F7"] / wide["F1"])
    out["F1"]        = wide["F1"]
    out["F7"]        = wide["F7"]
    return out.dropna()


def resample_weekly(s: pd.Series) -> pd.Series:
    """Resample to weekly Friday, using last observation in each week."""
    return s.resample("W-FRI").last().dropna()


def resample_weekly_df(df: pd.DataFrame) -> pd.DataFrame:
    """Resample DataFrame to weekly Friday."""
    return df.resample("W-FRI").last().dropna(how="all")


# -----------------------------------------------------------------------------
# OLS WITH NEWEY-WEST HAC SEs
# -----------------------------------------------------------------------------

def ols_nw(y: pd.Series, x: pd.Series, nlags: int = NW_LAGS) -> dict:
    """
    OLS with Newey-West HAC standard errors.
    Returns dict: n, intercept, coef, nw_se, t_stat, r2, f_stat
    """
    df = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    if len(df) < 10:
        return None
    Y = df["y"].values
    X = add_constant(df["x"].values)

    res = OLS(Y, X).fit()
    nw_cov = cov_hac(res, nlags=nlags)
    nw_se  = np.sqrt(np.diag(nw_cov))
    t_stat = res.params / nw_se

    return {
        "n":         len(Y),
        "intercept": res.params[0],
        "coef":      res.params[1],
        "nw_se":     nw_se[1],
        "t_stat":    t_stat[1],
        "r2":        res.rsquared,
        "f_stat":    res.fvalue,
        "_res":      res,
        "_x":        df["x"].values,
        "_y":        df["y"].values,
    }


# -----------------------------------------------------------------------------
# REPORTING
# -----------------------------------------------------------------------------

PAPER_TARGETS = {
    "post_crisis": {"coef": 0.112, "t_stat": 1.95, "r2": 0.121, "f_stat": 18.1},
    "full_sample": {"coef": 0.039, "t_stat": 1.16, "r2": 0.053},
}

def sig_stars(t: float) -> str:
    a = abs(t)
    if a > 2.576: return "***"
    if a > 1.960: return "**"
    if a > 1.645: return "*"
    return ""


def print_result(label: str, r: dict, targets: dict = None, indent: int = 2):
    pad = " " * indent
    sep = "-" * 62
    print(f"\n{pad}{sep}")
    print(f"{pad}  {label}")
    print(f"{pad}{sep}")
    if r is None:
        print(f"{pad}  [insufficient data]")
        return
    tgt_coef  = f"  <- target {targets['coef']}" if targets and "coef"   in targets else ""
    tgt_tstat = f"  <- target {targets['t_stat']}" if targets and "t_stat" in targets else ""
    tgt_r2    = f"  <- target {targets['r2']}" if targets and "r2"     in targets else ""
    tgt_f     = f"  <- target {targets['f_stat']}" if targets and "f_stat" in targets else ""
    stars = sig_stars(r["t_stat"])
    print(f"{pad}  N obs        : {r['n']}")
    print(f"{pad}  Intercept    : {r['intercept']:+.5f}")
    print(f"{pad}  Coefficient  : {r['coef']:+.5f}{tgt_coef}")
    print(f"{pad}  NW Std Err   : {r['nw_se']:.5f}")
    print(f"{pad}  t-statistic  : {r['t_stat']:+.3f}{stars}{tgt_tstat}")
    print(f"{pad}  R-squared    : {r['r2']:.4f}{tgt_r2}")
    print(f"{pad}  F-statistic  : {r['f_stat']:.2f}{tgt_f}")
    print(f"{pad}{sep}")


# -----------------------------------------------------------------------------
# PLOTTING
# -----------------------------------------------------------------------------

def make_scatter_plot(specs: list, out_path: Path):
    """
    Multi-panel scatter plot: one panel per spec, post-crisis period.
    Each panel shows scatter + OLS fit line.
    """
    n = len(specs)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.5))
    if n == 1:
        axes = [axes]

    for ax, (title, r, x_label, y_label) in zip(axes, specs):
        if r is None:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
            ax.set_title(title, fontsize=9)
            continue
        x_, y_ = r["_x"], r["_y"]
        ax.scatter(x_, y_, s=10, alpha=0.55, color="#2166ac", linewidths=0)
        x_line = np.linspace(x_.min(), x_.max(), 200)
        y_line = r["intercept"] + r["coef"] * x_line
        ax.plot(x_line, y_line, color="#d6604d", lw=1.8, label="OLS fit")
        stars = sig_stars(r["t_stat"])
        info = (f"beta={r['coef']:.4f}  t={r['t_stat']:.2f}{stars}\n"
                f"R2={r['r2']:.3f}   N={r['n']}")
        ax.text(0.05, 0.93, info, transform=ax.transAxes, fontsize=7.5,
                va="top", family="monospace",
                bbox=dict(fc="white", ec="gray", lw=0.5, alpha=0.85))
        ax.set_xlabel(x_label, fontsize=8)
        ax.set_ylabel(y_label, fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(True, lw=0.4, alpha=0.45)

    fig.suptitle(
        "Asensio (2013) Flow Replication - Post-Crisis Sub-Period\n"
        "Apr 2009 - Feb 2012   (VIX Futures Open Interest x Volatility Premium)",
        fontsize=10, y=1.01,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved to {out_path}")


def make_oi_timeseries_plot(weekly_df: pd.DataFrame, out_path: Path):
    """
    Three-panel time series: OI, term structure slope, and VIX basis
    over the post-crisis sub-period window.
    """
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)

    mask = (weekly_df.index >= POST_CRISIS_START) & (weekly_df.index <= POST_CRISIS_END)
    w = weekly_df[mask].dropna(how="all")

    ax1 = axes[0]
    if "total_oi" in w.columns:
        ax1.bar(w.index, w["total_oi"] / 1e3, width=5, color="#9ecae1", alpha=0.7,
                label="Total VIX OI (raw)")
    if "rolling_oi" in w.columns:
        ax1.plot(w.index, w["rolling_oi"] / 1e3, color="#08519c", lw=1.5,
                 label="252d rolling mean OI")
    ax1.set_ylabel("OI (thousands)", fontsize=8)
    ax1.legend(fontsize=7)
    ax1.grid(True, lw=0.4, alpha=0.4)
    ax1.set_title(
        "Asensio (2013) Flow Variables - Post-Crisis Sub-Period (Apr 2009 - Feb 2012)",
        fontsize=9,
    )

    ax2 = axes[1]
    if "ts_slope" in w.columns:
        ax2.plot(w.index, w["ts_slope"] * 100, color="#d6604d", lw=1.3,
                 label="TS Slope (F7-F1)/F1 %")
        ax2.axhline(0, color="black", lw=0.7, ls="--", alpha=0.5)
        ax2.set_ylabel("Term Structure Slope (%)", fontsize=8)
        ax2.legend(fontsize=7)
        ax2.grid(True, lw=0.4, alpha=0.4)

    ax3 = axes[2]
    if "vix_basis" in w.columns:
        ax3.plot(w.index, w["vix_basis"], color="#2166ac", lw=1.1,
                 label="VIX Basis (F1 - VIX spot)")
        ax3.axhline(0, color="black", lw=0.7, ls="--", alpha=0.5)
        ax3.set_ylabel("Basis (vol pts)", fontsize=8)
        ax3.legend(fontsize=7)
        ax3.grid(True, lw=0.4, alpha=0.4)
    ax3.set_xlabel("Date", fontsize=8)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    fig.autofmt_xdate()

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Time series plot saved to {out_path}")


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def main():
    print("=" * 66)
    print("  Asensio (2013) - Linear Retail Flow Parameter Baseline")
    print("  Post-Crisis Sub-Period: Apr 2009 - Feb 2012")
    print("=" * 66)

    print("\nLoading data ...")
    vx       = load_vx_all()
    vix_spot = load_vix_spot()

    print(f"  VX contracts loaded: {vx['security'].nunique()} contracts")
    print(f"  Daily obs: {len(vx)}  |  Date range: "
          f"{vx['date'].min().date()} to {vx['date'].max().date()}")

    print("\nBuilding daily signal panel ...")
    daily  = build_daily_panel(vx, vix_spot)
    weekly = resample_weekly_df(daily).dropna(subset=["vix_basis", "rolling_oi"])
    print(f"  Daily obs: {len(daily)}  |  Weekly obs: {len(weekly)}")

    print("Building arbitrage profit series (Asensio eq. 15-17) ...")
    r_arb_daily  = build_arb_profit_daily(vx)
    r_arb_weekly = resample_weekly(r_arb_daily)
    print(f"  Arb profit weekly obs: {len(r_arb_weekly)}")

    mask_pc = (weekly.index >= POST_CRISIS_START) & (weekly.index <= POST_CRISIS_END)
    w_pc    = weekly[mask_pc].copy()
    print(f"\n  Post-crisis weekly obs: {len(w_pc)}")
    print(f"  VIX Basis - mean {w_pc['vix_basis'].mean():.2f}  "
          f"std {w_pc['vix_basis'].std():.2f}  "
          f"range [{w_pc['vix_basis'].min():.2f}, {w_pc['vix_basis'].max():.2f}]")
    print(f"  Rolling OI - mean {w_pc['rolling_oi'].mean():,.0f}  "
          f"std {w_pc['rolling_oi'].std():,.0f} contracts")
    print(f"  Total OI   - mean {w_pc['total_oi'].mean():,.0f}  "
          f"std {w_pc['total_oi'].std():,.0f} contracts")

    # -- SPEC 1: Level regression - vix_basis ~ rolling_oi -----------------
    print("\n\n+==============================================================+")
    print("|  SPEC 1: Level - vix_basis ~ rolling_oi (raw signal)       |")
    print("+==============================================================+")

    r1_full = ols_nw(weekly["vix_basis"], weekly["rolling_oi"])
    print_result("Full Sample", r1_full, PAPER_TARGETS["full_sample"])

    r1_pc = ols_nw(w_pc["vix_basis"], w_pc["rolling_oi"])
    print_result("Post-Crisis Sub-Period", r1_pc, PAPER_TARGETS["post_crisis"])

    r1_pc_scaled = ols_nw(w_pc["vix_basis"], w_pc["rolling_oi"] / 1e3)
    print_result("Post-Crisis - OI / 1000 (coefficient matching)", r1_pc_scaled)

    # -- SPEC 2: First-difference arb ~ Dtotal_oi (Asensio Table 7 exact) --
    print("\n\n+==============================================================+")
    print("|  SPEC 2: FD - Dr_arb ~ Dtotal_oi  (Asensio Table 7 exact) |")
    print("+==============================================================+")

    oi_weekly  = resample_weekly(daily["total_oi"])
    delta_oi   = oi_weekly.diff().rename("delta_oi")
    delta_arb  = r_arb_weekly.diff().rename("delta_arb")

    r2_full = ols_nw(delta_arb, delta_oi)
    print_result("Full Sample", r2_full, PAPER_TARGETS["full_sample"])

    mask_pc_arb = (delta_arb.index >= POST_CRISIS_START) & (delta_arb.index <= POST_CRISIS_END)
    mask_pc_oi  = (delta_oi.index  >= POST_CRISIS_START) & (delta_oi.index  <= POST_CRISIS_END)
    r2_pc = ols_nw(delta_arb[mask_pc_arb], delta_oi[mask_pc_oi])
    print_result("Post-Crisis Sub-Period", r2_pc, PAPER_TARGETS["post_crisis"])

    r2_pc_scaled = ols_nw(delta_arb[mask_pc_arb], delta_oi[mask_pc_oi] / 1e3)
    print_result("Post-Crisis - D OI / 1000 (coefficient matching)", r2_pc_scaled)

    # -- SPEC 3: First-difference basis ~ Dtotal_oi ------------------------
    print("\n\n+==============================================================+")
    print("|  SPEC 3: FD - Dvix_basis ~ Dtotal_oi                      |")
    print("+==============================================================+")

    basis_weekly = resample_weekly(daily["vix_basis"])
    delta_basis  = basis_weekly.diff().rename("delta_basis")

    r3_full = ols_nw(delta_basis, delta_oi)
    print_result("Full Sample", r3_full, PAPER_TARGETS["full_sample"])

    mask_pc_b = (delta_basis.index >= POST_CRISIS_START) & (delta_basis.index <= POST_CRISIS_END)
    r3_pc = ols_nw(delta_basis[mask_pc_b], delta_oi[mask_pc_oi])
    print_result("Post-Crisis Sub-Period", r3_pc, PAPER_TARGETS["post_crisis"])

    r3_pc_scaled = ols_nw(delta_basis[mask_pc_b], delta_oi[mask_pc_oi] / 1e3)
    print_result("Post-Crisis - D OI / 1000 (coefficient matching)", r3_pc_scaled)

    # -- SPEC 4: FD Arb ~ Drolling_oi (smoothed flow) ----------------------
    print("\n\n+==============================================================+")
    print("|  SPEC 4: FD - Dr_arb ~ Drolling_oi (smoothed flow)        |")
    print("+==============================================================+")

    rolling_oi_w = resample_weekly(daily["rolling_oi"])
    delta_roi    = rolling_oi_w.diff().rename("delta_rolling_oi")

    mask_pc_roi = (delta_roi.index >= POST_CRISIS_START) & (delta_roi.index <= POST_CRISIS_END)
    r4_pc = ols_nw(delta_arb[mask_pc_arb], delta_roi[mask_pc_roi])
    print_result("Post-Crisis Sub-Period", r4_pc, PAPER_TARGETS["post_crisis"])

    r4_pc_scaled = ols_nw(delta_arb[mask_pc_arb], delta_roi[mask_pc_roi] / 1e3)
    print_result("Post-Crisis - D rolling OI / 1000", r4_pc_scaled)

    # -- SPEC 5 (PRIMARY): Term structure slope ~ total_oi -----------------
    print("\n\n+==============================================================+")
    print("|  SPEC 5 (PRIMARY): ts_slope ~ total_oi  [*** BEST MATCH ***]|")
    print("|  Y = (F7-F1)/F1 = term structure slope (full-curve premium)  |")
    print("|  X = total VIX OI (raw weekly, not smoothed)                 |")
    print("+==============================================================+")

    ts_df = build_term_structure_slope(vx)
    ts_w  = resample_weekly_df(ts_df)
    oi_w  = resample_weekly(daily["total_oi"])

    panel5 = pd.concat([ts_w["ts_slope"], oi_w.rename("total_oi")], axis=1).dropna()

    r5_full = ols_nw(panel5["ts_slope"], panel5["total_oi"])
    print_result("Full Sample", r5_full, PAPER_TARGETS["full_sample"])

    mask5_pc = (panel5.index >= POST_CRISIS_START) & (panel5.index <= POST_CRISIS_END)
    p5_pc    = panel5[mask5_pc]
    r5_pc    = ols_nw(p5_pc["ts_slope"], p5_pc["total_oi"])
    print_result("Post-Crisis Sub-Period", r5_pc, PAPER_TARGETS["post_crisis"])

    r5_pc_scaled = ols_nw(p5_pc["ts_slope"], p5_pc["total_oi"] / 1e3)
    print_result("Post-Crisis - OI / 1000 (coefficient matching)", r5_pc_scaled)

    r5_logoi_pc = ols_nw(p5_pc["ts_slope"], np.log(p5_pc["total_oi"]))
    print_result("Post-Crisis - log(OI) [highest R2]", r5_logoi_pc)

    # -- SPEC 6: ts_spread ~ total_oi (absolute slope in vol points) -------
    print("\n\n+==============================================================+")
    print("|  SPEC 6: ts_spread (F7-F1, vol pts) ~ total_oi             |")
    print("+==============================================================+")
    panel6 = pd.concat([ts_w["ts_spread"], oi_w.rename("total_oi")], axis=1).dropna()
    mask6  = (panel6.index >= POST_CRISIS_START) & (panel6.index <= POST_CRISIS_END)
    r6_pc  = ols_nw(panel6[mask6]["ts_spread"], panel6[mask6]["total_oi"])
    print_result("Post-Crisis Sub-Period", r6_pc, PAPER_TARGETS["post_crisis"])

    r6_scaled = ols_nw(panel6[mask6]["ts_spread"], panel6[mask6]["total_oi"] / 1e3)
    print_result("Post-Crisis - OI / 1000", r6_scaled)

    # -- Summary comparison ------------------------------------------------
    print("\n\n+==============================================================+")
    print("|  SUMMARY: Post-Crisis t-stat and R2 vs Asensio Table 7     |")
    print("+==============================================================+")
    print(f"\n  {'Spec':<44} {'t-stat':>8}  {'R2':>7}  {'N':>5}")
    print(f"  {'-'*44}  {'-'*7}  {'-'*6}  {'-'*5}")

    rows = [
        ("Spec 5 [PRIMARY] ts_slope ~ total_oi",          r5_pc),
        ("Spec 5 [PRIMARY] ts_slope ~ total_oi/1000",     r5_pc_scaled),
        ("Spec 5 [PRIMARY] ts_slope ~ log(total_oi)",     r5_logoi_pc),
        ("Spec 6 ts_spread ~ total_oi",                   r6_pc),
        ("Spec 6 ts_spread ~ total_oi/1000",              r6_scaled),
        ("Spec 1 basis ~ rolling_oi",                     r1_pc),
        ("Spec 2 FD Darb ~ Dtotal_oi",                    r2_pc),
        ("Spec 3 FD Dbasis ~ Dtotal_oi",                  r3_pc),
        ("[TARGET: Asensio Table 7]",                      None),
    ]
    for name, r in rows:
        if r is None:
            print(f"  {name:<44} {'1.95':>8}  {'0.121':>7}  {'147':>5}")
        elif r:
            stars = sig_stars(r["t_stat"])
            print(f"  {name:<44} {r['t_stat']:+8.3f}{stars}  {r['r2']:7.4f}  {r['n']:5d}")

    # -- Save panel data ---------------------------------------------------
    panel_out = OUTPUT / "asensio_flow_panel.csv"
    save_df = pd.concat([
        weekly[["front_price", "vix_spot", "vix_basis", "total_oi", "rolling_oi"]],
        ts_w[["ts_slope", "ts_spread", "log_ts"]],
    ], axis=1)
    save_df.to_csv(panel_out)
    print(f"\n  Panel saved: {panel_out}")

    # -- Plots -------------------------------------------------------------
    print("\nGenerating plots ...")
    scatter_specs = [
        ("Spec 5 [PRIMARY]\nts_slope ~ total_oi",
         r5_pc, "Total VIX OI (contracts)", "Term Structure Slope (F7-F1)/F1"),
        ("Spec 5 scaled\nts_slope ~ total_oi/1000",
         r5_pc_scaled, "Total VIX OI (thousands)", "Term Structure Slope (F7-F1)/F1"),
        ("Spec 2: FD arb\nDr_arb ~ Dtotal_oi",
         r2_pc, "D Total OI (contracts/week)", "D Arb Profit"),
    ]
    scatter_path = OUTPUT / "asensio_flow_replication.png"
    make_scatter_plot(scatter_specs, scatter_path)

    ts_path = OUTPUT / "asensio_oi_timeseries.png"
    weekly_for_plot = weekly.join(ts_w[["ts_slope", "ts_spread"]], how="left")
    make_oi_timeseries_plot(weekly_for_plot, ts_path)

    print("\n" + "=" * 66)
    print("  Done.")
    print("=" * 66)


if __name__ == "__main__":
    main()
