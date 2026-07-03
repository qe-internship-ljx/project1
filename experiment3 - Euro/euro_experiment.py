"""
euro_experiment.py
==================
Experiment 2 — Euro edition.  Runs the exact experiment2 expanding-window
pipeline (regressions.py, base_strategies.py, leveraged_strategies.py) on
European data by swapping in Euro inputs and delegating everything else to
``cross_market`` (which lives in this folder).

Euro inputs (the only market-specific code here):
  • Euro Stoxx 50 front-month (FX futures)        -> dependent return
  • Euro VRP via 1000-day rolling HAR on V2X²/12    -> "VP"
  • VSTOXX (DI) futures term-structure slope       -> "term_slope"
  • VV2TX (VSTOXX-of-VSTOXX) 5-day MA               -> "vvix_ma5" column

Models (VRP · VV2TX MA5 · VRP+Term Slope · VRP+VV2TX MA5), every base and
leveraged threshold variant, and the leveraged comparison are produced by the
shared runner.  Outputs/caches land in "experiment3 - Euro"/output.
"""

import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
DATA = ROOT.parent / "data"
EXP1_DIR = ROOT.parent / "experiment1 - VRP Computation"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "experiment2 - Return Regression"))
sys.path.insert(0, str(EXP1_DIR))

from helpers import compute_trend_quotient, build_master_panel
import cross_market
import experiment as exp1          # experiment1 production loop + summary plotter
from data_prep import compute_rv_components

ROLL_VRP_WIN = 1000


# ── Euro data loaders ─────────────────────────────────────────────────────────

def load_v2x_spot() -> pd.Series:
    df = pd.read_csv(DATA / "VolatilityIndexData.csv", parse_dates=["DATE"])
    s = (df[df["SECURITY"] == "V2X Index"].sort_values("DATE")
         .set_index("DATE")["INDEX_VALUE"])
    s.index.name = "date"
    return s


def load_vstoxx_futures() -> pd.DataFrame:
    """VSTOXX (DI) futures with time-to-maturity in years (non-expired only)."""
    sec_meta = pd.read_parquet(DATA / "VolatilityIndexFuture_security_meta.parquet")
    hist     = pd.read_parquet(DATA / "VolatilityIndexFuture_historical.parquet")

    di_secs = sec_meta[sec_meta["curve_group"] == "DI"][
        ["security", "last_trade_date"]].copy()
    di_secs["last_trade_date"] = pd.to_datetime(di_secs["last_trade_date"])

    di_hist = hist[hist["security"].isin(set(di_secs["security"]))].copy()
    di_hist["date"] = pd.to_datetime(di_hist["date"])

    di = di_hist.merge(di_secs, on="security")
    di["ttm_years"] = (di["last_trade_date"] - di["date"]).dt.days / 365.25
    di = di[di["ttm_years"] > 0].dropna(subset=["price", "ttm_years"])
    return di[["date", "security", "price", "ttm_years"]].sort_values("date")


def load_vv2tx() -> pd.Series:
    df = pd.read_csv(DATA / "VolatilityIndexData.csv", parse_dates=["DATE"])
    s = (df[df["SECURITY"] == "VV2TX Index"].sort_values("DATE")
         .set_index("DATE")["INDEX_VALUE"])
    s.index.name = "date"
    return s


def build_euro_vrp_panel(returns: pd.Series, v2x: pd.Series) -> pd.DataFrame:
    """Assemble a panel in experiment1's exact format (V2X substituted for VIX),
    so experiment1's production_loop / plot_combined_vrp_summary can be reused
    verbatim.  Mirrors experiment1._build_panel_from_ivar, except that rows with
    a missing forward target (the last 22 days) are kept: the production loop
    never trains on them but still emits a VP forecast there, so the trading
    panel gets VRP coverage right up to the end of the sample."""
    ivar  = (v2x ** 2 / 12.0).rename("IVar")
    rv    = compute_rv_components(returns)
    panel = rv.join(ivar, how="inner").dropna()
    panel["RV22_fwd"] = panel["RV22"].shift(-22)
    panel["VIX2_lag"] = panel["IVar"].shift(1)   # named VIX2_lag for har_model compat
    panel["RV22_lag"] = panel["RV22"].shift(1)
    panel["RV5_lag"]  = panel["RV5"].shift(1)
    panel["RV1_lag"]  = panel["RV1"].shift(1)
    return panel.dropna(subset=["VIX2_lag", "RV22_lag", "RV5_lag", "RV1_lag"])


def run_euro_vrp_summary(returns: pd.Series, v2x: pd.Series,
                         window: int = ROLL_VRP_WIN) -> pd.DataFrame:
    """Run experiment1's rolling-window production loop on Euro inputs, save the
    production-loop CSV, and render experiment1's combined VRP summary plot.
    Returns the production-loop DataFrame (columns include VP/CV/IVar) so the
    trading panel reuses the same VRP series — one code path, guaranteed
    consistency between the summary CSV/plot and the strategy inputs."""
    out_dir = ROOT / "output"
    out_dir.mkdir(exist_ok=True)

    panel = build_euro_vrp_panel(returns, v2x)
    print(f"  VRP panel: {panel.index.min().date()} – {panel.index.max().date()} "
          f"({len(panel):,} obs)")

    prod_df, stats_df = exp1.production_loop(panel, window=window, return_stats=True)
    prod_df.to_csv(out_dir / "production_loop_rolling.csv")
    print(f"  Saved production loop -> {out_dir / 'production_loop_rolling.csv'} "
          f"({len(prod_df):,} steps, VRP mean={prod_df['VP'].mean():.3f})")

    # Point experiment1's plotter at the Euro output dir, then call it verbatim.
    exp1.OUTPUT   = out_dir
    exp1.ROLL_WIN = window
    exp1.plot_combined_vrp_summary(
        prod_df, stats_df, tag="rolling",
        window_label=f"{window}-day Rolling OLS (Euro: V2X / Euro Stoxx 50)",
    )
    return prod_df


def compute_vstoxx_term_slope(vstoxx_df: pd.DataFrame) -> pd.Series:
    """Daily cross-sectional OLS  price ~ TtM  on VSTOXX futures (≥3 contracts);
    slope > 0 contango, < 0 backwardation."""
    slopes = {}
    for date, grp in vstoxx_df.groupby("date"):
        grp = grp.dropna(subset=["price", "ttm_years"])
        if len(grp) < 3:
            continue
        X = np.column_stack([np.ones(len(grp)), grp["ttm_years"].values])
        try:
            slopes[date] = float(np.linalg.lstsq(X, grp["price"].values, rcond=None)[0][1])
        except Exception:
            pass
    s = pd.Series(slopes, name="term_slope")
    s.index = pd.to_datetime(s.index); s.index.name = "date"
    return s.sort_index()


# ── Build panel + run ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 72)
    print("  Experiment 2 — Euro edition (Euro Stoxx 50 / V2X / VSTOXX / VV2TX)")
    print("=" * 72)

    stoxx      = cross_market.load_front_month("FX")   # Euro Stoxx 50
    v2x        = load_v2x_spot()

    # VRP production loop + experiment1-style summary plot (reuses experiment1
    # code); the same loop output feeds the trading panel below.
    print("\n[VRP] Rolling-window production loop + summary plot…")
    prod_df    = run_euro_vrp_summary(stoxx["returns"], v2x)

    vrp        = prod_df[["VP", "CV", "IVar"]]
    term_slope = compute_vstoxx_term_slope(load_vstoxx_futures())
    vv2tx_ma5  = load_vv2tx().rolling(5).mean().rename("vvix_ma5")
    trend_q    = compute_trend_quotient(stoxx)

    panel = build_master_panel(vrp, stoxx, term_slope, trend_q, vv2tx_ma5)
    print(f"  Panel: {panel.index.min().date()} – {panel.index.max().date()} "
          f"({len(panel):,} obs)")

    cross_market.run_all(
        panel,
        out_root=ROOT / "output",
        cache_dir=ROOT / "output" / "regression_cache",
        vv_label="VV2TX MA5",
    )
    print("\nDone — euro outputs in", ROOT / "output" / "plots")
