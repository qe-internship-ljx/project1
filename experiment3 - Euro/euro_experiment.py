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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "experiment2 - Return Regression"))

from helpers import compute_trend_quotient, build_master_panel
import cross_market

ROLL_VRP_WIN = 1000


# ── Euro data loaders ─────────────────────────────────────────────────────────

def load_stoxx_front_month() -> pd.DataFrame:
    """Continuous Euro Stoxx 50 (FX futures) front-month series with a
    returns-reconstructed price_level rebased to 1000."""
    sec_meta = pd.read_parquet(DATA / "EquityFuture_security_meta.parquet")
    hist     = pd.read_parquet(DATA / "EquityFuture_historical.parquet")

    fx_tickers = sec_meta[sec_meta["curve_group"] == "FX"]["security"].tolist()
    fx = hist[hist["security"].isin(fx_tickers)].copy()
    fx["date"] = pd.to_datetime(fx["date"])

    meta_fx = sec_meta[sec_meta["curve_group"] == "FX"][
        ["security", "expiry_yearmonth"]].copy()
    meta_fx["expiry_date"] = pd.to_datetime(meta_fx["expiry_yearmonth"], format="%Y-%m")
    fx = fx.merge(meta_fx[["security", "expiry_date"]], on="security")

    fx = fx.sort_values(["date", "expiry_date"])
    front = (fx.groupby("date").first().reset_index()
               [["date", "price", "returns"]].dropna(subset=["returns"]))
    front = front.sort_values("date").set_index("date")

    ret = front["returns"].dropna()
    front = front.join(((1 + ret).cumprod() * 1000).rename("price_level"), how="left")
    return front[["price", "price_level", "returns"]].dropna()


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


def compute_euro_vrp(returns: pd.Series, v2x: pd.Series,
                     window: int = ROLL_VRP_WIN) -> pd.DataFrame:
    """1000-day rolling HAR production loop for Euro VRP (mirrors
    vrp_experiment production_loop):  IVar = V2X²/12;  CV = HAR forecast of
    forward 22-day RV from rolling OLS;  VP = IVar − CV."""
    ivar = (v2x ** 2 / 12.0).rename("IVar")
    r100 = (returns * 100) ** 2
    df = pd.DataFrame({
        "IVar":     ivar,
        "IVar_lag": ivar.shift(1),
        "rv22":     r100.rolling(22).sum(),
        "rv22_lag": r100.rolling(22).sum().shift(1),
        "rv5_lag":  r100.rolling(5).sum().shift(1),
        "rv1_lag":  r100.shift(1),
    }).dropna()
    df["rv22_fwd"] = df["rv22"].shift(-22)

    feats  = ["IVar_lag", "rv22_lag", "rv5_lag", "rv1_lag"]
    N      = len(df)
    vp_arr = np.full(N, np.nan)
    cv_arr = np.full(N, np.nan)
    for i in range(window + 22, N - 22):
        train = df.iloc[i - window - 22 : i - 22].dropna(subset=["rv22_fwd"])
        if len(train) < 100:
            continue
        X_tr = np.column_stack([np.ones(len(train))] + [train[f].values for f in feats])
        try:
            beta = np.linalg.lstsq(X_tr, train["rv22_fwd"].values, rcond=None)[0]
        except Exception:
            continue
        x_i = np.array([1.0] + [float(df[f].iat[i]) for f in feats])
        cv_arr[i] = float(x_i @ beta)
        vp_arr[i] = float(df["IVar"].iat[i]) - cv_arr[i]

    return pd.DataFrame({"VP": vp_arr, "CV": cv_arr, "IVar": df["IVar"].values},
                        index=df.index).dropna(subset=["VP"])


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

    stoxx      = load_stoxx_front_month()
    vrp        = compute_euro_vrp(stoxx["returns"], load_v2x_spot())
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
