"""
data_prep.py
============
Builds the core daily panel used by the B&H (2014) replication:
  - SP500 returns   → daily proxy for realized variance (r²)
  - VIX daily close → VIX²/12 (monthly units, annualised %)
  - Multi-frequency RV aggregates: RV(1), RV(5), RV(22)

All variance quantities are expressed in MONTHLY PERCENTAGE-SQUARED units,
matching the paper's convention (e.g. VIX²/12 where VIX is annualised %).
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


def load_sp500_returns() -> pd.Series:
    """
    Construct a continuous daily SP500 return series from E-mini futures
    (front-month, roll to next nearest expiry each day).
    Returns decimal daily returns indexed by date.
    """
    meta = pd.read_parquet(DATA_DIR / "EquityFuture_security_meta.parquet")
    hist = pd.read_parquet(DATA_DIR / "EquityFuture_historical.parquet")

    es_tickers = meta[meta["curve_group"] == "ES"]["security"].tolist()
    es = hist[hist["security"].isin(es_tickers)].copy()
    es["date"] = pd.to_datetime(es["date"])

    # Merge expiry dates
    meta_es = meta[meta["curve_group"] == "ES"][["security", "expiry_yearmonth"]].copy()
    meta_es["expiry_date"] = pd.to_datetime(meta_es["expiry_yearmonth"], format="%Y-%m")
    es = es.merge(meta_es[["security", "expiry_date"]], on="security")

    # Front-month: on each date, take the contract with the soonest expiry
    es = es.sort_values(["date", "expiry_date"])
    front = es.groupby("date").first().reset_index()[["date", "returns"]].dropna()
    front = front.sort_values("date").set_index("date")["returns"]
    return front


def load_vix() -> pd.Series:
    """
    Load the VIX index (annualised %).  Returns a daily series indexed by date.
    """
    vix_df = pd.read_csv(DATA_DIR / "VolatilityIndexData.csv", parse_dates=["DATE"])
    vix = vix_df[vix_df["SECURITY"] == "VIX Index"][["DATE", "INDEX_VALUE"]].copy()
    vix = vix.sort_values("DATE").set_index("DATE")["INDEX_VALUE"]
    vix.index.name = "date"
    return vix


def compute_rv_components(ret: pd.Series, lags: tuple = (1, 5, 22)) -> pd.DataFrame:
    """
    Given daily decimal returns, compute RV proxies at daily/weekly/monthly
    frequencies in MONTHLY PERCENTAGE-SQUARED units.

    Since we lack 5-min intraday data, daily RV = (r_t × 100)².
    Monthly unit scaling: annualise would be ×252, but the paper keeps
    monthly units, so a 1-day RV is (r × 100)² and a 22-day aggregate
    is the sum of 22 daily RVs.

    The returned DataFrame columns:
        RV1   - daily  (1-day)  RV  [monthly % sq]
        RV5   - weekly (5-day)  RV  [monthly % sq]  — rolling 5-day avg × 22
        RV22  - monthly(22-day) RV  [monthly % sq]  — rolling 22-day sum
    """
    # daily RV in percentage-squared (pct² per day)
    rv_daily = (ret * 100) ** 2

    out = pd.DataFrame(index=ret.index)
    # RV(1): today's squared return, rescaled to monthly units (×22)
    out["RV1"] = rv_daily * 22
    # RV(5): 5-day rolling mean × 22
    out["RV5"] = rv_daily.rolling(5).mean() * 22
    # RV(22): 22-day rolling sum
    out["RV22"] = rv_daily.rolling(22).sum()
    return out


def build_panel() -> pd.DataFrame:
    """
    Assemble the full daily panel:
        date, RV22_fwd (target = next-month RV),
        VIX2_lag (VIX²/12 lagged 22 days),
        RV22_lag, RV5_lag, RV1_lag (lagged 22 days)
    Observations are overlapping (daily).
    """
    ret = load_sp500_returns()
    vix = load_vix()
    rv = compute_rv_components(ret)

    # Align on common dates
    panel = rv.join(vix.rename("VIX"), how="inner")
    panel = panel.dropna()

    # VIX² in monthly units (VIX is annualised %)
    panel["VIX2"] = panel["VIX"] ** 2 / 12.0

    # Forward-looking target: RV^(22)_{t+1..t+22} = sum of next 22 daily RVs.
    # Since RV22[t] = backward-looking sum ending at t, shifting by -22 gives
    # the backward-looking sum ending at t+22, which equals the forward sum
    # from t+1 to t+22.  This is the paper's RV^(22)_{t+1}.
    panel["RV22_fwd"] = panel["RV22"].shift(-22)

    # Predictors at time t (current, no additional lag beyond what's built into
    # the rolling windows): VIX_t^2, RV_t^(22), RV_t^(5), RV_t^(1).
    # Lag by 1 day so they are strictly predetermined vs RV22_fwd.
    panel["VIX2_lag"]  = panel["VIX2"].shift(1)
    panel["RV22_lag"]  = panel["RV22"].shift(1)
    panel["RV5_lag"]   = panel["RV5"].shift(1)
    panel["RV1_lag"]   = panel["RV1"].shift(1)

    panel = panel.dropna()
    return panel


if __name__ == "__main__":
    p = build_panel()
    print(f"Panel shape: {p.shape}")
    print(f"Date range:  {p.index.min().date()} – {p.index.max().date()}")
    print(p[["RV22_fwd", "VIX2_lag", "RV22_lag", "RV5_lag", "RV1_lag"]].describe().round(3))
