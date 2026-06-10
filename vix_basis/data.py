"""
Raw data loading.

All files are read once and returned as DataFrames with minimal
transformation (type coercions, column renames).  All joins and
derived columns live in panel.py.
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def load_vix_futures() -> pd.DataFrame:
    """
    VIX futures (curve_group='VX', Bloomberg tickers 'UX?') with
    last_trade_date merged from security metadata.

    Returned columns
    ----------------
    security, date, price, returns, last_trade_date
    """
    hist = pd.read_parquet(DATA_DIR / "VolatilityIndexFuture_historical.parquet")
    meta = pd.read_parquet(DATA_DIR / "VolatilityIndexFuture_security_meta.parquet")

    vx_meta = meta.loc[
        meta["curve_group"] == "VX", ["security", "last_trade_date"]
    ].assign(last_trade_date=lambda d: pd.to_datetime(d["last_trade_date"]))

    return (
        hist[hist["security"].isin(vx_meta["security"])]
        .assign(date=lambda d: pd.to_datetime(d["date"]))
        .merge(vx_meta, on="security")
        .sort_values(["date", "last_trade_date"])
        .reset_index(drop=True)
    )


def load_vix_spot() -> pd.DataFrame:
    """
    Daily CBOE VIX spot index levels.

    Returned columns
    ----------------
    date, vix_spot
    """
    df = pd.read_csv(DATA_DIR / "VolatilityIndexData.csv", parse_dates=["DATE"])
    return (
        df.loc[df["SECURITY"] == "VIX Index", ["DATE", "INDEX_VALUE"]]
        .rename(columns={"DATE": "date", "INDEX_VALUE": "vix_spot"})
        .sort_values("date")
        .reset_index(drop=True)
    )


def load_es_futures() -> pd.DataFrame:
    """
    S&P 500 E-mini futures (curve_group='ES') with last_trade_date.

    Returned columns
    ----------------
    security, date, price, returns, last_trade_date
    """
    hist = pd.read_parquet(DATA_DIR / "EquityFuture_historical.parquet")
    meta = pd.read_parquet(DATA_DIR / "EquityFuture_security_meta.parquet")

    es_meta = meta.loc[
        meta["curve_group"] == "ES", ["security", "last_trade_date"]
    ].assign(last_trade_date=lambda d: pd.to_datetime(d["last_trade_date"]))

    return (
        hist[hist["security"].isin(es_meta["security"])]
        .assign(date=lambda d: pd.to_datetime(d["date"]))
        .merge(es_meta, on="security")
        .sort_values(["date", "last_trade_date"])
        .reset_index(drop=True)
    )
