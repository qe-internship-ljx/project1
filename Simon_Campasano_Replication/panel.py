"""
Build the daily working panel used by all downstream analyses.

For each trading date the panel records the VIX spot level, front and
second VIX futures contracts, the *trade-eligible* front contract
(>= 10 business days to settlement, as required by the paper's trading
rules), and the front-month E-mini S&P 500 (ES) futures price / return.

Key derived columns
-------------------
front_basis   = front_price  - vix_spot
second_basis  = second_price - vix_spot
trade_basis   = trade_price  - vix_spot
daily_roll    = trade_basis  / trade_tts
                (expected daily decay toward spot assuming linear convergence)

Note on data differences from the paper
----------------------------------------
Simon & Campasano use intraday data (3:00-3:15 PM CST) to ensure
synchronous quotes.  We use daily settlement prices.  All methodology
is otherwise identical.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import load_es_futures, load_vix_futures, load_vix_spot


# ????????????????????????????????????????????????????????????????????????????
# Helpers
# ????????????????????????????????????????????????????????????????????????????

def _add_tts(df: pd.DataFrame, date_col: str, expiry_col: str) -> pd.DataFrame:
    """
    Add a 'tts' column = business-day count from date_col to expiry_col.
    Uses numpy.busday_count with the default Mon-Fri calendar.
    """
    df = df.copy()
    df["tts"] = np.busday_count(
        df[date_col].values.astype("datetime64[D]"),
        df[expiry_col].values.astype("datetime64[D]"),
    )
    return df


def _nth_contract(
    df: pd.DataFrame,
    n: int,
    min_tts: int = 0,
) -> pd.DataFrame:
    """
    Return the n-th nearest-to-expiry contract on each date (1 = front).
    Rows with tts < min_tts are excluded before ranking.
    """
    sub = df[df["tts"] >= min_tts].copy()
    sub["_rank"] = sub.groupby("date")["tts"].rank(method="first", ascending=True)
    return sub[sub["_rank"] == n].drop(columns="_rank")


# ????????????????????????????????????????????????????????????????????????????
# Public API
# ????????????????????????????????????????????????????????????????????????????

def build_daily_panel(
    start_date: str = "2004-01-01",
    end_date: str = "2026-12-31",
) -> pd.DataFrame:
    """
    Assemble the main daily panel.

    Parameters
    ----------
    start_date, end_date : ISO date strings bounding the output.

    Returns
    -------
    DataFrame indexed by date with one row per VIX-spot trading day.
    """
    vx   = load_vix_futures()
    spot = load_vix_spot()
    es   = load_es_futures()

    # ?? VIX futures: business-days-to-settlement ?????????????????????????
    vx = _add_tts(vx, "date", "last_trade_date")
    vx = vx[vx["tts"] >= 0]   # drop already-expired observations

    # Front (nearest) VIX contract -- no minimum TTS for display/regression
    front_vx = _nth_contract(vx, n=1)[
        ["date", "price", "tts", "last_trade_date", "security", "returns"]
    ].rename(columns={
        "price": "front_price",
        "tts": "front_tts",
        "last_trade_date": "front_expiry",
        "security": "front_sec",
        "returns": "front_ret",
    })

    # Second VIX contract
    second_vx = _nth_contract(vx, n=2)[
        ["date", "price"]
    ].rename(columns={"price": "second_price"})

    # Trade-eligible front contract: >= 10 business days remaining
    # (paper's requirement for entering new positions)
    trade_vx = _nth_contract(vx, n=1, min_tts=10)[
        ["date", "price", "tts", "last_trade_date", "security", "returns"]
    ].rename(columns={
        "price": "trade_price",
        "tts": "trade_tts",
        "last_trade_date": "trade_expiry",
        "security": "trade_sec",
        "returns": "trade_ret",
    })

    # ?? ES futures: front-month ???????????????????????????????????????????
    es = _add_tts(es, "date", "last_trade_date")
    es_front = _nth_contract(es[es["tts"] >= 0], n=1)[
        ["date", "price", "returns"]
    ].rename(columns={"price": "es_price", "returns": "es_ret"})

    # ?? Merge ????????????????????????????????????????????????????????????
    panel = (
        spot
        .merge(front_vx,  on="date", how="left")
        .merge(second_vx, on="date", how="left")
        .merge(trade_vx,  on="date", how="left")
        .merge(es_front,  on="date", how="left")
    )

    # ?? Derived columns ??????????????????????????????????????????????????
    panel["front_basis"]  = panel["front_price"]  - panel["vix_spot"]
    panel["second_basis"] = panel["second_price"]  - panel["vix_spot"]
    panel["trade_basis"]  = panel["trade_price"]   - panel["vix_spot"]
    # daily_roll: basis divided by business days to settlement
    # positive -> contango (futures premium), negative -> backwardation
    panel["daily_roll"]   = panel["trade_basis"]   / panel["trade_tts"]

    # Contango / backwardation flags
    panel["in_contango"]       = panel["trade_basis"] > 0
    panel["in_backwardation"]  = panel["trade_basis"] < 0

    # Date filter
    mask = (panel["date"] >= pd.Timestamp(start_date)) & (
            panel["date"] <= pd.Timestamp(end_date))
    return panel[mask].sort_values("date").reset_index(drop=True)
