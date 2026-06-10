"""
Section III: Trading simulation (Exhibits 5-8).

Trading rules (from the paper)
-------------------------------
Define:
  daily_roll = (trade_VIX_futures_price - VIX_spot) / business_days_to_settlement

SHORT (contango) trade
  Entry : daily_roll >  +0.10 VIX points/day
  Action: SELL 1 VIX futures at bid; SHORT hr ES mini contracts at mid
  Exit  : daily_roll <  +0.05  OR  9 business days have elapsed
          BUY 1 VIX futures at ask; CLOSE hr ES mini contracts at mid

LONG (backwardation) trade
  Entry : daily_roll <  -0.10 VIX points/day
  Action: BUY 1 VIX futures at ask; LONG hr ES mini contracts at mid
  Exit  : daily_roll >  -0.05  OR  9 business days have elapsed
          SELL 1 VIX futures at bid; CLOSE hr ES mini contracts at mid

Only the nearest contract with >= 10 business days to settlement is used.
Hedge ratio is locked at trade entry and not updated while the trade is live.

Transaction costs (matching the paper)
---------------------------------------
VIX futures  : full bid-ask spread + $3 round-trip brokerage per contract.
               Approximated as the sample average (0.062 VIX points front).
               Round-trip bid-ask cost = 0.062 x $1,000 = $62 per contract.
ES futures   : half of minimum tick (0.25 pt x $50) per side = $6.25/side
               i.e. $12.50 round-trip spread + $3 brokerage per contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

# -- Transaction-cost constants -------------------------------------------
VIX_MULTIPLIER   = 1_000
ES_MULTIPLIER    = 50
VIX_HALF_SPREAD  = 0.031          # half of avg 0.062 bid-ask (Exhibit 1)
ES_HALF_TICK     = 6.25
BROKERAGE_PER_RT = 3.0

VIX_RT_COST = (VIX_HALF_SPREAD * 2) * VIX_MULTIPLIER + BROKERAGE_PER_RT  # $65
ES_RT_COST  = ES_HALF_TICK * 2 + BROKERAGE_PER_RT                         # $15.50

ROLL_ENTRY_THRESHOLD = 0.10
ROLL_EXIT_THRESHOLD  = 0.05
MAX_HOLD_DAYS        = 9


# -------------------------------------------------------------------------
# Data class
# -------------------------------------------------------------------------

@dataclass
class Trade:
    direction: str
    entry_date: pd.Timestamp
    entry_vix_price: float
    entry_es_price: float
    hr: float
    entry_roll: float
    exit_date: Optional[pd.Timestamp]  = None
    exit_vix_price: Optional[float]    = None
    exit_es_price: Optional[float]     = None
    hold_days: Optional[int]           = None
    pnl_vix_futures: Optional[float]   = None
    pnl_es_hedge: Optional[float]      = None
    pnl_hedged: Optional[float]        = None
    pnl_unhedged: Optional[float]      = None
    pnl_roll: Optional[float]          = None
    daily_rolls: List[float]           = field(default_factory=list)


# -------------------------------------------------------------------------
# P&L accounting
# -------------------------------------------------------------------------

def _close_trade(
    trade: Trade,
    exit_date: pd.Timestamp,
    exit_vix_price: float,
    exit_es_price: float,
    hold_days: int,
) -> Trade:
    d  = trade.direction
    hr = abs(trade.hr)

    vix_price_change = exit_vix_price - trade.entry_vix_price
    pnl_vix_gross = (-vix_price_change if d == "short" else vix_price_change) * VIX_MULTIPLIER
    pnl_vix = pnl_vix_gross - VIX_RT_COST

    es_price_change = exit_es_price - trade.entry_es_price
    pnl_es_gross = (-es_price_change if d == "short" else es_price_change) * ES_MULTIPLIER * hr
    pnl_es = pnl_es_gross - hr * ES_RT_COST

    trade.exit_date       = exit_date
    trade.exit_vix_price  = exit_vix_price
    trade.exit_es_price   = exit_es_price
    trade.hold_days       = hold_days
    trade.pnl_vix_futures = pnl_vix
    trade.pnl_es_hedge    = pnl_es
    trade.pnl_hedged      = pnl_vix + pnl_es
    trade.pnl_unhedged    = pnl_vix
    trade.pnl_roll        = sum(trade.daily_rolls)
    return trade


# -------------------------------------------------------------------------
# Main simulation loop
# -------------------------------------------------------------------------

def run_simulation(
    panel: pd.DataFrame,
    hr_df: pd.DataFrame,
    start_date: str = "2007-01-01",
    end_date: str   = "2011-12-31",
) -> List[Trade]:
    """
    Run the full trading simulation.

    Parameters
    ----------
    panel      : daily panel from build_daily_panel()
    hr_df      : hedge ratios from compute_oos_hedge_ratios()
    start_date : first day eligible for new trade entries
    end_date   : last day eligible for new entries (open trades may close after)

    Returns
    -------
    List of completed Trade objects.
    """
    sim = (
        panel
        .merge(hr_df[["date", "hr"]], on="date", how="left")
        .sort_values("date")
        .reset_index(drop=True)
    )
    sim = sim[
        (sim["date"] >= pd.Timestamp(start_date)) &
        (sim["date"] <= pd.Timestamp(end_date) + pd.Timedelta(days=60))
    ].reset_index(drop=True)

    completed: List[Trade] = []
    active: Optional[Trade] = None
    biz_days_held: int = 0

    for _, row in sim.iterrows():
        date      = row["date"]
        roll      = row.get("daily_roll",  np.nan)
        vix_price = row.get("trade_price", np.nan)
        es_price  = row.get("es_price",    np.nan)
        hr        = row.get("hr",          np.nan)
        tts       = row.get("trade_tts",   np.nan)

        any_nan = any(pd.isna(v) for v in [roll, vix_price, es_price, hr, tts])

        # -- Manage open trade --------------------------------------------
        if active is not None:
            biz_days_held += 1
            active.daily_rolls.append(
                abs(roll) * VIX_MULTIPLIER if not pd.isna(roll) else 0.0
            )

            exit_triggered = False
            if not any_nan:
                if active.direction == "short" and roll < ROLL_EXIT_THRESHOLD:
                    exit_triggered = True
                elif active.direction == "long" and roll > -ROLL_EXIT_THRESHOLD:
                    exit_triggered = True

            if exit_triggered or biz_days_held >= MAX_HOLD_DAYS:
                active = _close_trade(
                    active,
                    exit_date=date,
                    exit_vix_price=vix_price if not pd.isna(vix_price) else active.entry_vix_price,
                    exit_es_price=es_price   if not pd.isna(es_price)  else active.entry_es_price,
                    hold_days=biz_days_held,
                )
                completed.append(active)
                active = None
                biz_days_held = 0
            continue

        # -- Check entry conditions (no open trade) -----------------------
        if date > pd.Timestamp(end_date):
            continue
        if any_nan:
            continue

        if roll > ROLL_ENTRY_THRESHOLD:
            active = Trade(direction="short", entry_date=date,
                           entry_vix_price=vix_price, entry_es_price=es_price,
                           hr=hr, entry_roll=roll)
            biz_days_held = 0

        elif roll < -ROLL_ENTRY_THRESHOLD:
            active = Trade(direction="long", entry_date=date,
                           entry_vix_price=vix_price, entry_es_price=es_price,
                           hr=hr, entry_roll=roll)
            biz_days_held = 0

    # Force-close any trade still open at the end of the window
    if active is not None:
        last = sim.iloc[-1]
        active = _close_trade(
            active,
            exit_date=last["date"],
            exit_vix_price=last.get("trade_price", active.entry_vix_price),
            exit_es_price=last.get("es_price",    active.entry_es_price),
            hold_days=biz_days_held,
        )
        completed.append(active)

    return completed


# -------------------------------------------------------------------------
# Results aggregation
# -------------------------------------------------------------------------

def trades_to_dataframe(trades: List[Trade]) -> pd.DataFrame:
    """Convert a list of Trade objects to a tidy DataFrame."""
    if not trades:
        return pd.DataFrame()
    return pd.DataFrame([{
        "direction":    t.direction,
        "entry_date":   t.entry_date,
        "exit_date":    t.exit_date,
        "hold_days":    t.hold_days,
        "entry_roll":   t.entry_roll,
        "hr":           t.hr,
        "pnl_hedged":   t.pnl_hedged,
        "pnl_unhedged": t.pnl_unhedged,
        "pnl_es_hedge": t.pnl_es_hedge,
        "pnl_roll":     t.pnl_roll,
    } for t in trades])
