"""
nasdaq_expanding_window.py
==========================
Experiment 2 — NASDAQ edition.  Runs the exact experiment2 expanding-window
pipeline (regressions.py, base_strategies.py, leveraged_strategies.py) with the
dependent variable swapped to NQ (NASDAQ-100 E-mini) front-month 20-day forward
return.  Every signal (VRP, VIX term-structure slope, VVIX MA5) is identical to
experiment2, so the only market-specific code here is the NQ price series; the
rest is delegated to ``cross_market`` (in "experiment3 - Euro").

Models (VRP · VVIX MA5 · VRP+Term Slope · VRP+VVIX MA5), every base and
leveraged threshold variant, and the leveraged comparison are produced by the
shared runner.  Outputs/caches land in "experiment3 - Nasdaq"/output.
"""

import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent
DATA = ROOT.parent / "data"
sys.path.insert(0, str(ROOT.parent / "experiment2 - Return Regression"))
sys.path.insert(0, str(ROOT.parent / "experiment3 - Euro"))

from helpers import (
    load_vrp_series_expanding, load_vix_futures_term_structure,
    load_vvix, compute_vvix_ma5, compute_trend_quotient, build_master_panel,
)
from fh_replication.fh_replication import compute_vix_term_slope
import cross_market


def load_nq_front_month() -> pd.DataFrame:
    """Continuous NASDAQ-100 E-mini (NQ, curve_group='NN') front-month series
    with a returns-reconstructed price_level rebased to 1000."""
    sec_meta = pd.read_parquet(DATA / "EquityFuture_security_meta.parquet")
    hist     = pd.read_parquet(DATA / "EquityFuture_historical.parquet")

    nq_tickers = sec_meta[sec_meta["curve_group"] == "NN"]["security"].tolist()
    nq = hist[hist["security"].isin(nq_tickers)].copy()
    nq["date"] = pd.to_datetime(nq["date"])

    meta_nq = sec_meta[sec_meta["curve_group"] == "NN"][
        ["security", "expiry_yearmonth"]].copy()
    meta_nq["expiry_date"] = pd.to_datetime(meta_nq["expiry_yearmonth"], format="%Y-%m")
    nq = nq.merge(meta_nq[["security", "expiry_date"]], on="security")

    nq = nq.sort_values(["date", "expiry_date"])
    front = (nq.groupby("date").first().reset_index()
               [["date", "price", "returns"]].dropna(subset=["returns"]))
    front = front.sort_values("date").set_index("date")

    ret = front["returns"].dropna()
    front = front.join(((1 + ret).cumprod() * 1000).rename("price_level"), how="left")
    return front[["price", "price_level", "returns"]].dropna()


if __name__ == "__main__":
    print("=" * 72)
    print("  Experiment 2 — NASDAQ edition (NQ front-month 20-day forward return)")
    print("=" * 72)

    nq         = load_nq_front_month()
    vrp        = load_vrp_series_expanding()
    term_slope = compute_vix_term_slope(load_vix_futures_term_structure())
    vvix_ma5   = compute_vvix_ma5(load_vvix())
    trend_q    = compute_trend_quotient(nq)

    panel = build_master_panel(vrp, nq, term_slope, trend_q, vvix_ma5)
    print(f"  Panel: {panel.index.min().date()} – {panel.index.max().date()} "
          f"({len(panel):,} obs)")

    cross_market.run_all(
        panel,
        out_root=ROOT / "output",
        cache_dir=ROOT / "output" / "regression_cache",
        vv_label="VVIX MA5",
    )
    print("\nDone — nasdaq outputs in", ROOT / "output" / "plots")
