"""
helpers.py
==========
Experiment 2: Equity Index Timing via VRP, Term Structure Geometry, VVIX, and Trend

Strategy: Systematic long/short timing of S&P 500 E-mini futures using four signals:
  1. HAR-derived Variance Risk Premium (VRP)  — loaded from Experiment 1
  2. VIX futures term structure slope          — daily cross-sectional regression
  3. Equity trend quotient                     — P_t / SMA(200)
  4. VVIX tail-risk indicator                  — 5-day SMA of VVIX index

Implementation follows the plan steps 1–8 strictly. Deviations from the plan are
flagged with [DEV-N] markers throughout the code and summarised in SUMMARY.md.

Steps
-----
1.  Signal aggregation: load VRP from Experiment 1, ES futures, VIX futures, VVIX
2.  VIX term structure slope via daily cross-sectional OLS (price ~ TtM)
3.  Equity trend quotient: front-month ES price / 200-day SMA
4.  VVIX tail-risk: 5-day SMA of VVIX index
5.  Bivariate predictive regression: forward 20-day S&P 500 return ~ VRP + signal
      Base / Model A / Model B / Model C — full-sample + rolling OLS positions
6.  Algorithmic logic gate: conjunctive long/short/flat entry rules
7.  Strategy simulation: daily P&L, net of 0.05% slippage per trade side
8.  Alpha isolation: compare vs buy-and-hold and 200-day MA trend baselines

Deviations from plan (summary)
-------------------------------
[DEV-1] Spot S&P 500 not available in data — front-month ES futures price level
         (reconstructed from cumulative daily returns, rebased to 1000 at start of
         available data) is used as a proxy for both F_t and S_t, making basis B_t ≈ 0.
[DEV-2] Full-signal overlap period: 2006-03-06 to 2025-12-31 (limited by VVIX start
         date 2006-03-06 and VRP production-loop data availability).
[DEV-5] Random forest exploration (Step 6 optional item) is not implemented; this
         is flagged as out-of-scope for the present experiment.

Outputs (all in ./output/)
--------------------------
  Plots: signals.png, regression_results.png, cumulative_returns.png,
         position_history.png, drawdown.png
  CSVs:  signals.csv, regression_results.csv, strategy_performance.csv,
         benchmark_comparison.csv
  MD:    EXPERIMENT_RESULTS.md
"""

import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats

ROOT    = Path(__file__).parent
OUTPUT  = ROOT / "output"
OUTPUT.mkdir(exist_ok=True)
DATA    = ROOT.parent / "data"
VRP_OUT = ROOT.parent / "experiment1 - VRP Computation" / "output"

BH_DIR = ROOT.parent / "bh_replication"
sys.path.insert(0, str(BH_DIR))
sys.path.insert(0, str(ROOT.parent))
from fh_replication.fh_replication import compute_vix_term_slope  # FH (2019) cross-sectional model

TCOST      = 0.0005  # 0.05% one-way slippage per contract side


# ═══════════════════════════════════════════════════════════════════════════
# STEP 1 — Signal Data Loading
# ═══════════════════════════════════════════════════════════════════════════

def load_vrp_series() -> pd.DataFrame:
    """
    Load VRP, conditional variance (CV), implied variance (IVar) from
    Experiment 1 production-loop output.  Falls back to CSV if parquet missing.
    Returns DataFrame indexed by date with columns [VRP, CV, IVar]. (column name in data is VP)
    """
    pq_path = VRP_OUT / "production_loop_rolling.parquet"
    csv_path = VRP_OUT / "production_loop_rolling.csv"
    if pq_path.exists():
        df = pd.read_parquet(pq_path)
    else:
        df = pd.read_csv(csv_path, parse_dates=["date"])
        df = df.set_index("date")
    return df[["VP", "CV", "IVar"]].copy()


def load_vrp_series_expanding() -> pd.DataFrame:
    """
    Load VRP from Experiment 1 expanding-window HAR production loop.
    Returns DataFrame indexed by date with columns [VP, CV, IVar].
    """
    csv_path = VRP_OUT / "production_loop_expanding.csv"
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df.set_index("date")
    return df[["VP", "CV", "IVar"]].copy()


def load_es_open_interest() -> pd.Series:
    """
    Total ES open interest (sum across tracked contracts), smoothed with a
    252-day backward rolling mean. The raw total still shows a quarterly
    sawtooth because the dataset covers only the front 1-2 contracts: when
    the expiring contract disappears the new front starts from near-zero,
    producing the same spike pattern as front-month OI alone. A 252-day
    rolling mean spans ~4 full roll cycles so the quarterly oscillation
    averages out, leaving a stable trend-following measure of aggregate
    market participation with no lookahead.
    Returns a Series indexed by date, named 'open_interest'.
    """
    sec_meta   = pd.read_parquet(DATA / "EquityFuture_security_meta.parquet")
    hist       = pd.read_parquet(DATA / "EquityFuture_historical.parquet")
    es_tickers = sec_meta[sec_meta["curve_group"] == "ES"]["security"].tolist()
    es = hist[hist["security"].isin(es_tickers)].copy()
    es["date"] = pd.to_datetime(es["date"])
    total_oi = (
        es.dropna(subset=["open_interest"])
          .groupby("date")["open_interest"].sum()
    )
    total_oi.index = pd.to_datetime(total_oi.index)
    total_oi = total_oi.sort_index()
    smooth_oi = total_oi.rolling(252, min_periods=63).mean()
    smooth_oi.name = "open_interest"
    return smooth_oi


def load_es_front_month() -> pd.DataFrame:
    """
    Build a continuous S&P 500 E-mini (ES) daily series.
    Front-month: on each date, take the contract with the earliest expiry.

    Returns DataFrame indexed by date with columns [price, returns].

    [DEV-1] The reconstructed continuous price level (cumulative product of
    returns, rebased to 1000 at the start) is used as proxy for spot S_t.
    The front-month futures price F_t is approximated by the same proxy;
    basis B_t = (F_t - S_t) / S_t ≈ 0.
    """
    sec_meta = pd.read_parquet(DATA / "EquityFuture_security_meta.parquet")
    hist     = pd.read_parquet(DATA / "EquityFuture_historical.parquet")

    es_tickers = sec_meta[sec_meta["curve_group"] == "ES"]["security"].tolist()
    es = hist[hist["security"].isin(es_tickers)].copy()
    es["date"] = pd.to_datetime(es["date"])

    meta_es = sec_meta[sec_meta["curve_group"] == "ES"][
        ["security", "expiry_yearmonth"]].copy()
    meta_es["expiry_date"] = pd.to_datetime(meta_es["expiry_yearmonth"], format="%Y-%m")
    es = es.merge(meta_es[["security", "expiry_date"]], on="security")

    # Front-month selection
    es = es.sort_values(["date", "expiry_date"])
    front = es.groupby("date").first().reset_index()[
        ["date", "price", "returns"]].dropna(subset=["returns"])
    front = front.sort_values("date").set_index("date")

    # Reconstruct continuous price level from returns (rebased to 1000)
    ret = front["returns"].dropna()
    price_level = (1 + ret).cumprod() * 1000
    price_level.name = "price_level"

    front = front.join(price_level, how="left")
    return front[["price", "price_level", "returns"]].dropna()


def load_vix_futures_term_structure() -> pd.DataFrame:
    """
    Load VIX futures (VX curve_group) with TtM in years for each contract/date.
    Returns DataFrame indexed by date with multi-row per date (one per contract).
    """
    sec_meta = pd.read_parquet(DATA / "VolatilityIndexFuture_security_meta.parquet")
    hist     = pd.read_parquet(DATA / "VolatilityIndexFuture_historical.parquet")

    vx_secs = sec_meta[sec_meta["curve_group"] == "VX"][
        ["security", "last_trade_date"]].copy()
    vx_secs["last_trade_date"] = pd.to_datetime(vx_secs["last_trade_date"])

    vx_hist = hist[hist["security"].isin(set(vx_secs["security"]))].copy()
    vx_hist["date"] = pd.to_datetime(vx_hist["date"])

    # Merge expiry info
    vx = vx_hist.merge(vx_secs, on="security")
    vx["ttm_years"] = (vx["last_trade_date"] - vx["date"]).dt.days / 365.25

    # Drop negative or zero TtM (expired contracts)
    vx = vx[vx["ttm_years"] > 0].dropna(subset=["price", "ttm_years"])
    return vx[["date", "security", "price", "ttm_years"]].sort_values("date")


def load_vvix() -> pd.Series:
    """Load VVIX (VIX-of-VIX) daily closing index values, converted to monthly units (÷√12)."""
    df = pd.read_csv(DATA / "VolatilityIndexData.csv", parse_dates=["DATE"])
    vvix = (df[df["SECURITY"] == "VVIX Index"]
            .sort_values("DATE")
            .set_index("DATE")["INDEX_VALUE"])
    vvix.index.name = "date"
    return vvix / np.sqrt(12)


def load_vix_spot() -> pd.Series:
    """Load VIX spot index level (annualised %)."""
    df = pd.read_csv(DATA / "VolatilityIndexData.csv", parse_dates=["DATE"])
    vix = (df[df["SECURITY"] == "VIX Index"]
           .sort_values("DATE")
           .set_index("DATE")["INDEX_VALUE"])
    vix.index.name = "date"
    return vix


def load_vix_basis() -> pd.Series:
    """
    Daily VIX basis = front VIX futures price − VIX spot.
    Front contract is the nearest-expiry VX contract with tts >= 0.
    Returns pd.Series indexed by date.
    """
    sec_meta = pd.read_parquet(DATA / "VolatilityIndexFuture_security_meta.parquet")
    hist     = pd.read_parquet(DATA / "VolatilityIndexFuture_historical.parquet")

    vx_secs = sec_meta[sec_meta["curve_group"] == "VX"][
        ["security", "last_trade_date"]].copy()
    vx_secs["last_trade_date"] = pd.to_datetime(vx_secs["last_trade_date"])

    vx_hist = hist[hist["security"].isin(set(vx_secs["security"]))].copy()
    vx_hist["date"] = pd.to_datetime(vx_hist["date"])
    vx = vx_hist.merge(vx_secs, on="security")
    vx["tts"] = np.busday_count(
        vx["date"].values.astype("datetime64[D]"),
        vx["last_trade_date"].values.astype("datetime64[D]"),
    )
    vx = vx[vx["tts"] >= 0]

    front_price = (vx.sort_values(["date", "tts"])
                     .groupby("date")["price"].first())
    front_price.index = pd.to_datetime(front_price.index)
    front_price.index.name = "date"

    spot = load_vix_spot()
    basis = front_price - spot
    basis.name = "vix_basis"
    return basis


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2 — VIX Term Structure Slope
# Sourced from fh_replication (Fassas & Hourvouliades, 2019) — see
# fh_replication/fh_replication.py for the full implementation.
# ═══════════════════════════════════════════════════════════════════════════
# compute_vix_term_slope is imported from fh_replication.fh_replication above.

# ═══════════════════════════════════════════════════════════════════════════
# STEP 3 — Equity Trend Quotient (200-day SMA)
# ═══════════════════════════════════════════════════════════════════════════

def compute_trend_quotient(es: pd.DataFrame) -> pd.Series:
    """
    Trend quotient = P_t / SMA_200(P_t).
    P_t is the reconstructed continuous ES front-month price level.
    Value > 1: price above SMA200 (positive trend).
    Value < 1: price below SMA200 (negative trend).
    """
    sma200 = es["price_level"].rolling(200).mean()
    tq = es["price_level"] / sma200
    tq.name = "trend_quotient"
    return tq


# ═══════════════════════════════════════════════════════════════════════════
# STEP 4 — VVIX Tail-Risk (5-day SMA)
# ═══════════════════════════════════════════════════════════════════════════

def compute_vvix_ma5(vvix: pd.Series) -> pd.Series:
    """5-day simple moving average of VVIX as tail-risk gauge."""
    ma5 = vvix.rolling(5).mean()
    ma5.name = "vvix_ma5"
    return ma5


def compute_vvix_ma10(vvix: pd.Series) -> pd.Series:
    """10-day simple moving average of VVIX as tail-risk gauge."""
    ma10 = vvix.rolling(10).mean()
    ma10.name = "vvix_ma10"
    return ma10


# ═══════════════════════════════════════════════════════════════════════════
# STEP 5 — Bivariate Predictive Regression
# ═══════════════════════════════════════════════════════════════════════════

def build_master_panel(vrp: pd.DataFrame,
                       es: pd.DataFrame,
                       term_slope: pd.Series,
                       trend_q: pd.Series,
                       vvix_ma5: pd.Series) -> pd.DataFrame:
    """
    Assemble daily panel with all signals and forward 20-day return target.

    Forward return: R_{t+20} = (P_{t+20} - P_t) / P_t (cumulative 20-day,
    computed as the 20-day forward rolling product of 1 + daily_return).
    """
    # Cumulative forward 20-day return on ES
    ret = es["returns"]
    fwd_20 = (ret + 1).rolling(20).apply(np.prod, raw=True).shift(-20) - 1
    fwd_20.name = "fwd_20d"

    # Daily ES return (for P&L simulation)
    daily_ret = ret.rename("daily_ret")

    panel = pd.DataFrame({
        "VP":           vrp["VP"],
        "CV":           vrp["CV"],
        "IVar":         vrp["IVar"],
        "term_slope":   term_slope,
        "trend_q":      trend_q,
        "vvix_ma5":     vvix_ma5,
        "fwd_20d":   fwd_20,
        "daily_ret":    daily_ret,
        "price_level":  es["price_level"],
    }).dropna(subset=["VP", "term_slope", "vvix_ma5"])

    return panel.sort_index()



# ═══════════════════════════════════════════════════════════════════════════
# STEP 7 — Strategy Simulation
# ═══════════════════════════════════════════════════════════════════════════

def simulate_strategy(positions: pd.Series,
                      daily_ret: pd.Series,
                      label: str = "",
                      tcost: float = TCOST) -> pd.DataFrame:
    """
    Simulate daily P&L for a given position series.

    Mechanics:
      - Position is set at close of day t, applied to next day t+1 return.
      - T-cost: tcost per unit when position changes.
        Rolling +1 → -1 (or reverse) is one trade (single change of |2 units|),
        as per plan note: "rolling over from +1 to -1 does not incur double cost."
      - P&L_{t+1} = pos_t × r_{t+1} − |Δpos_t| × tcost / 2
        (tcost halved since plan specifies 0.05% per trade, one direction)

    Returns DataFrame with columns [position, daily_ret, gross_pnl, net_pnl,
    cum_gross, cum_net].
    """
    pos    = positions.reindex(daily_ret.index).ffill().fillna(0)
    ret    = daily_ret.reindex(pos.index).fillna(0)

    # Align: pos_t applied to ret_{t+1}
    gross  = pos.shift(1).fillna(0) * ret
    pos_ch = pos.diff().abs().fillna(0)
    cost   = pos_ch * tcost        # 0.05% on each unit of change

    # Flatten +1 → -1 transition: |Δpos| = 2 → cost = 2 × 0.05% = 0.10%
    # Plan says "does not incur double trading cost" → single trade cost = 0.05%
    # Interpretation: a +1 to -1 flip is ONE trade, so cost = 0.05% (not 0.10%).
    # We approximate by capping Δpos cost at 0.05% per day.
    cost = cost.clip(upper=tcost)

    net    = gross - cost
    cum_g  = (1 + gross).cumprod()
    cum_n  = (1 + net).cumprod()

    return pd.DataFrame({
        "position":  pos,
        "daily_ret": ret,
        "gross_pnl": gross,
        "net_pnl":   net,
        "cum_gross": cum_g,
        "cum_net":   cum_n,
    })



def compute_performance_stats(sim: pd.DataFrame, label: str = "") -> dict:
    """
    Annualised return, volatility, Sharpe (3% risk-free), max drawdown.
    Based on net P&L series.
    """
    daily = sim["net_pnl"].dropna()
    n     = len(daily)
    ann   = float((1 + daily).prod() ** (252 / n) - 1) if n > 0 else np.nan
    vol   = float(daily.std() * np.sqrt(252))
    ann_excess = ann-0.03
    sharpe = ann_excess / vol if vol > 0 else np.nan

    cum_val = sim["cum_net"].dropna()
    roll_max = cum_val.cummax()
    dd = (cum_val - roll_max) / roll_max
    max_dd = float(dd.min())

    n_trades = int((sim["position"].diff().abs() > 0).sum())

    return {
        "label":   label,
        "ann_ret": round(ann,    4),
        "ann_vol": round(vol,    4),
        "sharpe":  round(sharpe, 3),
        "max_dd":  round(max_dd, 4),
        "n_obs":   n,
        "n_trades": n_trades,
        "total_ret": round(float(sim["cum_net"].iloc[-1]) - 1, 4),
    }


# ═══════════════════════════════════════════════════════════════════════════
# STEP 8 — Baseline Benchmarks
# ═══════════════════════════════════════════════════════════════════════════

def compute_buy_and_hold(daily_ret: pd.Series) -> pd.Series:
    """Always long (+1) in front-month ES futures."""
    return pd.Series(1, index=daily_ret.index, name="pos_bah")




