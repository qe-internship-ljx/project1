"""
Section III: Dynamic out-of-sample hedge ratio (Equations 3 & 4).

Equation (3) -- daily VIX futures price sensitivity to equity returns:

    dVIX_futures_t = b? + b??SPRET_t + b??(SPRET_t x TTS_t) + u_t

  where
    dVIX_futures_t = daily absolute price change (VIX points) of the
                     trade-eligible front contract (>=10 biz days to settle)
    SPRET_t        = daily percentage return of front ES futures
    TTS_t          = business days to settlement of that VIX contract

  b? < 0  (VIX futures move inversely to equities)
  b? > 0  (sensitivity attenuates for contracts further from settlement,
            consistent with mean-reversion in VIX)

Equation (4) -- hedge ratio (mini-S&P contracts per VIX contract):

    HR_t = (b??1000 + b??TTS_{t-1}?1000) / (0.01?ES_{t-1}?50)

  Numerator   = expected $ gain/loss on 1 VIX futures contract for 1% S&P move
  Denominator = $ gain/loss on 1 ES mini contract for 1% S&P move

Out-of-sample protocol (mirrors the paper exactly):
  ? Training window starts:  2006-01-01
  ? Trading starts:          2007-01-01
  ? Each trading day t: refit equation (3) on all data from 2006-01-01
    through t-1, then compute HR_t from the updated coefficients.
  ? HR is locked at trade entry and NOT updated while the trade is live.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ????????????????????????????????????????????????????????????????????????????
# Build the regression input series
# ????????????????????????????????????????????????????????????????????????????

def build_hedge_regression_data(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Construct the daily time series needed for Equation (3).

    The dependent variable is the *absolute* price change of the
    trade-eligible VIX futures contract (VIX points, not percent).
    On contract-roll days the change is computed within the new
    contract (same-contract price difference), which is equivalent
    to using the `returns` column already stored in the panel.

    Returns
    -------
    DataFrame with columns:
        date, delta_vix_fut, spret, tts
    Only rows where all three are non-NaN are included.
    """
    df = panel[["date", "trade_price", "trade_ret", "trade_tts",
                "es_ret", "trade_sec"]].copy()

    # Absolute VIX futures price change in VIX points (not percent):
    #   price_{t-1} = trade_price / (1 + trade_ret)  ->  delta = trade_ret * price_{t-1}
    # This is exact because the data stores 'returns' = (P_t - P_{t-1}) / P_{t-1}.
    df["delta_vix_fut"] = (df["trade_price"] / (1 + df["trade_ret"])) * df["trade_ret"]

    # CRITICAL: convert ES returns from decimal to *percentage* units.
    # The paper's Equation 3 uses SPRET as a percentage (1% move = 1.0), so
    # beta_1 comes out as ~-0.717 (VIX pts per 1 pct-pt S&P move).
    # This must match the 0.01 factor in the Equation 4 denominator.
    df["spret"] = df["es_ret"] * 100
    df = df.rename(columns={"trade_tts": "tts"})

    out = df[["date", "delta_vix_fut", "spret", "tts"]].dropna()
    return out.reset_index(drop=True)


# ????????????????????????????????????????????????????????????????????????????
# OLS via normal equations (fast, no external dependency)
# ????????????????????????????????????????????????????????????????????????????

def _ols_coefs(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Return OLS coefficients b = (X?X)?1 X?y.
    Returns array of NaNs if the system is singular or underdetermined.
    """
    try:
        return np.linalg.solve(X.T @ X, X.T @ y)
    except np.linalg.LinAlgError:
        return np.full(X.shape[1], np.nan)


# ????????????????????????????????????????????????????????????????????????????
# Rolling out-of-sample hedge ratio
# ????????????????????????????????????????????????????????????????????????????

def compute_oos_hedge_ratios(
    panel: pd.DataFrame,
    train_start: str = "2006-01-01",
    trade_start: str = "2007-01-01",
) -> pd.DataFrame:
    """
    For every trading day from trade_start onward, estimate Equation (3)
    on all available data from train_start through the *previous* day,
    then compute the hedge ratio HR_t (Equation 4).

    Parameters
    ----------
    panel       : daily panel from build_daily_panel()
    train_start : first date included in the training window
    trade_start : first date for which an out-of-sample HR is computed

    Returns
    -------
    DataFrame with columns: date, beta0, beta1, beta2, hr
      hr = HR_t = number of ES mini contracts to short/long per VIX contract.
      Hedge direction (sign of actual ES position) is applied in simulator.py.
    """
    reg_data = build_hedge_regression_data(panel)
    reg_data = reg_data[reg_data["date"] >= pd.Timestamp(train_start)]

    trading_days = panel.loc[
        panel["date"] >= pd.Timestamp(trade_start), "date"
    ].sort_values().values

    rows = []
    for date in trading_days:
        # Training window: [train_start, date)
        train = reg_data[reg_data["date"] < date]
        if len(train) < 10:          # need minimum history
            rows.append({"date": date, "beta0": np.nan,
                         "beta1": np.nan, "beta2": np.nan, "hr": np.nan})
            continue

        # Equation (3): dVIX_fut = b? + b??SPRET + b??(SPRET?TTS)
        y = train["delta_vix_fut"].values
        X = np.column_stack([
            np.ones(len(train)),          # b? (constant)
            train["spret"].values,        # b?
            train["spret"].values * train["tts"].values,   # b?
        ])
        beta = _ols_coefs(X, y)
        b0, b1, b2 = beta

        # Yesterday's ES price and TTS -- used for Equation (4)
        prev = panel[panel["date"] < date].iloc[-1] if len(panel[panel["date"] < date]) else None
        if prev is None or np.isnan(prev["es_price"]) or np.isnan(prev["trade_tts"]):
            hr = np.nan
        else:
            es_prev  = prev["es_price"]
            tts_prev = prev["trade_tts"]
            # Equation (4):
            #   Numerator  = expected $ change on 1 VIX contract per 1% S&P move
            #   Denominator= $ change on 1 ES contract per 1% S&P move
            numerator   = b1 * 1_000 + b2 * tts_prev * 1_000
            denominator = 0.01 * es_prev * 50
            hr = numerator / denominator if abs(denominator) > 1e-6 else np.nan

        rows.append({"date": pd.Timestamp(date),
                     "beta0": b0, "beta1": b1, "beta2": b2, "hr": hr})

    return pd.DataFrame(rows)


# ????????????????????????????????????????????????????????????????????????????
# Full-sample in-sample estimate (for display / validation only)
# ????????????????????????????????????????????????????????????????????????????

def fit_equation3_insample(panel: pd.DataFrame) -> dict:
    """
    Fit Equation (3) on the full sample for reporting purposes.
    Returns a dict with keys: beta, se, rbar2, nobs, dw.
    """
    import statsmodels.api as sm

    reg = build_hedge_regression_data(panel).dropna()
    y = reg["delta_vix_fut"].values
    X = sm.add_constant(np.column_stack([
        reg["spret"].values,
        reg["spret"].values * reg["tts"].values,
    ]))
    res = sm.OLS(y, X).fit()
    resid = res.resid
    dw = float(np.sum(np.diff(resid) ** 2) / (np.sum(resid ** 2) + 1e-12))
    return {
        "beta":  res.params,
        "se":    res.bse,
        "rbar2": res.rsquared_adj,
        "nobs":  int(res.nobs),
        "dw":    dw,
    }
