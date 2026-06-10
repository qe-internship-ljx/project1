"""
Section II: Predictive regressions (Exhibit 4).

Tests whether the VIX futures basis forecasts spot VIX changes or
VIX futures price changes.  Replicates equations (1) and (2):

  (1)  dVIX_spot(t->settle)    = a? + a? x basis_t  + u_t
  (2)  dVIX_futures(t->settle) = b? + b? x basis_t  + u_t

where:
  basis_t  = front_futures_price_t ? vix_spot_t   (on the last trading day
              of each month)
  settle   = settlement date of that front contract

Each observation represents one contract: we record the basis on the last
calendar-month-end trading day before settlement, then track that same
contract to its final price (settlement proxy).

Key findings reproduced from the paper:
  ? a?  ~ 0.23, not significant  -> basis has NO forecast power for spot VIX
  ? b?  ~ ?0.79, highly significant -> basis reliably predicts futures roll
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal

import numpy as np
import pandas as pd
import statsmodels.api as sm


# ????????????????????????????????????????????????????????????????????????????
# Data types
# ????????????????????????????????????????????????????????????????????????????

@dataclass
class RegressionResult:
    label: str          # e.g. "eq1_full", "eq2_contango"
    nobs: int
    intercept: float
    intercept_se: float
    slope: float
    slope_se: float
    slope_pvalue: float
    rbar2: float
    dw: float

    def stars(self) -> str:
        p = self.slope_pvalue
        return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


# ????????????????????????????????????????????????????????????????????????????
# Monthly data construction
# ????????????????????????????????????????????????????????????????????????????

def build_monthly_data(
    panel: pd.DataFrame,
    vx_raw: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build monthly observations for the predictive regressions.

    Algorithm
    ---------
    1.  Find the last available trading day within each calendar month.
    2.  On that date, identify the front VIX futures contract (front_sec).
    3.  Look up every subsequent price for that contract in vx_raw.
    4.  Use the *last* available price as the settlement proxy and the
        corresponding VIX spot as the settlement-day spot level.
    5.  Compute basis_t, dspot, and dfutures.

    Parameters
    ----------
    panel  : daily panel from build_daily_panel()
    vx_raw : raw VIX futures data from load_vix_futures()

    Returns
    -------
    DataFrame with one row per monthly observation.
    Columns: date, basis, settle_date, delta_vix, delta_futures, contango
    """
    # ?? Step 1: last trading day of each calendar month ??????????????????
    p = panel.copy()
    p["ym"] = p["date"].dt.to_period("M")
    month_end_dates = p.groupby("ym")["date"].max()

    me = (
        p[p["date"].isin(month_end_dates.values)]
        [["date", "vix_spot", "front_price", "front_sec", "front_expiry"]]
        .dropna(subset=["front_price", "front_sec"])
        .copy()
    )

    # ?? Step 2-5: for each month-end, find settlement values ?????????????
    vx = vx_raw.copy()
    vx["date"] = pd.to_datetime(vx["date"])

    # Build fast date->vix_spot lookup from the panel
    spot_lookup: Dict[pd.Timestamp, float] = dict(
        zip(p["date"], p["vix_spot"])
    )

    rows = []
    for _, row in me.iterrows():
        sec     = row["front_sec"]
        t       = row["date"]
        entry_p = row["front_price"]

        # Prices of this specific contract AFTER the observation date
        fwd = (
            vx.loc[(vx["security"] == sec) & (vx["date"] > t), ["date", "price"]]
            .sort_values("date")
        )
        if fwd.empty:
            continue

        # Last available price = settlement proxy
        settle_date       = fwd.iloc[-1]["date"]
        settle_fut_price  = fwd.iloc[-1]["price"]

        # VIX spot at settlement (fall back to nearest available date)
        settle_vix = spot_lookup.get(settle_date)
        if settle_vix is None:
            after = p[p["date"] >= settle_date]
            if after.empty:
                continue
            settle_vix = after.iloc[0]["vix_spot"]

        basis = entry_p - row["vix_spot"]
        rows.append({
            "date":          t,
            "basis":         basis,
            "settle_date":   settle_date,
            "delta_vix":     settle_vix - row["vix_spot"],
            "delta_futures": settle_fut_price - entry_p,
            "contango":      basis > 0,
        })

    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


# ????????????????????????????????????????????????????????????????????????????
# Regression estimation
# ????????????????????????????????????????????????????????????????????????????

def _fit_ols(
    y: pd.Series,
    x: pd.Series,
    label: str,
) -> RegressionResult:
    """OLS of y on [constant, x]; returns a RegressionResult."""
    mask = y.notna() & x.notna()
    y_, x_ = y[mask], x[mask]
    X = sm.add_constant(x_.values)
    res = sm.OLS(y_.values, X).fit()

    resid = res.resid
    dw = float(np.sum(np.diff(resid) ** 2) / (np.sum(resid ** 2) + 1e-12))

    return RegressionResult(
        label=label,
        nobs=int(res.nobs),
        intercept=float(res.params[0]),
        intercept_se=float(res.bse[0]),
        slope=float(res.params[1]),
        slope_se=float(res.bse[1]),
        slope_pvalue=float(res.pvalues[1]),
        rbar2=float(res.rsquared_adj),
        dw=dw,
    )


def run_regressions(
    monthly: pd.DataFrame,
) -> Dict[str, RegressionResult]:
    """
    Estimate equations (1) and (2) on the full sample and on
    contango / backwardation subsamples.

    Returns a dict keyed by '<eq>_<subset>', e.g. 'eq1_full'.
    """
    subsets = {
        "full":           monthly,
        "contango":       monthly[monthly["contango"]],
        "backwardation":  monthly[~monthly["contango"]],
    }
    results: Dict[str, RegressionResult] = {}
    for subset_name, df in subsets.items():
        results[f"eq1_{subset_name}"] = _fit_ols(
            df["delta_vix"],      df["basis"], f"eq1_{subset_name}"
        )
        results[f"eq2_{subset_name}"] = _fit_ols(
            df["delta_futures"],  df["basis"], f"eq2_{subset_name}"
        )
    return results


# ????????????????????????????????????????????????????????????????????????????
# Display
# ????????????????????????????????????????????????????????????????????????????

def print_exhibit4(results: Dict[str, RegressionResult]) -> None:
    """Print a formatted replication of Exhibit 4."""
    w = 80
    print("=" * w)
    print("EXHIBIT 4 -- Forecast power of the VIX futures basis")
    print("  Eq (1): d_VIX_spot    = a0 + a1*basis + u  [expect a1 insignificant]")
    print("  Eq (2): d_VIX_futures = b0 + b1*basis + u  [expect b1 ~ -0.79 ***]")
    print("  */**/*** = significant at 10/5/1 %")
    print("=" * w)
    hdr = f"{'Dep. var.':<16} {'Sample':<16} {'Const':>14} {'Basis coef':>16} {'RBAR2':>7} {'DW':>6} {'N':>5}"
    print(hdr)
    print("-" * w)
    order = [
        ("d_VIX_spot",    "eq1", ["full", "contango", "backwardation"]),
        ("d_VIX_futures", "eq2", ["full", "contango", "backwardation"]),
    ]
    for dep_label, eq_key, subsets in order:
        for s in subsets:
            r = results[f"{eq_key}_{s}"]
            const_str = f"{r.intercept:+.3f} ({r.intercept_se:.3f})"
            slope_str = f"{r.slope:+.3f}{r.stars()} ({r.slope_se:.3f})"
            print(
                f"{dep_label:<16} {s:<16} {const_str:>14} {slope_str:>16} "
                f"{r.rbar2:>7.3f} {r.dw:>6.2f} {r.nobs:>5}"
            )
        print()
