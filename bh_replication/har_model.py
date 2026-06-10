"""
har_model.py
============
Estimates the HAR-RV-VIX model (B&H 2014, "Model 8") via OLS with
Newey-West heteroskedasticity-and-autocorrelation-consistent (HAC)
standard errors.

Model 8 (levels):
    RV_{t+1}^(22) = c + α·VIX²_{t}/12
                    + β^m·RV_t^(22)
                    + β^w·RV_t^(5)
                    + β^d·RV_t^(1)
                    + ε_t

Paper's full-sample coefficients (Table 3 / eq.7):
    c=3.730, α=0.108, β^m=0.199, β^w=0.330, β^d=0.107
    Adj R²_OOS ≈ 0.555  (out-of-sample Mincer-Zarnowitz)

Because we use daily squared returns rather than 5-min intraday data,
we expect a lower R² – this module quantifies that gap.
"""

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.stats.sandwich_covariance import cov_hac

NW_LAGS = 44  # Newey-West lags used in the paper


def _nw_se(res, nlags: int = NW_LAGS) -> np.ndarray:
    """Return Newey-West standard errors for an OLS result."""
    cov = cov_hac(res, nlags=nlags)
    return np.sqrt(np.diag(cov))


def estimate_har(panel: pd.DataFrame, label: str = "full sample") -> dict:
    """
    Run Model 8 OLS on the supplied panel slice.
    Returns a dict with coefficients, NW SEs, t-stats, adj-R², and RMSE.
    """
    y = panel["RV22_fwd"]
    X = add_constant(panel[["VIX2_lag", "RV22_lag", "RV5_lag", "RV1_lag"]])

    res = OLS(y, X).fit()
    nw_ses = _nw_se(res)

    out = {
        "label":        label,
        "n":            int(res.nobs),
        "date_start":   panel.index.min().date(),
        "date_end":     panel.index.max().date(),
        "params":       dict(zip(X.columns, res.params)),
        "nw_se":        dict(zip(X.columns, nw_ses)),
        "t_stats":      dict(zip(X.columns, res.params / nw_ses)),
        "adj_r2":       res.rsquared_adj,
        "r2":           res.rsquared,
        "rmse_is":      np.sqrt(res.mse_resid),
        "fitted":       res.fittedvalues,
        "resid":        res.resid,
        "ols_result":   res,
    }
    return out


def out_of_sample_forecast(panel: pd.DataFrame,
                            train_end: str,
                            label: str = "OOS") -> dict:
    """
    Train on rows up to train_end, forecast the remainder.
    Returns Mincer-Zarnowitz R², RMSE, MAE, MAPE for the OOS window.
    """
    train = panel[panel.index <= train_end]
    test  = panel[panel.index >  train_end]

    y_tr = train["RV22_fwd"]
    X_tr = add_constant(train[["VIX2_lag", "RV22_lag", "RV5_lag", "RV1_lag"]])
    res  = OLS(y_tr, X_tr).fit()

    X_te = add_constant(test[["VIX2_lag", "RV22_lag", "RV5_lag", "RV1_lag"]],
                        has_constant="add")
    y_te   = test["RV22_fwd"]
    y_hat  = res.predict(X_te)

    # Mincer-Zarnowitz R²: regress actual on fitted
    mz = OLS(y_te.values, add_constant(y_hat.values)).fit()

    err   = y_te - y_hat
    rmse  = float(np.sqrt((err**2).mean()))
    mae   = float(err.abs().mean())
    mape  = float((err.abs() / y_te).mean())

    # IS stats
    is_res = estimate_har(train, label=f"{label} in-sample")

    return {
        "label":      label,
        "train_end":  train_end,
        "n_train":    len(train),
        "n_test":     len(test),
        "is_adj_r2":  is_res["adj_r2"],
        "oos_mz_r2":  float(mz.rsquared),
        "oos_rmse":   rmse,
        "oos_mae":    mae,
        "oos_mape":   mape,
        "params":     is_res["params"],
        "nw_se":      is_res["nw_se"],
        "t_stats":    is_res["t_stats"],
        "y_test":     y_te,
        "y_hat":      y_hat,
        "test_panel": test,
    }


def rolling_r2(panel: pd.DataFrame,
               window: int = 252,
               step: int = 22) -> pd.DataFrame:
    """
    Compute rolling in-sample Adj-R² and RMSE of Model 8.
    window: trading days; step: stride in days.
    """
    rows = []
    idx  = panel.index
    dates = idx[::step]

    for end_date in dates:
        start_date = end_date - pd.DateOffset(days=int(window * 365 / 252))
        sub = panel[(panel.index >= start_date) & (panel.index <= end_date)]
        if len(sub) < window // 2:
            continue
        r = estimate_har(sub)
        rows.append({
            "date":     end_date,
            "adj_r2":   r["adj_r2"],
            "rmse":     r["rmse_is"],
            "n":        r["n"],
        })
    return pd.DataFrame(rows).set_index("date")


if __name__ == "__main__":
    from data_prep import build_panel
    panel = build_panel()
    res = estimate_har(panel)
    print(f"\n=== HAR-RV-VIX  {res['label']}  (n={res['n']}) ===")
    print(f"  Adj-R²  : {res['adj_r2']:.4f}")
    print(f"  RMSE    : {res['rmse_is']:.4f}")
    print("\n  Coefficients (Newey-West SEs, 44 lags):")
    print(f"  {'Var':<12} {'Coef':>10} {'NW-SE':>10} {'t-stat':>10}")
    for v in res["params"]:
        print(f"  {v:<12} {res['params'][v]:>10.4f} {res['nw_se'][v]:>10.4f} {res['t_stats'][v]:>10.3f}")
