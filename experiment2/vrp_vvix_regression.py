"""
vrp_vvix_regression.py
======================
Regress daily VRP (VP) onto the 5-day rolling mean of VVIX.

Model:  VP_t = α + β · VVIX_MA5_t + ε

Newey-West SEs with 20 lags to account for VRP autocorrelation (HAR model
residuals carry multi-day persistence).  Output: console table + scatter plot.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

ROOT    = Path(__file__).parent
OUTPUT  = ROOT / "output"
OUTPUT.mkdir(exist_ok=True)
DATA    = ROOT.parent / "data"
VRP_OUT = ROOT.parent / "vrp_experiment" / "output"

sys.path.insert(0, str(ROOT.parent / "bh_replication"))
sys.path.insert(0, str(ROOT.parent))
from har_model import _nw_se

NW_LAGS = 20


# ── data loading ─────────────────────────────────────────────────────────────

def load_vrp() -> pd.Series:
    pq = VRP_OUT / "production_loop_full.parquet"
    csv = VRP_OUT / "production_loop_full.csv"
    if pq.exists():
        df = pd.read_parquet(pq)
    else:
        df = pd.read_csv(csv, parse_dates=["date"]).set_index("date")
    return df["VP"].dropna()


def load_vvix_ma5() -> pd.Series:
    df = pd.read_csv(DATA / "VolatilityIndexData.csv", parse_dates=["DATE"])
    vvix = (df[df["SECURITY"] == "VVIX Index"]
            .sort_values("DATE")
            .set_index("DATE")["INDEX_VALUE"])
    vvix.index.name = "date"
    ma5 = vvix.rolling(5).mean()
    ma5.name = "vvix_ma5"
    return ma5.dropna()


# ── regression ───────────────────────────────────────────────────────────────

def ols_nw(y: pd.Series, X: pd.DataFrame, nlags: int = NW_LAGS) -> dict:
    X_c  = add_constant(X, has_constant="skip")
    res  = OLS(y, X_c).fit()
    nw   = _nw_se(res, nlags=nlags)
    names = list(X_c.columns)
    params = dict(zip(names, res.params))
    se     = dict(zip(names, nw))
    tstat  = {k: params[k] / se[k] for k in names}
    return {
        "n":      int(res.nobs),
        "params": params,
        "nw_se":  se,
        "t_stat": tstat,
        "r2":     float(res.rsquared),
        "adj_r2": float(res.rsquared_adj),
    }


def main():
    vp      = load_vrp()
    vvix_m5 = load_vvix_ma5()

    panel = pd.DataFrame({"VP": vp, "vvix_ma5": vvix_m5}).dropna()
    print(f"Sample: {panel.index[0].date()} – {panel.index[-1].date()}  "
          f"(N = {len(panel):,} days)\n")

    res = ols_nw(panel["VP"], panel[["vvix_ma5"]])

    # ── print results ─────────────────────────────────────────────────────────
    hdr  = f"{'':12s}  {'Coef':>12s}  {'NW SE':>12s}  {'t-stat':>10s}"
    sep  = "-" * len(hdr)
    rows = []
    for k in ["const", "vvix_ma5"]:
        rows.append(
            f"{k:12s}  {res['params'][k]:12.6f}  {res['nw_se'][k]:12.6f}  "
            f"{res['t_stat'][k]:10.3f}"
        )

    print("OLS: VP ~ VVIX_MA5  (Newey-West SEs, 20 lags)")
    print(sep)
    print(hdr)
    print(sep)
    for r in rows:
        print(r)
    print(sep)
    print(f"{'R²':12s}  {res['r2']:12.6f}")
    print(f"{'Adj R²':12s}  {res['adj_r2']:12.6f}")
    print(f"{'N':12s}  {res['n']:12,d}")

    # ── scatter plot ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(panel["vvix_ma5"], panel["VP"],
               s=4, alpha=0.35, color="steelblue", linewidths=0)

    # OLS fit line
    x_grid = np.linspace(panel["vvix_ma5"].min(), panel["vvix_ma5"].max(), 200)
    y_fit  = res["params"]["const"] + res["params"]["vvix_ma5"] * x_grid
    ax.plot(x_grid, y_fit, color="firebrick", lw=1.8,
            label=f"β = {res['params']['vvix_ma5']:.4f}  "
                  f"(t = {res['t_stat']['vvix_ma5']:.2f})\n"
                  f"R² = {res['r2']:.4f}")

    ax.axhline(0, color="black", lw=0.6, ls="--")
    ax.set_xlabel("VVIX 5-day MA", fontsize=11)
    ax.set_ylabel("Daily VRP (VP)", fontsize=11)
    ax.set_title("VRP regressed on VVIX 5-day MA", fontsize=12)
    ax.legend(fontsize=9)
    plt.tight_layout()

    out_path = OUTPUT / "vrp_on_vvix_ma5.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nPlot saved: {out_path}")


if __name__ == "__main__":
    main()
