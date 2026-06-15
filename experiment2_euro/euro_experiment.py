"""
euro_experiment.py
==================
Replicates the experiment2 expanding-window regression approach for European
markets using Euro Stoxx 50 (FX futures), VSTOXX (V2X), and VV2TX (VSTOXX
of VSTOXX).  All code is written from scratch; no project modules are imported.

Models evaluated:
  Base       -- Univariate Euro-VRP
  Model_A    -- VRP + VSTOXX term structure slope
  Model_C    -- VRP + VV2TX 5-day MA
  Model_VV2X -- Univariate VV2TX MA5

Euro VRP is computed via a 500-day rolling HAR production loop:
  IVar = SX5E 1-month variance swap IV²/12 (≥ 2008-11-04); V2X²/12 before that.
  Physical CV = HAR forecast from rolling OLS: RV22_{t+1} ~ IVar + RV22 + RV5 + RV1
  VP_t = IVar_t − CV_t

VSTOXX term slope: daily cross-sectional OLS  price ~ TtM  on VSTOXX (DI) futures.
VV2TX MA5: 5-day simple moving average of VV2TX index.

Two EW threshold variants per model (fixed and rolling-mu) following experiment2 spec.
All plots follow plot.md format.  Positions and betas are cached in output/regression_cache/.

OOS: 2012-01-01 to latest data.  Min training window: 500 days.  NW-HAC lags: 20.
T-gate: |t| > 1.28 on all non-intercept betas.  Slippage: 0.05%.
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

# --- Paths --------------------------------------------------------------------
ROOT      = Path(__file__).parent
OUTPUT    = ROOT / "output"
CACHE_DIR = OUTPUT / "regression_cache"
DATA      = ROOT.parent / "data"
DIR_EW    = OUTPUT / "expanding_window"

EW_MODEL_DIR = {
    "Base":       DIR_EW / "VRP",
    "Model_A":    DIR_EW / "VRP + Term Slope",
    "Model_C":    DIR_EW / "VRP + VV2TX MA5",
    "Model_VV2X": DIR_EW / "VV2TX MA5",
}
DIR_EW_CMP = DIR_EW / "comparisons"

for _d in [CACHE_DIR, DIR_EW_CMP] + list(EW_MODEL_DIR.values()):
    _d.mkdir(parents=True, exist_ok=True)

# --- Constants ----------------------------------------------------------------
OOS_START    = "2012-01-01"
MIN_WIN      = 500
T_THRESH     = 1.28
NW_LAGS      = 20
TCOST        = 0.0005
ROLL_VRP_WIN = 500

DELTAS    = [0.002, 0.005, 0.0075, 0.010]
DELTA_LBL = ["d=0.2%", "d=0.5%", "d=0.75%", "d=1.0%"]
BAH_COLOR = "#d62728"

MODEL_FEATURES = {
    "Base":       ["VP"],
    "Model_A":    ["VP", "term_slope"],
    "Model_C":    ["VP", "vv2tx_ma5"],
    "Model_VV2X": ["vv2tx_ma5"],
}

MODEL_LABEL = {
    "Base":       "Base -- Univariate Euro-VRP",
    "Model_A":    "Model A -- VRP + VSTOXX Term Slope",
    "Model_C":    "Model C -- VRP + VV2TX MA5",
    "Model_VV2X": "Model VV2TX -- Univariate VV2TX MA5",
}

MODEL_PALETTE = {
    "Base":       ["#08306b", "#2171b5", "#6baed6", "#9ecae1"],
    "Model_A":    ["#00441b", "#238b45", "#74c476", "#c7e9c0"],
    "Model_C":    ["#3f007d", "#6a51a3", "#9e9ac8", "#dadaeb"],
    "Model_VV2X": ["#7f2704", "#d94801", "#fd8d3c", "#fdbe85"],
}

FEAT_DISPLAY = {
    "VP":         "VRP",
    "term_slope": "Term Slope",
    "vv2tx_ma5":  "VV2TX MA5",
}

LINESTYLE = ["-", "--", "-.", ":"]


# =============================================================================
# 1. DATA LOADING
# =============================================================================

def load_stoxx_front_month() -> pd.DataFrame:
    """
    Build continuous Euro Stoxx 50 (FX futures, curve_group='FX') daily series.
    Front-month = contract with nearest expiry on each date.
    Returns DataFrame indexed by date with columns [price, price_level, returns].
    price_level is reconstructed from cumulative returns, rebased to 1000.
    """
    sec_meta = pd.read_parquet(DATA / "EquityFuture_security_meta.parquet")
    hist     = pd.read_parquet(DATA / "EquityFuture_historical.parquet")

    fx_tickers = sec_meta[sec_meta["curve_group"] == "FX"]["security"].tolist()
    fx = hist[hist["security"].isin(fx_tickers)].copy()
    fx["date"] = pd.to_datetime(fx["date"])

    meta_fx = sec_meta[sec_meta["curve_group"] == "FX"][
        ["security", "expiry_yearmonth"]].copy()
    meta_fx["expiry_date"] = pd.to_datetime(
        meta_fx["expiry_yearmonth"], format="%Y-%m")
    fx = fx.merge(meta_fx[["security", "expiry_date"]], on="security")

    fx = fx.sort_values(["date", "expiry_date"])
    front = (fx.groupby("date").first().reset_index()
               [["date", "price", "returns"]].dropna(subset=["returns"]))
    front = front.sort_values("date").set_index("date")

    ret = front["returns"].dropna()
    price_level = (1 + ret).cumprod() * 1000
    price_level.name = "price_level"
    front = front.join(price_level, how="left")
    return front[["price", "price_level", "returns"]].dropna()


def load_v2x_spot() -> pd.Series:
    """Load V2X (VSTOXX spot) index in annual vol % terms."""
    df = pd.read_csv(DATA / "VolatilityIndexData.csv", parse_dates=["DATE"])
    s = (df[df["SECURITY"] == "V2X Index"]
         .sort_values("DATE")
         .set_index("DATE")["INDEX_VALUE"])
    s.index.name = "date"
    return s


def load_vstoxx_futures() -> pd.DataFrame:
    """
    Load VSTOXX futures (curve_group='DI') with time-to-maturity in years.
    Returns DataFrame with columns [date, security, price, ttm_years].
    Only rows with ttm_years > 0 (non-expired contracts) are kept.
    """
    sec_meta = pd.read_parquet(DATA / "VolatilityIndexFuture_security_meta.parquet")
    hist     = pd.read_parquet(DATA / "VolatilityIndexFuture_historical.parquet")

    di_secs = sec_meta[sec_meta["curve_group"] == "DI"][
        ["security", "last_trade_date"]].copy()
    di_secs["last_trade_date"] = pd.to_datetime(di_secs["last_trade_date"])

    di_hist = hist[hist["security"].isin(set(di_secs["security"]))].copy()
    di_hist["date"] = pd.to_datetime(di_hist["date"])

    di = di_hist.merge(di_secs, on="security")
    di["ttm_years"] = (di["last_trade_date"] - di["date"]).dt.days / 365.25
    di = di[di["ttm_years"] > 0].dropna(subset=["price", "ttm_years"])
    return di[["date", "security", "price", "ttm_years"]].sort_values("date")


def load_vv2tx() -> pd.Series:
    """Load VV2TX (VSTOXX of VSTOXX) daily index values."""
    df = pd.read_csv(DATA / "VolatilityIndexData.csv", parse_dates=["DATE"])
    s = (df[df["SECURITY"] == "VV2TX Index"]
         .sort_values("DATE")
         .set_index("DATE")["INDEX_VALUE"])
    s.index.name = "date"
    return s


# =============================================================================
# 2. EURO VRP  (HAR rolling-window production loop)
# =============================================================================

def compute_euro_ivar(v2x: pd.Series) -> pd.Series:
    """
    Implied variance from V2X: IVar = V2X²/12  (monthly %²-units).
    V2X is used throughout -- no variance swap mixing, exactly mirroring the
    US experiment's use of VIX²/12 exclusively.
    """
    ivar = (v2x ** 2 / 12.0).rename("IVar")
    return ivar


def compute_euro_vrp(returns: pd.Series, ivar: pd.Series,
                     window: int = ROLL_VRP_WIN) -> pd.DataFrame:
    """
    500-day rolling HAR production loop for Euro VRP.
    Matches vrp_experiment/experiment.py production_loop() exactly:

      Panel construction:
        RV22_fwd[t] = RV22[t+22]  -- forward 22-day realized variance (target)
        IVar_lag[t] = IVar[t-1]   -- lagged implied variance predictor
        RV22_lag[t] = RV22[t-1]
        RV5_lag[t]  = RV5[t-1]
        RV1_lag[t]  = RV1[t-1]

      At each step i in [window+22, N-22):
        Train: RV22_fwd[s] ~ const + IVar_lag[s] + RV22_lag[s] + RV5_lag[s] + RV1_lag[s]
               for s in [i-window-22, i-22]   (strict OOS: last label RV22_fwd[i-23]
               uses returns through day i-1, all fully realised before prediction date i)
        Predict CV_i = beta @ [1, IVar_lag[i], RV22_lag[i], RV5_lag[i], RV1_lag[i]]
        VP_i = IVar[i] - CV_i   (current IVar minus forecasted conditional variance)

    RV22 = 22-day rolling sum of (r*100)^2  (monthly realized variance proxy)
    RV5  = 5-day rolling sum
    RV1  = daily squared return in %^2
    """
    r100 = (returns * 100) ** 2
    rv1  = r100
    rv5  = r100.rolling(5).sum()
    rv22 = r100.rolling(22).sum()

    df = pd.DataFrame({
        "IVar":     ivar,
        "IVar_lag": ivar.shift(1),
        "rv22":     rv22,
        "rv22_lag": rv22.shift(1),
        "rv5_lag":  rv5.shift(1),
        "rv1_lag":  rv1.shift(1),
    }).dropna()

    df["rv22_fwd"] = df["rv22"].shift(-22)   # forward 22-day RV target

    feats  = ["IVar_lag", "rv22_lag", "rv5_lag", "rv1_lag"]
    N      = len(df)
    vp_arr = np.full(N, np.nan)
    cv_arr = np.full(N, np.nan)

    for i in range(window + 22, N - 22):
        train = df.iloc[i - window - 22 : i - 22].dropna(subset=["rv22_fwd"])
        if len(train) < 100:
            continue
        y_tr = train["rv22_fwd"].values
        X_tr = np.column_stack(
            [np.ones(len(train))] + [train[f].values for f in feats])
        try:
            beta = np.linalg.lstsq(X_tr, y_tr, rcond=None)[0]
        except Exception:
            continue

        x_i = np.array([1.0] + [float(df[f].iat[i]) for f in feats])
        cv  = float(x_i @ beta)
        cv_arr[i] = cv
        vp_arr[i] = float(df["IVar"].iat[i]) - cv   # current IVar, not lagged

    result = pd.DataFrame({
        "VP":   vp_arr,
        "CV":   cv_arr,
        "IVar": df["IVar"].values,
    }, index=df.index)
    return result.dropna(subset=["VP"])


# =============================================================================
# 3. SIGNALS
# =============================================================================

def compute_vstoxx_term_slope(vstoxx_df: pd.DataFrame) -> pd.Series:
    """
    Fassas & Hourvouliades (2019) style cross-sectional OLS on VSTOXX futures:
        price_i = α + β × TtM_i + ε_i
    β > 0: contango; β < 0: backwardation.
    Requires ≥ 3 contracts per day.
    Returns Series of slopes indexed by date.
    """
    slopes = {}
    for date, grp in vstoxx_df.groupby("date"):
        grp = grp.dropna(subset=["price", "ttm_years"])
        if len(grp) < 3:
            continue
        y = grp["price"].values
        X = np.column_stack([np.ones(len(grp)), grp["ttm_years"].values])
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            slopes[date] = float(beta[1])
        except Exception:
            pass
    s = pd.Series(slopes, name="term_slope")
    s.index = pd.to_datetime(s.index)
    s.index.name = "date"
    return s.sort_index()


def compute_vv2tx_ma5(vv2tx: pd.Series) -> pd.Series:
    """5-day simple moving average of VV2TX (VSTOXX-of-VSTOXX) index."""
    ma5 = vv2tx.rolling(5).mean()
    ma5.name = "vv2tx_ma5"
    return ma5


# =============================================================================
# 4. PANEL
# =============================================================================

def build_euro_panel(vrp: pd.DataFrame, stoxx: pd.DataFrame,
                     term_slope: pd.Series,
                     vv2tx_ma5: pd.Series) -> pd.DataFrame:
    """
    Assemble the daily signal panel.  No joint dropna is applied here -- each
    model's run_expanding_window call drops NaN on its own feature columns.

    Forward 20-day return: cumulative product of daily returns shifted -20.
    """
    ret   = stoxx["returns"]
    fwd20 = (ret + 1).rolling(20).apply(np.prod, raw=True).shift(-20) - 1
    fwd20.name = "fwd_ret_20"

    panel = pd.DataFrame({
        "VP":          vrp["VP"],
        "CV":          vrp["CV"],
        "IVar":        vrp["IVar"],
        "term_slope":  term_slope,
        "vv2tx_ma5":   vv2tx_ma5,
        "fwd_ret_20":  fwd20,
        "daily_ret":   ret.rename("daily_ret"),
        "price_level": stoxx["price_level"],
    })
    return panel.sort_index()


# =============================================================================
# 5. NEWEY-WEST HAC STANDARD ERRORS
# =============================================================================

def _nw_se(res, nlags: int) -> np.ndarray:
    """
    Newey-West HAC standard errors (Bartlett kernel) for a statsmodels OLS result.
    Returns array of length k (one SE per coefficient, including intercept).

    V_NW = (X'X)^{-1}  S  (X'X)^{-1}
    S = Σ_t (x_t e_t)(x_t e_t)' + Σ_{l=1}^{L} w_l [Σ_t(x_t e_t)(x_{t-l} e_{t-l})' + T]
    w_l = 1 − l/(L+1)   (Bartlett)
    """
    X  = np.asarray(res.model.exog)   # n × k
    e  = np.asarray(res.resid)        # n
    Xe = X * e[:, None]               # n × k  score vectors

    S = Xe.T @ Xe                     # lag-0 outer product
    for l in range(1, nlags + 1):
        w  = 1.0 - l / (nlags + 1)
        G  = Xe[l:].T @ Xe[:-l]      # k × k cross-lag
        S += w * (G + G.T)

    XtX_inv = np.linalg.inv(X.T @ X)
    V = XtX_inv @ S @ XtX_inv
    return np.sqrt(np.maximum(np.diag(V), 0.0))


# =============================================================================
# 6. EXPANDING-WINDOW REGRESSION
# =============================================================================

def run_expanding_window(panel: pd.DataFrame, model: str, delta: float,
                         t_threshold: float = T_THRESH) -> pd.Series:
    """
    Expanding OLS -- fixed-delta threshold.

    At prediction row i (>= OOS_START and >= MIN_WIN+20):
      • Training: sub.iloc[0 : i-20]  -- all history, OOS gap = 20 rows
      • Gate: all non-intercept |t_NW| > t_threshold  (if any fail -> flat)
      • Position: +1 if ŷ > delta, -1 if ŷ < -delta, else 0

    Results are cached in CACHE_DIR as parquet keyed by model/delta/OOS_START.
    """
    feat_cols  = MODEL_FEATURES[model]
    cache_name = (f"euro_pos_EW_{model}_d{int(delta*10000)}bps"
                  f"_t{int(t_threshold*100)}_oos{OOS_START}.parquet")
    cache_path = CACHE_DIR / cache_name
    if cache_path.exists():
        s = pd.read_parquet(cache_path).squeeze()
        s.name = f"pos_EW_{model}_d{delta}"
        return s

    sub = panel.dropna(subset=feat_cols + ["fwd_ret_20"]).copy()
    N   = len(sub)
    pos = pd.Series(0.0, index=sub.index, name=f"pos_EW_{model}_d{delta}")

    oos_idx = sub.index.searchsorted(pd.Timestamp(OOS_START))
    start_i = max(MIN_WIN + 20, oos_idx)

    for i in range(start_i, N):
        train  = sub.iloc[0 : i - 20]
        X_tr   = add_constant(train[feat_cols], has_constant="skip")
        res    = OLS(train["fwd_ret_20"], X_tr).fit()
        nw     = _nw_se(res, nlags=NW_LAGS)
        t_vals = res.params.values[1:] / nw[1:]
        if not np.all(np.abs(t_vals) > t_threshold):
            continue
        row    = sub.iloc[[i]][feat_cols].copy()
        row.insert(0, "const", 1.0)
        y_hat  = float(res.predict(row).iloc[0])
        if   y_hat >  delta: pos.iloc[i] =  1.0
        elif y_hat < -delta: pos.iloc[i] = -1.0

    pos.to_frame().to_parquet(cache_path)
    return pos


def run_expanding_window_rolmu(panel: pd.DataFrame, model: str, delta: float,
                                t_threshold: float = T_THRESH,
                                rolling_window: int = 500) -> pd.Series:
    """
    Expanding OLS -- rolling-mean-adjusted threshold.

    μ_i = mean of fwd_ret_20 over the most recent `rolling_window` training rows
    Position: +1 if ŷ > μ + delta, -1 if ŷ < μ − delta, else 0.
    Allows two-sided signals benchmarked against recent average returns.
    """
    feat_cols  = MODEL_FEATURES[model]
    cache_name = (f"euro_pos_EW_rolmu_{model}_d{int(delta*10000)}bps"
                  f"_t{int(t_threshold*100)}_rw{rolling_window}_oos{OOS_START}.parquet")
    cache_path = CACHE_DIR / cache_name
    if cache_path.exists():
        s = pd.read_parquet(cache_path).squeeze()
        s.name = f"pos_EWrm_{model}_d{delta}"
        return s

    sub  = panel.dropna(subset=feat_cols + ["fwd_ret_20"]).copy()
    N    = len(sub)
    pos  = pd.Series(0.0, index=sub.index, name=f"pos_EWrm_{model}_d{delta}")
    fwd  = sub["fwd_ret_20"].values

    oos_idx = sub.index.searchsorted(pd.Timestamp(OOS_START))
    start_i = max(MIN_WIN + 20, oos_idx)

    for i in range(start_i, N):
        train  = sub.iloc[0 : i - 20]
        X_tr   = add_constant(train[feat_cols], has_constant="skip")
        res    = OLS(train["fwd_ret_20"], X_tr).fit()
        nw     = _nw_se(res, nlags=NW_LAGS)
        t_vals = res.params.values[1:] / nw[1:]
        if not np.all(np.abs(t_vals) > t_threshold):
            continue

        lo = max(0, i - 20 - rolling_window)
        mu = float(np.mean(fwd[lo : i - 20]))

        row   = sub.iloc[[i]][feat_cols].copy()
        row.insert(0, "const", 1.0)
        y_hat = float(res.predict(row).iloc[0])
        if   y_hat > mu + delta: pos.iloc[i] =  1.0
        elif y_hat < mu - delta: pos.iloc[i] = -1.0

    pos.to_frame().to_parquet(cache_path)
    return pos


def compute_ew_betas(panel: pd.DataFrame, model: str) -> pd.DataFrame:
    """
    Record how the OLS coefficients and NW t-stats evolve as the expanding
    window grows.  Cached per model.

    Columns: beta_alpha, se_alpha, t_alpha,
             beta_VP (first predictor), se_VP, t_VP,
             beta_sec, se_sec, t_sec  (only for bivariate models),
             r2_insample (all models).

    Note: 'beta_VP' refers to the first non-intercept predictor regardless of
    whether it is VP or vv2tx_ma5 (for Model_VV2X). Labels in plots use
    FEAT_DISPLAY to show the correct name.
    If cached parquet lacks r2_insample, recompute.
    """
    feat_cols = MODEL_FEATURES[model]
    bivariate = len(feat_cols) >= 2
    cache_path = CACHE_DIR / f"euro_betas_EW_{model}_oos{OOS_START}.parquet"
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        if "r2_insample" in df.columns:
            return df
        cache_path.unlink()   # recompute — missing r2_insample column

    print(f"    Computing beta evolution for {model} (one-time)...")
    sub     = panel.dropna(subset=feat_cols + ["fwd_ret_20"]).copy()
    N       = len(sub)
    oos_idx = sub.index.searchsorted(pd.Timestamp(OOS_START))
    start_i = max(MIN_WIN + 20, oos_idx)

    records = []
    for i in range(start_i, N):
        train  = sub.iloc[0 : i - 20]
        X_tr   = add_constant(train[feat_cols], has_constant="skip")
        res    = OLS(train["fwd_ret_20"], X_tr).fit()
        nw     = _nw_se(res, nlags=NW_LAGS)

        b_a, se_a = float(res.params.iloc[0]), float(nw[0])
        b_v, se_v = float(res.params.iloc[1]), float(nw[1])
        rec = {
            "beta_alpha":  b_a,
            "se_alpha":    se_a,
            "t_alpha":     b_a / se_a if se_a > 0 else 0.0,
            "beta_VP":     b_v,
            "se_VP":       se_v,
            "t_VP":        b_v / se_v if se_v > 0 else 0.0,
            "r2_insample": float(res.rsquared),
        }
        if bivariate:
            b_s, se_s = float(res.params.iloc[2]), float(nw[2])
            rec.update({
                "beta_sec": b_s,
                "se_sec":   se_s,
                "t_sec":    b_s / se_s if se_s > 0 else 0.0,
            })
        records.append(rec)

    df = pd.DataFrame(records, index=sub.index[start_i:])
    df.to_parquet(cache_path)
    return df


# =============================================================================
# 7. STRATEGY SIMULATION
# =============================================================================

def simulate_strategy(positions: pd.Series, daily_ret: pd.Series,
                      tcost: float = TCOST) -> pd.DataFrame:
    """
    Daily P&L simulation.
    • Position set at close of t; earns return at t+1 (pos.shift(1)).
    • T-cost: 0.05% per unit of |Δpos|, capped at 0.05% per day
      (a ±1->∓1 flip costs 0.05%, not 0.10% -- consistent with experiment2).
    """
    pos   = positions.reindex(daily_ret.index).ffill().fillna(0)
    ret   = daily_ret.reindex(pos.index).fillna(0)
    gross = pos.shift(1).fillna(0) * ret
    cost  = pos.diff().abs().fillna(0).clip(upper=1.0) * tcost
    net   = gross - cost
    return pd.DataFrame({
        "position":  pos,
        "daily_ret": ret,
        "gross_pnl": gross,
        "net_pnl":   net,
        "cum_gross": (1 + gross).cumprod(),
        "cum_net":   (1 + net).cumprod(),
    })


def compute_performance_stats(sim: pd.DataFrame, label: str = "") -> dict:
    """Annualised return, vol, Sharpe (0% rf), max drawdown, trade count."""
    daily = sim["net_pnl"].dropna()
    n     = len(daily)
    ann   = float((1 + daily).prod() ** (252 / n) - 1) if n > 0 else np.nan
    vol   = float(daily.std() * np.sqrt(252))
    sharpe = ann / vol if vol > 0 else np.nan
    cum   = sim["cum_net"].dropna()
    max_dd = float(((cum - cum.cummax()) / cum.cummax()).min())
    n_tr  = int((sim["position"].diff().abs() > 0).sum())
    return {
        "label":    label,
        "ann_ret":  round(ann,    4),
        "ann_vol":  round(vol,    4),
        "sharpe":   round(sharpe, 3) if not np.isnan(sharpe) else float("nan"),
        "max_dd":   round(max_dd, 4),
        "n_obs":    n,
        "n_trades": n_tr,
    }


def compute_buy_and_hold(daily_ret: pd.Series) -> pd.Series:
    return pd.Series(1.0, index=daily_ret.index, name="pos_bah")


# =============================================================================
# 8. PLOT HELPERS
# =============================================================================

def _shade(ax, s_dt, e_dt):
    """Grey bands for COVID drawdown and 2022 rate-hike regime."""
    for a, b in [("2020-02-01", "2020-06-01"), ("2022-01-01", "2022-12-31")]:
        a, b = pd.Timestamp(a), pd.Timestamp(b)
        if b > s_dt and a < e_dt:
            ax.axvspan(max(a, s_dt), min(b, e_dt), alpha=0.08, color="grey", lw=0)


def _oos_cumret(sim_df, start=OOS_START):
    net = sim_df["net_pnl"]
    s   = net[net.index >= start]
    return (1 + s).cumprod()


def _best_delta(sim_dict, model):
    """Delta index (0-3) with highest OOS Sharpe for this model."""
    return max(range(len(DELTAS)),
               key=lambda di: sim_dict[(model, di)][0].get("sharpe", -99) or -99)


def _setup_year_axis(axes, interval=2):
    for ax in axes:
        ax.xaxis.set_major_locator(mdates.YearLocator(interval))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.tick_params(axis="x", which="major", labelbottom=True, labelsize=7, pad=2)


# =============================================================================
# 9. PLOT A: Per-model expanding-window detail  (follows plot.md spec)
# =============================================================================

def plot_expanding_detail(model, out_path, ew_dict, ew_sim_dict, panel,
                          bah_sim, bah_st, s_dt, e_dt, extra_title=""):
    pal         = MODEL_PALETTE[model]
    feat_cols   = MODEL_FEATURES[model]
    bivariate   = len(feat_cols) >= 2
    pred_labels = [FEAT_DISPLAY.get(f, f) for f in feat_cols]
    pred_str    = " + ".join(pred_labels)
    n_d         = len(DELTAS)
    oos_dt      = pd.Timestamp(OOS_START)

    fig, axes = plt.subplots(
        2 + n_d, 1,
        figsize=(14, 11 + 2.2 * n_d),
        sharex=True,
        gridspec_kw={"height_ratios": [2.5, 1.2] + [1.0] * n_d, "hspace": 0.35},
    )
    fig.subplots_adjust(top=0.955, bottom=0.03, left=0.10, right=0.93)

    # Compute OOS R² (Goyal-Welch) using betas to reconstruct y_hat
    _betas_euro = compute_ew_betas(panel, model)
    _sub_euro   = panel.dropna(subset=feat_cols + ["fwd_ret_20"]).copy()
    _idx_euro   = _betas_euro.index.intersection(_sub_euro.index)
    _oos_r2_euro = float("nan")
    if len(_idx_euro) > 0:
        _y_act_e  = _sub_euro.loc[_idx_euro, "fwd_ret_20"]
        _y_all_e  = _sub_euro["fwd_ret_20"]
        _prev_m_e = _y_all_e.expanding().mean().shift(21).loc[_idx_euro]
        _alpha_e  = _betas_euro.loc[_idx_euro, "beta_alpha"]
        _x_prim_e = _sub_euro.loc[_idx_euro, feat_cols[0]]
        _b_prim_e = _betas_euro.loc[_idx_euro, "beta_VP"]
        _y_hat_e  = _alpha_e + _b_prim_e * _x_prim_e
        if bivariate and "beta_sec" in _betas_euro.columns:
            _x_sec_e  = _sub_euro.loc[_idx_euro, feat_cols[1]]
            _b_sec_e  = _betas_euro.loc[_idx_euro, "beta_sec"]
            _y_hat_e  = _y_hat_e + _b_sec_e * _x_sec_e
        _ss_res_e = float(((_y_act_e - _y_hat_e) ** 2).sum())
        _ss_tot_e = float(((_y_act_e - _prev_m_e) ** 2).sum())
        if _ss_tot_e > 0:
            _oos_r2_euro = 1.0 - _ss_res_e / _ss_tot_e
    _oos_r2_euro_str = f"\nOOS R² = {_oos_r2_euro:.4f}" if not np.isnan(_oos_r2_euro) else ""

    fig.suptitle(
        f"{pred_str} -> Euro Stoxx 50  20-day Forward Return  "
        f"(Expanding Window, OOS from {OOS_START}){extra_title}{_oos_r2_euro_str}\n"
        f"Training grows daily; OOS gap = 20 days; "
        f"NW-HAC {NW_LAGS} lags; |t| > {T_THRESH:.2f} gate; 0.05% slippage",
        fontsize=10, y=0.998,
    )

    ax_ret  = axes[0]
    ax_tb   = axes[1]
    ax_poss = axes[2:]
    xlim    = (oos_dt, e_dt)

    # -- Panel 1: Cumulative Net Return ------------------------------------
    ax_ret.set_xlim(*xlim)
    _shade(ax_ret, s_dt, e_dt)

    # Find earliest activation across all deltas (for stat-window consistency)
    _activation = None
    for _di in range(n_d):
        _p = ew_dict.get((model, _di))
        if _p is None:
            continue
        _act = _p[(_p.index >= OOS_START) & (_p != 0)]
        if len(_act):
            _cand = _act.index.min()
            if _activation is None or _cand < _activation:
                _activation = _cand

    _rebase_start = None
    if _activation is not None and _activation > pd.Timestamp("2020-01-01"):
        _bah_idx  = bah_sim.index
        _act_iloc = _bah_idx.searchsorted(_activation)
        _rebase_start = _bah_idx[min(_act_iloc + 1, len(_bah_idx) - 1)]

    _stat_start = _rebase_start if _rebase_start is not None else pd.Timestamp(OOS_START)
    _stat_lbl   = (f" · stats from {_stat_start.strftime('%Y-%m-%d')}"
                   if _rebase_start is not None else "")

    # Main B&H — stats from _stat_start
    _bah_st_plot = compute_performance_stats(
        bah_sim[bah_sim.index >= _stat_start], "BaH_plot")
    bah_oos = _oos_cumret(bah_sim)
    ax_ret.plot(bah_oos.index, bah_oos.values,
                color=BAH_COLOR, lw=1.5, ls="-.", alpha=0.6,
                label=(f"Buy-and-Hold{_stat_lbl}  "
                       f"[SR={_bah_st_plot['sharpe']:+.2f}  "
                       f"ret={_bah_st_plot['ann_ret']*100:+.1f}%  "
                       f"DD={_bah_st_plot['max_dd']*100:.1f}%]"))

    # Rebased B&H from first activation if that activation is post-2020
    if _rebase_start is not None:
        _net     = bah_sim["net_pnl"]
        _bah_act = (1 + _net[_net.index >= _rebase_start]).cumprod()
        _bah_act_st = compute_performance_stats(
            bah_sim[bah_sim.index >= _rebase_start], "BaH_act")
        ax_ret.plot(_bah_act.index, _bah_act.values,
                    color=BAH_COLOR, lw=1.2, ls=":", alpha=0.85,
                    label=(f"Buy-and-Hold from {_rebase_start.strftime('%Y-%m-%d')}  "
                           f"[SR={_bah_act_st['sharpe']:+.2f}  "
                           f"ret={_bah_act_st['ann_ret']*100:+.1f}%  "
                           f"DD={_bah_act_st['max_dd']*100:.1f}%]"))

    for di in range(n_d):
        if (model, di) not in ew_sim_dict:
            continue
        _, sim_di = ew_sim_dict[(model, di)]
        st_plot   = compute_performance_stats(
            sim_di[sim_di.index >= _stat_start],
            f"euro_{model}_{DELTA_LBL[di]}_plot")
        cum        = _oos_cumret(sim_di)
        pos_oos    = ew_dict[(model, di)]
        pos_oos    = pos_oos[pos_oos.index >= _stat_start]
        pL = float((pos_oos ==  1).mean() * 100)
        pS = float((pos_oos == -1).mean() * 100)
        ax_ret.plot(cum.index, cum.values,
                    color=pal[di], lw=1.8, alpha=0.9,
                    label=(f"{DELTA_LBL[di]}  "
                           f"[SR={st_plot['sharpe']:+.2f}  "
                           f"ret={st_plot['ann_ret']*100:+.1f}%  "
                           f"DD={st_plot['max_dd']*100:.1f}%  "
                           f"L{pL:.0f}%/S{pS:.0f}%]"))

    ax_ret.axhline(1, color="black", lw=0.4, ls=":")
    ax_ret.set_yscale("log")
    ax_ret.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{v:.2f}x"))
    ax_ret.set_ylabel("Cumulative Net Return (log)", fontsize=9)
    ax_ret.legend(fontsize=8, loc="upper left", framealpha=0.92)
    ax_ret.grid(axis="y", alpha=0.2, lw=0.6)
    ax_ret.spines[["top", "right"]].set_visible(False)

    # -- Panel 2: NW t-stat and Beta Over Time ----------------------------
    betas  = compute_ew_betas(panel, model)
    ax_tb.set_xlim(*xlim)
    _shade(ax_tb, s_dt, e_dt)

    t_prim = betas["t_VP"]
    b_prim = betas["beta_VP"]
    ax_tb.plot(t_prim.index, t_prim.values,
               color=pal[0], lw=1.0, alpha=0.85,
               label=f"NW t-stat ({pred_labels[0]})")
    if bivariate and "t_sec" in betas.columns:
        ax_tb.plot(betas["t_sec"].index, betas["t_sec"].values,
                   color=pal[1], lw=1.0, alpha=0.85, ls="--",
                   label=f"NW t-stat ({pred_labels[1]})")

    ax_tb.fill_between(t_prim.index, -T_THRESH, T_THRESH,
                       color="firebrick", alpha=0.05, label="Below gate (flat zone)")
    ax_tb.axhline( T_THRESH, color="firebrick", lw=1.2, ls="--",
                  label=f"|t| = {T_THRESH:.2f} gate")
    ax_tb.axhline(-T_THRESH, color="firebrick", lw=1.2, ls="--")
    ax_tb.axhline(0, color="black", lw=0.5, ls=":")
    ax_tb.set_ylabel("NW t-stat", fontsize=9)
    ax_tb.grid(axis="y", alpha=0.2, lw=0.6)
    ax_tb.spines["top"].set_visible(False)

    ax_tb2 = ax_tb.twinx()
    ax_tb2.plot(b_prim.index, b_prim.values,
                color="dimgrey", lw=1.0, ls="--", alpha=0.60,
                label=f"Beta ({pred_labels[0]})")
    if bivariate and "beta_sec" in betas.columns:
        ax_tb2.plot(betas["beta_sec"].index, betas["beta_sec"].values,
                    color="dimgrey", lw=1.0, ls=":", alpha=0.50,
                    label=f"Beta ({pred_labels[1]})")
    ax_tb2.axhline(0, color="dimgrey", lw=0.4, ls=":")
    ax_tb2.set_ylabel("Beta", fontsize=8, color="dimgrey")
    ax_tb2.tick_params(axis="y", labelcolor="dimgrey", labelsize=7)
    ax_tb2.spines["top"].set_visible(False)

    # In-sample R² on the right axis (shared with beta axis = ax_tb2)
    if "r2_insample" in betas.columns:
        r2_s = betas["r2_insample"]
        ax_tb2.plot(r2_s.index, r2_s.values,
                    color="forestgreen", lw=0.9, ls=":", alpha=0.75,
                    label="In-sample R²")

    h1, l1 = ax_tb.get_legend_handles_labels()
    h2, l2 = ax_tb2.get_legend_handles_labels()
    ax_tb.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")

    # -- Panels 3-6: Position Over Time -----------------------------------
    for di, ax_p in enumerate(ax_poss):
        if (model, di) not in ew_dict:
            ax_p.axis("off")
            continue
        pos_full = ew_dict[(model, di)]
        pos = pos_full[pos_full.index >= OOS_START]
        _shade(ax_p, s_dt, e_dt)
        ax_p.fill_between(pos.index, pos.where(pos ==  1, 0), 0,
                          color=pal[di], alpha=0.75)
        ax_p.fill_between(pos.index, pos.where(pos == -1, 0), 0,
                          color=pal[di], alpha=0.30, hatch="///")
        ax_p.axhline(0, color="black", lw=0.4)
        ax_p.set_ylim(-1.5, 1.5)
        ax_p.set_yticks([-1, 0, 1])
        ax_p.set_yticklabels(["Short", "Flat", "Long"], fontsize=8)
        ax_p.set_ylabel(DELTA_LBL[di], fontsize=9, rotation=0,
                        ha="right", va="center", labelpad=56, color=pal[di])
        pL = float((pos ==  1).mean() * 100)
        pS = float((pos == -1).mean() * 100)
        pF = float((pos ==  0).mean() * 100)
        ax_p.text(0.01, 0.97,
                  f"Long {pL:.1f}%  Short {pS:.1f}%  Flat {pF:.1f}%  "
                  f"AvgPos={float(pos.mean()):+.3f}",
                  transform=ax_p.transAxes, fontsize=7.5, va="top", color=pal[di])
        ax_p.spines[["top", "right"]].set_visible(False)

    _setup_year_axis([ax_ret, ax_tb] + list(ax_poss))
    fig.savefig(out_path, dpi=155, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# =============================================================================
# 10. PLOT B: Post-activation comparison (all models × best δ)
# =============================================================================

def plot_post2020_comparison(out_path, ew_dict, ew_sim_dict, models,
                              bah_sim, bah_st, s_dt, e_dt):
    _MODEL_COLOR = {
        "Base":       "#08306b",
        "Model_A":    "#006d2c",
        "Model_C":    "#3f007d",
        "Model_VV2X": "#7f2704",
    }

    # Start = first date any secondary-signal model (A, C) takes a position
    first_dates = []
    for m in [m for m in ["Model_A", "Model_C"] if (m, 0) in ew_sim_dict]:
        di  = _best_delta(ew_sim_dict, m)
        pos = ew_dict.get((m, di))
        if pos is not None:
            active = pos[pos != 0]
            if len(active):
                first_dates.append(active.index.min())
    START = min(first_dates) if first_dates else pd.Timestamp("2020-01-01")
    start = START.strftime("%Y-%m-%d")

    def _rebase(sim_df):
        net = sim_df["net_pnl"]
        s   = net[net.index >= start]
        return (1 + s).cumprod()

    def _post_st(sim_df, lbl):
        s = sim_df[sim_df.index >= start].copy()
        s["cum_net"] = (1 + s["net_pnl"]).cumprod()
        return compute_performance_stats(s, lbl)

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.suptitle(
        f"Euro Stoxx 50  — Performance from First Secondary-Signal Activation ({start})  ·  "
        f"Best δ  ·  Expanding Window\nRebased to 1.0 at {start}  ·  "
        f"|t| > {T_THRESH:.2f} gate  ·  0.05% slippage",
        fontsize=10,
    )

    bah_post = _post_st(bah_sim, "BaH")
    bah_cum  = _rebase(bah_sim)
    ax.plot(bah_cum.index, bah_cum.values,
            color=BAH_COLOR, lw=1.8, ls="-.", alpha=0.65,
            label=(f"Buy-and-Hold  "
                   f"[SR={bah_post['sharpe']:+.2f}  "
                   f"ret={bah_post['ann_ret']*100:+.1f}%  "
                   f"DD={bah_post['max_dd']*100:.1f}%]"))

    for m in models:
        col  = _MODEL_COLOR.get(m, "#333333")
        di   = _best_delta(ew_sim_dict, m)
        _, s = ew_sim_dict[(m, di)]
        st   = _post_st(s, m)
        cum  = _rebase(s)
        pos  = ew_dict[(m, di)]
        pfr  = pos[pos.index >= start]
        pL   = int((pfr ==  1).mean() * 100)
        pS   = int((pfr == -1).mean() * 100)
        short_lbl = MODEL_LABEL[m].split("--")[1].strip()
        ax.plot(cum.index, cum.values,
                color=col, lw=2.2, ls="-", alpha=0.92,
                label=(f"{short_lbl} ({DELTA_LBL[di]})  "
                       f"[SR={st['sharpe']:+.2f}  "
                       f"ret={st['ann_ret']*100:+.1f}%  "
                       f"DD={st['max_dd']*100:.1f}%  "
                       f"L{pL}%/S{pS}%]"))

    for a, b in [("2020-02-01", "2020-06-01"), ("2022-01-01", "2022-12-31")]:
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), alpha=0.07, color="grey", lw=0)

    ax.axhline(1, color="black", lw=0.5, ls=":")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{v:.2f}x"))
    ax.set_ylabel("Cumulative Net Return (log, rebased to 1.0)", fontsize=10)
    ax.set_xlim(START, e_dt)
    ax.legend(fontsize=7.5, loc="upper left", framealpha=0.92)
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax.get_xticklabels(), visible=True, fontsize=9)
    ax.grid(axis="y", alpha=0.2, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)

    fig.savefig(out_path, dpi=155, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# =============================================================================
# 11. PLOT C: Beta evolution (N-row × 2-col, one row per model)
# =============================================================================

def plot_ew_beta_evolution(out_path, models, panel, s_dt, e_dt):
    SEC_COLOR = {
        "Model_A":    "#006d2c",
        "Model_C":    "#3f007d",
        "Model_VV2X": "#7f2704",
    }
    n_rows = len(models)
    fig, axes = plt.subplots(n_rows, 2, figsize=(16, 5 * n_rows),
                             gridspec_kw={"wspace": 0.15, "hspace": 0.38},
                             squeeze=False)
    fig.suptitle(
        "Expanding Window -- Coefficient Estimates and NW t-statistics Over Time "
        "(Euro Markets)\n"
        f"Training = all history up to t−20  ·  NW-HAC {NW_LAGS} lags  ·  "
        f"Position gate: |t| > {T_THRESH:.2f}  ·  Shaded band = ±2 NW-SE",
        fontsize=10,
    )

    for row, model in enumerate(models):
        betas     = compute_ew_betas(panel, model)
        bivariate = "beta_sec" in betas.columns
        pal_vp    = MODEL_PALETTE[model][0]
        sec_col   = SEC_COLOR.get(model, "darkorange")
        feat_cols = MODEL_FEATURES[model]
        prim_name = FEAT_DISPLAY.get(feat_cols[0], feat_cols[0])
        sec_name  = FEAT_DISPLAY.get(feat_cols[1], feat_cols[1]) if bivariate else ""

        ax_b = axes[row, 0]
        ax_t = axes[row, 1]

        # Beta panel
        b_vp  = betas["beta_VP"]
        se_vp = betas["se_VP"]
        ax_b.plot(b_vp.index, b_vp.values, color=pal_vp, lw=1.2,
                  label=f"β({prim_name})")
        ax_b.fill_between(b_vp.index, b_vp - 2*se_vp, b_vp + 2*se_vp,
                          color=pal_vp, alpha=0.15)

        if bivariate:
            b_sec  = betas["beta_sec"]
            se_sec = betas["se_sec"]
            ax_b.plot(b_sec.index, b_sec.values, color=sec_col, lw=1.2,
                      label=f"β({sec_name})")
            ax_b.fill_between(b_sec.index, b_sec - 2*se_sec, b_sec + 2*se_sec,
                              color=sec_col, alpha=0.12)

        ax_b.axhline(0, color="black", lw=0.4, ls=":")
        ax_b.set_title(f"{MODEL_LABEL[model]} -- β ± 2 NW-SE", fontsize=9, loc="left")
        ax_b.set_ylabel("Signal β", fontsize=9)
        ax_b.grid(axis="y", alpha=0.2, lw=0.6)
        ax_b.spines["top"].set_visible(False)

        ax_b2 = ax_b.twinx()
        b_a  = betas["beta_alpha"]
        se_a = betas["se_alpha"]
        ax_b2.plot(b_a.index, b_a.values, color="grey", lw=1.0,
                   ls=":", alpha=0.85, label="α (intercept)")
        ax_b2.fill_between(b_a.index, b_a - 2*se_a, b_a + 2*se_a,
                           color="grey", alpha=0.08)
        ax_b2.set_ylabel("α", fontsize=9, color="grey")
        ax_b2.tick_params(axis="y", labelcolor="grey", labelsize=7)
        ax_b2.spines["top"].set_visible(False)

        h1, l1 = ax_b.get_legend_handles_labels()
        h2, l2 = ax_b2.get_legend_handles_labels()
        ax_b.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")

        # t-stat panel
        ax_t.plot(betas["t_alpha"].index, betas["t_alpha"].values,
                  color="grey", lw=1.0, ls=":", alpha=0.7, label="t(α)")
        ax_t.plot(betas["t_VP"].index, betas["t_VP"].values,
                  color=pal_vp, lw=1.0, ls="--", alpha=0.8,
                  label=f"t({prim_name})")
        if bivariate and "t_sec" in betas.columns:
            ax_t.plot(betas["t_sec"].index, betas["t_sec"].values,
                      color=sec_col, lw=1.5, label=f"t({sec_name})")

        ax_t.axhline( T_THRESH, color="firebrick", lw=1.2, ls="--",
                     label=f"|t| = {T_THRESH:.2f} (gate)")
        ax_t.axhline(-T_THRESH, color="firebrick", lw=1.2, ls="--")
        ax_t.axhline(0, color="black", lw=0.5, ls=":")
        ax_t.fill_between(betas["t_VP"].index, -T_THRESH, T_THRESH,
                          color="firebrick", alpha=0.05, label="Below gate -> flat")
        ax_t.set_title(f"{MODEL_LABEL[model]} -- NW t-statistics", fontsize=9, loc="left")
        ax_t.set_ylabel("NW t-statistic", fontsize=9)
        ax_t.legend(fontsize=8, loc="upper left")
        ax_t.grid(axis="y", alpha=0.2, lw=0.6)
        ax_t.spines[["top", "right"]].set_visible(False)

        for ax in [ax_b, ax_t]:
            ax.set_xlim(pd.Timestamp(OOS_START), e_dt)
            ax.xaxis.set_major_locator(mdates.YearLocator(2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            plt.setp(ax.get_xticklabels(), visible=True, fontsize=8)
            for a_s, b_s in [("2020-02-01","2020-06-01"), ("2022-01-01","2022-12-31")]:
                ax.axvspan(pd.Timestamp(a_s), pd.Timestamp(b_s),
                           alpha=0.07, color="grey", lw=0)

    fig.savefig(out_path, dpi=155, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# =============================================================================
# 12. PLOT D: Performance summary table
# =============================================================================

def plot_summary_table(out_path, ew_sim_dict, models, method_label, bah_st):
    TINT = {
        "Base":       "#deebf7",
        "Model_A":    "#e5f5e0",
        "Model_C":    "#efedf5",
        "Model_VV2X": "#feedde",
    }
    headers = ["Model", "Method", "Best δ",
               "Ann. Ret", "Ann. Vol", "Sharpe", "Max DD",
               "Trades", "Long%", "Short%", "Flat%"]

    rows, colors = [], []
    rows.append(["Buy-and-Hold (SX5E)", "--", "--",
                 f"{bah_st['ann_ret']*100:.2f}%",
                 f"{bah_st['ann_vol']*100:.2f}%",
                 f"{bah_st['sharpe']:.3f}",
                 f"{bah_st['max_dd']*100:.1f}%",
                 "--", "100%", "0%", "0%"])
    colors.append(["#fde0d0"] * len(headers))

    for m in models:
        di = _best_delta(ew_sim_dict, m)
        st, _ = ew_sim_dict[(m, di)]
        rows.append([
            MODEL_LABEL[m].split("--")[0].strip(),
            method_label,
            DELTA_LBL[di],
            f"{st['ann_ret']*100:.2f}%",
            f"{st['ann_vol']*100:.2f}%",
            f"{st['sharpe']:.3f}",
            f"{st['max_dd']*100:.1f}%",
            str(st["n_trades"]),
            f"{st.get('pct_long', 0):.1f}%",
            f"{st.get('pct_short', 0):.1f}%",
            f"{st.get('pct_flat', 0):.1f}%",
        ])
        colors.append([TINT.get(m, "#f0f0f0")] * len(headers))

    fig, ax = plt.subplots(figsize=(16, max(4, 1 + 1.1 * len(rows))))
    fig.suptitle(
        f"Performance Summary -- {method_label}  (Euro Stoxx 50)\n"
        "Best delta (highest OOS Sharpe) shown per model  ·  "
        "Net of 0.05% slippage  ·  0% risk-free rate",
        fontsize=11,
    )
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=headers,
                   cellLoc="center", loc="center",
                   cellColours=colors)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.6)

    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# =============================================================================
# 13. MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  Euro Experiment -- Loading and preparing data")
    print("=" * 60)

    stoxx      = load_stoxx_front_month()
    v2x        = load_v2x_spot()
    vstoxx_fut = load_vstoxx_futures()
    vv2tx_raw  = load_vv2tx()

    print(f"  Euro Stoxx 50 (FX): {stoxx.index.min().date()} to {stoxx.index.max().date()}")
    print(f"  V2X spot:           {v2x.index.min().date()} to {v2x.index.max().date()}")
    print(f"  VSTOXX futures:     {vstoxx_fut['date'].min().date()} to {vstoxx_fut['date'].max().date()}")
    print(f"  VV2TX:              {vv2tx_raw.index.min().date()} to {vv2tx_raw.index.max().date()}")

    print("\nComputing Euro implied variance (V2X^2/12 throughout)...")
    ivar = compute_euro_ivar(v2x)

    print("Computing Euro VRP via rolling HAR (500-day window) ...")
    vrp = compute_euro_vrp(stoxx["returns"], ivar, window=ROLL_VRP_WIN)
    print(f"  VRP series: {vrp.index.min().date()} to {vrp.index.max().date()}, "
          f"n={len(vrp)}, mean VP={vrp['VP'].mean():.2f}")

    print("Computing VSTOXX term slope (FH cross-sectional OLS)...")
    term_slope = compute_vstoxx_term_slope(vstoxx_fut)
    print(f"  Term slope: {term_slope.index.min().date()} to {term_slope.index.max().date()}, "
          f"n={len(term_slope)}, mean={term_slope.mean():.2f}")

    print("Computing VV2TX MA5...")
    vv2tx_ma5 = compute_vv2tx_ma5(vv2tx_raw)

    print("Building panel...")
    panel = build_euro_panel(vrp, stoxx, term_slope, vv2tx_ma5)
    print(f"  Panel shape: {panel.shape}, "
          f"dates: {panel.index.min().date()} to {panel.index.max().date()}")
    panel.to_csv(OUTPUT / "signals.csv")
    print("  Saved: signals.csv")

    # Coverage check per model
    print("\nSignal coverage per model:")
    for m, fc in MODEL_FEATURES.items():
        sub = panel.dropna(subset=fc + ["fwd_ret_20"])
        oos = sub[sub.index >= OOS_START]
        print(f"  {m:<12} total={len(sub):5d}  oos={len(oos):5d}  "
              f"({sub.index.min().date()} to {sub.index.max().date()})")

    # Reference dates (from daily_ret -- aligned to stoxx returns range)
    daily_ret = panel["daily_ret"].dropna()
    s_dt      = daily_ret.index[0]
    e_dt      = daily_ret.index[-1]
    oos_dt    = pd.Timestamp(OOS_START)

    # Buy-and-Hold
    bah_pos = compute_buy_and_hold(daily_ret)
    bah_sim = simulate_strategy(bah_pos, daily_ret)
    bah_st  = compute_performance_stats(bah_sim[bah_sim.index >= OOS_START],
                                        "Buy-and-Hold")
    bah_st.update(avg_position=1.0, pct_long=100.0, pct_short=0.0, pct_flat=0.0)
    print(f"\nBuy-and-Hold (OOS): SR={bah_st['sharpe']:.3f}  "
          f"ret={bah_st['ann_ret']*100:.2f}%  "
          f"DD={bah_st['max_dd']*100:.1f}%")

    MODELS     = ["Base", "Model_A", "Model_C"]
    ALL_MODELS = ["Base", "Model_A", "Model_C", "Model_VV2X"]

    # =========================================================================
    # EW -- Fixed-threshold
    # =========================================================================
    print("\n" + "-"*55)
    print("  EW Fixed-threshold positions")
    print("-"*55)
    EW, EW_SIM = {}, {}
    for m in MODELS:
        for di, delta in enumerate(DELTAS):
            print(f"  EW  {m:<12}  {DELTA_LBL[di]}  ...", end=" ", flush=True)
            pos     = run_expanding_window(panel, m, delta)
            sim     = simulate_strategy(pos, daily_ret)
            sim_oos = sim[sim.index >= OOS_START]
            st      = compute_performance_stats(sim_oos, f"EW_{m}_{DELTA_LBL[di]}")
            oos_pos = pos[pos.index >= OOS_START]
            st["pct_long"]  = float((oos_pos ==  1).mean()) * 100
            st["pct_short"] = float((oos_pos == -1).mean()) * 100
            st["pct_flat"]  = float((oos_pos ==  0).mean()) * 100
            EW[(m, di)]     = pos
            EW_SIM[(m, di)] = (st, sim)
            print(f"SR={st['sharpe']:+.3f}")

    print("\n--- EW fixed-threshold detail plots ---")
    for m in ["Base", "Model_A", "Model_C"]:
        fname = "symmetric_" + EW_MODEL_DIR[m].name.replace(" ", "_") + ".png"
        plot_expanding_detail(
            m, EW_MODEL_DIR[m] / fname,
            EW, EW_SIM, panel, bah_sim, bah_st, s_dt, e_dt,
        )

    print("\n--- EW comparisons: post-activation, fixed-threshold ---")
    plot_post2020_comparison(
        DIR_EW_CMP / "symmetric_comparisons.png",
        EW, EW_SIM, MODELS, bah_sim, bah_st, s_dt, e_dt,
    )
    plot_summary_table(
        DIR_EW_CMP / "performance_summary.png",
        EW_SIM, MODELS, "EW Fixed Threshold", bah_st,
    )

    # =========================================================================
    # EW -- Rolling-mu threshold
    # =========================================================================
    print("\n" + "-"*55)
    print("  EW Rolling-mu positions")
    print("-"*55)
    EW_RM, EW_RM_SIM = {}, {}
    for m in MODELS:
        for di, delta in enumerate(DELTAS):
            print(f"  EW-rolmu  {m:<12}  {DELTA_LBL[di]}  ...", end=" ", flush=True)
            pos     = run_expanding_window_rolmu(panel, m, delta)
            sim     = simulate_strategy(pos, daily_ret)
            sim_oos = sim[sim.index >= OOS_START]
            st      = compute_performance_stats(sim_oos, f"EWrm_{m}_{DELTA_LBL[di]}")
            oos_pos = pos[pos.index >= OOS_START]
            st["pct_long"]  = float((oos_pos ==  1).mean()) * 100
            st["pct_short"] = float((oos_pos == -1).mean()) * 100
            st["pct_flat"]  = float((oos_pos ==  0).mean()) * 100
            EW_RM[(m, di)]     = pos
            EW_RM_SIM[(m, di)] = (st, sim)
            print(f"SR={st['sharpe']:+.3f}")

    print("\n--- EW rolling-mu detail plots ---")
    for m in ["Base", "Model_A", "Model_C"]:
        fname = "base_return_shift_" + EW_MODEL_DIR[m].name.replace(" ", "_") + ".png"
        plot_expanding_detail(
            m, EW_MODEL_DIR[m] / fname,
            EW_RM, EW_RM_SIM, panel, bah_sim, bah_st, s_dt, e_dt,
            extra_title=" · Threshold = rolling-avg(20d return) ± δ",
        )

    # =========================================================================
    # Univariate VV2TX -- rolling-mu only
    # =========================================================================
    print("\n" + "-"*55)
    print("  EW-rolmu Univariate VV2TX positions")
    print("-"*55)
    EW_VV2X, EW_VV2X_SIM = {}, {}
    for di, delta in enumerate(DELTAS):
        print(f"  EW-rolmu  Model_VV2X  {DELTA_LBL[di]}  ...", end=" ", flush=True)
        pos     = run_expanding_window_rolmu(panel, "Model_VV2X", delta)
        sim     = simulate_strategy(pos, daily_ret)
        sim_oos = sim[sim.index >= OOS_START]
        st      = compute_performance_stats(sim_oos, f"EWrm_VV2X_{DELTA_LBL[di]}")
        oos_pos = pos[pos.index >= OOS_START]
        st["pct_long"]  = float((oos_pos ==  1).mean()) * 100
        st["pct_short"] = float((oos_pos == -1).mean()) * 100
        st["pct_flat"]  = float((oos_pos ==  0).mean()) * 100
        EW_VV2X[di]     = pos
        EW_VV2X_SIM[di] = (st, sim)
        print(f"SR={st['sharpe']:+.3f}")

    EW_VV2X_DICT     = {("Model_VV2X", di): EW_VV2X[di]     for di in range(4)}
    EW_VV2X_SIM_DICT = {("Model_VV2X", di): EW_VV2X_SIM[di] for di in range(4)}

    print("\n--- VV2TX MA5 / expanding detail plot ---")
    plot_expanding_detail(
        "Model_VV2X",
        EW_MODEL_DIR["Model_VV2X"] / "base_return_shift_VV2TX_MA5.png",
        EW_VV2X_DICT, EW_VV2X_SIM_DICT, panel,
        bah_sim, bah_st, s_dt, e_dt,
        extra_title=" · Threshold = rolling-avg(20d return) ± δ",
    )

    # =========================================================================
    # Combined rolling-mu comparisons
    # =========================================================================
    _comb     = {**EW_RM,     **EW_VV2X_DICT}
    _comb_sim = {**EW_RM_SIM, **EW_VV2X_SIM_DICT}

    print("\n--- comparisons: post-activation, rolling-mu ---")
    plot_post2020_comparison(
        DIR_EW_CMP / "base_return_shift_comparisons.png",
        _comb, _comb_sim, ALL_MODELS,
        bah_sim, bah_st, s_dt, e_dt,
    )
    plot_summary_table(
        DIR_EW_CMP / "performance_summary_rolmu.png",
        _comb_sim, ALL_MODELS, "EW Rolling-Mu Threshold", bah_st,
    )

    # =========================================================================
    # Beta evolution (all models)
    # =========================================================================
    print("\n--- beta evolution ---")
    plot_ew_beta_evolution(
        DIR_EW_CMP / "ew_beta_evolution.png",
        ALL_MODELS, panel, s_dt, e_dt,
    )

    print("\n" + "=" * 60)
    print("  Euro Experiment -- COMPLETE")
    print("=" * 60)
