"""Compute all VS replication numbers for RESULTS.md update."""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from pathlib import Path
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.stats.sandwich_covariance import cov_hac

DATA_DIR = Path(__file__).parent.parent / 'data'
NW_LAGS  = 44

# SP500 front-month returns
meta = pd.read_parquet(DATA_DIR / 'EquityFuture_security_meta.parquet')
hist = pd.read_parquet(DATA_DIR / 'EquityFuture_historical.parquet')
es_tickers = meta[meta['curve_group'] == 'ES']['security'].tolist()
es = hist[hist['security'].isin(es_tickers)].copy()
es['date'] = pd.to_datetime(es['date'])
meta_es = meta[meta['curve_group'] == 'ES'][['security','expiry_yearmonth']].copy()
meta_es['expiry_date'] = pd.to_datetime(meta_es['expiry_yearmonth'], format='%Y-%m')
es = es.merge(meta_es[['security','expiry_date']], on='security').sort_values(['date','expiry_date'])
sp_ret = (es.groupby('date').first().reset_index()[['date','returns']]
          .dropna().sort_values('date').set_index('date')['returns'])

# VIX
vix_df = pd.read_csv(DATA_DIR / 'VolatilityIndexData.csv', parse_dates=['DATE'])
vix = (vix_df[vix_df['SECURITY']=='VIX Index']
       .sort_values('DATE').set_index('DATE')['INDEX_VALUE'])
vix.index.name = 'date'

# Variance swap 1m SPX
vs_raw = pd.read_csv(DATA_DIR / 'EquityIndexVarianceSwapData.csv', parse_dates=['DATE'])
vs = (vs_raw[(vs_raw['UNDERLYING']=='SPX') & (vs_raw['TENOR_MONTHS']==1.0)]
      .sort_values('DATE').set_index('DATE')['IMPLIED_VOLATILITY'])
vs.index.name = 'date'

# RV components
rv_daily = (sp_ret * 100)**2
rv = pd.DataFrame(index=sp_ret.index)
rv['RV1']  = rv_daily * 22
rv['RV5']  = rv_daily.rolling(5).mean() * 22
rv['RV22'] = rv_daily.rolling(22).sum()

# Build panel with VS
vs2  = vs**2 / 12.0
vix2 = vix**2 / 12.0

panel = rv.join(vs2.rename('VS2'), how='inner').dropna()
panel['VIX2']    = vix2
panel['RV22_fwd'] = panel['RV22'].shift(-22)
panel['VS2_lag']  = panel['VS2'].shift(1)
panel['VIX2_lag'] = panel['VIX2'].shift(1)
panel['RV22_lag'] = panel['RV22'].shift(1)
panel['RV5_lag']  = panel['RV5'].shift(1)
panel['RV1_lag']  = panel['RV1'].shift(1)
panel = panel.dropna()

print(f'Panel: {panel.shape}  {panel.index.min().date()} -> {panel.index.max().date()}')

# Helpers
def nw_se(res, nlags=NW_LAGS):
    return np.sqrt(np.diag(cov_hac(res, nlags=nlags)))

def estimate(pnl, xcol):
    y   = pnl['RV22_fwd']
    X   = add_constant(pnl[[xcol,'RV22_lag','RV5_lag','RV1_lag']])
    res = OLS(y, X).fit()
    ses = nw_se(res)
    vs_ = X.columns.tolist()
    return dict(n=int(res.nobs), params=dict(zip(vs_,res.params)),
                nwse=dict(zip(vs_,ses)), tstat=dict(zip(vs_,res.params/ses)),
                adj_r2=res.rsquared_adj, rmse=float(np.sqrt(res.mse_resid)),
                fitted=res.fittedvalues)

def oos_forecast(pnl, split, xcol):
    tr, te = pnl[pnl.index<=split], pnl[pnl.index>split]
    y_tr   = tr['RV22_fwd']
    X_tr   = add_constant(tr[[xcol,'RV22_lag','RV5_lag','RV1_lag']])
    res    = OLS(y_tr, X_tr).fit()
    X_te   = add_constant(te[[xcol,'RV22_lag','RV5_lag','RV1_lag']], has_constant='add')
    y_te   = te['RV22_fwd']
    y_hat  = res.predict(X_te)
    mz     = OLS(y_te.values, add_constant(y_hat.values)).fit()
    err    = y_te - y_hat
    isr    = estimate(tr, xcol)
    return dict(n_train=len(tr), n_test=len(te),
                is_adj_r2=isr['adj_r2'], is_rmse=isr['rmse'],
                oos_mz_r2=float(mz.rsquared), oos_rmse=float(np.sqrt((err**2).mean())),
                oos_mae=float(err.abs().mean()), oos_mape=float((err.abs()/y_te).mean()))

def pred_reg(pnl, sp_returns, horizon, vp_col):
    monthly = pnl.resample('ME').last()[[vp_col]].dropna()
    sp_m    = sp_returns.resample('ME').agg(lambda x:(1+x).prod()-1)
    log_sp  = np.log(1+sp_m)
    fwd     = (np.exp(log_sp.rolling(horizon).sum().shift(-horizon))-1)*(12/horizon)*100
    monthly['ret'] = fwd
    monthly = monthly.dropna()
    if len(monthly) < 20: return None
    y   = monthly['ret']
    X   = add_constant(monthly[[vp_col]])
    res = OLS(y, X).fit()
    nw  = np.sqrt(np.diag(cov_hac(res, nlags=max(3,2*horizon))))
    vi  = list(X.columns).index(vp_col)
    return dict(h=horizon, n=len(monthly), coef=round(float(res.params[vp_col]),4),
                nwse=round(float(nw[vi]),4), tstat=round(float(res.params[vp_col]/nw[vi]),2),
                adj_r2=round(float(res.rsquared_adj),4))

# Run all
res_vs  = estimate(panel, 'VS2_lag')
res_vix = estimate(panel, 'VIX2_lag')

split_date = panel.index[int(0.75*len(panel))]
oos_vs  = oos_forecast(panel, split_date, 'VS2_lag')
oos_vix = oos_forecast(panel, split_date, 'VIX2_lag')

panel['CV_vs']  = res_vs['fitted']
panel['CV_vix'] = res_vix['fitted']
panel['VP_vs']  = panel['VS2']  - panel['CV_vs']
panel['VP_vix'] = panel['VIX2'] - panel['CV_vix']

# Print
print()
print('=== IS RESULTS ===')
for xcol, r, label in [('VS2_lag',res_vs,'VS'),('VIX2_lag',res_vix,'VIX')]:
    print(f'  HAR-RV-{label}  n={r["n"]}  adj_r2={r["adj_r2"]:.4f}  rmse={r["rmse"]:.3f}')
    for v in ['const', xcol, 'RV22_lag', 'RV5_lag', 'RV1_lag']:
        print(f'    {v:<14}  coef={r["params"][v]:>8.4f}  nwse={r["nwse"][v]:>7.4f}  t={r["tstat"][v]:>6.2f}')

print()
print(f'Split date: {split_date.date()}  (75% of {len(panel)} obs)')
print('=== OOS RESULTS ===')
for label, o in [('VS',oos_vs),('VIX',oos_vix)]:
    print(f'  HAR-RV-{label}  train={o["n_train"]}  test={o["n_test"]}')
    print(f'    IS  adj_r2={o["is_adj_r2"]:.4f}  IS  rmse={o["is_rmse"]:.3f}')
    print(f'    OOS mz_r2={o["oos_mz_r2"]:.4f}  OOS rmse={o["oos_rmse"]:.3f}  mae={o["oos_mae"]:.3f}  mape={o["oos_mape"]:.4f}')

print()
print('=== VP DESCRIPTIVE ===')
for col, label in [('VP_vs','VS'),('VP_vix','VIX')]:
    s = panel[col]
    print(f'  VP_{label}  mean={s.mean():.3f}  std={s.std():.3f}  min={s.min():.3f}  max={s.max():.3f}')
print(f'  VS2 mean={panel["VS2"].mean():.3f}  VIX2 mean={panel["VIX2"].mean():.3f}')
print(f'  diff(VIX-VS) mean={(panel["VIX2"]-panel["VS2"]).mean():.3f}  corr={panel["VS2"].corr(panel["VIX2"]):.4f}')

print()
print('=== RETURN PREDICTABILITY ===')
for h in [1, 3, 12]:
    r_vs  = pred_reg(panel, sp_ret, h, 'VP_vs')
    r_vix = pred_reg(panel, sp_ret, h, 'VP_vix')
    if r_vs and r_vix:
        print(f'  h={h:>2}m  n={r_vs["n"]}')
        print(f'         VP_VS : coef={r_vs["coef"]:>7.4f}  nwse={r_vs["nwse"]:>6.4f}  t={r_vs["tstat"]:>5.2f}  adjR2={r_vs["adj_r2"]:>6.4f}')
        print(f'         VP_VIX: coef={r_vix["coef"]:>7.4f}  nwse={r_vix["nwse"]:>6.4f}  t={r_vix["tstat"]:>5.2f}  adjR2={r_vix["adj_r2"]:>6.4f}')
