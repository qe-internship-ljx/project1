"""Full VIX vs VS comparison including DM test and forecast encompassing."""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from pathlib import Path
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.stats.sandwich_covariance import cov_hac

DATA_DIR = Path(__file__).parent.parent / 'data'
NW_LAGS  = 44

# ── Load data ─────────────────────────────────────────────────────────────────
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

vix_df = pd.read_csv(DATA_DIR / 'VolatilityIndexData.csv', parse_dates=['DATE'])
vix = vix_df[vix_df['SECURITY']=='VIX Index'].sort_values('DATE').set_index('DATE')['INDEX_VALUE']
vix.index.name = 'date'

vs_raw = pd.read_csv(DATA_DIR / 'EquityIndexVarianceSwapData.csv', parse_dates=['DATE'])
vs = (vs_raw[(vs_raw['UNDERLYING']=='SPX') & (vs_raw['TENOR_MONTHS']==1.0)]
      .sort_values('DATE').set_index('DATE')['IMPLIED_VOLATILITY'])
vs.index.name = 'date'

# ── Build unified panel ───────────────────────────────────────────────────────
rv_daily = (sp_ret * 100)**2
rv = pd.DataFrame(index=sp_ret.index)
rv['RV1']  = rv_daily * 22
rv['RV5']  = rv_daily.rolling(5).mean() * 22
rv['RV22'] = rv_daily.rolling(22).sum()

panel = rv.join((vs**2/12).rename('VS2'), how='inner')
panel['VIX2']    = vix**2 / 12
panel['RV22_fwd'] = panel['RV22'].shift(-22)
panel['VS2_lag']  = panel['VS2'].shift(1)
panel['VIX2_lag'] = panel['VIX2'].shift(1)
panel['RV22_lag'] = panel['RV22'].shift(1)
panel['RV5_lag']  = panel['RV5'].shift(1)
panel['RV1_lag']  = panel['RV1'].shift(1)
panel = panel.dropna()

print(f'Panel: {len(panel):,} obs  {panel.index.min().date()} -> {panel.index.max().date()}')

# ── Helpers ───────────────────────────────────────────────────────────────────
def nw_se(res, nlags=NW_LAGS):
    return np.sqrt(np.diag(cov_hac(res, nlags=nlags)))

def estimate(pnl, xcol):
    y   = pnl['RV22_fwd']
    X   = add_constant(pnl[[xcol,'RV22_lag','RV5_lag','RV1_lag']])
    res = OLS(y, X).fit()
    ses = nw_se(res)
    vrs = X.columns.tolist()
    return dict(n=int(res.nobs), params=dict(zip(vrs,res.params)),
                nwse=dict(zip(vrs,ses)), tstat=dict(zip(vrs,res.params/ses)),
                adj_r2=res.rsquared_adj, rmse=float(np.sqrt(res.mse_resid)),
                fitted=res.fittedvalues)

def oos_forecast(pnl, split, xcol):
    tr, te = pnl[pnl.index<=split], pnl[pnl.index>split]
    res    = OLS(tr['RV22_fwd'], add_constant(tr[[xcol,'RV22_lag','RV5_lag','RV1_lag']])).fit()
    X_te   = add_constant(te[[xcol,'RV22_lag','RV5_lag','RV1_lag']], has_constant='add')
    y_te   = te['RV22_fwd']
    y_hat  = res.predict(X_te)
    mz     = OLS(y_te.values, add_constant(y_hat.values)).fit()
    err    = y_te - y_hat
    isr    = estimate(tr, xcol)
    return dict(n_train=len(tr), n_test=len(te),
                is_adj_r2=isr['adj_r2'], is_rmse=isr['rmse'],
                oos_mz_r2=float(mz.rsquared), oos_rmse=float(np.sqrt((err**2).mean())),
                oos_mae=float(err.abs().mean()), oos_mape=float((err.abs()/y_te).mean()),
                y_test=y_te, y_hat=y_hat, err=err)

# ── IS estimation ─────────────────────────────────────────────────────────────
res_vs  = estimate(panel, 'VS2_lag')
res_vix = estimate(panel, 'VIX2_lag')

# ── OOS ───────────────────────────────────────────────────────────────────────
split = panel.index[int(0.75*len(panel))]
oos_vs  = oos_forecast(panel, split, 'VS2_lag')
oos_vix = oos_forecast(panel, split, 'VIX2_lag')

# ── Diebold-Mariano test ──────────────────────────────────────────────────────
# H0: equal MSE.  d_t = e_VS_t^2 - e_VIX_t^2.  t = mean(d) / NW-SE(d)
e_vs  = oos_vs['err']
e_vix = oos_vix['err']
d = e_vs**2 - e_vix**2           # positive = VS worse
d_mean = d.mean()
# NW SE of d using regression on constant
dm_res = OLS(d.values, np.ones(len(d))).fit()
dm_nw  = float(np.sqrt(cov_hac(dm_res, nlags=NW_LAGS)[0,0]))
dm_t   = d_mean / dm_nw
print(f'\nDiebold-Mariano test (H0: equal MSE):')
print(f'  mean(d) = {d_mean:.4f}  NW-SE = {dm_nw:.4f}  t = {dm_t:.3f}')
print(f'  {"VS worse" if d_mean>0 else "VS better"}, {"reject H0 at 5%" if abs(dm_t)>1.96 else "fail to reject H0"}')

# ── Forecast encompassing ─────────────────────────────────────────────────────
# Regress actual on BOTH forecasts: y = a + b1*f_VS + b2*f_VIX + e
y_te    = oos_vs['y_test']
f_vs    = oos_vs['y_hat']
f_vix   = oos_vix['y_hat']
X_enc   = add_constant(pd.DataFrame({'f_VS': f_vs.values, 'f_VIX': f_vix.values}))
enc_res = OLS(y_te.values, X_enc).fit()
enc_nw  = nw_se(enc_res)
enc_vars = X_enc.columns.tolist()
print(f'\nForecast encompassing regression (actual ~ const + f_VS + f_VIX):')
for v, c, se in zip(enc_vars, enc_res.params, enc_nw):
    print(f'  {v:<8}  coef={c:.4f}  NW-SE={se:.4f}  t={c/se:.2f}')
print(f'  Adj-R²={enc_res.rsquared_adj:.4f}')

# ── Combined model IS ─────────────────────────────────────────────────────────
y_all   = panel['RV22_fwd']
X_both  = add_constant(panel[['VS2_lag','VIX2_lag','RV22_lag','RV5_lag','RV1_lag']])
res_both = OLS(y_all, X_both).fit()
ses_both = nw_se(res_both)
print(f'\nCombined model (both VS2_lag + VIX2_lag):  adj_r2={res_both.rsquared_adj:.4f}  rmse={np.sqrt(res_both.mse_resid):.3f}')
for v, c, se in zip(X_both.columns, res_both.params, ses_both):
    print(f'  {v:<12}  coef={c:.4f}  t={c/se:.2f}')

# ── VP & return predictability ────────────────────────────────────────────────
panel['CV_vs']  = res_vs['fitted']
panel['CV_vix'] = res_vix['fitted']
panel['VP_vs']  = panel['VS2']  - panel['CV_vs']
panel['VP_vix'] = panel['VIX2'] - panel['CV_vix']

def pred_reg(pnl, sp_returns, horizon, vp_col):
    monthly = pnl.resample('ME').last()[[vp_col]].dropna()
    sp_m    = sp_returns.resample('ME').agg(lambda x:(1+x).prod()-1)
    log_sp  = np.log(1+sp_m)
    fwd     = (np.exp(log_sp.rolling(horizon).sum().shift(-horizon))-1)*(12/horizon)*100
    monthly['ret'] = fwd
    monthly = monthly.dropna()
    if len(monthly)<20: return None
    y   = monthly['ret']
    X   = add_constant(monthly[[vp_col]])
    res = OLS(y,X).fit()
    nw  = np.sqrt(np.diag(cov_hac(res, nlags=max(3,2*horizon))))
    vi  = list(X.columns).index(vp_col)
    return dict(h=horizon, n=len(monthly), coef=round(float(res.params[vp_col]),4),
                nwse=round(float(nw[vi]),4), tstat=round(float(res.params[vp_col]/nw[vi]),2),
                adj_r2=round(float(res.rsquared_adj),4))

print(f'\nReturn predictability:')
print(f'  {"h":>4}  {"n":>5}  {"VP_VS coef":>12}  {"t":>6}  {"adjR2":>7}  {"VP_VIX coef":>13}  {"t":>6}  {"adjR2":>7}')
for h in [1,3,12]:
    rv = pred_reg(panel, sp_ret, h, 'VP_vs')
    rk = pred_reg(panel, sp_ret, h, 'VP_vix')
    if rv and rk:
        print(f'  {h:>4}m {rv["n"]:>5}  {rv["coef"]:>12.4f}  {rv["tstat"]:>6.2f}  {rv["adj_r2"]:>7.4f}  {rk["coef"]:>13.4f}  {rk["tstat"]:>6.2f}  {rk["adj_r2"]:>7.4f}')

print(f'\nVP summary:')
for col,label in [('VP_vs','VS'),('VP_vix','VIX')]:
    s=panel[col]
    print(f'  VP_{label}: mean={s.mean():.3f}  std={s.std():.3f}  min={s.min():.3f}  max={s.max():.3f}')

print(f'\nIS summary:')
for xcol,r,label in [('VS2_lag',res_vs,'VS'),('VIX2_lag',res_vix,'VIX')]:
    print(f'  HAR-RV-{label}: adj_r2={r["adj_r2"]:.4f}  rmse={r["rmse"]:.3f}')
    for v in ['const',xcol,'RV22_lag','RV5_lag','RV1_lag']:
        print(f'    {v:<14} coef={r["params"][v]:>8.4f}  t={r["tstat"][v]:>6.2f}')

print(f'\nOOS summary (split={split.date()}):')
for label,o in [('VS',oos_vs),('VIX',oos_vix)]:
    print(f'  HAR-RV-{label}: IS_adj_r2={o["is_adj_r2"]:.4f}  OOS_mzr2={o["oos_mz_r2"]:.4f}  rmse={o["oos_rmse"]:.3f}  mae={o["oos_mae"]:.3f}  mape={o["oos_mape"]:.4f}')
