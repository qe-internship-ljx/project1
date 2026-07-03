"""Full VIX vs VS comparison including DM test and forecast encompassing.

Data loading, RV construction and HAR estimation are shared with the rest of
the replication: panels come from data_prep, estimators from har_model
(parameterised by ``xcol`` to switch the implied-variance predictor).
"""
import warnings; warnings.filterwarnings('ignore')
import sys
import numpy as np, pandas as pd
from pathlib import Path
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

sys.path.insert(0, str(Path(__file__).parent))
from data_prep import load_sp500_returns, load_vix, load_variance_swap, compute_rv_components
from har_model import estimate_har, out_of_sample_forecast, _nw_se

# ── Build unified panel (VS-restricted sample, both predictors) ───────────────
sp_ret = load_sp500_returns()
panel = compute_rv_components(sp_ret).join(
    (load_variance_swap()**2/12).rename('VS2'), how='inner')
panel['VIX2']     = load_vix()**2 / 12
panel['RV22_fwd'] = panel['RV22'].shift(-22)
for c in ['VS2', 'VIX2', 'RV22', 'RV5', 'RV1']:
    panel[f'{c}_lag'] = panel[c].shift(1)
panel = panel.dropna()

print(f'Panel: {len(panel):,} obs  {panel.index.min().date()} -> {panel.index.max().date()}')

# ── IS estimation ─────────────────────────────────────────────────────────────
res_vs  = estimate_har(panel, 'VS IS',  xcol='VS2_lag')
res_vix = estimate_har(panel, 'VIX IS', xcol='VIX2_lag')

# ── OOS ───────────────────────────────────────────────────────────────────────
split = panel.index[int(0.75*len(panel))]
oos_vs  = out_of_sample_forecast(panel, split, 'VS OOS',  xcol='VS2_lag')
oos_vix = out_of_sample_forecast(panel, split, 'VIX OOS', xcol='VIX2_lag')

# ── Diebold-Mariano test ──────────────────────────────────────────────────────
# H0: equal MSE.  d_t = e_VS_t^2 - e_VIX_t^2.  t = mean(d) / NW-SE(d)
e_vs  = oos_vs['y_test'] - oos_vs['y_hat']
e_vix = oos_vix['y_test'] - oos_vix['y_hat']
d = e_vs**2 - e_vix**2           # positive = VS worse
d_mean = d.mean()
# NW SE of d using regression on constant
dm_res = OLS(d.values, np.ones(len(d))).fit()
dm_nw  = float(_nw_se(dm_res)[0])
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
enc_nw  = _nw_se(enc_res)
enc_vars = X_enc.columns.tolist()
print(f'\nForecast encompassing regression (actual ~ const + f_VS + f_VIX):')
for v, c, se in zip(enc_vars, enc_res.params, enc_nw):
    print(f'  {v:<8}  coef={c:.4f}  NW-SE={se:.4f}  t={c/se:.2f}')
print(f'  Adj-R²={enc_res.rsquared_adj:.4f}')

# ── Combined model IS ─────────────────────────────────────────────────────────
y_all   = panel['RV22_fwd']
X_both  = add_constant(panel[['VS2_lag','VIX2_lag','RV22_lag','RV5_lag','RV1_lag']])
res_both = OLS(y_all, X_both).fit()
ses_both = _nw_se(res_both)
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
    nw  = _nw_se(res, nlags=max(3,2*horizon))
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
    print(f'  HAR-RV-{label}: adj_r2={r["adj_r2"]:.4f}  rmse={r["rmse_is"]:.3f}')
    for v in ['const',xcol,'RV22_lag','RV5_lag','RV1_lag']:
        print(f'    {v:<14} coef={r["params"][v]:>8.4f}  t={r["t_stats"][v]:>6.2f}')

print(f'\nOOS summary (split={split.date()}):')
for label,o in [('VS',oos_vs),('VIX',oos_vix)]:
    print(f'  HAR-RV-{label}: IS_adj_r2={o["is_adj_r2"]:.4f}  OOS_mzr2={o["oos_mz_r2"]:.4f}  rmse={o["oos_rmse"]:.3f}  mae={o["oos_mae"]:.3f}  mape={o["oos_mape"]:.4f}')
