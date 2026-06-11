import pandas as pd
import numpy as np
import os

# ── Load data ──────────────────────────────────────────────────────────────────
data_path = os.path.join(os.path.dirname(__file__), '..', 'spx_total_returns.csv')
df = pd.read_csv(data_path, parse_dates=['date'])
df = df.dropna(subset=['monthly_spx_total_return', 'monthly_1m_tbill_return'])
df = df.sort_values('date').reset_index(drop=True)

# ── Statistics ─────────────────────────────────────────────────────────────────
def compute_stats(sub: pd.DataFrame) -> dict:
    r  = sub['monthly_spx_total_return'].values
    rf = sub['monthly_1m_tbill_return'].values
    n  = len(r)

    mean_m   = r.mean()
    std_m    = r.std(ddof=1)
    excess   = r - rf
    mean_ex  = excess.mean()
    std_ex   = excess.std(ddof=1)

    ann_mean   = mean_m * 12
    ann_vol    = std_m  * np.sqrt(12)
    ann_se     = ann_vol / np.sqrt(n)
    ann_sharpe = mean_ex * np.sqrt(12) / std_ex   # annualised SR
    t_stat     = mean_m / (std_m / np.sqrt(n))    # H0: mean = 0

    return dict(
        N          = n,
        Ann_Mean   = ann_mean,
        Ann_Vol    = ann_vol,
        Ann_SE     = ann_se,
        Ann_Sharpe = ann_sharpe,
        T_stat     = t_stat,
    )

# ── Sample periods ─────────────────────────────────────────────────────────────
periods = [
    ('2016-05-31', '2026-05-31', '10Y  2016-05 to 2026-05'),
    ('2006-05-31', '2026-05-31', '20Y  2006-05 to 2026-05'),
    ('1986-05-31', '2026-05-31', '40Y  1986-05 to 2026-05'),
    ('1946-05-31', '2026-05-31', '80Y  1946-05 to 2026-05'),
]

rows = []
for start, end, label in periods:
    mask = (df['date'] >= start) & (df['date'] <= end)
    stats = compute_stats(df.loc[mask])
    rows.append({'Period': label, **stats})

results = pd.DataFrame(rows).set_index('Period')

# ── Format & display ───────────────────────────────────────────────────────────
fmt = {
    'N':          '{:>6.0f}',
    'Ann_Mean':   '{:>9.4%}',
    'Ann_Vol':    '{:>9.4%}',
    'Ann_SE':     '{:>9.4%}',
    'Ann_Sharpe': '{:>9.4f}',
    'T_stat':     '{:>9.4f}',
}

print('\nSP500 Total Return - Summary Statistics')
print('=' * 76)
header = f"{'Period':<26} {'N':>6} {'Ann Mean':>9} {'Ann Vol':>9} {'Ann SE':>9} {'Sharpe':>9} {'T-stat':>9}"
print(header)
print('-' * 76)
for idx, row in results.iterrows():
    line = (f"{idx:<26} "
            f"{row['N']:>6.0f} "
            f"{row['Ann_Mean']:>9.4%} "
            f"{row['Ann_Vol']:>9.4%} "
            f"{row['Ann_SE']:>9.4%} "
            f"{row['Ann_Sharpe']:>9.4f} "
            f"{row['T_stat']:>9.4f}")
    print(line)
print('=' * 76)
print()
print('Notes:')
print('  Ann Mean   = mean(monthly return) * 12')
print('  Ann Vol    = std(monthly return)  * sqrt(12)   [sample std, ddof=1]')
print('  Ann SE     = Ann Vol / sqrt(N)                 [SE of annualised mean]')
print('  Ann Sharpe = mean(excess return) * sqrt(12) / std(excess return)')
print('               excess return = SPX return - 1m T-bill return')
print('  T-stat     = mean(monthly return) / (std / sqrt(N)),  H0: mean = 0')

# ── Save CSV ───────────────────────────────────────────────────────────────────
out_path = os.path.join(os.path.dirname(__file__), 'results.csv')
results.to_csv(out_path, float_format='%.6f')
print(f'\nResults saved to {out_path}')
