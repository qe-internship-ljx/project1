"""
Diagnostic: why does the short strategy fail outside 2006-2011?

Four root causes investigated:
  1. Entry roll size over time      -- is the carry buffer shrinking?
  2. VIX-move distribution          -- are spikes larger / more frequent?
  3. Hedge effectiveness by era     -- does ES still offset VIX moves?
  4. Roll-to-cost coverage          -- does roll income clear fixed costs?
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from vix_basis.panel import build_daily_panel
from vix_basis.hedge_ratio import compute_oos_hedge_ratios
from vix_basis.simulator import run_simulation, trades_to_dataframe, VIX_RT_COST, ES_RT_COST

# ── Build data ────────────────────────────────────────────────────────────
panel  = build_daily_panel("2004-01-01", "2026-12-31")
hr     = compute_oos_hedge_ratios(panel, "2004-04-05", "2005-01-01")
trades = run_simulation(panel, hr, "2005-01-01", "2026-03-31")

df    = trades_to_dataframe(trades)
short = df[df["direction"] == "short"].copy()
short["year"] = short["entry_date"].dt.year

# VIX-move P&L = what the unhedged VIX position earned beyond the roll
# positive = VIX fell more than expected (good for short)
# negative = VIX rose during the hold (bad for short)
short["vix_move_pnl"] = short["pnl_unhedged"] - short["pnl_roll"]

# Attach entry-day panel data
panel_idx = panel.set_index("date")
short["entry_vix"]  = short["entry_date"].map(panel_idx["vix_spot"])
short["entry_es"]   = short["entry_date"].map(panel_idx["es_price"])

ERAS = [
    ("2005-2007", 2005, 2007),
    ("2008-2011", 2008, 2011),
    ("2012-2014", 2012, 2014),
    ("2015-2017", 2015, 2017),
    ("2018-2020", 2018, 2020),
    ("2021-2026", 2021, 2026),
]

SEP = "-" * 78

# ── 1. P&L waterfall decomposition ───────────────────────────────────────
print()
print("=" * 78)
print("1. P&L WATERFALL BY ERA")
print("   Roll income = carry earned while open (always +)")
print("   VIX-move    = residual loss/gain from actual VIX price change")
print("   ES hedge    = gain/loss on the S&P500 futures hedge")
print("   Net hedged  = sum of all three, minus transaction costs")
print("=" * 78)
print(f"  {'Era':<14} {'N':>4}  {'Roll':>8}  {'VIX-move':>10}  {'ES hedge':>9}  {'Net hedged':>10}")
print("  " + SEP[:72])
for lbl, y0, y1 in ERAS:
    g = short[(short["year"] >= y0) & (short["year"] <= y1)]
    if len(g) == 0:
        continue
    print(
        f"  {lbl:<14} {len(g):>4}"
        f"  {g.pnl_roll.mean():>+8,.0f}"
        f"  {g.vix_move_pnl.mean():>+10,.0f}"
        f"  {g.pnl_es_hedge.mean():>+9,.0f}"
        f"  {g.pnl_hedged.mean():>+10,.0f}"
    )

# ── 2. Entry roll size (carry buffer available per trade) ─────────────────
print()
print("=" * 78)
print("2. ENTRY ROLL SIZE BY ERA")
print("   Roll at entry = basis / TTS  (daily carry rate, VIX points/day)")
print("   Smaller entry roll = less buffer to absorb costs + VIX moves")
print("=" * 78)
print(f"  {'Era':<14} {'N':>4}  {'Mean roll':>10}  {'Median roll':>11}  "
      f"{'pct10 roll':>10}  {'VIX mean':>9}")
print("  " + SEP[:72])
for lbl, y0, y1 in ERAS:
    g = short[(short["year"] >= y0) & (short["year"] <= y1)]
    if len(g) == 0:
        continue
    print(
        f"  {lbl:<14} {len(g):>4}"
        f"  {g.entry_roll.mean():>+10.3f}"
        f"  {g.entry_roll.median():>+11.3f}"
        f"  {g.entry_roll.quantile(0.10):>+10.3f}"
        f"  {g.entry_vix.mean():>9.1f}"
    )

# ── 3. VIX-move distribution (spike risk) ────────────────────────────────
print()
print("=" * 78)
print("3. VIX-MOVE P&L DISTRIBUTION (during short hold period)")
print("   Negative skew = fat left tail = occasional large spike losses dominate")
print("=" * 78)
print(f"  {'Era':<14} {'N':>4}  {'Mean':>8}  {'Median':>8}  "
      f"{'% adverse':>10}  {'Mean loss':>10}  {'Mean gain':>9}  {'Skew':>6}")
print("  " + SEP[:76])
for lbl, y0, y1 in ERAS:
    g     = short[(short["year"] >= y0) & (short["year"] <= y1)]
    vm    = g["vix_move_pnl"]
    if len(g) == 0:
        continue
    loss  = vm[vm < 0]
    gain  = vm[vm > 0]
    print(
        f"  {lbl:<14} {len(g):>4}"
        f"  {vm.mean():>+8,.0f}"
        f"  {vm.median():>+8,.0f}"
        f"  {len(loss)/len(vm)*100:>9.0f}%"
        f"  {loss.mean():>+10,.0f}"
        f"  {gain.mean() if len(gain) else 0:>+9,.0f}"
        f"  {vm.skew():>6.2f}"
    )

# ── 4. Hedge effectiveness: ES offset of VIX move ────────────────────────
print()
print("=" * 78)
print("4. HEDGE EFFECTIVENESS: did the ES hedge offset VIX-move losses?")
print("   Offset ratio = ES hedge P&L / VIX-move loss (100% = perfect hedge)")
print("   HR = average |hedge ratio| (ES contracts per VIX contract)")
print("=" * 78)
print(f"  {'Era':<14} {'N':>4}  {'VIX-move':>10}  {'ES hedge':>9}  "
      f"{'Offset':>9}  {'HR mean':>8}  {'HR std':>7}")
print("  " + SEP[:72])
for lbl, y0, y1 in ERAS:
    g        = short[(short["year"] >= y0) & (short["year"] <= y1)]
    if len(g) == 0:
        continue
    vix_loss = g["vix_move_pnl"].mean()
    hedge    = g["pnl_es_hedge"].mean()
    offset   = hedge / (-vix_loss) if vix_loss < 0 else float("nan")
    print(
        f"  {lbl:<14} {len(g):>4}"
        f"  {vix_loss:>+10,.0f}"
        f"  {hedge:>+9,.0f}"
        f"  {offset:>8.1%}"
        f"  {abs(g.hr).mean():>8.2f}"
        f"  {abs(g.hr).std():>7.2f}"
    )

# ── 5. Roll vs cost coverage ──────────────────────────────────────────────
avg_hr      = abs(short["hr"]).mean()
fixed_cost  = VIX_RT_COST + avg_hr * ES_RT_COST
breakeven   = fixed_cost  # roll must exceed this just to cover transaction costs

print()
print("=" * 78)
print("5. ROLL vs TRANSACTION COST COVERAGE")
print(f"   VIX round-trip cost  : ${VIX_RT_COST:.0f}")
print(f"   ES  round-trip cost  : ${ES_RT_COST:.2f} x |HR|")
print(f"   Avg |HR|             : {avg_hr:.2f}  -> ES cost = ${avg_hr * ES_RT_COST:.0f}")
print(f"   Total cost floor     : ${fixed_cost:.0f}  (roll must beat this just to break even on costs)")
print("=" * 78)
print(f"  {'Era':<14} {'Roll mean':>11}  {'Cost floor':>10}  {'Coverage':>10}  {'Profitable after cost?':>22}")
print("  " + SEP[:72])
for lbl, y0, y1 in ERAS:
    g    = short[(short["year"] >= y0) & (short["year"] <= y1)]
    if len(g) == 0:
        continue
    roll = g["pnl_roll"].mean()
    hr_g = abs(g["hr"]).mean()
    cost = VIX_RT_COST + hr_g * ES_RT_COST
    cov  = roll / cost
    ok   = "YES" if roll > cost else "NO  <-- costs eat the carry"
    print(f"  {lbl:<14} {roll:>+11,.0f}  {cost:>+10,.0f}  {cov:>9.2f}x  {ok}")

# ── 6. VIX regime: contango depth vs VIX level ───────────────────────────
print()
print("=" * 78)
print("6. CONTANGO DEPTH IN CALM REGIMES (the structural problem)")
print("   When VIX is low, contango is shallow => tiny roll => roll barely covers costs")
print("=" * 78)
vix_bins  = [0, 15, 20, 25, 35, 100]
vix_lbls  = ["VIX<15", "15-20", "20-25", "25-35", "VIX>35"]
short["vix_bucket"] = pd.cut(
    short["entry_vix"], bins=vix_bins, labels=vix_lbls, right=True
)
print(f"  {'VIX bucket':<12} {'N':>5}  {'Roll mean':>11}  {'Net hedged':>11}  {'Win rate':>10}")
print("  " + SEP[:58])
for bkt in vix_lbls:
    g = short[short["vix_bucket"] == bkt]
    if len(g) == 0:
        continue
    print(
        f"  {bkt:<12} {len(g):>5}"
        f"  {g.pnl_roll.mean():>+11,.0f}"
        f"  {g.pnl_hedged.mean():>+11,.0f}"
        f"  {(g.pnl_hedged > 0).mean()*100:>9.1f}%"
    )

print()
