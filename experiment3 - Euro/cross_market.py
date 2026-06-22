"""
cross_market.py
===============
Thin driver that runs the experiment2 expanding-window pipeline — regressions.py,
base_strategies.py and leveraged_strategies.py (all in "experiment2 - Return
Regression") — on an arbitrary market panel.

This module lives in "experiment3 - Euro"; the NASDAQ experiment imports it from
here (see nasdaq_expanding_window.py).

The euro / nasdaq experiment files only build a panel with the standard
experiment2 column names ("VP", "term_slope", "vvix_ma5", "fwd_20d",
"daily_ret") from their own data and then call ``run_all``.  Every regression,
position rule, simulation and plot is reused verbatim from the three modules;
only the input data changes.

Caches and figure outputs are redirected to the calling market's own folders
(``_redirect``) so the US (experiment2) caches are never read or overwritten —
the beta/position caches are keyed by predictor name only, which would otherwise
collide across markets.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent
EXP2 = ROOT.parent / "experiment2 - Return Regression"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXP2))
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT.parent / "bh_replication"))

import regressions as reg
import base_strategies as bs
import leveraged_strategies as lev
from helpers import compute_buy_and_hold, simulate_strategy

FWD = "fwd_20d"

# Base-plot 4-delta palettes (reused from base_strategies.main in this folder).
PAL = {
    "VP":         ["#08306b", "#2171b5", "#4292c6", "#6baed6"],
    "vvix_ma5":   ["#3f007d", "#6a51a3", "#807dba", "#9e9ac8"],
    "term_slope": ["#00441b", "#006d2c", "#31a354", "#74c476"],
    "biv":        ["#3f007d", "#6a51a3", "#9e9ac8", "#dadaeb"],
}

# Base unit-position strategies: (position fn, plot fn, filename stem).
_BASE_UNI = [
    (bs.run_ew,       bs.plot_2panel, "symmetric"),
    (bs.run_ew_asym,  bs.plot_asym,   "asymmetric"),
    (bs.run_ew_rolmu, bs.plot_rolmu,  "base_return_shift"),
]
_BASE_BIV = [
    (bs.run_ew_bivariate,       bs.plot_2panel_bivariate, "symmetric"),
    (bs.run_ew_asym_bivariate,  bs.plot_asym_bivariate,   "asymmetric"),
    (bs.run_ew_rolmu_bivariate, bs.plot_rolmu_bivariate,  "base_return_shift"),
]
# Leveraged (leveraged) strategies: type -> (uni fn, biv fn, filename stem).
_LEV = {
    "sym":   (lev.run_ew_leveraged_sym,   lev.run_ew_biv_leveraged_sym,   "symmetric"),
    "asym":  (lev.run_ew_leveraged_asym,  lev.run_ew_biv_leveraged_asym,  "asymmetric"),
    "rolmu": (lev.run_ew_leveraged_rolmu, lev.run_ew_biv_leveraged_rolmu, "base_return_shift"),
}


def _redirect(out_root: Path, cache_dir: Path):
    """Point every module's cache + output globals at this market's folders."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    for mod in (reg, bs, lev):
        mod.CACHE_DIR = cache_dir
        mod.OUTPUT = out_root


def _fname(stem, label):
    return f"{stem}_{label.replace(' ', '_')}.png"


def run_univariate(panel, daily_ret, bah_sim, pred, label, out_dir, color):
    """All base + leveraged univariate plots for one predictor. Returns the
    leveraged-asymmetric sim (used by the cross-model comparison)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pal   = PAL.get(pred, PAL["VP"])
    betas = reg.compute_betas(panel, pred, FWD, reg.OOS_GAP, reg.NW_LAGS)

    for runner, plot, stem in _BASE_UNI:
        sim_dict = {di: (d, simulate_strategy(
                        runner(panel, pred, FWD, reg.OOS_GAP, reg.NW_LAGS, d), daily_ret))
                    for di, d in enumerate(reg.DELTAS)}
        plot(pred_label=label, horizon_label="20-day",
             oos_gap=reg.OOS_GAP, nw_lags=reg.NW_LAGS, color_palette=pal,
             sim_dict=sim_dict, betas_df=betas, bah_sim=bah_sim,
             out_path=out_dir / _fname(stem, label))

    yh = reg._yhat_univariate(panel, pred, FWD, betas)
    mu = reg._rolling_mu(panel, FWD, reg.OOS_GAP, reg.RW, predictor=pred)
    asym_sim = None
    for t, (uni_fn, _, stem) in _LEV.items():
        sim = simulate_strategy(uni_fn(panel, pred, FWD, reg.OOS_GAP, reg.NW_LAGS), daily_ret)
        lev.plot_leveraged_univariate(label, "20-day", t, color, sim, betas, bah_sim,
                                    yh, mu, out_dir / _fname("leveraged_" + stem, label))
        if t == "asym":
            asym_sim = sim
    return asym_sim


def run_bivariate(panel, daily_ret, bah_sim, p1, p2, l1, l2, out_dir, color):
    """All base + leveraged bivariate plots for one predictor pair."""
    out_dir.mkdir(parents=True, exist_ok=True)
    label = f"{l1} + {l2}"
    betas = reg.compute_betas_bivariate(panel, p1, p2, FWD, reg.OOS_GAP, reg.NW_LAGS)

    for runner, plot, stem in _BASE_BIV:
        sim_dict = {di: (d, simulate_strategy(
                        runner(panel, p1, p2, FWD, reg.OOS_GAP, reg.NW_LAGS, d), daily_ret))
                    for di, d in enumerate(reg.DELTAS)}
        plot(pred1_label=l1, pred2_label=l2, horizon_label="20-day",
             oos_gap=reg.OOS_GAP, nw_lags=reg.NW_LAGS, color_palette=PAL["biv"],
             sim_dict=sim_dict, betas_df=betas, bah_sim=bah_sim,
             out_path=out_dir / _fname(stem, label))

    yh = reg._yhat_bivariate(panel, p1, p2, FWD, betas)
    mu = reg._rolling_mu(panel, FWD, reg.OOS_GAP, reg.RW, predictor=[p1, p2])
    asym_sim = None
    for t, (_, biv_fn, stem) in _LEV.items():
        sim = simulate_strategy(biv_fn(panel, p1, p2, FWD, reg.OOS_GAP, reg.NW_LAGS), daily_ret)
        lev.plot_leveraged_bivariate(l1, l2, "20-day", t, color, sim, betas, bah_sim,
                                   yh, mu, out_dir / _fname("leveraged_" + stem, label))
        if t == "asym":
            asym_sim = sim
    return asym_sim


def run_all(panel, out_root: Path, cache_dir: Path, vv_label: str):
    """Run VRP, vol-of-vol (``vv_label``), VRP+Term Slope and VRP+vol-of-vol
    base + leveraged strategies, plus the leveraged-asymmetric comparison.

    ``vv_label`` is the display/folder name for the vol-of-vol signal carried in
    the panel's ``vvix_ma5`` column ("VVIX MA5" for NASDAQ, "VV2TX MA5" for euro).
    """
    _redirect(out_root, cache_dir)
    daily_ret = panel["daily_ret"].dropna()
    bah_sim   = simulate_strategy(compute_buy_and_hold(daily_ret), daily_ret)
    ew        = out_root / "expanding_window"

    print("\n[VRP]"); run_univariate(panel, daily_ret, bah_sim, "VP", "VRP",
                                     ew / "VRP", lev.C_VRP)
    print(f"\n[{vv_label}]"); vv_asym = run_univariate(
        panel, daily_ret, bah_sim, "vvix_ma5", vv_label, ew / vv_label, lev.C_VVIX)
    print("\n[VRP + Term Slope]"); term_asym = run_bivariate(
        panel, daily_ret, bah_sim, "VP", "term_slope", "VRP", "Term Slope",
        ew / "VRP + Term Slope", lev.C_TERM)
    print(f"\n[VRP + {vv_label}]"); biv_asym = run_bivariate(
        panel, daily_ret, bah_sim, "VP", "vvix_ma5", "VRP", vv_label,
        ew / f"VRP + {vv_label}", lev.C_BIV)

    print("\n[Comparison] leveraged asymmetric")
    cmp_dir = ew / "comparisons"; cmp_dir.mkdir(parents=True, exist_ok=True)
    lev.plot_leveraged_asymmetric_comparison(
        vv_asym, biv_asym, term_asym, bah_sim,
        cmp_dir / "leveraged_asymmetric_comparison.png")
