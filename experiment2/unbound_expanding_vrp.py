"""
unbound_expanding_vrp.py
========================
Runs the three unbound position-sizing strategies (symmetric, asymmetric,
base_return_shift) on the Expanding VRP model and saves the plots to:

  output/expanding_window/poor_correlation/Expanding VRP/
    unbound_symmetric_VRP.png
    unbound_asymmetric_VRP.png
    unbound_base_return_shift_VRP.png

Uses the same cached positions and betas as unbound_strategies.py (VP predictor).
"""

import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT.parent / "bh_replication"))
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT))

from experiment2 import (
    load_vrp_series_expanding,
    load_es_front_month, load_vvix, compute_vvix_ma5,
    load_vix_spot, load_vix_futures_term_structure, load_es_open_interest,
    compute_buy_and_hold, simulate_strategy,
)
from fh_replication.fh_replication import compute_vix_term_slope
from horizon_regression import (
    build_panel, compute_betas, OOS_START,
)
from unbound_strategies import (
    run_ew_unbound_sym, run_ew_unbound_asym, run_ew_unbound_rolmu,
    _yhat_univariate, _rolling_mu,
    plot_unbound_univariate,
    _report,
    OOS_GAP as UB_OOS_GAP, NW_LAGS, RW, C_VRP,
)

OUTPUT = ROOT / "output"


def main():
    print("=" * 72)
    print("  Unbound Strategies — Expanding VRP")
    print("  sym / asym / base_return_shift  ->  output/expanding_window/poor_correlation/Expanding VRP/")
    print("=" * 72)

    print("\n[1] Loading data...")
    vrp        = load_vrp_series_expanding()
    es         = load_es_front_month()
    vvix_ma5   = compute_vvix_ma5(load_vvix())
    vix_spot   = load_vix_spot()
    term_slope = compute_vix_term_slope(load_vix_futures_term_structure())
    oi         = load_es_open_interest()
    panel      = build_panel(vrp, es, vvix_ma5, vix_spot, term_slope, oi)
    # Rename VP -> VP_exp so cache keys are distinct from the rolling-window VRP analyses.
    panel      = panel.rename(columns={"VP": "VP_exp"})
    print(f"    {len(panel):,} obs  "
          f"[{panel.index.min().date()} - {panel.index.max().date()}]")

    daily_ret = panel["daily_ret"].dropna()
    bah_sim   = simulate_strategy(compute_buy_and_hold(daily_ret), daily_ret)

    print("\n[2] Loading / computing betas (cached)...")
    betas_vrp = compute_betas(panel, "VP_exp", "fwd_20d", oos_gap=UB_OOS_GAP, nw_lags=NW_LAGS)
    print("    Done.")

    yh_vrp = _yhat_univariate(panel, "VP_exp", "fwd_20d", betas_vrp)
    mu_20d = _rolling_mu(panel, "fwd_20d", UB_OOS_GAP, RW, predictor="VP_exp")

    out_dir = OUTPUT / "expanding_window" / "poor_correlation" / "Expanding VRP"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── symmetric ───────────────────────────────────────────────────────────────
    print("\n[A] Expanding VRP unbound symmetric...")
    pos = run_ew_unbound_sym(panel, "VP_exp", "fwd_20d", UB_OOS_GAP, NW_LAGS)
    sim = _report(pos, daily_ret, "ub_sym_VRP_exp")
    plot_unbound_univariate(
        "Expanding VRP", "20-day", "sym", C_VRP,
        sim, betas_vrp, bah_sim, yh_vrp, mu_20d,
        out_dir / "unbound_symmetric_VRP.png",
    )

    # ── asymmetric ──────────────────────────────────────────────────────────────
    print("\n[B] Expanding VRP unbound asymmetric...")
    pos = run_ew_unbound_asym(panel, "VP_exp", "fwd_20d", UB_OOS_GAP, NW_LAGS)
    sim = _report(pos, daily_ret, "ub_asym_VRP_exp")
    plot_unbound_univariate(
        "Expanding VRP", "20-day", "asym", C_VRP,
        sim, betas_vrp, bah_sim, yh_vrp, mu_20d,
        out_dir / "unbound_asymmetric_VRP.png",
    )

    # ── base_return_shift ───────────────────────────────────────────────────────
    print("\n[C] Expanding VRP unbound base_return_shift...")
    pos = run_ew_unbound_rolmu(panel, "VP_exp", "fwd_20d", UB_OOS_GAP, NW_LAGS)
    sim = _report(pos, daily_ret, "ub_rolmu_VRP_exp")
    plot_unbound_univariate(
        "Expanding VRP", "20-day", "rolmu", C_VRP,
        sim, betas_vrp, bah_sim, yh_vrp, mu_20d,
        out_dir / "unbound_base_return_shift_VRP.png",
    )

    print("\nDone.")
    print(f"  {out_dir}/unbound_symmetric_VRP.png")
    print(f"  {out_dir}/unbound_asymmetric_VRP.png")
    print(f"  {out_dir}/unbound_base_return_shift_VRP.png")
    print("=" * 72)


if __name__ == "__main__":
    main()
