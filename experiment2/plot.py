"""
plot.py
=======
Regenerates all non-poor-correlation plots under output/expanding_window/ in one shot.

Poor-correlation baselines (Vol Trend, VIX, Term Slope univariate, Expanding VRP)
are intentionally excluded.  Run horizon_regression.py directly to regenerate those.

Generated outputs
-----------------
  === Non-leveraged (unit-position) strategies ===
  output/expanding_window/VRP/symmetric_VRP.png
  output/expanding_window/VRP/asymmetric_VRP.png
  output/expanding_window/VRP/base_return_shift_VRP.png
  output/expanding_window/VVIX MA5/symmetric_VVIX_MA5.png
  output/expanding_window/VVIX MA5/asymmetric_VVIX_MA5.png
  output/expanding_window/VVIX MA5/base_return_shift_VVIX_MA5.png
  output/expanding_window/VVIX MA10/symmetric_VVIX_MA10.png
  output/expanding_window/VVIX MA10/asymmetric_VVIX_MA10.png
  output/expanding_window/VVIX MA10/base_return_shift_VVIX_MA10.png
  output/expanding_window/VRP + VVIX MA5/symmetric_VRP_+_VVIX_MA5.png
  output/expanding_window/VRP + VVIX MA5/asymmetric_VRP_+_VVIX_MA5.png
  output/expanding_window/VRP + VVIX MA5/base_return_shift_VRP_+_VVIX_MA5.png
  output/expanding_window/VRP + VVIX MA10/symmetric_VRP_+_VVIX_MA10.png
  output/expanding_window/VRP + VVIX MA10/asymmetric_VRP_+_VVIX_MA10.png
  output/expanding_window/VRP + VVIX MA10/base_return_shift_VRP_+_VVIX_MA10.png
  output/expanding_window/VRP + Term Slope/symmetric_VRP_+_Term_Slope.png
  output/expanding_window/VRP + Term Slope/asymmetric_VRP_+_Term_Slope.png
  output/expanding_window/VRP + Term Slope/base_return_shift_VRP_+_Term_Slope.png
  output/expanding_window/VRP + Open Interest/symmetric_VRP_+_Open_Interest.png
  output/expanding_window/VRP + Open Interest/asymmetric_VRP_+_Open_Interest.png
  output/expanding_window/VRP + Open Interest/base_return_shift_VRP_+_Open_Interest.png

  === Leveraged (unbound) strategies ===
  output/expanding_window/VVIX MA5/leveraged_{symmetric,asymmetric,base_return_shift}_VVIX_MA5.png
  output/expanding_window/VVIX MA10/leveraged_{symmetric,asymmetric,base_return_shift}_VVIX_MA10.png
  output/expanding_window/VRP/leveraged_{symmetric,asymmetric,base_return_shift}_VRP.png
  output/expanding_window/VRP + VVIX MA5/leveraged_{symmetric,asymmetric,base_return_shift}_VRP_+_VVIX_MA5.png
  output/expanding_window/VRP + VVIX MA10/leveraged_{symmetric,asymmetric,base_return_shift}_VRP_+_VVIX_MA10.png
  output/expanding_window/VRP + Term Slope/leveraged_{symmetric,asymmetric,base_return_shift}_VRP_+_Term_Slope.png
  output/expanding_window/VRP + Open Interest/leveraged_{symmetric,asymmetric,base_return_shift}_VRP_+_Open_Interest.png
  output/expanding_window/trivariate/leveraged_{symmetric,asymmetric,base_return_shift}_VRP_+_VVIX_MA5_+_Term_Slope.png
  output/expanding_window/comparisons/leveraged_asymmetric_vvix_vs_vrp_vvix.png

All regression positions and betas are cached in output/regression_cache/, so
re-runs only redo the plot rendering.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

import base_strategies
import leveraged_strategies


def main():
    base_strategies.main()
    leveraged_strategies.main()


if __name__ == "__main__":
    main()
