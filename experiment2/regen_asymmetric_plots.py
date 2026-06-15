"""
regen_asymmetric_plots.py
=========================
Regenerates only the asymmetric-threshold plots using cached positions and betas.
Adds post-signal activation B&H baseline per updated plot.md spec.
"""
import sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT.parent / "bh_replication"))
sys.path.insert(0, str(ROOT))

# fixed_split_eval runs everything at module level — all positions and betas
# are loaded from cache, so this import is fast after the first run.
import fixed_split_eval as fse

_ASYM_MODELS = ["Base", "Model_A", "Model_C", "Model_VVIX", "Model_G", "Model_H"]

print("\nRegenerating asymmetric-threshold plots with activation B&H baseline...")
for m in _ASYM_MODELS:
    pos      = fse.EW_ASYM[m]
    st, sim  = fse.EW_ASYM_SIM[m]
    fname    = "asymmetric_" + fse.EW_MODEL_DIR[m].name.replace(" ", "_") + ".png"
    out_path = fse.EW_MODEL_DIR[m] / fname
    fse.plot_expanding_asymmetric(m, pos, sim, st, out_path)

print("\nDone.")
