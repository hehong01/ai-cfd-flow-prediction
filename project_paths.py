from pathlib import Path
import os

# ============================================================
# Repository root
# ============================================================

REPO_ROOT = Path(__file__).resolve().parent


# ============================================================
# Local data root
#
# Default local structure:
#
# ai-cfd-flow-prediction/
# ├── github/
# └── ai-cfd-data/
#
# 다른 PC에서는 AI_CFD_DATA_ROOT 환경변수로 변경 가능
# ============================================================

_default_data_root = REPO_ROOT.parent / "ai-cfd-data"

DATA_ROOT = Path(
    os.environ.get("AI_CFD_DATA_ROOT", _default_data_root)
).resolve()


# ============================================================
# Data directories
# ============================================================

IMAGE_DIR = DATA_ROOT / "01_images"
STL_DIR = DATA_ROOT / "02_stl"
SPACECLAIM_DIR = DATA_ROOT / "03_spaceclaim"
FLUENT_DIR = DATA_ROOT / "04_fluent"
CFD_CSV_DIR = DATA_ROOT / "05_cfd_csv"

PREDICTION_DIR = DATA_ROOT / "07_predictions"
MLP_PREDICTION_DIR = PREDICTION_DIR / "mlp"
DGCNN_PREDICTION_DIR = PREDICTION_DIR / "dgcnn"

RESULT_DIR = DATA_ROOT / "08_results"
METRICS_DIR = RESULT_DIR / "metrics"
FIGURES_DIR = RESULT_DIR / "figures"
