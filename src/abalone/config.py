"""Shared paths and reproducibility settings."""

from pathlib import Path

RANDOM_STATE = 42
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
_ORGANIZED_DATA_PATH = DATA_DIR / "raw" / "abalone.csv"
# Keep the workflow usable while an existing source CSV is still at repository root.
RAW_DATA_PATH = _ORGANIZED_DATA_PATH if _ORGANIZED_DATA_PATH.exists() else ROOT_DIR / "abalone.csv"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MONITORING_DIR = REPORTS_DIR / "monitoring"

MODEL_PATH = MODELS_DIR / "best_model.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
REFERENCE_PROFILE_PATH = MODELS_DIR / "reference_profile.json"


def ensure_output_directories() -> None:
    """Create the project output directories if they do not exist."""
    for path in (DATA_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR, MONITORING_DIR):
        path.mkdir(parents=True, exist_ok=True)
