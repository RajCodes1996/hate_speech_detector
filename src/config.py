from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent

# Use /var/data on Render, fall back to local BASE_DIR for localhost
PERSISTENT_DIR = Path(os.environ.get("PERSISTENT_STORAGE_PATH", BASE_DIR))

DATA_DIR = PERSISTENT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PERSISTENT_DIR / "models"
REPORTS_DIR = PERSISTENT_DIR / "reports"

DEFAULT_DATASET = RAW_DATA_DIR / "hate_speech.csv"
DEFAULT_MODEL_PATH = MODELS_DIR / "hate_speech_model.joblib"
DEFAULT_METRICS_PATH = REPORTS_DIR / "metrics.json"

RANDOM_STATE = 42
TEST_SIZE = 0.2
PREDICTION_THRESHOLD = 0.50
BORDERLINE_MARGIN = 0.10


def ensure_directories() -> None:
    for path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)