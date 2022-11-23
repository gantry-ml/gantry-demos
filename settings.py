from pathlib import Path

from dotenv import dotenv_values

TUTORIAL_ROOT_DIR = Path(__file__).parent.resolve()
ENV_VARS = dotenv_values(TUTORIAL_ROOT_DIR / ".env")

# Gantry settings
GANTRY_API_KEY = "you must have a valid api key"
GANTRY_APPLICATION_NAME = "Gantry Tutorial Application"
GANTRY_APPLICATION_VERSION = 0

# Model settings
MODEL_DIR = TUTORIAL_ROOT_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)
MODEL_FILENAME = MODEL_DIR / "model.pkl"

# Data settings
DATA_DIR = TUTORIAL_ROOT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
DATASET_NAMES = [
    (DATA_DIR / "train.csv"),
    (DATA_DIR / "test.csv"),
    (DATA_DIR / "feature_shift.csv"),
    (DATA_DIR / "label_shift.csv"),
]
FILENAME = DATA_DIR / "loans.json"
FEEDBACK_FILENAME = DATA_DIR / "feedback.json"
TRAIN_FILENAME = DATA_DIR / "train.csv"

