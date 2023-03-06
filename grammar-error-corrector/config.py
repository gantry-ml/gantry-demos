import os
import logging
from dataclasses import dataclass
import datetime
from dotenv import dotenv_values

# Basic
# PROJECT_NAME = "gantry-demo-data-backfill"

config = dotenv_values(".env")
if config:
    os.environ.update(config)

GANTRY_API_KEY = os.environ.get("GANTRY_API_KEY")
if not GANTRY_API_KEY:
    raise ValueError("GANTRY_API_KEY not set in environment or .env file.")

GANTRY_APP_NAME = os.environ.get("GANTRY_APP_NAME", "gec-demo-app")


@dataclass
class GantryConfig:
    GANTRY_API_KEY: str = GANTRY_API_KEY
    GANTRY_APP_NAME: str = GANTRY_APP_NAME
    GANTRY_PROD_ENV: str = "prod"
    GANTRY_EVAL_ENV: str = "eval"


# Setup logging
logger = logging.getLogger(GantryConfig.GANTRY_APP_NAME)

logger.setLevel(logging.INFO)

# A location for pulling the model
@dataclass
class ModelConfig:
    HF_MODEL_PATH: str = "prithivida/grammar_error_correcter_v1"
    MODEL_DIR = "./.temp_model"


# Storage Config
@dataclass
class DataStorageConfig:
    DATASET_URL = (
        "https://gantry-demo-data.s3.us-west-2.amazonaws.com/gec-demo-data.csv"
    )
    MIN_DATE: datetime = datetime.datetime(2022, 3, 30, 0, 0)
    MAX_DATE: datetime = datetime.datetime(2022, 5, 1, 0, 0)
