import os
import logging
from pathlib import Path
from dataclasses import dataclass
import datetime
# Basic
# PROJECT_NAME = "gantry-demo-data-backfill"

@dataclass
class GantryConfig:
    GANTRY_API_KEY: str = os.environ.get("GANTRY_API_KEY")
    GANTRY_APP_NAME: str = "gec-demo-app"
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
    S3_BUCKET: str = "gantry-demo-data"
    S3_OBJECT: str = "gec-demo-data.csv"
    MIN_DATE: datetime = datetime.datetime(2022, 3, 30, 0 ,0)
    MAX_DATE: datetime = datetime.datetime(2022, 5, 1, 0, 0)