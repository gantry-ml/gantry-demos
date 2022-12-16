import os
import logging
from pathlib import Path
from dataclasses import dataclass

# Basic
# PROJECT_NAME = "gantry-demo-data-backfill"

# Setup logging
logger = logging.getLogger("gantry-example-app")
logger.setLevel(logging.INFO)

@dataclass
class GantryConfig:
    GANTRY_API_KEY: str = os.environ.get("GANTRY_API_KEY")
    GANTRY_APP_NAME: str = "demo-gec-app"
    GANTRY_PROD_ENV: str = "prod"
    GANTRY_EVAL_ENV: str = "eval"

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
