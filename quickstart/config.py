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
    GANTRY_APP_NAME: str = "my-gec-app"
    GANTRY_PROD_ENV: str = "prod"
    GANTRY_EVAL_ENV: str = "eval"

# A location for pulling the model
HF_MODEL_PATH = "prithivida/grammar_error_correcter_v1"

# Storage Config
@dataclass
class DataStorageConfig:
    S3_BUCKET: str = "gantry-demo-data"
    S3_OBJECT: str = "gec-demo-data.csv"
