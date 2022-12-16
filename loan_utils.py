import os
import pickle

from loguru import logger
import numpy as np
import pandas as pd

from generate_data import generate_all_data
from settings import MODEL_FILENAME, DATASET_NAMES, TRAIN_FILENAME


def loan_data():
    if not TRAIN_FILENAME.exists():
        generate_all_data()

    return {
        dataset_name.stem: pd.read_csv(dataset_name) for dataset_name in DATASET_NAMES
    }


def loan_features(input_data):
    features = ["college_degree", "loan_amount"]
    X = input_data[features].copy()
    X["college_degree"] = X["college_degree"].astype(bool)
    X = X.fillna(False)
    return X


def loan_labels(input_data):
    y = input_data["loan_repaid"].copy().astype(int)
    return y


def normalize_features(X, mean, std):
    return (X - mean) / std


def preprocess_features(loan_id, college_degree, loan_amount, X_mean, X_std):
    # First process & normalize the features
    college_degree = int(college_degree)
    raw_features = np.array([college_degree, loan_amount])
    raw_features = np.nan_to_num(raw_features)  # fills nans as 0
    preprocessed_features = (raw_features - X_mean) / X_std
    return preprocessed_features


def save_model_and_artifacts(model, X_train_mean, X_train_std):
    model_and_artifacts = (model, X_train_mean, X_train_std)
    with MODEL_FILENAME.open("wb") as f:
        pickle.dump(model_and_artifacts, f)
    logger.info(f"Saved model to {str(MODEL_FILENAME)}")


def load_model_and_artifacts():
    with MODEL_FILENAME.open("rb") as f:
        model, X_train_mean, X_train_std = pickle.load(f)
    return (model, X_train_mean, X_train_std)
