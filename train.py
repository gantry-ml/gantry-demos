from loguru import logger
from sklearn.linear_model import LogisticRegression
import gantry
import pandas as pd

from loan_utils import (
    loan_data,
    loan_features,
    loan_labels,
    save_model_and_artifacts,
    normalize_features,
)
from settings import (
    GANTRY_API_KEY,
    GANTRY_APPLICATION_NAME,
    GANTRY_APPLICATION_VERSION,
)

# Initialize Gantry
gantry.init(
    api_key=GANTRY_API_KEY,
    environment=f"train-{pd.Timestamp.now().isoformat()}",
)


def train():

    # Load data
    train = loan_data()["train"]
    X_train = loan_features(train).astype(float)
    y_true = loan_labels(train)

    X_train_norm = normalize_features(
        X_train,
        X_train.mean(),
        X_train.std(),
    )

    # Train and save model
    model = LogisticRegression()
    model.fit(X_train_norm, y_true)
    score = model.score(X_train_norm, y_true)
    logger.info("Fit a model with training accuracy {}".format(score))
    save_model_and_artifacts(model, X_train.mean().values, X_train.std().values)
    logger.info("\U0001F389 Model training successful! \U0001F389")

    # Assign predictions
    y_pred = (
        pd.Series(model.predict(X_train_norm).astype(bool))
        .rename("loan_repaid_pred")
        .to_frame()
    )

    # Log to Gantry
    gantry.log_records(
        GANTRY_APPLICATION_NAME,
        version=GANTRY_APPLICATION_VERSION,
        inputs=X_train.to_dict("records"),
        outputs=y_pred.to_dict("records"),
        as_batch=True,
    )


if __name__ == "__main__":
    train()
    
