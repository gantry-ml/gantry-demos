import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from settings import DATA_DIR


def create_data(N, id_col, corrupt_features=False, corrupt_labels=False, p_nan=0):
    p = 0.7  # proportion that are educated (education=1)
    p_var = p * (1 - p)
    p_std = np.sqrt(p_var)
    education = np.random.uniform(low=0, high=1, size=N) <= p
    # As education goes up, loan status should go down

    loan_amount = np.random.normal(6000, 1500, size=N)  # ev=6000, std=1500
    noise_term = np.random.normal(0, 0.5, size=N)
    # As loan_amount goes up, loan_status should go down
    loan_status = (
        (education - p) / p_std - (loan_amount - 6000) / 1500 + noise_term
    ) >= 0  # noqa

    if corrupt_features:
        loan_amount += (np.random.uniform(low=0, high=1, size=N) <= 0.15) * 10000
    if corrupt_labels:
        size = int(0.25 * len(loan_amount))
        idx = np.random.choice(N, size, replace=False)
        loan_status = loan_status.astype(np.int64)
        loan_status[idx] -= 1  # Turn 0 -> -1, 1 -> 0
        loan_status = np.clip(loan_status, 0, 1)  # -1 -> 0
    if p_nan > 0:  # percentage of elements to set to nan
        # default to setting loan_amount feature to nan
        tot_nan = int(p_nan * N)
        nan_idx = np.random.choice(N, size=tot_nan)
        loan_amount[nan_idx] = np.nan

    d = {
        "loan_id": id_col,
        "college_degree": education,
        "loan_amount": loan_amount,
        "loan_repaid": loan_status,
    }
    return pd.DataFrame(data=d)


def compute_regression_score(res):
    res = res.copy()
    model = LogisticRegression()
    features = res[["college_degree", "loan_amount"]].fillna(0.0)
    model.fit(features, res["loan_repaid"])
    return model.score(features, res["loan_repaid"])


def generate_all_data():
    total = 0
    for name, N, corrupt_features, corrupt_labels in [
        ("train", 853, False, False),
        ("test", 428, False, False),
        ("feature_shift", 613, True, False),
        ("label_shift", 275, False, True),
    ]:
        id_col = ["ID%04d" % x for x in range(total, total + N)]
        total += N
        # TODO set p_nan > 0 when MomentSketch supports nans.
        df = create_data(N, id_col, corrupt_features, corrupt_labels)
        print(f"Generated data {name}")
        regression_score = compute_regression_score(df)
        print(f"In-sample regression accuracy: {regression_score}")
        save_path = DATA_DIR / f"{name}.csv"
        df.to_csv(save_path, index=False)
        print(f"Saved to path {save_path}")


if __name__ == "__main__":
    generate_all_data()


"""
# Inspired by the following dataset & pre-processing

dataset_loc = "https://raw.githubusercontent.com/shrikant-temburwar/Loan-Prediction-Dataset/master/"  # noqa
train_loc = dataset_loc + "train.csv"
test_loc = dataset_loc + "test.csv"

save_loc = "raw_data/loans/"

def download_loan_data():
    os.makedirs(save_loc, exist_ok=True)
    urllib.request.urlretrieve(train_loc, save_loc + "train.csv")
    urllib.request.urlretrieve(test_loc, save_loc + "test.csv")


def loan_data():
    if not os.path.exists(save_loc + "train.csv"):
        download_loan_data()

    return (
        pd.read_csv(save_loc + "train.csv"), pd.read_csv(save_loc + "test.csv")
    )


def load_train_feedback():
    return pd.read_csv(save_loc + "train.csv")[["Loan_ID", "Loan_Status"]]


def loan_features(input_data):
    features = ['Education', 'Self_Employed', 'ApplicantIncome',
                'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
                'Credit_History', 'Property_Area']
    X = input_data[features].copy()
    X['Education'] = (X['Education'] == 'Graduate').astype(int)
    X['Self_Employed'] = (X['Self_Employed'] == "Yes").astype(int)
    X['Property_Area'] = (X['Property_Area'] == "Urban").astype(int)
    X = X.fillna(0.)
    return X


def loan_labels(input_data):
    y = input_data['Loan_Status'].copy()
    y = (y == 'Y').astype(int)
    return y


def loan_postprocess(raw_labels):
    y = np.where(raw_labels == 1, "Y", "N")
    return y
"""
