from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT_DIR / "data" / "bank" / "bank-full.csv"
TEST_FILE = ROOT_DIR / "test_data.csv"

NUMERICAL_COLUMNS = [
    "age",
    "balance",
    "day",
    "duration",
    "campaign",
    "pdays",
    "previous",
]

CATEGORICAL_COLUMNS = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "poutcome",
]


def load_data():
    data = pd.read_csv(DATA_FILE, sep=";")
    return data


def split_data(data):
    X = data.drop("y", axis=1)
    y = data["y"].map({"no": 0, "yes": 1})

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test


def create_preprocessor():
    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", StandardScaler(), NUMERICAL_COLUMNS),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_COLUMNS,
            ),
        ]
    )
    return preprocessor


def save_test_data(X_test, y_test):
    test_data = X_test.copy()
    test_data["y"] = y_test.map({0: "no", 1: "yes"})
    test_data.to_csv(TEST_FILE, sep=";", index=False)


if __name__ == "__main__":
    bank_data = load_data()
    X_train, X_test, y_train, y_test = split_data(bank_data)
    save_test_data(X_test, y_test)

    print("Training rows:", len(X_train))
    print("Test rows:", len(X_test))
