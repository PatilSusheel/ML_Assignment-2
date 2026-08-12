from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT_DIR / "data" / "bank" / "bank-full.csv"
TEST_FILE = ROOT_DIR / "test_data.csv"

# contacted_before is created by engineer_features() and added to the
# numerical columns. It is NOT part of the raw input columns.
NUMERICAL_COLUMNS = [
    "age",
    "balance",
    "day",
    "duration",
    "campaign",
    "pdays",
    "previous",
    "contacted_before",  # new binary flag from feature engineering
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


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    pdays = -1 means the client was NEVER previously contacted.
    StandardScaler treats -1 as a large negative outlier, which skews
    the entire column. We split it into two signals instead:
      - contacted_before: 1 if pdays != -1, else 0
      - pdays: replace -1 with 0 (harmless once contacted_before exists)
    """
    df = data.copy()
    df["contacted_before"] = (df["pdays"] != -1).astype(int)
    df["pdays"] = df["pdays"].replace(-1, 0)
    return df


def split_data(data):
    X = data.drop("y", axis=1)
    y = data["y"].map({"no": 0, "yes": 1})

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,  # keeps class ratio in both splits
    )

    return X_train, X_test, y_train, y_test


def create_preprocessor():
    """
    Build the preprocessing transformer (scaling + one-hot encoding).

    Feature engineering (engineer_features) is applied as a separate step in
    the training pipeline so that the SAME transformation is applied during
    training and at inference time in the Streamlit app. The app passes raw
    test data (no contacted_before column) and the pipeline creates it
    automatically.
    """
    return ColumnTransformer(
        transformers=[
            ("numerical", StandardScaler(), NUMERICAL_COLUMNS),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_COLUMNS,
            ),
        ]
    )


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
    print(
        "Class balance (test):",
        y_test.value_counts(normalize=True).round(3).to_dict(),
    )
