import pickle
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


MODEL_DIR = Path(__file__).resolve().parent / "model"
TEST_DATA_FILE = Path(__file__).resolve().parent / "test_data.csv"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
}

FEATURE_COLUMNS = [
    "age",
    "job",
    "marital",
    "education",
    "default",
    "balance",
    "housing",
    "loan",
    "contact",
    "day",
    "month",
    "duration",
    "campaign",
    "pdays",
    "previous",
    "poutcome",
]


@st.cache_resource
def load_model(model_name):
    model_path = MODEL_DIR / MODEL_FILES[model_name]
    with open(model_path, "rb") as model_file:
        return pickle.load(model_file)


def calculate_metrics(actual, predicted, probabilities):
    return {
        "Accuracy": accuracy_score(actual, predicted),
        "AUC": roc_auc_score(actual, probabilities),
        "Precision": precision_score(actual, predicted, zero_division=0),
        "Recall": recall_score(actual, predicted, zero_division=0),
        "F1 Score": f1_score(actual, predicted, zero_division=0),
        "MCC": matthews_corrcoef(actual, predicted),
    }


st.set_page_config(page_title="Bank Marketing Classification", layout="wide")

st.title("Bank Marketing Classification")
st.write(
    "Select a classification model to view its evaluation results on the "
    "default test dataset. You may also upload another compatible test CSV."
)

selected_model = st.selectbox("Select a model", list(MODEL_FILES.keys()))
uploaded_file = st.file_uploader("Upload another test dataset (optional)", type="csv")

try:
    if uploaded_file is not None:
        test_data = pd.read_csv(uploaded_file, sep=";")
        st.caption("Evaluation dataset: uploaded CSV")
    else:
        test_data = pd.read_csv(TEST_DATA_FILE, sep=";")
        st.caption("Evaluation dataset: test_data.csv included in this repository")

    required_columns = FEATURE_COLUMNS + ["y"]
    missing_columns = [
        column for column in required_columns if column not in test_data.columns
    ]

    if missing_columns:
        st.error("Missing columns: " + ", ".join(missing_columns))
    else:
        X_test = test_data[FEATURE_COLUMNS]
        y_test = test_data["y"].map({"no": 0, "yes": 1})

        if y_test.isna().any():
            st.error("The target column must contain only 'yes' and 'no'.")
        else:
            model = load_model(selected_model)
            predictions = model.predict(X_test)
            probabilities = model.predict_proba(X_test)[:, 1]
            metrics = calculate_metrics(y_test, predictions, probabilities)

            st.subheader(selected_model + " Results")
            metric_columns = st.columns(6)
            for column, (metric_name, metric_value) in zip(
                metric_columns, metrics.items()
            ):
                column.metric(metric_name, f"{metric_value:.4f}")

            st.subheader("Confusion Matrix")
            matrix = confusion_matrix(y_test, predictions)
            matrix_df = pd.DataFrame(
                matrix,
                index=["Actual No", "Actual Yes"],
                columns=["Predicted No", "Predicted Yes"],
            )
            st.dataframe(matrix_df)

            st.subheader("Classification Report")
            report = classification_report(
                y_test,
                predictions,
                target_names=["No", "Yes"],
                output_dict=True,
                zero_division=0,
            )
            st.dataframe(pd.DataFrame(report).transpose().round(4))

except Exception as error:
    st.error(f"Unable to evaluate the test data: {error}")
