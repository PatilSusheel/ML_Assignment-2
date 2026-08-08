import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
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

# The saved pipelines embed a FunctionTransformer that references
# data_pipeline.engineer_features. Add the model dir to sys.path so the
# function is importable when the pickles are unpickled from the app root.
sys.path.insert(0, str(MODEL_DIR))

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
    """Load a saved model pipeline (preprocessor + classifier) from disk."""
    model_path = MODEL_DIR / MODEL_FILES[model_name]
    with open(model_path, "rb") as model_file:
        return pickle.load(model_file)


def calculate_metrics(actual, predicted, probabilities):
    """Compute all six evaluation metrics live from the predictions."""
    return {
        "Accuracy": accuracy_score(actual, predicted),
        "AUC": roc_auc_score(actual, probabilities),
        "Precision": precision_score(actual, predicted, zero_division=0),
        "Recall": recall_score(actual, predicted, zero_division=0),
        "F1": f1_score(actual, predicted, zero_division=0),
        "MCC": matthews_corrcoef(actual, predicted),
    }


def evaluate_model(model_name, X_test, y_test):
    """Run a single model on the test data and return metrics + predictions."""
    model = load_model(model_name)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    metrics = calculate_metrics(y_test, predictions, probabilities)
    return metrics, predictions, probabilities


def render_metrics(metrics):
    """Display the six metrics as metric cards."""
    metric_columns = st.columns(6)
    for column, (metric_name, metric_value) in zip(metric_columns, metrics.items()):
        column.metric(metric_name, f"{metric_value:.4f}")


def render_confusion_matrix(y_test, predictions):
    """Render the confusion matrix as a seaborn heatmap."""
    matrix = confusion_matrix(y_test, predictions)
    figure, axis = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Predicted No", "Predicted Yes"],
        yticklabels=["Actual No", "Actual Yes"],
        ax=axis,
        cbar=False,
    )
    axis.set_title("Confusion Matrix")
    axis.set_ylabel("Actual")
    axis.set_xlabel("Predicted")
    st.pyplot(figure)
    plt.close(figure)


def render_classification_report(y_test, predictions):
    """Render the classification report as a table."""
    report = classification_report(
        y_test,
        predictions,
        target_names=["No", "Yes"],
        output_dict=True,
        zero_division=0,
    )
    st.dataframe(pd.DataFrame(report).transpose().round(4))


def load_test_data(uploaded_file):
    """Load the test dataset from the uploaded file or the default CSV."""
    if uploaded_file is not None:
        test_data = pd.read_csv(uploaded_file, sep=";")
        source = "uploaded CSV"
    else:
        test_data = pd.read_csv(TEST_DATA_FILE, sep=";")
        source = "test_data.csv (included in this repository)"
    return test_data, source


def validate_test_data(test_data):
    """Check that the test data has all required columns and a valid target."""
    required_columns = FEATURE_COLUMNS + ["y"]
    missing_columns = [
        column for column in required_columns if column not in test_data.columns
    ]
    if missing_columns:
        return None, None, "Missing columns: " + ", ".join(missing_columns)

    X_test = test_data[FEATURE_COLUMNS]
    y_test = test_data["y"].map({"no": 0, "yes": 1})

    if y_test.isna().any():
        return None, None, "The target column must contain only 'yes' and 'no'."

    return X_test, y_test, None


st.set_page_config(page_title="Bank Marketing Classification", layout="wide")

st.title("Bank Marketing Classification")
st.write(
    "Upload a test CSV (or use the pre-loaded `test_data.csv`), select a model, "
    "and click **Evaluate** to compute the metrics live. Use **Compare All** to "
    "evaluate every model and view a comparison table."
)

# --- Sidebar: dataset upload and model selection ---
with st.sidebar:
    st.header("Configuration")
    uploaded_file = st.file_uploader(
        "Upload test dataset (CSV)", type="csv", help="Leave empty to use test_data.csv"
    )
    selected_model = st.selectbox("Select a model", list(MODEL_FILES.keys()))

    evaluate_clicked = st.button("Evaluate", type="primary", use_container_width=True)
    compare_clicked  = st.button("Compare All", use_container_width=True)

# --- Load and validate the test data ---
test_data, source = load_test_data(uploaded_file)
X_test, y_test, validation_error = validate_test_data(test_data)

if validation_error:
    st.error(validation_error)
    st.stop()

st.caption(f"Evaluation dataset: {source} ({len(test_data)} rows)")

# --- Single model evaluation ---
if evaluate_clicked:
    with st.spinner(f"Evaluating {selected_model} on the test data..."):
        metrics, predictions, _ = evaluate_model(selected_model, X_test, y_test)

    st.subheader(f"{selected_model} Results")
    render_metrics(metrics)

    st.subheader("Confusion Matrix")
    render_confusion_matrix(y_test, predictions)

    st.subheader("Classification Report")
    render_classification_report(y_test, predictions)

# --- Compare all models ---
if compare_clicked:
    st.subheader("Model Comparison (evaluated live on the test data)")
    comparison_rows = []
    progress = st.progress(0.0)
    total = len(MODEL_FILES)

    for index, model_name in enumerate(MODEL_FILES.keys()):
        with st.spinner(f"Evaluating {model_name}..."):
            metrics, _, _ = evaluate_model(model_name, X_test, y_test)
        comparison_rows.append({"ML Model Name": model_name, **metrics})
        progress.progress((index + 1) / total)
    progress.empty()

    comparison_df = pd.DataFrame(comparison_rows)
    st.dataframe(comparison_df.round(4), use_container_width=True)

    # Highlight the best model per metric
    st.markdown("#### Best model per metric")
    best_rows = []
    for metric in ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]:
        best_model = comparison_df.loc[comparison_df[metric].idxmax(), "ML Model Name"]
        best_rows.append(
            {
                "Metric": metric,
                "Best Model": best_model,
                "Value": comparison_df[metric].max(),
            }
        )
    st.dataframe(pd.DataFrame(best_rows).round(4), use_container_width=True)

if not evaluate_clicked and not compare_clicked:
    st.info(
        "Select a model and click **Evaluate** to see its live metrics, "
        "or click **Compare All** to evaluate every model at once."
    )
