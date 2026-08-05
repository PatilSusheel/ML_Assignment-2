import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from data_pipeline import create_preprocessor, load_data, split_data


MODEL_DIR = Path(__file__).resolve().parent

MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "kNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
}

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
}


def calculate_metrics(y_test, predictions, probabilities):
    return {
        "Accuracy": accuracy_score(y_test, predictions),
        "AUC": roc_auc_score(y_test, probabilities),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1": f1_score(y_test, predictions, zero_division=0),
        "MCC": matthews_corrcoef(y_test, predictions),
    }


def train_and_evaluate_models():
    data = load_data()
    X_train, X_test, y_train, y_test = split_data(data)
    results = []

    for model_name, classifier in MODELS.items():
        print("Training", model_name)

        pipeline = Pipeline(
            steps=[
                ("preprocessor", create_preprocessor()),
                ("classifier", classifier),
            ]
        )

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)[:, 1]

        metrics = calculate_metrics(y_test, predictions, probabilities)
        results.append({"ML Model Name": model_name, **metrics})

        model_path = MODEL_DIR / MODEL_FILES[model_name]
        with open(model_path, "wb") as model_file:
            pickle.dump(pipeline, model_file)

    results_df = pd.DataFrame(results)
    results_df.to_csv(MODEL_DIR / "model_metrics.csv", index=False)
    return results_df


if __name__ == "__main__":
    model_results = train_and_evaluate_models()
    print("\nModel comparison")
    print(model_results.round(4).to_string(index=False))
