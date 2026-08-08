# Bank Marketing Classification

## Problem Statement

The objective of this project is to predict whether a bank customer will
subscribe to a term deposit based on information collected during a marketing
campaign.

## Dataset Description

The Bank Marketing dataset is taken from the UCI Machine Learning Repository.
The complete dataset contains 45,211 records, 16 input features, and the target
column `y`. The target has two classes: `yes` and `no`.

Dataset source: https://archive.ics.uci.edu/dataset/222/bank+marketing

The complete dataset is stored in `data/bank/bank-full.csv`. A stratified 20%
test split is stored as `test_data.csv` for model evaluation and the Streamlit
application. The application loads this test file automatically, while also
providing an optional CSV upload for another compatible test dataset.

## GitHub Repository Link

https://github.com/PatilSusheel/ML_Assignment-2

## Models Used

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors Classifier
4. Gaussian Naive Bayes Classifier
5. Random Forest Classifier

### Model Comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.8447 | 0.9079 | 0.4166 | 0.8166 | 0.5517 | 0.5085 |
| Decision Tree | 0.8071 | 0.8955 | 0.3641 | 0.8686 | 0.5131 | 0.4784 |
| kNN | 0.9015 | 0.8840 | 0.6640 | 0.3195 | 0.4314 | 0.4157 |
| Naive Bayes | 0.8465 | 0.8069 | 0.3849 | 0.5217 | 0.4430 | 0.3619 |
| Random Forest | 0.8439 | 0.9228 | 0.4188 | 0.8629 | 0.5639 | 0.5292 |

### Model Observations

| ML Model Name | Observation about model performance |
| --- | --- |
| Logistic Regression | With `class_weight="balanced"` and feature engineering, recall jumps to 0.82 and F1 to 0.55, giving the best F1/MCC among linear models. |
| Decision Tree | Balanced class weights and depth/leaf constraints lift recall to 0.87 while keeping the tree from overfitting. |
| kNN | Highest accuracy (0.90) and precision (0.66), but the lowest recall — it still struggles to catch subscribers. |
| Naive Bayes | Moderate recall (0.52) but the lowest AUC, precision, and MCC of all models. |
| Random Forest | Best overall: highest AUC (0.92), F1 (0.56), and MCC (0.53) with strong recall (0.86). |
| Overall Winner | Random Forest gives the best overall performance on this dataset. |

### Preprocessing & Feature Engineering

To improve the evaluation metrics, the following techniques were applied:

- **Feature engineering** — `pdays = -1` (client never previously contacted) was
  split into a binary `contacted_before` flag, and `pdays` was set to `0`. This
  removes a large negative outlier that skewed `StandardScaler`.
- **Class balancing** — `class_weight="balanced"` (and
  `"balanced_subsample"` for Random Forest) up-weights the minority "yes" class,
  dramatically improving recall, F1, and MCC.
- **Regularisation / tuning** — Logistic Regression `C=0.1`, Decision Tree
  `max_depth=8` + `min_samples_leaf=20`, kNN `n_neighbors=15` +
  `weights="distance"`, Random Forest `max_depth=12` + `min_samples_leaf=10`.

The feature engineering is embedded **inside** the saved pipeline (via
`FunctionTransformer`), so the Streamlit app applies the exact same
transformation at inference time without any extra code.

## Streamlit Application

The interactive web application is built with Streamlit (`app.py`) and evaluates
the saved models **live** on the test data every time a button is clicked — it
does not display pre-computed metrics.

### Running the application

```bash
streamlit run app.py
```

### Repository structure

```
model/
  ├── logistic_regression.pkl
  ├── decision_tree.pkl
  ├── knn.pkl
  ├── naive_bayes.pkl
  ├── random_forest.pkl
  ├── data_pipeline.py
  └── train_models.py
app.py
test_data.csv
data/bank/bank-full.csv
```

Each saved model is a complete `sklearn` pipeline that bundles the feature
engineering (`FunctionTransformer`), the preprocessor (scaler/encoder), and the
classifier, so the Streamlit app can evaluate any uploaded test CSV without
re-fitting.
