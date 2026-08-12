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
| Logistic Regression | 0.8467 | 0.9067 | 0.4194 | 0.8062 | 0.5517 | 0.5068 |
| Decision Tree | 0.8400 | 0.8872 | 0.4050 | 0.7836 | 0.5340 | 0.4847 |
| kNN | 0.8307 | 0.8864 | 0.3918 | 0.8091 | 0.5279 | 0.4825 |
| Naive Bayes | 0.7877 | 0.7900 | 0.3101 | 0.6654 | 0.4231 | 0.3479 |
| Random Forest | 0.8673 | 0.9192 | 0.4603 | 0.7788 | 0.5787 | 0.5307 |

### Model Observations

| ML Model Name | Observation about model performance |
| --- | --- |
| Logistic Regression | Captures 80.6% of actual subscribers (recall) while keeping precision at 0.42, meaning it flags many non-subscribers as false positives. Its high AUC (0.91) shows it ranks subscribers well, and it achieves the best F1 (0.55) and MCC (0.51) among the linear models. |
| Decision Tree | Correctly identifies 78.4% of subscribers (recall) with a precision of 0.41. Its AUC (0.89) is the lowest among the tree-based models, indicating weaker probability ranking, but depth and leaf constraints keep it from overfitting. |
| kNN | Reaches the highest recall of the distance-based models at 0.81, catching most subscribers, but its precision (0.39) is the second lowest — it over-predicts the "yes" class. Its AUC (0.89) trails the linear and ensemble models. |
| Naive Bayes | Has the weakest overall ranking ability with the lowest AUC (0.79). It recalls 66.5% of subscribers but at the cost of the lowest precision (0.31), producing the most false positives and the lowest MCC (0.35) of all models. |
| Random Forest | Leads every metric: highest accuracy (0.87), AUC (0.92), precision (0.46), F1 (0.58), and MCC (0.53). It recalls 77.9% of subscribers while maintaining the best precision, giving the most reliable overall predictions. |
| Overall Winner | Random Forest gives the best overall performance on this dataset. |

## Streamlit Application

The interactive web application is built with Streamlit (`app.py`) and evaluates
the saved models **live** on the test data. Selecting a model from the dropdown
immediately evaluates it on the test data and shows its metrics, confusion
matrix, and classification report. The **Compare All** button evaluates every
model at once and displays a comparison table.

### Running the application

```bash
streamlit run app.py
```

Each saved model is a complete `sklearn` pipeline that bundles the feature
engineering (`FunctionTransformer`), the preprocessor (scaler/encoder), and the
classifier, so the Streamlit app can evaluate any uploaded test CSV without
re-fitting.
