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
| Logistic Regression | 0.9012 | 0.9056 | 0.6445 | 0.3478 | 0.4518 | 0.4261 |
| Decision Tree | 0.8746 | 0.7015 | 0.4649 | 0.4754 | 0.4701 | 0.3990 |
| kNN | 0.8962 | 0.8277 | 0.5990 | 0.3403 | 0.4340 | 0.4001 |
| Naive Bayes | 0.8548 | 0.8101 | 0.4059 | 0.5198 | 0.4559 | 0.3774 |
| Random Forest | 0.9045 | 0.9263 | 0.6506 | 0.3960 | 0.4924 | 0.4597 |

### Model Observations

| ML Model Name | Observation about model performance |
| --- | --- |
| Logistic Regression | It gives high accuracy and AUC, but its recall for term-deposit subscribers is low. |
| Decision Tree | It identifies more subscribers than Logistic Regression, but has lower accuracy and AUC. |
| kNN | It gives good accuracy, but its recall and F1 score are lower than the tree-based models. |
| Naive Bayes | It has the highest recall, but also the lowest accuracy and precision. |
| Random Forest | It produces the highest accuracy, AUC, precision, F1 score, and MCC. |
| Overall Winner | Random Forest gives the best overall performance on this dataset. |
