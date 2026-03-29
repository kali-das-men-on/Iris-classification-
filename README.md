# Random Forest & Support Vector Machine Classification

## Overview

This project implements and compares two widely used machine learning models—Random Forest and Support Vector Machine (SVM)—for a supervised classification task. The goal is to evaluate model performance, understand trade-offs between approaches, and identify the most important features driving predictions.

The project uses 4 features - sepal_length, sepal_width, petal_length, petal_width to predict the species of the iris flower (whether it is Setosa, Versicolor, or Virginica) and walks through the full machine learning pipeline, including data preprocessing, model training, hyperparameter tuning, evaluation, and interpretation.

---

## Objectives

* Train a Random Forest classifier and tune key hyperparameters
* Implement an SVM model with different kernels
* Evaluate both models using cross-validation
* Compare performance using precision, recall, and F1-score
* Analyze feature importance to interpret model behavior

---

## Dataset

* **Source:** iris.csv
* **Samples:** 150 observations
* **Features:** 4 Features - sepal_length, sepal_width, petal_length, petal_width	
* **Target Variable:** species

Data preprocessing steps included:

* Checking for missing values
* Feature scaling (for SVM)
* Train-test split

---

## Methods

### 1. Random Forest

An ensemble learning method that builds multiple decision trees and aggregates their predictions.

**Key hyperparameters tuned:**

* Number of trees (`n_estimators`)
* Maximum tree depth (`max_depth`)
* Minimum samples per split

---

### 2. Support Vector Machine (SVM)

A model that finds the optimal hyperplane to separate classes.

**Key configurations:**

* Kernels: Linear, RBF
* Regularization parameter (`C`)
* Kernel coefficient (`gamma`)

---

## Model Evaluation

Both models were evaluated using cross-validation and the following metrics:

* **Precision** – Accuracy of positive predictions
* **Recall** – Ability to find all positive instances
* **F1-score** – Balance between precision and recall

---

## Key Insights

* Random Forest performed better on (e.g., noisy / nonlinear data)
* SVM performed better when (e.g., data was well-separated)
* Feature importance analysis showed that **(top features)** were the most influential
* Hyperparameter tuning significantly improved model performance

---

## Visualizations

The project includes:

* Decision boundary plots (for SVM)
* Feature importance plots (Random Forest)
* Model comparison charts

---

## Project Structure

```
project/
│── data/
│── notebooks/
│── src/
│── results/
│── README.md
│── requirements.txt
```

## Technologies Used

* Python
* scikit-learn
* pandas
* numpy
* matplotlib

---

## Future Improvements

* Test additional models (e.g., Gradient Boosting, Neural Networks)
* Perform deeper hyperparameter optimization (GridSearch / RandomSearch)
* Use a larger or more complex dataset
* Deploy the model as an API or web app

---

## Author

Kalidas Menon
University of Massachusetts Amherst
Computer Science & Economics

---
