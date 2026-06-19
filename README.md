# Credit Risk Dashboard

An interactive web application that predicts the probability a borrower will default on a loan, comparing a logistic regression baseline against a feedforward neural network. Built with Python, Scikit-learn, PyTorch, and Streamlit.

## Overview

This project uses the "Give Me Some Credit" Kaggle dataset (150,000 borrowers) to predict serious delinquency within two years. It compares two modeling approaches head to head: a simple, explainable logistic regression and a small feedforward neural network, both trained on the same cleaned, SMOTE-balanced data for a fair comparison.

## Features

- Upload a CSV of borrowers and get instant default probability scores
- Choose between Logistic Regression or Neural Network models
- Adjustable risk threshold slider, since the "best" cutoff depends on risk appetite, not a fixed rule
- Visualizations of risk distribution and predicted default probability
- Downloadable report of flagged high-risk borrowers

## Data Cleaning

The raw dataset required real cleaning before training, not just plug-and-play:

- Dropped 1 row with `age == 0` (clear data entry error)
- Filled missing `MonthlyIncome` (20% of rows) and `NumberOfDependents` (2.6% of rows) with the median, chosen specifically because both fields are right-skewed
- Removed ~28,877 rows with `DebtRatio > 10`, found through investigation to be a mechanical artifact (a broken ratio when income is missing or near-zero), not genuine extreme debt
- Final cleaned dataset: 121,122 rows, true default rate of 6.95%

## Model Performance (on held-out test set, threshold = 0.75)

| Model | Precision | Recall | F1 Score |
|---|---|---|---|
| Logistic Regression + SMOTE | 0.48 | 0.24 | 0.32 |
| Neural Network + SMOTE | 0.33-0.37 | 0.50-0.55 | 0.41-0.43 |

At a consistent threshold, the two models trade off differently rather than one strictly beating the other: logistic regression is more precise (fewer false alarms among flagged borrowers), while the neural network catches meaningfully more actual defaulters (higher recall) and wins on overall F1. Which model is "better" depends on whether the priority is minimizing false alarms or catching more real risk.

A second, deeper neural network architecture (2 hidden layers, 32 units) was also tested and did not meaningfully outperform the simpler 1-layer, 16-unit network, suggesting the dataset's 10 features don't have enough complexity for extra model capacity to exploit. Training was capped at 200 epochs after confirming (via testing up to 500) that further training reduced loss only marginally without improving precision or recall on the test set.

## Threshold Tuning

Rather than using the default 0.5 cutoff, a systematic sweep (0.2 to 0.9) on the neural network found F1 peaks around threshold 0.75-0.80. Lower thresholds favor recall (catch more defaulters, more false alarms); higher thresholds favor precision (fewer false alarms, miss more risk). The dashboard exposes this as an adjustable slider for both models rather than hardcoding one choice, since the right tradeoff depends on business context.

## A Note on Interpretability

Logistic regression's coefficients revealed a counterintuitive result: `NumberOfTime60-89DaysPastDueNotWorse` has a negative coefficient, suggesting more late payments in that window reduce default risk. Investigation showed this is caused by severe multicollinearity (correlation of 0.98+ between this feature and two other delinquency-count columns) — a known statistical artifact where highly correlated features distort individual coefficient signs without harming overall predictive accuracy. This is a real limitation worth disclosing, not a bug to hide.

## Tech Stack

- Python
- Pandas
- Scikit-learn (Logistic Regression, StandardScaler, SMOTE via imbalanced-learn)
- PyTorch (neural network)
- Streamlit (dashboard)
- Plotly (visualizations)

## How to Run Locally

git clone https://github.com/RyOnFire/credit-risk.git
cd credit-risk
pip install -r requirements.txt
streamlit run app.py

You will need to download `cs-training.csv` from the Kaggle "Give Me Some Credit" competition separately and place it in the project folder.

## Future Improvements

- Combine the three highly correlated delinquency features into a single engineered feature to resolve the multicollinearity issue
- Try class weighting (`pos_weight`) as an alternative to SMOTE for the neural network
- Deploy via API for real-time scoring rather than batch CSV upload
