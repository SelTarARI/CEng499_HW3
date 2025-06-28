# CENG499 - Homework 3

This repository contains my solutions for Homework 3 of the CENG499 Machine Learning course at METU. The assignment focuses on classical machine learning algorithms including decision trees, support vector machines, and model comparison via nested cross-validation.

## 📂 Contents

### 🌳 Part 1 – ID3 Decision Tree
- Manual implementation of the ID3 algorithm from scratch using information gain.
- Includes tree building, prediction, and evaluation on a toy dataset.

### 📐 Part 2 – Support Vector Machines
- Binary classification using SVM with:
  - Linear kernel
  - RBF (Gaussian) kernel
- Evaluation metrics: Accuracy, 95% confidence intervals
- Experiments conducted on two synthetic datasets

### 📊 Part 3 – Model Comparison
- Applied multiple classifiers on the ECG dataset:
  - KNN
  - SVM
  - ID3 Decision Tree
  - Random Forest
  - MLP
  - Gradient Boosting
- Used **nested cross-validation** for unbiased model selection and performance estimation.
- Report includes metric summaries and interpretation.

## 📋 Report

The complete analysis and results are presented in `report.pdf`.

## 🧰 Requirements

- Python 3.9+
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
