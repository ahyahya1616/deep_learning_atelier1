# Deep Learning — Lab 1: Regression & Multi-Class Classification

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)

## Academic Information

- **Institution:** Faculty of Sciences and Techniques of Tangier (FSTT) — Abdelmalek Essaâdi University
- **Degree Program:** Software and Intelligent Systems Engineering (*Cycle d'Ingénieur : Génie Logiciel et Systèmes Intelligents — LSI*)
- **Module:** Deep Learning
- **Course Instructor:** Pr. Lotfi ELAACHAK
- **Student:** Yahya Ahmane
- **Academic Year:** 2026–2027

---

## Overview

This repository contains the practical implementation of **Lab 1: Deep Neural Networks with PyTorch**, divided into two core real-world problems:

1. **Continuous Regression (NYSE Stock Prediction):** Forecasting closing stock prices (`close`) from intraday market indicators (`open`, `low`, `high`, `volume`) using Deep Neural Networks (DNN) with hyperparameter grid search and regularization analysis.
2. **Multi-Class Classification under Imbalance (Predictive Maintenance):** Diagnosing 6 equipment failure modes on the **AI4I 2020 Predictive Maintenance** dataset using thermodynamic feature engineering, SMOTE oversampling, Cross-Entropy loss, and Macro-ROC / Confusion Matrix evaluation.

---

## Repository Structure

```bash
lab1/
├── Atelier1.pdf             # Official lab assignment sheet (FSTT)
├── README.md                # Project summary and benchmark report
├── data/                    # Datasets (NYSE and AI4I Maintenance)
│   ├── nyse/
│   └── predictive_maintenance/
├── figures/                 # Visualizations and performance plots
├── notebooks/               # Interactive Jupyter notebooks
│   ├── part1_regression.ipynb
│   └── part2_classification.ipynb
├── scripts/                 # Utility scripts (data download)
│   └── download_data.py
└── src/                     # Standalone Python pipelines
    ├── part1_regression.py
    └── part2_classification.py
```

---

## Quickstart

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install scikit-learn pandas numpy matplotlib seaborn imbalanced-learn jupyter

# 3. Run the standalone scripts
python3 src/part1_regression.py
python3 src/part2_classification.py

# Or launch Jupyter Notebooks
jupyter notebook notebooks/
```

---

## Part 1: Stock Price Prediction (Regression)

### Objective & Setup
- **Goal:** Predict the continuous target `close` price using historical stock metrics.
- **Dataset:** New York Stock Exchange (NYSE) daily prices (850,000+ records).
- **Preprocessing:** Min-Max scaling, temporal and correlation analysis.
- **Architecture & Optimization:**
  - Fully connected DNN (`Input(4) -> 64 -> 32 -> Output(1)`) with ReLU activations.
  - Hyperparameter tuning via 3-Fold Cross-Validation: **Adam** (`lr=0.01`, `batch_size=64`) selected.
  - Loss function: Mean Squared Error (MSE).

### Key Results

| Model Configuration | Test MSE | Test RMSE ($) | Test MAE ($) | $R^2$ Score | Tolerance Accuracy ($\pm 5\%$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline DNN** | **0.00018** | **$1.82** | **$0.95** | **0.9984** | **98.2%** |
| **Regularized (Dropout + BN + L2)** | 0.00942 | $26.96$ | $16.15$ | 0.8781 | 84.3% |

<p align="center">
  <img src="figures/part1_q4_loss_epochs.png" width="48%" alt="Loss over Epochs" />
  <img src="figures/part1_q5_actual_vs_predicted.png" width="48%" alt="Actual vs Predicted" />
</p>

> **Key Finding:** For highly correlated, deterministic input features (`open`, `low`, `high`), aggressive Dropout ($p=0.2$) introduces unnecessary shrinkage on extreme price values. The unregularized baseline achieves exceptional precision ($R^2 > 0.998$), while regularization provides lower gradient variance and increased stability against distributional drift.

---

## Part 2: Predictive Maintenance (Multi-Class Classification)

### Objective & Setup
- **Goal:** Classify machine condition into 6 classes: *No Failure*, *Heat Dissipation*, *Power Failure*, *Overstrain*, *Tool Wear*, and *Random Failure*.
- **Dataset:** AI4I 2020 Predictive Maintenance (10,000 instances, continuous physical sensors).
- **Challenge:** Severe class imbalance (96.5% normal operation, only 3.5% cumulative failure instances).
- **Preprocessing & Strategy:**
  - Feature engineering: derived temperature differential $\text{Temp\_Diff} = T_{\text{process}} - T_{\text{air}}$.
  - Removed identifiers and target proxy columns to avoid data leakage.
  - **SMOTE Oversampling applied strictly to the training split** (the test set preserves real-world distribution).
  - Selected model via Stratified CV: `Input(7) -> 128 -> 64 -> Output(6)` with LeakyReLU and Adam (`lr=0.001`, `batch_size=32`).

### Key Results

- **Overall Test Accuracy:** **97.3%** on unaugmented test data.
- **Macro-Average ROC-AUC:** **0.911** across all 6 classes.

<p align="center">
  <img src="figures/part2_q7_confusion_matrix.png" width="48%" alt="Confusion Matrix" />
  <img src="figures/part2_q8_roc_curves.png" width="48%" alt="ROC Curves" />
</p>

#### Per-Class Performance Breakdown

| Class | Fault Mode | Precision | Recall | F1-Score | ROC-AUC (OvR) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **0** | No Failure | 0.997 | 0.982 | 0.990 | 0.942 |
| **1** | Heat Dissipation Failure | 0.440 | **1.000** | 0.611 | **0.997** |
| **2** | Power Failure | 0.484 | 0.789 | 0.600 | **0.991** |
| **3** | Overstrain Failure | 0.609 | 0.875 | 0.718 | **0.994** |
| **4** | Tool Wear Failure | 0.160 | 0.444 | 0.235 | 0.916 |
| **5** | Random Failures | 0.000 | 0.000 | 0.000 | 0.628 |
| **-** | **Macro Average** | **0.448** | **0.682** | **0.526** | **0.911** |

> **Key Finding:** In mission-critical predictive maintenance, **Recall is paramount**—missing a catastrophic failure is significantly more costly than inspecting a false alarm. With SMOTE and calibrated Cross-Entropy, our model captures **100% of heat dissipation failures** and **87.5% of overstrain failures**.

---

## Core Takeaways & Best Practices

1. **Metric Selection under Imbalance:** Overall accuracy is misleading when 96.5% of samples belong to the majority class. Prioritize Macro-Recall, F1-score, and ROC-AUC.
2. **Data Leakage Hygiene:** Resampling (SMOTE) and feature scaling must be fitted strictly on training folds. Resampling before splitting creates false overoptimistic benchmarks.
3. **Regularization Trade-offs:** While Batch Normalization accelerates deep network convergence, Dropout rates on structured tabular data should be calibrated conservatively to avoid underfitting high-value regions.
