# NBA Career Longevity Prediction Documentation

Welcome to the documentation for the NBA Career Longevity Prediction model. This project is developed by **Aryan Goel (Student ID: `26040826`)** for UTS Coursework (Assessment Task 1 - Advanced Machine Learning, Group 38).

---

## 1. Project Background & Objective
In the NBA, first-round draft picks receive two-year guaranteed rookie contracts (typically $5M–$10M/year), with team options for years 3 and 4. Drafting prospects who fail to establish themselves in the league incurs severe salary cap penalties and lost competitive potential.

This project delivers a machine learning solution to predict whether a collegiate prospect will achieve a **sustained NBA career (playing $\ge 3$ seasons in the NBA)** based on their collegiate performance statistics, recruiting pedigree, and multi-season development trajectory.

* **Target Label (`label`):** Binary indicator (`1` = $\ge 3$ seasons in NBA, `0` = $<3$ seasons or never made the league).
* **Primary Evaluation Metric:** **Area Under the Precision-Recall Curve (AUPRC)** due to the extreme class imbalance (~3.74% base rate).

---

## 2. Experimental Progression

The research and modeling pipeline is documented across three sequential Jupyter notebooks:

1. **Experiment 1 (Baseline):** [`notebooks/36120-26SP-group38-26040826-AT1-experiment-1.ipynb`](../../notebooks/36120-26SP-group38-26040826-AT1-experiment-1.ipynb)
   * Initial data exploration, missing value imputation, and baseline LightGBM model.
2. **Experiment 2 (Model Comparison):** [`notebooks/36120-26SP-group38-26040826-AT1-experiment-2.ipynb`](../../notebooks/36120-26SP-group38-26040826-AT1-experiment-2.ipynb)
   * Systematic architecture comparison of Random Forest, HistGradientBoosting, and Tuned LightGBM.
3. **Experiment 3 (Feature Selection & Ensembling):** [`notebooks/36120-26SP-group38-26040826-AT1-experiment-3.ipynb`](../../notebooks/36120-26SP-group38-26040826-AT1-experiment-3.ipynb)
   * Advanced cumulative career metrics, LightGBM gain feature selection, and a 3-model GBDT ensemble (LightGBM + XGBoost + CatBoost) trained on peak final-college seasons.

---

## 3. Model Benchmark & Results

All models were evaluated using **5-Fold Stratified Group Cross-Validation** (grouped strictly by `pid` to eliminate data leakage):

| Model Architecture | Out-of-Fold (OOF) AUPRC | Lift vs Random Chance (0.034) |
| :--- | :---: | :---: |
| **Random Chance Baseline** | `0.03396` | 1.0x |
| **LightGBM** | `0.48416` | 14.3x |
| **Logistic Regression** | `0.48983` | 14.4x |
| **Extra Trees** | `0.49197` | 14.5x |
| **XGBoost** | `0.50970` | 15.0x |
| **CatBoost** | `0.51188` | 15.1x |
| **Blended Ensemble (Final)** | **`0.52035`** | **15.3x** |

---

## 4. Key Project Commands
* **Preprocess Data & Engineer Features:** `python -m AT1_AML.dataset`
* **Train All Models & Output Predictions:** `python -m AT1_AML.modeling.train`
* **Generate High-Resolution Figures:** `python -m AT1_AML.plots`
* **Run Test Suite:** `pytest 36120-26SP-group38-26040826-package/tests`

