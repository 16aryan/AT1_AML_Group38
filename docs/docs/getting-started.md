# Getting Started Guide

This guide walks through setting up the environment, running the preprocessing script, training the ensembled models, executing the Jupyter notebooks, and running the automated test suite.

---

## 1. Project Setup & Package Installation
Clone the repository and install all dependencies (which installs the local feature engineering package `nba_prep` in editable mode):

```bash
pip install -r requirements.txt
```

*(Alternatively, install the published package directly from TestPyPI:)*
```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ nba_prep_26040826==0.3.3
```

---

## 2. Running the Data Preprocessing Pipeline
Execute the complete data cleaning and feature engineering pipeline:

```bash
python -m AT1_AML.dataset
```

**Pipeline Stages:**
1. **Phase 1 (Data Cleaning):** Converts height strings to numeric inches, maps college study years to ordinal numbers (1–4), computes chronological player age from `dob`, and saves interim cleaned datasets.
2. **Phase 2 (Feature Engineering):** Calculates advanced basketball metrics (True Shooting %, Defensive Impact score, Assist-to-Turnover ratio, scoring efficiency), computes cumulative career totals, expanding career maximums, year-over-year progression deltas, and saves the final datasets to `data/processed/train_processed.csv` and `data/processed/test_processed.csv`.

---

## 3. Training Models & Generating Predictions
Run the 5-Fold Stratified Group Cross-Validation training script:

```bash
python -m AT1_AML.modeling.train
```

**Output:**
1. Evaluates 5 models (LightGBM, XGBoost, CatBoost, Extra Trees, Logistic Regression) using Stratified Group K-Fold on unique player final seasons.
2. Generates the final ensembled `submission.csv` (containing 6,261 unique player IDs with zero duplicates and zero null values).
3. Saves the full-dataset model binaries into the `models/` directory:
   * `final_lightgbm.joblib` / `36120-26SP-group38-26040826-final_lightgbm.joblib`
   * `final_xgboost.joblib` / `36120-26SP-group38-26040826-final_xgboost.joblib`
   * `final_catboost.joblib` / `36120-26SP-group38-26040826-final_catboost.joblib`
   * `36120-26SP-group38-26040826-experiment3-model.joblib`

---

## 4. Running Experimental Notebooks
To batch execute all experimental walkthrough notebooks via the command line:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/36120-26SP-group38-26040826-AT1-experiment-1.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/36120-26SP-group38-26040826-AT1-experiment-2.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/36120-26SP-group38-26040826-AT1-experiment-3.ipynb
```

---

## 5. Generating Evaluation Plots & Figures
To regenerate all high-resolution reporting figures into `reports/figures/`:

```bash
python -m AT1_AML.plots
```

---

## 6. Running Unit Tests
To run all 13 unit tests verifying data cleaning, feature calculations, cumulative tracking, model metrics, and business draft ROI:

```bash
# Run unit tests
pytest 36120-26SP-group38-26040826-package/tests

# Run verification demo script
python 36120-26SP-group38-26040826-package/verify_package.py
```

