# Getting Started

This guide explains how to set up the project on a clean install, run the preprocessing script, train the ensembled models, and run the test suite.

---

## 1. Project Setup
Clone the repository and install all dependencies (including the local feature engineering package `nba_prep`) using the unified requirements file:

```bash
# Install dependencies and local custom package
pip install -r requirements.txt
```

---

## 2. Running the Data Pipeline
Run the preprocessing script to clean raw columns and generate the advanced basketball features:

```bash
python -m AT1_AML.dataset
```

This command runs in two phases:
1. **Phase 1 (Data Cleaning)**: Converts heights to inches, cleans college year classifications, estimates ages, and outputs the intermediate data to `data/interim/train_cleaned.csv` and `data/interim/test_cleaned.csv`.
2. **Phase 2 (Feature Engineering)**: Calculates advanced statistics, cumulative career totals, recruit ranks, and conference checks, then outputs the final files to `data/processed/train_processed.csv` and `data/processed/test_processed.csv`.

---

## 3. Training the Models
Run the training script to run 5-Fold Stratified Group Cross-Validation, blend the predictions, and serialize the finalized models:

```bash
python -m AT1_AML.modeling.train
```

This command will:
1. Load features and apply robust categorical encoding.
2. Train LightGBM, XGBoost, and CatBoost models across 5 folds.
3. Blend the predictions (40% LightGBM, 30% XGBoost, 30% CatBoost).
4. Apply player-level average post-processing and save `submission.csv` (deduplicated by `pid`).
5. Fit and serialize finalized full-dataset models to the `models/` directory.

---

## 4. Running the Tests
To run all unit tests for the data cleaning, basketball features, cumulative stats, and processed datasets:

```bash
# Run unit tests in the package directory
python -m pytest 36120-26SP-group-student-package/tests

# Run verification tests in the project directory
python -m pytest AT1_AML/tests
```
