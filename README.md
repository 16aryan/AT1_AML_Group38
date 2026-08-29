# AT1_AML

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

# NBA Career Longevity Prediction

## Background & Prediction Target
The goal of this project is to predict whether a U.S. college basketball player will go on to have a sustained NBA career, defined as:

**Playing at least 3 seasons in the NBA.**

The binary target variable (`label` in `train.csv`) represents exactly this definition:
- **`label = 1`**: The player achieved a sustained NBA career (played at least 3 seasons in the league).
- **`label = 0`**: The player did not achieve a sustained NBA career (played less than 3 seasons, or never made the league).

The machine learning models (LightGBM, XGBoost, and CatBoost) are trained using `y = train["label"]` as the target and predict the continuous probability of this label being 1. The evaluation metric is Area Under the Precision-Recall Curve (AUPRC).

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         AT1_AML and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── AT1_AML   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes AT1_AML a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

## Reproducibility Instructions

### 1. Install Environment & Package Dependencies
To set up the project environment and install all required machine learning libraries alongside the custom `nba_prep` feature engineering package:
```bash
pip install -r requirements.txt
```
*(Alternatively, install `nba_prep` directly from TestPyPI: `pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ nba_prep_26040826==0.3.3`)*

### 2. Preprocess Raw Data & Generate Processed CSVs
To generate all fresh, cleaned, and feature-engineered datasets from raw inputs:
```bash
python -m AT1_AML.dataset
```
**Output Files Generated in `data/processed/`:**
* `train_processed.csv`: Cleaned multi-season training data with 110+ engineered features (cumulative statistics, True Shooting %, defensive impact, age, and recruit pedigree).
* `test_processed.csv`: Cleaned test set with aligned feature columns and zero missing feature values.
* `interim/` datasets: Intermediate cleaned and imputed tables.

### 3. Train Models, Output Metrics & Generate Predictions
Run the complete automated 5-Fold Stratified Group Cross-Validation and model training pipeline:
```bash
python -m AT1_AML.modeling.train
```
**What this accomplishes:**
1. **Cross-Validation Evaluation:** Evaluates 5 separate algorithms (LightGBM, XGBoost, CatBoost, Extra Trees, Logistic Regression) and their ensemble using Stratified Group K-Fold (grouped strictly by `pid` to eliminate data leakage).
2. **Generates Fresh Model Files in `models/`:** Fits final estimators on full peak-season data and saves serialized model weights:
   * **LightGBM:** `models/final_lightgbm.joblib`, `models/final_lightgbm.txt`, `models/36120-26SP-group38-26040826-final_lightgbm.joblib`
   * **XGBoost:** `models/final_xgboost.joblib`, `models/final_xgboost.json`, `models/36120-26SP-group38-26040826-final_xgboost.joblib`
   * **CatBoost:** `models/final_catboost.joblib`, `models/final_catboost.cbm`, `models/36120-26SP-group38-26040826-final_catboost.joblib`
   * **Final Ensemble Model:** `models/36120-26SP-group38-26040826-experiment3-model.joblib`
3. **Generates Deduplicated Kaggle Submission (`submission.csv`):** Evaluates test prospects strictly on their final college seasons, ensuring exactly 6,261 unique player IDs with zero duplicates and zero null values.

### 4. Running Experimental Walkthrough Notebooks
The complete, structured experimental progression is documented across three dedicated Jupyter notebooks:

* **Experiment 1 (Baseline Model):** [`notebooks/36120-26SP-group38-26040826-AT1-experiment-1.ipynb`](notebooks/36120-26SP-group38-26040826-AT1-experiment-1.ipynb)
  * Implements baseline LightGBM model, initial data exploratory checks, and saves `models/36120-26SP-group38-26040826-experiment1-model.joblib`.
* **Experiment 2 (Model Architecture Exploration):** [`notebooks/36120-26SP-group38-26040826-AT1-experiment-2.ipynb`](notebooks/36120-26SP-group38-26040826-AT1-experiment-2.ipynb)
  * Rigorous head-to-head comparison of Random Forest, HistGradientBoosting, and Tuned LightGBM. Saves `models/36120-26SP-group38-26040826-experiment2-model.joblib`.
* **Experiment 3 (Feature Selection & Ensembling):** [`notebooks/36120-26SP-group38-26040826-AT1-experiment-3.ipynb`](notebooks/36120-26SP-group38-26040826-AT1-experiment-3.ipynb)
  * Comprehensive feature engineering, correlation analysis, LightGBM gain-importance feature selection, and a 3-model GBDT ensemble. Generates processed split files (`X_train.csv`, `y_train.csv`, `X_val.csv`, `y_val.csv`, `X_test.csv`, `y_test.csv`), saves `models/36120-26SP-group38-26040826-experiment3-model.joblib`, and outputs the final blended `submission.csv`.

#### To re-execute all notebooks via CLI:
```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/36120-26SP-group38-26040826-AT1-experiment-1.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/36120-26SP-group38-26040826-AT1-experiment-2.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/36120-26SP-group38-26040826-AT1-experiment-3.ipynb
```
*(Or launch interactively using `jupyter lab` or `jupyter notebook`).*

## Latest Model Performance & Credit

This training run and model architecture was developed by **Aryan Goel (26040826)**.

### Out-of-Fold (OOF) Performance Scores (AUPRC)
* **LightGBM**: `0.48416`
* **Extra Trees**: `0.49197`
* **Logistic Regression**: `0.48983`
* **XGBoost**: `0.50970`
* **CatBoost**: `0.51188`
* **Blended Ensemble**: **`0.52035`** *(Peak Performance)*

### Generated Model Artifacts (`models/`)
* **LightGBM**: `models/final_lightgbm.joblib`, `models/final_lightgbm.txt`, `models/36120-26SP-group38-26040826-final_lightgbm.joblib`
* **XGBoost**: `models/final_xgboost.joblib`, `models/final_xgboost.json`, `models/36120-26SP-group38-26040826-final_xgboost.joblib`
* **CatBoost**: `models/final_catboost.joblib`, `models/final_catboost.cbm`, `models/36120-26SP-group38-26040826-final_catboost.joblib`
* **Experiment 3 Final Model**: `models/36120-26SP-group38-26040826-experiment3-model.joblib`
* **Predictions**: `submission.csv` (6,261 unique player rows, 0 duplicates, 0 nulls, ready for Kaggle submission)


