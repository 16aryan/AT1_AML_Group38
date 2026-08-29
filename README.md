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

### 1. Install dependencies and local package
To set up the project environment and install the required libraries:
```bash
pip install -r requirements.txt
```
This will automatically install my custom `nba_prep` feature engineering package in editable mode.

### 2. Preprocess Data and Engineer Features
Run the dataset pipeline to clean the raw data (height, year, age) and generate the advanced basketball features:
```bash
python -m AT1_AML.dataset
```
This will save the processed training and test features to `data/processed/train_processed.csv` and `data/processed/test_processed.csv`.

### 3. Train Model and Generate Predictions
Run the cross-validated model training script:
```bash
python -m AT1_AML.modeling.train
```
This will:
- Execute a 5-Fold Stratified Group Cross-Validation (grouped by `pid` to prevent player-level leakage) using an ensemble of **LightGBM**, **XGBoost**, and **CatBoost**.
- Output overall AUPRC scores (target metric) for each individual model and the blended ensemble.
- Save the predicted probabilities on each player's final college season to `submission.csv` (deduplicated by `pid` and ready for Kaggle upload).

### 4. Interactive Analysis Notebooks
The complete series of experimental notebooks conducted for this assessment:
* **Experiment 1 (Baseline):** [`notebooks/36120-26SP-group38-26040826-AT1-experiment-1.ipynb`](notebooks/36120-26SP-group38-26040826-AT1-experiment-1.ipynb) — Initial baseline LightGBM model and data cleaning exploration.
* **Experiment 2 (Model Exploration):** [`notebooks/36120-26SP-group38-26040826-AT1-experiment-2.ipynb`](notebooks/36120-26SP-group38-26040826-AT1-experiment-2.ipynb) — Systematic architecture comparison between Random Forest, HistGradientBoosting, and Tuned LightGBM.
* **Experiment 3 (Feature Selection & Ensembling):** [`notebooks/36120-26SP-group38-26040826-AT1-experiment-3.ipynb`](notebooks/36120-26SP-group38-26040826-AT1-experiment-3.ipynb) — Multi-model GBDT ensemble (LightGBM, XGBoost, CatBoost) trained on peak final-college seasons with correlation and gain-based feature selection.

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


