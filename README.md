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

### 4. Interactive Analysis
My interactive end-to-end walkthrough notebook is available at `notebooks/36120-26SP-group38-26040826-AT1-experiment-1.ipynb`.

## Latest Model Performance & Credit

This training run was performed by **Aryan Goel (26040826)**.

### Performance Scores (OOF AUPRC)
* **LightGBM**: `0.48416`
* **XGBoost**: `0.50970`
* **CatBoost**: `0.51188`
* **Blended Ensemble**: `0.51557`

### Generated Artifacts
* **Ensembled Models**: `models/final_lightgbm.txt`, `models/final_xgboost.json`, and `models/final_catboost.cbm`
* **Predictions**: `submission.csv` (12,061 rows, fully aligned to raw test.csv)


