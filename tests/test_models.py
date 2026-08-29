import os
import pytest
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from AT1_AML.config import PROCESSED_DATA_DIR, MODELS_DIR, PROJ_ROOT

def test_models_exist_and_load():
    """
    Verifies that the trained model binary files exist in the models directory
    (in both native booster format and joblib serialization) and can be loaded.
    """
    lgb_path = MODELS_DIR / "final_lightgbm.txt"
    xgb_path = MODELS_DIR / "final_xgboost.json"
    cat_path = MODELS_DIR / "final_catboost.cbm"
    
    lgb_joblib = MODELS_DIR / "final_lightgbm.joblib"
    xgb_joblib = MODELS_DIR / "final_xgboost.joblib"
    cat_joblib = MODELS_DIR / "final_catboost.joblib"
    exp3_joblib = MODELS_DIR / "36120-26SP-group38-26040826-experiment3-model.joblib"
    
    # Assert all model files exist
    assert lgb_path.exists(), f"LightGBM model not found at {lgb_path}"
    assert xgb_path.exists(), f"XGBoost model not found at {xgb_path}"
    assert cat_path.exists(), f"CatBoost model not found at {cat_path}"
    
    assert lgb_joblib.exists(), f"LightGBM joblib not found at {lgb_joblib}"
    assert xgb_joblib.exists(), f"XGBoost joblib not found at {xgb_joblib}"
    assert cat_joblib.exists(), f"CatBoost joblib not found at {cat_joblib}"
    assert exp3_joblib.exists(), f"Experiment 3 model not found at {exp3_joblib}"
    
    # Test loading native boosters
    model_lgb = lgb.Booster(model_file=str(lgb_path))
    assert model_lgb is not None, "Failed to load LightGBM booster"
    
    model_xgb = xgb.Booster()
    model_xgb.load_model(str(xgb_path))
    assert model_xgb is not None, "Failed to load XGBoost booster"
    
    model_cat = CatBoostClassifier()
    model_cat.load_model(str(cat_path))
    assert model_cat is not None, "Failed to load CatBoost model"
    
    # Test loading joblib models
    loaded_exp3 = joblib.load(str(exp3_joblib))
    assert loaded_exp3 is not None, "Failed to load Experiment 3 joblib model"

def test_model_predictions():
    """
    Verifies that all three loaded models can successfully make predictions 
    on a sample of processed features, and that predicted probabilities are valid.
    """
    # Load sample features
    test_path = PROCESSED_DATA_DIR / "test_processed.csv"
    assert test_path.exists(), "Processed test features file not found"
    
    test_df = pd.read_csv(test_path, nrows=5)
    
    # Drop non-feature columns
    drop_cols = ["pid", "label", "ht", "yr", "dob", "type", "num"]
    feature_cols = [col for col in test_df.columns if col not in drop_cols]
    
    # Replicate categorical encoding done during training
    X_sample = test_df[feature_cols].copy()
    categorical_cols = ["team", "conf", "role"]
    for col in categorical_cols:
        if col in X_sample.columns:
            X_sample[col] = (pd.Categorical(X_sample[col]).codes + 1).astype(int)
            
    # Convert inputs to float32 for LightGBM
    X_sample_arr = X_sample.values.astype(np.float32)
    
    # Load boosters
    model_lgb = lgb.Booster(model_file=str(MODELS_DIR / "final_lightgbm.txt"))
    model_xgb = xgb.Booster()
    model_xgb.load_model(str(MODELS_DIR / "final_xgboost.json"))
    model_cat = CatBoostClassifier()
    model_cat.load_model(str(MODELS_DIR / "final_catboost.cbm"))
    
    # 1. Test LightGBM predictions
    preds_lgb = model_lgb.predict(X_sample_arr)
    assert len(preds_lgb) == 5
    assert np.all((preds_lgb >= 0.0) & (preds_lgb <= 1.0))
    
    # 2. Test XGBoost predictions
    dtest = xgb.DMatrix(X_sample)
    preds_xgb = model_xgb.predict(dtest)
    assert len(preds_xgb) == 5
    assert np.all((preds_xgb >= 0.0) & (preds_xgb <= 1.0))
    
    # 3. Test CatBoost predictions
    preds_cat = model_cat.predict_proba(X_sample)[:, 1]
    assert len(preds_cat) == 5
    assert np.all((preds_cat >= 0.0) & (preds_cat <= 1.0))
    
    # 4. Test Ensembled/Blended predictions
    blend_preds = 0.4 * preds_lgb + 0.3 * preds_xgb + 0.3 * preds_cat
    assert len(blend_preds) == 5
    assert np.all((blend_preds >= 0.0) & (blend_preds <= 1.0))

def test_submission_file_integrity():
    """
    Verifies that submission.csv exists, has exactly 6,261 rows, 
    contains zero duplicate player IDs, zero missing values, and valid probabilities.
    """
    sub_path = PROJ_ROOT / "submission.csv"
    assert sub_path.exists(), f"submission.csv not found at {sub_path}"
    
    sub = pd.read_csv(sub_path)
    assert len(sub) == 6261, f"Expected 6261 rows, got {len(sub)}"
    assert sub["pid"].nunique() == 6261, "Duplicate player IDs found in submission.csv"
    assert sub.isna().sum().sum() == 0, "Null values found in submission.csv"
    assert set(sub.columns) == {"pid", "label"}, "Columns must be exactly ['pid', 'label']"
    assert np.all((sub["label"] >= 0.0) & (sub["label"] <= 1.0)), "Probabilities out of bounds [0, 1]"

