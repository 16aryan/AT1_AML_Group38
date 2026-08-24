import os
import pytest
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from AT1_AML.config import PROCESSED_DATA_DIR, MODELS_DIR

def test_models_exist_and_load():
    """
    Verifies that the trained model binary files exist in the models directory
    and can be successfully loaded by their respective libraries.
    """
    lgb_path = MODELS_DIR / "final_lightgbm.txt"
    xgb_path = MODELS_DIR / "final_xgboost.json"
    cat_path = MODELS_DIR / "final_catboost.cbm"
    
    assert lgb_path.exists(), f"LightGBM model not found at {lgb_path}"
    assert xgb_path.exists(), f"XGBoost model not found at {xgb_path}"
    assert cat_path.exists(), f"CatBoost model not found at {cat_path}"
    
    # Test loading LightGBM
    model_lgb = lgb.Booster(model_file=str(lgb_path))
    assert model_lgb is not None, "Failed to load LightGBM booster"
    
    # Test loading XGBoost
    model_xgb = xgb.Booster()
    model_xgb.load_model(str(xgb_path))
    assert model_xgb is not None, "Failed to load XGBoost booster"
    
    # Test loading CatBoost
    model_cat = CatBoostClassifier()
    model_cat.load_model(str(cat_path))
    assert model_cat is not None, "Failed to load CatBoost model"

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
            X_sample[col] = X_sample[col].astype('category').cat.codes + 1
            
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
    
    # 4. Test Ensembled/Blended predictions (mean blending)
    blend_preds = (preds_lgb + preds_xgb + preds_cat) / 3.0
    assert len(blend_preds) == 5
    assert np.all((blend_preds >= 0.0) & (blend_preds <= 1.0))
