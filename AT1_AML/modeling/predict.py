import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import catboost as cat
from pathlib import Path
from loguru import logger
import typer
from AT1_AML.config import MODELS_DIR, PROCESSED_DATA_DIR, PROJ_ROOT

app = typer.Typer()

@app.command()
def main(
    test_features_path: Path = PROCESSED_DATA_DIR / "test_processed.csv",
    train_features_path: Path = PROCESSED_DATA_DIR / "train_processed.csv",
    predictions_output_path: Path = PROJ_ROOT / "submission.csv",
):
    logger.info("Loading processed datasets for inference...")
    if not test_features_path.exists() or not train_features_path.exists():
        logger.error("Processed datasets not found. Please run preprocessing first.")
        return
        
    train = pd.read_csv(train_features_path, low_memory=False)
    test = pd.read_csv(test_features_path, low_memory=False)
    
    # Identify feature columns
    drop_cols = ["pid", "label", "ht", "yr", "dob", "type", "num"]
    feature_cols = [col for col in train.columns if col not in drop_cols]
    
    X_train = train[feature_cols]
    X_test = test[feature_cols]
    
    # Encode categorical features matching the training scheme
    categorical_cols = ["team", "conf", "role"]
    for col in categorical_cols:
        if col in X_train.columns:
            categories = pd.concat([X_train[col], X_test[col]]).dropna().unique()
            X_test[col] = (pd.Categorical(X_test[col], categories=categories).codes + 1).astype(int)
            
    # Load the 3 finalized ensembled models
    lgb_model_path = MODELS_DIR / "final_lightgbm.txt"
    xgb_model_path = MODELS_DIR / "final_xgboost.json"
    cat_model_path = MODELS_DIR / "final_catboost.cbm"
    
    if not (lgb_model_path.exists() and xgb_model_path.exists() and cat_model_path.exists()):
        logger.error("Trained models not found. Please run model training first.")
        return
        
    logger.info("Loading trained models...")
    
    # 1. LightGBM Booster
    model_lgb = lgb.Booster(model_file=str(lgb_model_path))
    X_test_lgb = X_test.copy()
    for col in categorical_cols:
        X_test_lgb[col] = X_test_lgb[col].astype("category")
    preds_lgb = model_lgb.predict(X_test_lgb)
    
    # 2. XGBoost Booster
    model_xgb = xgb.Booster()
    model_xgb.load_model(str(xgb_model_path))
    dtest = xgb.DMatrix(X_test)
    preds_xgb = model_xgb.predict(dtest)
    
    # 3. CatBoost Classifier
    model_cat = cat.CatBoostClassifier()
    model_cat.load_model(str(cat_model_path), format="cbm")
    X_test_cat = X_test.copy()
    for col in categorical_cols:
        X_test_cat[col] = X_test_cat[col].astype(str)
    preds_cat = model_cat.predict_proba(X_test_cat)[:, 1]
    
    # Perform blending (40% LGB + 30% XGB + 30% CatBoost)
    logger.info("Blending ensembled model predictions...")
    blend_preds = 0.4 * preds_lgb + 0.3 * preds_xgb + 0.3 * preds_cat
    
    # Format submission file
    submission = pd.DataFrame({
        "pid": test["pid"],
        "label": blend_preds
    })
    
    # Apply player-level average post-processing
    logger.info("Applying player-level average post-processing...")
    submission['label'] = submission.groupby('pid')['label'].transform('mean')
    
    # Deduplicate by player ID (pid) to match unique test player count
    logger.info("Deduplicating predictions by player ID...")
    submission = submission.drop_duplicates(subset=["pid"])
    
    submission.to_csv(predictions_output_path, index=False)
    logger.success(f"Predictions successfully generated and saved to {predictions_output_path}")

if __name__ == "__main__":
    app()
