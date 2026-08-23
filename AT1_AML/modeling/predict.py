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
    
    X_train = train[feature_cols].copy()
    X_test = test[feature_cols].copy()
    
    # Encode categorical features matching the training scheme
    categorical_cols = ["team", "conf", "role"]
    for col in categorical_cols:
        if col in X_train.columns:
            categories = pd.concat([X_train[col], X_test[col]]).dropna().unique()
            X_train[col] = (pd.Categorical(X_train[col], categories=categories).codes + 1).astype(int)
            X_test[col] = (pd.Categorical(X_test[col], categories=categories).codes + 1).astype(int)
            
    # Filter test set for last college seasons
    test_last_indices = test.groupby('pid')['career_season_num'].idxmax()
    X_test_last = X_test.loc[test_last_indices].copy()
    test_last_pids = test.loc[test_last_indices, 'pid'].values
    
    logger.info("Loading trained models for Last-Season Ensemble...")
    # 1. Last-Season LGBM
    model_lgb = lgb.Booster(model_file=str(MODELS_DIR / "final_lightgbm.txt"))
    X_test_lgb = X_test_last.copy()
    for col in categorical_cols:
        X_test_lgb[col] = X_test_lgb[col].astype("category")
    preds_lgb = model_lgb.predict(X_test_lgb)
    
    # 2. Last-Season XGBoost
    model_xgb = xgb.Booster()
    model_xgb.load_model(str(MODELS_DIR / "final_xgboost.json"))
    dtest_last = xgb.DMatrix(X_test_last)
    preds_xgb = model_xgb.predict(dtest_last)
    
    # 3. Last-Season CatBoost
    model_cat = cat.CatBoostClassifier()
    model_cat.load_model(str(MODELS_DIR / "final_catboost.cbm"), format="cbm")
    X_test_cat = X_test_last.copy()
    for col in categorical_cols:
        X_test_cat[col] = X_test_cat[col].astype(str)
    preds_cat = model_cat.predict_proba(X_test_cat)[:, 1]
    
    # Blend predictions (Last-Season only)
    blend_preds = 0.4 * preds_lgb + 0.3 * preds_xgb + 0.3 * preds_cat
    
    # Save final predictions
    submission = pd.DataFrame({
        "pid": test_last_pids,
        "label": blend_preds
    })
    
    submission.to_csv(predictions_output_path, index=False)
    logger.success(f"Predictions successfully generated and saved to {predictions_output_path} ({len(submission)} rows)")

if __name__ == "__main__":
    app()
