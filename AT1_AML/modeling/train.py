import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import catboost as cat
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import average_precision_score
from pathlib import Path
from loguru import logger
from AT1_AML.config import PROCESSED_DATA_DIR, MODELS_DIR, PROJ_ROOT

def train_ensemble():
    logger.info("Loading processed datasets...")
    train_path = PROCESSED_DATA_DIR / "train_processed.csv"
    test_path = PROCESSED_DATA_DIR / "test_processed.csv"
    
    if not train_path.exists() or not test_path.exists():
        logger.error("Processed datasets not found. Please run the preprocessing script first.")
        return
        
    train = pd.read_csv(train_path, low_memory=False)
    test = pd.read_csv(test_path, low_memory=False)
    
    # Identify feature columns
    drop_cols = ["pid", "label", "ht", "yr", "dob", "type", "num"]
    feature_cols = [col for col in train.columns if col not in drop_cols]
    
    logger.info(f"Using {len(feature_cols)} features for model training.")
    
    X = train[feature_cols]
    y = train["label"]
    groups = train["pid"]
    X_test = test[feature_cols]
    
    # Configure categorical features robustly across all models
    categorical_cols = ["team", "conf", "role"]
    for col in categorical_cols:
        if col in X.columns:
            # Fit categories on combined train and test values to ensure consistent mapping
            categories = pd.concat([X[col], X_test[col]]).dropna().unique()
            X[col] = (pd.Categorical(X[col], categories=categories).codes + 1).astype(int)
            X_test[col] = (pd.Categorical(X_test[col], categories=categories).codes + 1).astype(int)
            
    # CV setup (Stratified Group K-Fold to prevent player-level leakage)
    gkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Out of fold predictions
    oof_lgb = np.zeros(len(train))
    oof_xgb = np.zeros(len(train))
    oof_cat = np.zeros(len(train))
    
    # Test predictions
    test_lgb = np.zeros(len(test))
    test_xgb = np.zeros(len(test))
    test_cat = np.zeros(len(test))
    
    logger.info("Starting 5-Fold Stratified Group Cross-Validation...")
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        # 1. Train LightGBM
        logger.info(f"--- Fold {fold + 1}: Training LightGBM ---")
        # For LightGBM, convert categorical cols to category dtype
        X_tr_lgb = X_train.copy()
        X_v_lgb = X_val.copy()
        X_t_lgb = X_test.copy()
        for col in categorical_cols:
            X_tr_lgb[col] = X_tr_lgb[col].astype("category")
            X_v_lgb[col] = X_v_lgb[col].astype("category")
            X_t_lgb[col] = X_t_lgb[col].astype("category")
            
        model_lgb = lgb.LGBMClassifier(
            n_estimators=1500,
            learning_rate=0.01,
            num_leaves=31,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight=None,
            random_state=42 + fold,
            verbose=-1
        )
        model_lgb.fit(
            X_tr_lgb, y_train,
            eval_set=[(X_v_lgb, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
        )
        val_lgb_preds = model_lgb.predict_proba(X_v_lgb)[:, 1]
        oof_lgb[val_idx] = val_lgb_preds
        test_lgb += model_lgb.predict_proba(X_t_lgb)[:, 1] / gkf.n_splits
        
        # 2. Train XGBoost
        logger.info(f"--- Fold {fold + 1}: Training XGBoost ---")
        # Treat category codes as simple integers for XGBoost
        model_xgb = xgb.XGBClassifier(
            n_estimators=1500,
            learning_rate=0.01,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42 + fold,
            early_stopping_rounds=100
        )
        model_xgb.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        val_xgb_preds = model_xgb.predict_proba(X_val)[:, 1]
        oof_xgb[val_idx] = val_xgb_preds
        test_xgb += model_xgb.predict_proba(X_test)[:, 1] / gkf.n_splits
        
        # 3. Train CatBoost
        logger.info(f"--- Fold {fold + 1}: Training CatBoost ---")
        # Treat codes as strings for CatBoost categorical features
        X_train_cat = X_train.copy()
        X_val_cat = X_val.copy()
        X_test_cat = X_test.copy()
        for col in categorical_cols:
            X_train_cat[col] = X_train_cat[col].astype(str)
            X_val_cat[col] = X_val_cat[col].astype(str)
            X_test_cat[col] = X_test_cat[col].astype(str)
            
        model_cat = cat.CatBoostClassifier(
            iterations=1500,
            learning_rate=0.02,
            depth=6,
            cat_features=categorical_cols,
            random_seed=42 + fold,
            early_stopping_rounds=100,
            verbose=0
        )
        model_cat.fit(
            X_train_cat, y_train,
            eval_set=(X_val_cat, y_val)
        )
        val_cat_preds = model_cat.predict_proba(X_val_cat)[:, 1]
        oof_cat[val_idx] = val_cat_preds
        test_cat += model_cat.predict_proba(X_test_cat)[:, 1] / gkf.n_splits
        
        # Score fold AUPRCs
        fold_lgb_score = average_precision_score(y_val, val_lgb_preds)
        fold_xgb_score = average_precision_score(y_val, val_xgb_preds)
        fold_cat_score = average_precision_score(y_val, val_cat_preds)
        logger.info(f"Fold {fold + 1} AUPRC - LGB: {fold_lgb_score:.5f}, XGB: {fold_xgb_score:.5f}, CAT: {fold_cat_score:.5f}")

    # Evaluate individual models and blend
    score_lgb = average_precision_score(y, oof_lgb)
    score_xgb = average_precision_score(y, oof_xgb)
    score_cat = average_precision_score(y, oof_cat)
    
    oof_blend = 0.4 * oof_lgb + 0.3 * oof_xgb + 0.3 * oof_cat
    score_blend = average_precision_score(y, oof_blend)
    
    # Evaluate player-averaged post-processing
    temp_df = pd.DataFrame({'pid': groups, 'pred': oof_blend, 'label': y})
    pid_mean_preds = temp_df.groupby('pid')['pred'].transform('mean')
    pp_score_blend = average_precision_score(y, pid_mean_preds)
    
    logger.info("--------------------------------------")
    logger.success(f"Overall OOF AUPRC - LightGBM: {score_lgb:.5f}")
    logger.success(f"Overall OOF AUPRC - XGBoost:  {score_xgb:.5f}")
    logger.success(f"Overall OOF AUPRC - CatBoost: {score_cat:.5f}")
    logger.success(f"Overall OOF AUPRC - Blend:    {score_blend:.5f}")
    logger.success(f"Overall OOF AUPRC - Blend (player-averaged post-processing): {pp_score_blend:.5f}")
    logger.info("--------------------------------------")
    
    # Save submission file with ensembled predictions and post-processing
    submission_path = PROJ_ROOT / "submission.csv"
    test_blend = 0.4 * test_lgb + 0.3 * test_xgb + 0.3 * test_cat
    
    submission = pd.DataFrame({
        "pid": test["pid"],
        "label": test_blend
    })
    
    # Apply player-level average post-processing
    logger.info("Applying player-level average post-processing to test predictions...")
    submission['label'] = submission.groupby('pid')['label'].transform('mean')
    
    # Deduplicate by player ID to prevent Kaggle duplicate ID error
    logger.info("Deduplicating submission by player ID (pid)...")
    submission = submission.drop_duplicates(subset=["pid"])
    
    submission.to_csv(submission_path, index=False)
    logger.success(f"Kaggle submission file saved to {submission_path}")

if __name__ == "__main__":
    train_ensemble()


