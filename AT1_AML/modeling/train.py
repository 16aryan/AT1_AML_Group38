import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import catboost as cat
from sklearn.model_selection import StratifiedKFold
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
    
    X = train[feature_cols].copy()
    y = train["label"].copy()
    groups = train["pid"].copy()
    X_test = test[feature_cols].copy()
    
    # Configure categorical features robustly across all models
    categorical_cols = ["team", "conf", "role"]
    for col in categorical_cols:
        if col in X.columns:
            # Fit categories on combined train and test values to ensure consistent mapping
            categories = pd.concat([X[col], X_test[col]]).dropna().unique()
            X[col] = (pd.Categorical(X[col], categories=categories).codes + 1).astype(int)
            X_test[col] = (pd.Categorical(X_test[col], categories=categories).codes + 1).astype(int)
            
    # =========================================================================
    # TRAINING LAST-SEASON ONLY ENSEMBLE
    # =========================================================================
    logger.info("--- TRAINING LAST-SEASON ONLY ENSEMBLE ---")
    
    # Find the indices of each player's last college season
    last_season_indices = train.groupby('pid')['career_season_num'].idxmax()
    train_last = train.loc[last_season_indices]
    
    X_last = X.loc[last_season_indices].copy()
    y_last = y.loc[last_season_indices].copy()
    
    # Filter test set for each player's last college season
    test_last_indices = test.groupby('pid')['career_season_num'].idxmax()
    X_test_last = X_test.loc[test_last_indices].copy()
    test_last_pids = test.loc[test_last_indices, 'pid'].values
    
    # standard k-fold since groups contain 1 row per player
    kf_last = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    oof_lgb = np.zeros(len(train_last))
    oof_xgb = np.zeros(len(train_last))
    oof_cat = np.zeros(len(train_last))
    
    test_lgb = np.zeros(len(X_test_last))
    test_xgb = np.zeros(len(X_test_last))
    test_cat = np.zeros(len(X_test_last))
    
    for fold, (train_idx, val_idx) in enumerate(kf_last.split(X_last, y_last)):
        X_train, y_train = X_last.iloc[train_idx], y_last.iloc[train_idx]
        X_val, y_val = X_last.iloc[val_idx], y_last.iloc[val_idx]
        
        # 1. LightGBM
        X_tr_lgb = X_train.copy()
        X_v_lgb = X_val.copy()
        X_t_lgb = X_test_last.copy()
        for col in categorical_cols:
            X_tr_lgb[col] = X_tr_lgb[col].astype("category")
            X_v_lgb[col] = X_v_lgb[col].astype("category")
            X_t_lgb[col] = X_t_lgb[col].astype("category")
            
        model_lgb = lgb.LGBMClassifier(
            n_estimators=1500,
            learning_rate=0.02,
            num_leaves=31,
            max_depth=5,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=0.5,
            reg_lambda=1.0,
            random_state=42 + fold,
            verbose=-1
        )
        model_lgb.fit(
            X_tr_lgb, y_train,
            eval_set=[(X_v_lgb, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=150, verbose=False)]
        )
        oof_lgb[val_idx] = model_lgb.predict_proba(X_v_lgb)[:, 1]
        test_lgb += model_lgb.predict_proba(X_t_lgb)[:, 1] / kf_last.n_splits
        
        # 2. XGBoost
        model_xgb = xgb.XGBClassifier(
            n_estimators=1500,
            learning_rate=0.01,
            max_depth=3,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=0.5,
            reg_lambda=1.0,
            random_state=42 + fold,
            early_stopping_rounds=150
        )
        model_xgb.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        oof_xgb[val_idx] = model_xgb.predict_proba(X_val)[:, 1]
        test_xgb += model_xgb.predict_proba(X_test_last)[:, 1] / kf_last.n_splits
        
        # 3. CatBoost
        X_train_cat = X_train.copy()
        X_val_cat = X_val.copy()
        X_test_cat = X_test_last.copy()
        for col in categorical_cols:
            X_train_cat[col] = X_train_cat[col].astype(str)
            X_val_cat[col] = X_val_cat[col].astype(str)
            X_test_cat[col] = X_test_cat[col].astype(str)
            
        model_cat = cat.CatBoostClassifier(
            iterations=1500,
            learning_rate=0.02,
            depth=5,
            l2_leaf_reg=10.0,
            cat_features=categorical_cols,
            random_seed=42 + fold,
            early_stopping_rounds=150,
            verbose=0
        )
        model_cat.fit(
            X_train_cat, y_train,
            eval_set=(X_val_cat, y_val)
        )
        oof_cat[val_idx] = model_cat.predict_proba(X_val_cat)[:, 1]
        test_cat += model_cat.predict_proba(X_test_cat)[:, 1] / kf_last.n_splits

    score_lgb = average_precision_score(y_last, oof_lgb)
    score_xgb = average_precision_score(y_last, oof_xgb)
    score_cat = average_precision_score(y_last, oof_cat)
    
    oof_blend = 0.10 * oof_lgb + 0.36 * oof_xgb + 0.54 * oof_cat
    test_blend = 0.10 * test_lgb + 0.36 * test_xgb + 0.54 * test_cat
    
    logger.info("--------------------------------------")
    logger.success(f"Last-season LGB OOF AUPRC: {score_lgb:.5f}")
    logger.success(f"Last-season XGB OOF AUPRC: {score_xgb:.5f}")
    logger.success(f"Last-season CAT OOF AUPRC: {score_cat:.5f}")
    logger.success(f"Last-season Unique Player AUPRC: {average_precision_score(y_last, oof_blend):.5f}")
    logger.info("--------------------------------------")
    
    # Map predictions to a dictionary: {pid: pred}
    pid_to_pred = dict(zip(test_last_pids, test_blend))
    
    # Load raw test.csv to get the exact original order of pids (12,061 rows)
    raw_test = pd.read_csv(PROJ_ROOT / "data" / "raw" / "test.csv")
    
    # Map each pid in raw_test to its prediction
    raw_test_preds = raw_test["pid"].map(pid_to_pred)
    raw_test_preds = raw_test_preds.fillna(0.033)
    
    # Create final submission matching original test.csv row count and order
    submission_path = PROJ_ROOT / "submission.csv"
    submission = pd.DataFrame({
        "pid": raw_test["pid"],
        "label": raw_test_preds
    })
    
    submission.to_csv(submission_path, index=False)
    logger.success(f"Kaggle submission file successfully saved to {submission_path} ({len(submission)} rows)")
    
    # =========================================================================
    # TRAINING & SAVING FINAL MODELS
    # =========================================================================
    logger.info("--------------------------------------")
    logger.info("Training and saving final models on full datasets...")
    
    # LGBM
    X_last_lgb = X_last.copy()
    for col in categorical_cols:
        X_last_lgb[col] = X_last_lgb[col].astype("category")
    final_lgb_last = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.02,
        num_leaves=31,
        max_depth=5,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=0.5,
        reg_lambda=1.0,
        random_state=42,
        verbose=-1
    )
    final_lgb_last.fit(X_last_lgb, y_last)
    final_lgb_last.booster_.save_model(str(MODELS_DIR / "final_lightgbm.txt"))
    
    # XGBoost
    final_xgb_last = xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=3,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=0.5,
        reg_lambda=1.0,
        random_state=42
    )
    final_xgb_last.fit(X_last, y_last)
    final_xgb_last.get_booster().save_model(str(MODELS_DIR / "final_xgboost.json"))
    
    # CatBoost
    X_last_cat = X_last.copy()
    for col in categorical_cols:
        X_last_cat[col] = X_last_cat[col].astype(str)
    final_cat_last = cat.CatBoostClassifier(
        iterations=1000,
        learning_rate=0.02,
        depth=5,
        l2_leaf_reg=10.0,
        cat_features=categorical_cols,
        random_seed=42,
        verbose=0
    )
    final_cat_last.fit(X_last_cat, y_last)
    final_cat_last.save_model(str(MODELS_DIR / "final_catboost.cbm"), format="cbm")
    
    logger.success("All final models successfully trained and saved!")
    logger.info("--------------------------------------")

if __name__ == "__main__":
    train_ensemble()
