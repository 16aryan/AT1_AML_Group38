import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import average_precision_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from AT1_AML.config import PROCESSED_DATA_DIR

# Load data
train_path = PROCESSED_DATA_DIR / "train_processed.csv"
train = pd.read_csv(train_path)

drop_cols = ["pid", "label", "ht", "yr", "dob", "type", "num"]
feature_cols = [col for col in train.columns if col not in drop_cols]

X = train[feature_cols].copy()
y = train["label"].copy()
groups = train["pid"].copy()

# Categorical mapping
categorical_cols = ["team", "conf", "role"]
for col in categorical_cols:
    if col in X.columns:
        X[col] = X[col].astype('category').cat.codes + 1

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

oof_lgb = np.zeros(len(train))
oof_xgb = np.zeros(len(train))
oof_cat = np.zeros(len(train))

for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
    # LGBM
    X_tr_lgb = X_train.copy()
    X_v_lgb = X_val.copy()
    for col in categorical_cols:
        X_tr_lgb[col] = X_tr_lgb[col].astype("category")
        X_v_lgb[col] = X_v_lgb[col].astype("category")
        
    model_lgb = lgb.LGBMClassifier(
        n_estimators=1000, learning_rate=0.01, num_leaves=31, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42+fold, verbose=-1
    )
    model_lgb.fit(X_tr_lgb, y_train, eval_set=[(X_v_lgb, y_val)], callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)])
    oof_lgb[val_idx] = model_lgb.predict_proba(X_v_lgb)[:, 1]
    
    # XGBoost
    model_xgb = xgb.XGBClassifier(
        n_estimators=1000, learning_rate=0.01, max_depth=5, subsample=0.8, colsample_bytree=0.8, random_state=42+fold, early_stopping_rounds=100
    )
    model_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    oof_xgb[val_idx] = model_xgb.predict_proba(X_val)[:, 1]
    
    # CatBoost
    X_train_cat = X_train.copy()
    X_val_cat = X_val.copy()
    for col in categorical_cols:
        X_train_cat[col] = X_train_cat[col].astype(str)
        X_val_cat[col] = X_val_cat[col].astype(str)
    model_cat = CatBoostClassifier(
        iterations=1000, learning_rate=0.02, depth=6, cat_features=categorical_cols, random_seed=42+fold, early_stopping_rounds=100, verbose=0
    )
    model_cat.fit(X_train_cat, y_train, eval_set=[(X_val_cat, y_val)])
    oof_cat[val_idx] = model_cat.predict_proba(X_val_cat)[:, 1]

score_lgb = average_precision_score(y, oof_lgb)
score_xgb = average_precision_score(y, oof_xgb)
score_cat = average_precision_score(y, oof_cat)
score_blend = average_precision_score(y, 0.4 * oof_lgb + 0.3 * oof_xgb + 0.3 * oof_cat)

print(f"Full-val LGB AUPRC: {score_lgb:.5f}")
print(f"Full-val XGB AUPRC: {score_xgb:.5f}")
print(f"Full-val CAT AUPRC: {score_cat:.5f}")
print(f"Full-val Blend AUPRC: {score_blend:.5f}")
