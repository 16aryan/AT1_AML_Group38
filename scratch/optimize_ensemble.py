import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
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

# Categorical mapping
categorical_cols = ["team", "conf", "role"]
for col in categorical_cols:
    if col in X.columns:
        X[col] = X[col].astype('category').cat.codes + 1

# Filter for last season
last_season_indices = train.groupby('pid')['career_season_num'].idxmax()
X_last = X.loc[last_season_indices].copy()
y_last = y.loc[last_season_indices].copy()

# Stratified K-Fold
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_lgb = np.zeros(len(X_last))
oof_xgb = np.zeros(len(X_last))
oof_cat = np.zeros(len(X_last))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_last, y_last)):
    X_train, y_train = X_last.iloc[train_idx], y_last.iloc[train_idx]
    X_val, y_val = X_last.iloc[val_idx], y_last.iloc[val_idx]
    
    # 1. LightGBM (with regularization)
    X_tr_lgb = X_train.copy()
    X_v_lgb = X_val.copy()
    for col in categorical_cols:
        X_tr_lgb[col] = X_tr_lgb[col].astype("category")
        X_v_lgb[col] = X_v_lgb[col].astype("category")
        
    model_lgb = lgb.LGBMClassifier(
        n_estimators=1500,
        learning_rate=0.015,
        num_leaves=15,          # Restrict num_leaves to prevent overfitting on small samples
        max_depth=4,            # Shallower trees
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=2.0,          # L1 regularization
        reg_lambda=5.0,         # L2 regularization
        min_child_samples=30,   # Prevent tiny leaf splits
        random_state=42 + fold,
        verbose=-1
    )
    model_lgb.fit(
        X_tr_lgb, y_train,
        eval_set=[(X_v_lgb, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=150, verbose=False)]
    )
    oof_lgb[val_idx] = model_lgb.predict_proba(X_v_lgb)[:, 1]
    
    # 2. XGBoost (with regularization)
    model_xgb = xgb.XGBClassifier(
        n_estimators=1500,
        learning_rate=0.015,
        max_depth=4,            # Shallower trees
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=2.0,          # L1 regularization
        reg_lambda=5.0,         # L2 regularization
        min_child_weight=5,     # Prevent overfitting on rare features
        random_state=42 + fold,
        early_stopping_rounds=150
    )
    model_xgb.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    oof_xgb[val_idx] = model_xgb.predict_proba(X_val)[:, 1]
    
    # 3. CatBoost (with regularization)
    X_train_cat = X_train.copy()
    X_val_cat = X_val.copy()
    for col in categorical_cols:
        X_train_cat[col] = X_train_cat[col].astype(str)
        X_val_cat[col] = X_val_cat[col].astype(str)
        
    model_cat = CatBoostClassifier(
        iterations=1500,
        learning_rate=0.02,
        depth=5,                # Shallower trees
        l2_leaf_reg=10.0,       # Higher L2 regularization
        cat_features=categorical_cols,
        random_seed=42 + fold,
        early_stopping_rounds=150,
        verbose=0
    )
    model_cat.fit(X_train_cat, y_train, eval_set=[(X_val_cat, y_val)])
    oof_cat[val_idx] = model_cat.predict_proba(X_val_cat)[:, 1]

# Calculate individual scores
score_lgb = average_precision_score(y_last, oof_lgb)
score_xgb = average_precision_score(y_last, oof_xgb)
score_cat = average_precision_score(y_last, oof_cat)

print(f"Regularized LGB AUPRC: {score_lgb:.5f}")
print(f"Regularized XGB AUPRC: {score_xgb:.5f}")
print(f"Regularized CAT AUPRC: {score_cat:.5f}")

# Grid Search for optimal blending weights
best_score = 0
best_weights = None

for w_lgb in np.linspace(0, 1, 11):
    for w_xgb in np.linspace(0, 1 - w_lgb, 11):
        w_cat = 1.0 - w_lgb - w_xgb
        if w_cat < 0 or w_cat > 1:
            continue
        
        blend = w_lgb * oof_lgb + w_xgb * oof_xgb + w_cat * oof_cat
        score = average_precision_score(y_last, blend)
        
        if score > best_score:
            best_score = score
            best_weights = (w_lgb, w_xgb, w_cat)

print(f"\nBest Blend AUPRC: {best_score:.5f}")
print(f"Best Weights (LGB, XGB, CAT): {best_weights}")
