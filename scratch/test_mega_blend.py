import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
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

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_lgb = np.zeros(len(X_last))
oof_xgb = np.zeros(len(X_last))
oof_cat = np.zeros(len(X_last))
oof_et = np.zeros(len(X_last))
oof_lr = np.zeros(len(X_last))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_last, y_last)):
    X_train, y_train = X_last.iloc[train_idx], y_last.iloc[train_idx]
    X_val, y_val = X_last.iloc[val_idx], y_last.iloc[val_idx]
    
    # 1. LGBM
    X_tr_lgb = X_train.copy()
    X_v_lgb = X_val.copy()
    for col in categorical_cols:
        X_tr_lgb[col] = X_tr_lgb[col].astype("category")
        X_v_lgb[col] = X_v_lgb[col].astype("category")
    model_lgb = lgb.LGBMClassifier(
        n_estimators=1000, learning_rate=0.02, num_leaves=31, max_depth=5, subsample=0.7, colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=1.0, random_state=42+fold, verbose=-1
    )
    model_lgb.fit(X_tr_lgb, y_train, eval_set=[(X_v_lgb, y_val)], callbacks=[lgb.early_stopping(stopping_rounds=150, verbose=False)])
    oof_lgb[val_idx] = model_lgb.predict_proba(X_v_lgb)[:, 1]
    
    # 2. XGBoost
    model_xgb = xgb.XGBClassifier(
        n_estimators=1000, learning_rate=0.01, max_depth=3, subsample=0.7, colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=1.0, random_state=42+fold, early_stopping_rounds=150
    )
    model_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    oof_xgb[val_idx] = model_xgb.predict_proba(X_val)[:, 1]
    
    # 3. CatBoost
    X_train_cat = X_train.copy()
    X_val_cat = X_val.copy()
    for col in categorical_cols:
        X_train_cat[col] = X_train_cat[col].astype(str)
        X_val_cat[col] = X_val_cat[col].astype(str)
    model_cat = CatBoostClassifier(
        iterations=1000, learning_rate=0.02, depth=5, l2_leaf_reg=10.0, cat_features=categorical_cols, random_seed=42+fold, early_stopping_rounds=150, verbose=0
    )
    model_cat.fit(X_train_cat, y_train, eval_set=[(X_val_cat, y_val)])
    oof_cat[val_idx] = model_cat.predict_proba(X_val_cat)[:, 1]
    
    # 4. Extra Trees
    pipe_et = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('et', ExtraTreesClassifier(n_estimators=500, max_depth=8, random_state=42+fold, n_jobs=-1))
    ])
    pipe_et.fit(X_train, y_train)
    oof_et[val_idx] = pipe_et.predict_proba(X_val)[:, 1]
    
    # 5. Logistic Regression
    pipe_lr = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(C=0.05, max_iter=1000, random_state=42+fold))
    ])
    pipe_lr.fit(X_train, y_train)
    oof_lr[val_idx] = pipe_lr.predict_proba(X_val)[:, 1]

# Find best weights using simple random search
best_score = 0
best_w = None

np.random.seed(42)
for i in range(10000):
    w = np.random.dirichlet(np.ones(5))
    blend = w[0] * oof_lgb + w[1] * oof_xgb + w[2] * oof_cat + w[3] * oof_et + w[4] * oof_lr
    score = average_precision_score(y_last, blend)
    if score > best_score:
        best_score = score
        best_w = w

print(f"Best Mega-Blend AUPRC: {best_score:.5f}")
print(f"Weights (LGB, XGB, CAT, ET, LR): {best_w}")
