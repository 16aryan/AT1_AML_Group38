import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
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

# 1. Random Forest
oof_rf = np.zeros(len(X_last))
for fold, (train_idx, val_idx) in enumerate(kf.split(X_last, y_last)):
    X_train, y_train = X_last.iloc[train_idx], y_last.iloc[train_idx]
    X_val, y_val = X_last.iloc[val_idx], y_last.iloc[val_idx]
    
    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('rf', RandomForestClassifier(n_estimators=500, max_depth=8, random_state=42+fold, n_jobs=-1))
    ])
    pipe.fit(X_train, y_train)
    oof_rf[val_idx] = pipe.predict_proba(X_val)[:, 1]

score_rf = average_precision_score(y_last, oof_rf)
print(f"Random Forest OOF AUPRC: {score_rf:.5f}")

# 2. Extra Trees
oof_et = np.zeros(len(X_last))
for fold, (train_idx, val_idx) in enumerate(kf.split(X_last, y_last)):
    X_train, y_train = X_last.iloc[train_idx], y_last.iloc[train_idx]
    X_val, y_val = X_last.iloc[val_idx], y_last.iloc[val_idx]
    
    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('et', ExtraTreesClassifier(n_estimators=500, max_depth=8, random_state=42+fold, n_jobs=-1))
    ])
    pipe.fit(X_train, y_train)
    oof_et[val_idx] = pipe.predict_proba(X_val)[:, 1]

score_et = average_precision_score(y_last, oof_et)
print(f"Extra Trees OOF AUPRC: {score_et:.5f}")

# 3. Logistic Regression
oof_lr = np.zeros(len(X_last))
for fold, (train_idx, val_idx) in enumerate(kf.split(X_last, y_last)):
    X_train, y_train = X_last.iloc[train_idx], y_last.iloc[train_idx]
    X_val, y_val = X_last.iloc[val_idx], y_last.iloc[val_idx]
    
    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(C=0.05, max_iter=1000, random_state=42+fold))
    ])
    pipe.fit(X_train, y_train)
    oof_lr[val_idx] = pipe.predict_proba(X_val)[:, 1]

score_lr = average_precision_score(y_last, oof_lr)
print(f"Logistic Regression OOF AUPRC: {score_lr:.5f}")
