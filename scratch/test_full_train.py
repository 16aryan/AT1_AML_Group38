import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import average_precision_score
import lightgbm as lgb
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

# Identify last season indices for validation
last_season_indices = train.groupby('pid')['career_season_num'].idxmax()
is_last_season = np.zeros(len(train), dtype=bool)
is_last_season[last_season_indices] = True

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

oof_lgb = np.zeros(len(last_season_indices))
pid_to_idx = {pid: i for i, pid in enumerate(train.loc[last_season_indices, 'pid'])}
y_last_val = train.loc[last_season_indices, 'label'].values

for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
    # Filter validation set to last season only
    val_idx_last = val_idx[is_last_season[val_idx]]
    
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx_last], y.iloc[val_idx_last]
    val_pids = groups.iloc[val_idx_last].values
    
    # Train LightGBM
    X_tr_lgb = X_train.copy()
    X_v_lgb = X_val.copy()
    for col in categorical_cols:
        X_tr_lgb[col] = X_tr_lgb[col].astype("category")
        X_v_lgb[col] = X_v_lgb[col].astype("category")
        
    model_lgb = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.01,
        num_leaves=31,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        verbose=-1
    )
    model_lgb.fit(
        X_tr_lgb, y_train,
        eval_set=[(X_v_lgb, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
    )
    
    preds = model_lgb.predict_proba(X_v_lgb)[:, 1]
    for pid, pred in zip(val_pids, preds):
        oof_lgb[pid_to_idx[pid]] = pred

score = average_precision_score(y_last_val, oof_lgb)
print("Out-of-fold AUPRC on last season when training on all seasons:", score)
