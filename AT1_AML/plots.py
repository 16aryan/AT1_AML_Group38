import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from pathlib import Path
from loguru import logger
import typer
from AT1_AML.config import FIGURES_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

@app.command()
def main(
    train_features_path: Path = PROCESSED_DATA_DIR / "train_processed.csv",
    importance_output: Path = FIGURES_DIR / "feature_importances.png",
    distribution_output: Path = FIGURES_DIR / "class_distribution.png",
):
    logger.info("Loading processed datasets for plotting...")
    if not train_features_path.exists():
        logger.error("Processed training data not found. Please run preprocessing first.")
        return
        
    train = pd.read_csv(train_features_path, low_memory=False)
    
    # 1. Generate Class Distribution Plot
    logger.info("Generating target class distribution plot...")
    plt.figure(figsize=(7, 5))
    class_counts = train["label"].value_counts()
    colors = ["#3498db", "#e74c3c"]
    
    plt.bar(["No Longevity (label=0)", "NBA Longevity (label=1)"], class_counts.values, color=colors, width=0.6)
    plt.ylabel("Number of College Seasons")
    plt.title("NBA Career Longevity Label Distribution (Class Imbalance)")
    for i, val in enumerate(class_counts.values):
        plt.text(i, val + 500, f"{val}\n({val/len(train)*100:.1f}%)", ha="center", va="bottom", fontweight="bold")
    plt.tight_layout()
    plt.savefig(distribution_output, dpi=300)
    plt.close()
    logger.success(f"Class distribution plot saved to {distribution_output}")
    
    # 2. Generate Feature Importances Plot
    logger.info("Generating feature importance plot from LightGBM...")
    # Identify feature columns
    drop_cols = ["pid", "label", "ht", "yr", "dob", "type", "num"]
    feature_cols = [col for col in train.columns if col not in drop_cols]
    
    X = train[feature_cols].copy()
    y = train["label"]
    
    # Convert categorical cols to category for LightGBM
    categorical_cols = ["team", "conf", "role"]
    for col in categorical_cols:
        if col in X.columns:
            X[col] = X[col].astype("category")
            
    # Fit a LightGBM model to get importances
    model = lgb.LGBMClassifier(n_estimators=300, random_state=42, verbose=-1)
    model.fit(X, y)
    
    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=True)
    top_importances = importances.tail(15) # top 15 features
    
    plt.figure(figsize=(10, 6))
    top_importances.plot(kind="barh", color="#2ecc71")
    plt.xlabel("LightGBM Feature Importance Split Count")
    plt.ylabel("Features")
    plt.title("Top 15 Most Influential Features for NBA Career Longevity")
    plt.tight_layout()
    plt.savefig(importance_output, dpi=300)
    plt.close()
    logger.success(f"Feature importance plot saved to {importance_output}")

if __name__ == "__main__":
    app()
