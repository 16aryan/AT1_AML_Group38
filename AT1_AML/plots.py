import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from pathlib import Path
from loguru import logger
import typer
from sklearn.metrics import precision_recall_curve, average_precision_score, confusion_matrix
from AT1_AML.config import FIGURES_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

@app.command()
def main(
    train_features_path: Path = PROCESSED_DATA_DIR / "train_processed.csv",
):
    logger.info("Loading processed datasets for plotting...")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    if not train_features_path.exists():
        logger.error("Processed training data not found. Please run preprocessing first.")
        return
        
    train = pd.read_csv(train_features_path, low_memory=False)
    
    # Filter to final college seasons for player-level analysis
    last_season_indices = train.groupby('pid')['career_season_num'].idxmax()
    train_last = train.loc[last_season_indices].copy()
    
    # 1. Target Class Distribution Plot
    logger.info("Generating target class distribution plot...")
    plt.figure(figsize=(7, 5))
    class_counts = train_last["label"].value_counts()
    colors = ["#4C72B0", "#C44E52"]
    
    plt.bar(["No Longevity (label=0)", "NBA Longevity (label=1)"], class_counts.values, color=colors, width=0.55)
    plt.ylabel("Number of Unique College Players", fontsize=11)
    plt.title("NBA Career Longevity Label Distribution (Unique Players)", fontsize=13, fontweight="bold")
    for i, val in enumerate(class_counts.values):
        plt.text(i, val + 150, f"{val:,}\n({val/len(train_last)*100:.2f}%)", ha="center", va="bottom", fontweight="bold")
    plt.ylim(0, max(class_counts.values) * 1.15)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "class_distribution.png", dpi=300)
    plt.close()
    logger.success(f"Saved: {FIGURES_DIR / 'class_distribution.png'}")
    
    # 2. Model Comparison AUPRC Barplot
    logger.info("Generating model architecture comparison plot...")
    models_scores = {
        "Logistic Reg": 0.48983,
        "LightGBM": 0.48416,
        "Extra Trees": 0.49197,
        "XGBoost": 0.50970,
        "CatBoost": 0.51188,
        "Blended Ensemble": 0.52035
    }
    plt.figure(figsize=(9, 5))
    bar_colors = ["#95a5a6", "#3498db", "#9b59b6", "#e67e22", "#1abc9c", "#e74c3c"]
    bars = plt.bar(models_scores.keys(), models_scores.values(), color=bar_colors, width=0.6)
    plt.ylabel("Out-of-Fold AUPRC Score", fontsize=11)
    plt.title("Model Architecture Performance Comparison (AUPRC)", fontsize=13, fontweight="bold")
    plt.ylim(0.40, 0.56)
    plt.axhline(0.034, color='navy', linestyle='--', label='Random Chance Baseline (0.034)')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.003, f"{height:.5f}", ha='center', va='bottom', fontweight='bold')
    plt.legend(loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "model_comparison_auprc.png", dpi=300)
    plt.close()
    logger.success(f"Saved: {FIGURES_DIR / 'model_comparison_auprc.png'}")

    # 3. Recruiting Pedigree Distribution Boxplot
    logger.info("Generating recruiting rank distribution boxplot...")
    plt.figure(figsize=(7, 5))
    rec_data = train_last[train_last['Rec Rank'].notna()]
    sns.boxplot(data=rec_data, x='label', y='Rec Rank', palette='mako', width=0.5)
    plt.xticks([0, 1], ['No Longevity (0)', 'NBA Longevity (1)'])
    plt.title('High School Recruiting Rank vs Professional Longevity', fontsize=12, fontweight='bold')
    plt.xlabel('Target Label', fontsize=11)
    plt.ylabel('High School Recruiting Rank (0-100)', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "recruiting_pedigree.png", dpi=300)
    plt.close()
    logger.success(f"Saved: {FIGURES_DIR / 'recruiting_pedigree.png'}")

    # 4. Feature Importances Plot from LightGBM
    logger.info("Generating LightGBM feature importance plot...")
    drop_cols = ["pid", "label", "ht", "yr", "dob", "type", "num"]
    feature_cols = [col for col in train_last.columns if col not in drop_cols]
    
    X = train_last[feature_cols].copy()
    y = train_last["label"]
    
    categorical_cols = ["team", "conf", "role"]
    for col in categorical_cols:
        if col in X.columns:
            X[col] = X[col].astype("category")
            
    model = lgb.LGBMClassifier(n_estimators=300, random_state=42, verbose=-1)
    model.fit(X, y)
    
    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=True)
    top_importances = importances.tail(15)
    
    plt.figure(figsize=(10, 6))
    top_importances.plot(kind="barh", color="#2ecc71")
    plt.xlabel("LightGBM Feature Importance (Gain / Split Count)", fontsize=11)
    plt.ylabel("Features", fontsize=11)
    plt.title("Top 15 Most Influential Features for NBA Career Longevity", fontsize=13, fontweight="bold")
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "feature_importances.png", dpi=300)
    plt.close()
    logger.success(f"Saved: {FIGURES_DIR / 'feature_importances.png'}")
    
    logger.success("All reporting figures successfully generated!")

if __name__ == "__main__":
    app()
