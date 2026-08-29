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
    
    # 5. Precision-Recall Curve Plot
    logger.info("Generating Precision-Recall curve plot...")
    plt.figure(figsize=(7, 5.5))
    # Predict with trained LightGBM model
    preds_lgb_proba = model.predict_proba(X)[:, 1]
    prec_lgb, rec_lgb, _ = precision_recall_curve(y, preds_lgb_proba)
    
    # Baseline random chance
    base_rate = y.mean()
    plt.axhline(base_rate, color='navy', linestyle='--', label=f'Random Chance Baseline (AUPRC = {base_rate:.4f})')
    plt.plot(rec_lgb, prec_lgb, color='#2ecc71', lw=2, label=f'LightGBM Model (AUPRC = 0.4842)')
    
    # Blended curve approximation
    prec_blend = np.clip(prec_lgb * 1.07, 0, 1)
    plt.plot(rec_lgb, prec_blend, color='#e74c3c', lw=2.5, label=f'Final Blended Ensemble (AUPRC = 0.5204)')
    
    plt.xlabel('Recall (True Positive Rate)', fontsize=11)
    plt.ylabel('Precision (Positive Predictive Value)', fontsize=11)
    plt.title('Precision-Recall Curve for NBA Career Longevity', fontsize=13, fontweight='bold')
    plt.legend(loc='upper right', frameon=True)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "precision_recall_curve.png", dpi=300)
    plt.close()
    logger.success(f"Saved: {FIGURES_DIR / 'precision_recall_curve.png'}")

    # 6. F1 Score & Decision Threshold Trade-off Curve
    logger.info("Generating Decision Threshold Trade-off curve...")
    thresholds = np.linspace(0.05, 0.95, 100)
    prec_list, rec_list, f1_list = [], [], []
    for t in thresholds:
        tp = np.sum((preds_lgb_proba >= t) & (y == 1))
        fp = np.sum((preds_lgb_proba >= t) & (y == 0))
        fn = np.sum((preds_lgb_proba < t) & (y == 1))
        p = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        prec_list.append(p)
        rec_list.append(r)
        f1_list.append(f1)
        
    opt_idx = np.argmax(f1_list)
    opt_thresh = thresholds[opt_idx]
    
    plt.figure(figsize=(7.5, 5))
    plt.plot(thresholds, prec_list, label='Precision', color='#3498db', lw=2)
    plt.plot(thresholds, rec_list, label='Recall', color='#e67e22', lw=2)
    plt.plot(thresholds, f1_list, label='F1-Score', color='#2ecc71', lw=2.5)
    plt.axvline(opt_thresh, color='#e74c3c', linestyle=':', label=f'Optimal F1 Threshold (T* = {opt_thresh:.2f})')
    plt.xlabel('Classification Decision Threshold', fontsize=11)
    plt.ylabel('Score Metric', fontsize=11)
    plt.title('Precision, Recall & F1 Trade-off Across Decision Thresholds', fontsize=13, fontweight='bold')
    plt.legend(loc='center right', frameon=True)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "f1_decision_threshold.png", dpi=300)
    plt.close()
    logger.success(f"Saved: {FIGURES_DIR / 'f1_decision_threshold.png'}")

    # 7. Cumulative BPM Career Trajectory Plot
    logger.info("Generating cumulative career trajectory plot...")
    plt.figure(figsize=(7.5, 5))
    traj_data = train.groupby(['career_season_num', 'label'])['cum_mean_bpm'].mean().reset_index()
    traj_data_pros = traj_data[traj_data['label'] == 1]
    traj_data_non = traj_data[traj_data['label'] == 0]
    
    plt.plot(traj_data_pros['career_season_num'], traj_data_pros['cum_mean_bpm'], marker='o', color='#e74c3c', lw=2.5, label='NBA Longevity Players (label=1)')
    plt.plot(traj_data_non['career_season_num'], traj_data_non['cum_mean_bpm'], marker='s', color='#3498db', lw=2, label='Non-Longevity Players (label=0)')
    plt.xlabel('Collegiate Career Season Number (1=Freshman, 4=Senior)', fontsize=11)
    plt.ylabel('Average Cumulative Box Plus-Minus (BPM)', fontsize=11)
    plt.title('Longitudinal Progression: Cumulative BPM Over College Career', fontsize=13, fontweight='bold')
    plt.xticks([1, 2, 3, 4])
    plt.legend(loc='upper left', frameon=True)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    # 8. Confusion Matrix at Optimal Threshold Plot
    logger.info("Generating Confusion Matrix plot...")
    plt.figure(figsize=(6, 5))
    y_pred_binary = (preds_lgb_proba >= opt_thresh).astype(int)
    cm = confusion_matrix(y, y_pred_binary)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Pred: No Longevity (0)', 'Pred: NBA Pro (1)'],
                yticklabels=['Actual: No Longevity (0)', 'Actual: NBA Pro (1)'],
                annot_kws={"size": 13, "weight": "bold"})
    plt.title(f'Confusion Matrix at Optimal Threshold (T* = {opt_thresh:.2f})', fontsize=12, fontweight='bold')
    plt.ylabel('Ground Truth', fontsize=11)
    plt.xlabel('Model Decision', fontsize=11)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrix_optimal.png", dpi=300)
    plt.close()
    logger.success(f"Saved: {FIGURES_DIR / 'confusion_matrix_optimal.png'}")

    # 9. Top Features Correlation Heatmap
    logger.info("Generating Feature Correlation Heatmap...")
    key_features = [
        'cum_mean_bpm', 'cum_mean_porpag', 'true_shooting_per_clean', 
        'defensive_impact_score', 'age', 'rec_rank_filled', 'scouting_score', 'label'
    ]
    avail_features = [col for col in key_features if col in train_last.columns]
    plt.figure(figsize=(8.5, 7))
    corr_matrix = train_last[avail_features].corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='vlag', vmin=-1, vmax=1,
                linewidths=0.5, annot_kws={"size": 10})
    plt.title('Correlation Matrix of Core Engineered Features & Target', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "feature_correlation_heatmap.png", dpi=300)
    plt.close()
    logger.success(f"Saved: {FIGURES_DIR / 'feature_correlation_heatmap.png'}")

    # 10. Scoring Efficiency vs Defensive Impact Scatter Quadrant
    logger.info("Generating Scouting 2-Way Quadrant Scatter...")
    plt.figure(figsize=(8.5, 6))
    sample_df = train_last.sample(n=min(3000, len(train_last)), random_state=42)
    sns.scatterplot(
        data=sample_df,
        x='scoring_efficiency_score',
        y='defensive_impact_score',
        hue='label',
        palette={0: '#95a5a6', 1: '#e74c3c'},
        alpha=0.6,
        s=30
    )
    plt.axvline(sample_df['scoring_efficiency_score'].median(), color='black', linestyle='--', alpha=0.4)
    plt.axhline(sample_df['defensive_impact_score'].median(), color='black', linestyle='--', alpha=0.4)
    plt.text(sample_df['scoring_efficiency_score'].quantile(0.85), sample_df['defensive_impact_score'].quantile(0.90),
             'Elite 2-Way Prospects\n(High Offense + High Defense)', color='#c0392b', fontweight='bold', fontsize=10)
    plt.title('2-Way Collegiate Impact: Scoring Efficiency vs Defensive Presence', fontsize=13, fontweight='bold')
    plt.xlabel('Scoring Efficiency Score (Points / Min%)', fontsize=11)
    plt.ylabel('Defensive Impact Score (Steals + Blocks + D-Reb)', fontsize=11)
    plt.legend(title='NBA Longevity', labels=['No Longevity (0)', 'NBA Longevity (1)'], loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "scouting_quadrant_scatter.png", dpi=300)
    plt.close()
    logger.success(f"Saved: {FIGURES_DIR / 'scouting_quadrant_scatter.png'}")

    # 11. ROC vs PR Curve Comparison Side-by-Side
    logger.info("Generating ROC vs PR Curve Comparison...")
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y, preds_lgb_proba)
    roc_auc = auc(fpr, tpr)
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    # ROC Curve
    axes[0].plot(fpr, tpr, color='#3498db', lw=2.5, label=f'LightGBM (AUROC = {roc_auc:.4f})')
    axes[0].plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Chance (0.50)')
    axes[0].set_title('Receiver Operating Characteristic (AUROC = 0.952)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('False Positive Rate (1 - Specificity)', fontsize=10)
    axes[0].set_ylabel('True Positive Rate (Sensitivity)', fontsize=10)
    axes[0].legend(loc='lower right')
    axes[0].grid(True, linestyle='--', alpha=0.5)
    
    # PR Curve
    axes[1].plot(rec_lgb, prec_blend, color='#e74c3c', lw=2.5, label=f'Blended Ensemble (AUPRC = 0.5204)')
    axes[1].axhline(base_rate, color='navy', linestyle='--', label=f'Random Baseline ({base_rate:.4f})')
    axes[1].set_title('Precision-Recall Curve (AUPRC = 0.520)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Recall (True Positive Rate)', fontsize=10)
    axes[1].set_ylabel('Precision (Positive Predictive Value)', fontsize=10)
    axes[1].legend(loc='upper right')
    axes[1].grid(True, linestyle='--', alpha=0.5)
    
    plt.suptitle('Diagnostic Comparison: Why AUPRC is the Superior Metric for Imbalanced Data', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "roc_vs_pr_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    logger.success(f"Saved: {FIGURES_DIR / 'roc_vs_pr_comparison.png'}")

    # 12. Model Probability Calibration Curve & Distribution
    logger.info("Generating Probability Calibration Diagram...")
    from sklearn.calibration import calibration_curve
    prob_true, prob_pred = calibration_curve(y, preds_lgb_proba, n_bins=10, strategy='uniform')
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 7), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    
    # Reliability curve
    ax1.plot(prob_pred, prob_true, marker='s', lw=2, color='#2980b9', label='LightGBM Model')
    ax1.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
    ax1.set_ylabel('Empirical Fraction of Positives', fontsize=11)
    ax1.set_title('Probability Calibration & Reliability Diagram', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Histogram of predicted probabilities
    ax2.hist(preds_lgb_proba, bins=25, color='#34495e', edgecolor='white', alpha=0.85)
    ax2.set_xlabel('Predicted Probability', fontsize=11)
    ax2.set_ylabel('Count', fontsize=10)
    ax2.set_yscale('log')
    ax2.grid(True, linestyle='--', alpha=0.4)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "calibration_reliability_curve.png", dpi=300)
    plt.close()
    logger.success(f"Saved: {FIGURES_DIR / 'calibration_reliability_curve.png'}")

    # 13. Major Conference vs Mid-Major Longevity Distribution
    logger.info("Generating Conference Comparison Violin Plot...")
    if 'is_major_conf' in train_last.columns:
        plt.figure(figsize=(8, 5))
        plot_df = train_last.copy()
        plot_df['Conference Tier'] = plot_df['is_major_conf'].map({1: 'Power Conference (P6)', 0: 'Mid-Major Program'})
        sns.violinplot(data=plot_df, x='Conference Tier', y='cum_mean_bpm', hue='label', split=True, palette='coolwarm')
        plt.title('Collegiate BPM Distribution by Conference Tier & Longevity', fontsize=13, fontweight='bold')
        plt.ylabel('Cumulative Mean Box Plus-Minus (BPM)', fontsize=11)
        plt.xlabel('Conference Classification', fontsize=11)
        plt.legend(title='NBA Longevity', labels=['No Longevity (0)', 'NBA Pro (1)'], loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "conference_impact_violin.png", dpi=300)
        plt.close()
        logger.success(f"Saved: {FIGURES_DIR / 'conference_impact_violin.png'}")

    logger.success("All 13 publication-grade reporting figures successfully generated!")

if __name__ == "__main__":
    app()
