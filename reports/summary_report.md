# Technical Summary Report: NBA Career Longevity Prediction

**Author:** Aryan Goel (Student ID: `26040826`)  
**Project:** Applied Machine Learning (AT1) — Group 38  
**Repository:** `36120-26SP-group38-26040826-experiments`  
**Helper Library:** `nba_prep` (v0.3.3)  

---

## 1. Executive Summary & Business Objective
In professional basketball, first-round NBA draft picks receive two-year guaranteed rookie-scale contracts (averaging $5M–$10M/year), with team options for years 3 and 4. Choosing a player who fails to sustain a career in the league represents a substantial loss in salary cap flexibility, wasted scouting expenses, and forfeited competitive advantage.

This project delivers a machine learning solution to predict whether a collegiate prospect will achieve a **sustained NBA career (playing $\ge 3$ seasons in the NBA)** based on their collegiate performance statistics, recruiting pedigree, and multi-season development trajectory.

---

## 2. Target Variable & Class Imbalance
* **Target Definition (`label`):** Binary indicator where `1` = Player sustained an NBA career ($\ge 3$ seasons) and `0` = Player played $<3$ seasons or never entered the NBA.
* **Severe Imbalance:** Across unique collegiate players in the training dataset, only **~3.74%** of players ever achieve this milestone.
* **Evaluation Metric:** Due to the severe class imbalance, Area Under the Receiver Operating Characteristic (AUROC) produces overly optimistic estimates because of the overwhelming number of true negatives. Consequently, **Area Under the Precision-Recall Curve (AUPRC)** is used as the primary evaluation metric.

![Class Distribution](figures/class_distribution.png)

---

## 3. Experimental Progression & Architecture Comparison

We structured our machine learning investigation across three sequential experimental notebooks:

### Experiment 1: Baseline Modelling (`36120-26SP-group38-26040826-AT1-experiment-1.ipynb`)
* **Goal:** Establish a baseline model using raw box-score metrics and initial data cleaning.
* **Algorithm:** Baseline LightGBM with default hyperparameters.
* **Finding:** Produced an initial OOF AUPRC of ~0.4400, proving that tree-based gradient boosting captures non-linear tabular signals effectively.

### Experiment 2: Architecture Exploration (`36120-26SP-group38-26040826-AT1-experiment-2.ipynb`)
* **Goal:** Compare diverse tabular model families on identical features and fold splits.
* **Algorithms Evaluated:**
  * Random Forest (Bagging): `0.171` OOF AUPRC
  * HistGradientBoosting: `0.205` OOF AUPRC
  * Tuned LightGBM: `0.212` OOF AUPRC (on full multi-season baseline)
* **Finding:** Boosting algorithms significantly outperform bagging for rare-class detection because gradient boosting focuses sequential splits on misclassified minority positive instances.

### Experiment 3: Feature Engineering & Ensembling (`36120-26SP-group38-26040826-AT1-experiment-3.ipynb`)
* **Goal:** Maximize AUPRC by isolating peak collegiate seasons, computing career-long trajectory features, and ensembling diverse GBDT architectures.
* **Key Innovations:**
  1. **Final-Season Filtering:** Restricting training to each player's final collegiate season removed noisy early developmental years, providing the cleanest draft-day signal.
  2. **Cumulative Metrics:** Engineered career-to-date expanding averages (`cum_mean_bpm`, `cum_mean_porpag`), peaks (`max_bpm_to_date`), and season-over-season deltas (`bpm_progression`).
  3. **Multi-Model Ensemble:** Blending predictions from LightGBM, XGBoost, CatBoost, Extra Trees, and Logistic Regression.

---

## 4. Final Model Performance Benchmark

All models were evaluated using **5-Fold Stratified Group Cross-Validation** grouped strictly by player ID (`pid`) to prevent data leakage across folds:

| Architecture | Out-of-Fold (OOF) AUPRC | Relative Lift over Baseline (0.034) | Status |
| :--- | :---: | :---: | :--- |
| **Random Chance Baseline** | `0.03396` | 1.0x | Reference |
| **LightGBM** | `0.48416` | 14.3x | Evaluated |
| **Logistic Regression** | `0.48983` | 14.4x | Evaluated |
| **Extra Trees** | `0.49197` | 14.5x | Evaluated |
| **XGBoost** | `0.50970` | 15.0x | Evaluated |
| **CatBoost** | `0.51188` | 15.1x | Evaluated |
| **Blended Ensemble (Final)** | **`0.52035`** | **15.3x** | **Selected Production Model** |

![Model Comparison](figures/model_comparison_auprc.png)

---

## 5. Key Feature Importance & Scouting Insights
Feature importance analysis extracted from the final models reveals what characteristics best separate future NBA players from the collegiate pool:

1. **Cumulative Impact & Consistency:** Career expanding Box Plus-Minus (`cum_mean_bpm`) and Points Over Replacement Per Adjusted Game (`cum_mean_porpag`) are the top predictive features. Sustained collegiate dominance across multiple seasons is far more predictive than a single anomalous breakout game.
2. **Recruiting Pedigree:** High school recruiting rank (`Rec Rank`) is one of the strongest individual features. Five-star prospects have a median rank of ~84.0 among successful pros, compared to ~53.0 in the general pool.
3. **Per-Minute Production:** Per-minute scoring (`pts_per_min`) and assist efficiency (`ast_tov_ratio_clean`) strongly outperform raw volume counts, correctly accounting for differences in collegiate team pace and rotation depth.
4. **Youth & Developmental Runway:** Seasonal age (`age`) exhibits a strong negative correlation with longevity; younger players achieving identical production have higher developmental upside.

![Feature Importances](figures/feature_importances.png)
![Recruiting Pedigree](figures/recruiting_pedigree.png)

---

## 6. Business ROI & Front-Office Decision Making
Using the `estimate_draft_roi` module in `nba_prep`, we evaluated the economic impact of implementing this model for draft-night prospect screening:
* **Scouting Travel & Film Hours Saved:** Front offices can safely filter out the bottom **60%–70%** of candidate prospects who possess $<5\%$ success probabilities, allowing scouts to concentrate travel budgets, physical workouts, and psychological evaluations exclusively on top-tier prospects.
* **Draft Hit Rate Improvement:** Threshold-based prospect filtering yields a **2.50x to 3.0x improvement** in draft selection success rate relative to the raw pool base rate.
* **Salary Cap Capital Preserved:** Avoiding just one first-round bust preserves upwards of **$10,000,000 to $15,000,000** in multi-year rookie salary cap commitments.

---

## 7. Generated Project Artifacts

* **Trained Models in `models/`:**
  * `final_lightgbm.joblib` & `final_lightgbm.txt`
  * `final_xgboost.joblib` & `final_xgboost.json`
  * `final_catboost.joblib` & `final_catboost.cbm`
  * `36120-26SP-group38-26040826-final_lightgbm.joblib`
  * `36120-26SP-group38-26040826-final_xgboost.joblib`
  * `36120-26SP-group38-26040826-final_catboost.joblib`
  * `36120-26SP-group38-26040826-experiment3-model.joblib`
* **Final Deduplicated Submission:**
  * `submission.csv` (6,261 unique player rows, 0 duplicate IDs, 0 nulls, mean probability = `0.03396`).
