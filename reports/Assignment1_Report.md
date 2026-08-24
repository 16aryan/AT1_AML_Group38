# Assignment 1: Kaggle Competition

**Group**: group38  
**Student Name**: Aryan Goel  
**Student ID**: 26040826  
**GitHub ID**: 16aryan  
**Kaggle ID**: 16aryan  
**Semester**: 2026 SPRING  

| Link Type | Target Repository Link |
| :--- | :--- |
| **Group Experimentation Repo** | [https://github.com/16aryan/AT1_AML_Group38](https://github.com/16aryan/AT1_AML_Group38) |
| **Custom Package Repo** | [https://github.com/16aryan/36120-26SP-group38-26040826-Python-package-Aryan-Goel](https://github.com/16aryan/36120-26SP-group38-26040826-Python-package-Aryan-Goel) |

---
*36120 - Advanced Machine Learning Application*  
*Master of Data Science and Innovation*  
*University of Technology of Sydney*  

---

## Executive Summary & Objective

The objective of this project is to construct a robust machine learning pipeline to predict collegiate basketball players' career longevity in the NBA. Specifically, the target variable (`label`) is binary, representing whether a player will go on to play **at least three seasons** in the NBA. 

From a business standpoint, NBA franchises commit significant financial resources ($2M to $10M+ per rookie contract) and draft capital on college prospects. Selecting a player who fails to sustain an NBA career (a draft "bust") represents a major loss of capital and opportunity. By leveraging historical collegiate performance records (containing over 100 raw box score and advanced attributes), this project builds an ensemble model to maximize the **Area Under the Precision-Recall Curve (AUPRC)**. This metric is chosen over standard AUROC because the dataset is imbalanced (longevity success represents a minority class of ~30%), and the front office prioritizes high precision (minimizing false positives/draft busts) while maintaining reasonable recall.

---

## 1. Exploratory Data Analysis (EDA)

Exploratory analysis of the historical collegiate dataset revealed several key insights regarding player longevity:
1. **On-Court Efficiency**: Advanced composite impact statistics like Box Plus-Minus (`bpm`) show a strong correlation with future NBA success. Players with a college career-peak BPM of `+8.0` or higher have a baseline NBA success rate of over 65%, whereas those below `+3.0` rarely sustain three seasons in the league.
2. **Pedigree and Recruiting Rank**: High school recruiting ranks (`Rec Rank`) are highly predictive but suffer from heavy missingness (often exceeding 60% for non-elite prospects). Elite prospects ranked in the top 25 have a massive probability of entering the NBA, suggesting recruiting rank must be imputed robustly rather than discarded.
3. **Age and Experience**: Players who achieve longevity often enter the NBA younger, implying that younger collegiate players with high production profiles represent higher-ceiling prospects than seniors who dominate due to physical maturity.
4. **Conference Strength**: Production in Power Conferences (e.g., ACC, Big Ten, Big 12, SEC, Pac-12, Big East) carries significantly more weight than production in mid-major conferences, demanding pace-adjusted and opponent-strength-aware features.

---

## 2. Data Cleaning & Preprocessing

Raw collegiate data is notoriously noisy, containing formatting inconsistencies and structural anomalies. The cleaning process was modularized into the `nba_prep.cleaning` module:
* **Height Standardisation (`clean_height`)**: Height records were stored as string objects in various formats (e.g., `"6-7"`, `"6'7"`). A regex parser was developed to isolate feet and inches, converting them into a clean, continuous numeric float representing total height in inches (e.g., `"6-7"` becomes `79.0`).
* **Collegiate Year Alignment (`clean_year`)**: Player academic years were listed as categorical strings (`Fr`, `So`, `Jr`, `Sr`). These were mapped directly to ordinal integers (`1`, `2`, `3`, `4`) to enable tree-based algorithms to leverage experience as a monotonic feature.
* **Age Calculation (`calculate_age`)**: Birthdates (`dob`) were cleaned to resolve floating-point string formats, and players' ages during the respective collegiate seasons were calculated relative to the season calendar year to capture developmental progression.

---

## 3. Advanced Feature Engineering & Taxonomy

A rich taxonomy of feature engineering was designed across three distinct phases to capture different facets of a prospect's profile:

### Group A: Advanced Basketball Analytics
Exposed via the package to calculate composite scoring and efficiency:
* **True Shooting Percentage (`TS%`)**: Captures overall shooting efficiency by weighting three-pointers, two-pointers, and free throws:
  $$\text{TS\%} = \frac{\text{PTS}}{2 \times (\text{FGA} + 0.44 \times \text{FTA})}$$
* **Assist-to-Turnover Ratio (`AST/TOV`)**: Tracks ball security and playmaking IQ, handling divide-by-zero errors safely with a small epsilon fallback.
* **Defensive Impact Score**: Combines defensive Box Plus-Minus (`dbpm`), defensive rating (`drtg`), steals, and blocks to represent defensive utility.
* **Usage-to-Efficiency Interaction**: Multiplies player usage rate (`usg`) by offensive rating (`ortg`) to identify high-volume players who maintain high efficiency.

### Group B: Cumulative Career Metrics
Trees cannot easily capture historical sequence information. To resolve this, cumulative features were engineered across a player's career:
* **Career Season Count (`career_season_num`)**: Tracks the running index of a player's collegiate seasons (1 for Freshman, 2 for Sophomore, etc.).
* **Running Career Averages**: Tracks running means of Box Plus-Minus (`bpm`), points (`pts`), assists (`ast`), and rebounds (`treb`) up to the current season.
* **Peak Career Maximums**: Tracks the maximum BPM, points, and assists achieved by the player to date.
* **Season-over-Season progression (`progression_deltas`)**: Calculates the difference in production (e.g., $\text{BPM}_{\text{current}} - \text{BPM}_{\text{previous}}$) to identify upward-trending prospects.

### Group C: Recruiting & Pace-Adjusted Conference stats
* **Recruiting Imputation**: Missing recruiting ranks (`Rec Rank`) were imputed to a default value (`150.0` or median) and paired with a binary indicator flag (`has_rec_rank`) to prevent tree distortion.
* **Major Conference Flags (`is_major_conf`)**: Flagged programs playing in the Power Conferences (ACC, Big Ten, Big 12, SEC, Pac-12, Big East).
* **Per-Minute Volume Rates**: Calculated points, rebounds, assists, blocks, and steals per minute played (`pts_per_min`, `treb_per_min`, etc.) to account for varying team paces and playing times.

---

## 4. Modeling Strategy & Cross-Validation

The project progressed through three sequential experimental phases:

### Experiment 1: Baseline Models
* **Data**: Standard box-score features with minimal cleaning.
* **Algorithm**: Logistic Regression and basic Decision Tree.
* **Validation**: K-Fold cross-validation.
* **Result**: Achieved low AUPRC (~0.58-0.62) due to severe overfitting on senior players and inability to capture non-linear interactions.

### Experiment 2: Advanced Feature Expansion
* **Data**: Incorporated Group A (Analytics) and Group C (Conference/Recruiting) features.
* **Algorithms**: Trained gradient-boosted trees: LightGBM, XGBoost, and CatBoost.
* **Validation**: 5-Fold Stratified Cross-Validation grouped by player ID (`pid`) to prevent data leakage (a player's multiple seasons appearing in both training and validation folds).
* **Result**: Validation AUPRC rose significantly to ~0.67-0.68.

### Experiment 3: Peak Performance & Career Sequence (Final Strategy)
* **Data**: Added Group B (Cumulative Career Stats & Progression Deltas).
* **Targeting Peak Potential**: Instead of predicting on every season, the modeling was targeted at each player's **final/last collegiate season** (using `train.groupby('pid')['career_season_num'].idxmax()`). This aligns directly with the real-world scouting process: a front office evaluates a player at the point they declare for the draft, not their freshman year.
* **Cross-Validation**: 5-Fold Stratified Cross-Validation on last-season data.
* **Result**: Validation AUPRC peaked at ~0.70-0.71, demonstrating that modeling peak production combined with cumulative career growth patterns is highly effective.

---

## 5. Model Ensembling & Blending

To achieve maximum prediction stability and generalize effectively to the unseen Kaggle test set, a probability blending ensemble was implemented. 
Three diverse gradient boosting frameworks were blended:
1. **LightGBM Classifier**: Handles sparse features and categorical splits exceptionally fast.
2. **XGBoost Classifier**: Highly regulated tree growth via L1/L2 regularization to prevent overfitting on noisy collegiate features.
3. **CatBoost Classifier**: Excelled at handling categorical variables (team, conference, role) without manual target encoding.

The final ensemble was constructed by extracting predictions from all three models and applying a weighted average blend:
$$\hat{y}_{\text{blend}} = w_{1} \cdot \hat{y}_{\text{LGB}} + w_{2} \cdot \hat{y}_{\text{XGB}} + w_{3} \cdot \hat{y}_{\text{Cat}}$$

The weights were optimized based on out-of-fold validation scores:
* **LightGBM weight**: `0.30`
* **XGBoost weight**: `0.30`
* **CatBoost weight**: `0.40` (due to superior handling of team/conference features)

This blending strategy yielded a final validation **AUPRC of 0.7105**, outperforming any single model.

---

## 6. Results, Feature Importance & Analysis

### Performance Metrics Comparison
| Model Strategy | Validation AUPRC | Validation AUROC |
| :--- | :--- | :--- |
| **Baseline (Exp 1)** | 0.6012 | 0.7145 |
| **LightGBM (Exp 3)** | 0.7038 | 0.8124 |
| **XGBoost (Exp 3)** | 0.7011 | 0.8098 |
| **CatBoost (Exp 3)** | 0.7062 | 0.8152 |
| **Blended Ensemble** | **0.7105** | **0.8201** |

### Feature Importance Analysis
Extraction of feature importances across the ensemble models revealed the following top predictors:
1. **`cum_mean_bpm` (Career Mean BPM)**: The absolute strongest predictor. A player's average Box Plus-Minus across their entire college career represents consistent, sustained impact.
2. **`age` (Draft Age)**: Younger prospects with high production scores are heavily favored to achieve longevity.
3. **`true_shooting_per_clean` (TS%)**: Highlights the premium that NBA front offices place on scoring efficiency.
4. **`is_major_conf`**: Proving production in a Power Conference acts as a critical filter.
5. **`progression_deltas`**: Players showing year-over-year statistical growth indicate a high developmental ceiling.

---

## 7. Financial & Business ROI Analysis

To translate the model's AUPRC score into tangible business value for an NBA front office, a financial ROI model was developed (implemented in the package's `nba_prep.evaluation` module):

* **Baseline Scenario**: collegiate players drafted have a baseline NBA success rate of approximately **30.0%**. This means 7 out of 10 selections fail to sustain a 3-year career, resulting in massive wasted capital.
* **Model-Driven Scouting**: Selecting prospects using our model at a probability threshold of **0.50** increases the draft success rate to **75.0%** (a **2.50x improvement** in draft yield).

### Wasted Cap Space Reduction
Assuming a standard first-round rookie contract guaranteed cost is **$5,000,000**:
* **Without the Model**: Out of 10 picks, 7 busts cost the team **$35,000,000** in wasted salary cap space.
* **With the Model**: At threshold 0.50, the model filters out high-risk prospects, reducing busts to 1 out of 4 drafted. This cuts wasted cap space down to **$5,000,000**—representing a **$30,000,000 cap space savings** that can be reallocated to signing veteran free agents.

### Scouting Efficiency
By applying the model's threshold filters, the front office can filter out **60% of the prospect pool** automatically. This allows scouts and video analysts to focus 100% of their travel budget, live viewings, and interview hours on the top 40% high-probability targets, increasing qualitative evaluation depth.

---

## 8. Conclusion & Future Directions

This project demonstrated that predicting NBA career longevity from collegiate statistics requires modeling peak production patterns (using final-season aggregation) and incorporating sequential career trends (cumulative stats). By blending LightGBM, XGBoost, and CatBoost, we built a highly stable classifier achieving a validation AUPRC of **0.7105**.

### Future Enhancements:
1. **Combine & Athletic Measurements**: Integrating physical metrics (wingspan, vertical leap, lane agility times) to complement box score metrics.
2. **Injury History Tracking**: Incorporating games missed due to injury as a feature, since health is a critical determinant of career longevity.
3. **Deep Learning Sequence Models**: Utilizing Recurrent Neural Networks (RNNs) or LSTMs to process season-by-season statistics sequentially instead of manually engineering cumulative features.
