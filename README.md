# 36120-26SP-group38-experiments

Group experimentation repository for UTS 36120 AT1 — predicting whether a U.S. college basketball player will sustain an NBA career (3+ seasons) using only college-level statistics, as part of a private Kaggle competition.

##  Overview
This repository contains our team's experimentation, modeling, and analysis work for Assignment 1 of 36120 Advanced Machine Learning Application. The goal is to build a model that predicts NBA career longevity (3+ seasons) based solely on a player's college basketball statistics, submitted to a private Kaggle competition.

##  Project Objective
- Predict whether a college basketball player will sustain a 3+ season NBA career
- Use only college-level performance statistics as input features
- Optimize and validate models against the Kaggle competition leaderboard

## 🛠 Tech Stack
- **Language:** Python
- **Core libraries:** pandas, numpy, scikit-learn (update as your stack evolves)
- **Environment:** Jupyter Notebook
- **Version control:** Git / GitHub
- **Dependencies:** see `requirements.txt`

##  Getting Started

### Setup
```bash
# Clone the repository
git clone https://github.com/AT1_AML_Group38/36120-26SP-group38-experiments.git
cd 36120-26SP-group38-experiments

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

## 📁 Repository Structure
```
├── data/           # Raw and processed datasets (train/test splits, Kaggle data)
├── models/         # Saved/trained model artifacts
├── notebooks/      # EDA, feature engineering, and modeling notebooks
├── .gitignore
├── README.md
└── requirements.txt
```

## 🌿 Branching Strategy
- `main` — stable, reviewed work merged from all members' experiments
- `<student-id>-AT1-experiments` — individual branches for each member's experimentation (e.g. `26173702-AT1-experiments`)

Each member experiments independently on their own branch, then merges validated improvements into `main` once reviewed.

## ✅ Contribution Workflow
1. Pull latest changes from `main` before starting new work:
   ```bash
   git checkout main
   git pull origin main
   ```
2. Switch to (or create) your personal experiment branch:
   ```bash
   git checkout -b <your-student-id>-AT1-experiments
   ```
3. Commit changes with clear messages describing the experiment or change:
   ```bash
   git commit -m "Add feature engineering for shooting efficiency stats"
   ```
4. Push your branch and merge into `main` once your approach is validated:
   ```bash
   git push origin <your-student-id>-AT1-experiments
   ```
5. Regularly merge `main` into your branch to stay up to date and avoid large conflicts later.

## Workflow Summary
1. **Data exploration** — understand available college stats and target variable distribution
2. **Feature engineering** — derive relevant features from raw college statistics
3. **Model experimentation** — each member tries different models/approaches on their branch
4. **Validation** — compare results locally and via Kaggle leaderboard submissions
5. **Final model selection** — merge the best-performing, well-validated approach into `main`

## Communication
Team updates, blockers, and result comparisons are shared through our group chat, with regular check-ins to track leaderboard progress and avoid duplicated experimentation.

## Notes
- Keep raw Kaggle data out of version control if it's large — reference `.gitignore`
- Document key modeling decisions and results (e.g. in notebooks or a shared log) so the final report can reference what was tried and why
- Avoid overwriting teammates' saved models in `models/` — use descriptive filenames
