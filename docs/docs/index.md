# NBA Career Longevity Prediction Documentation

Welcome to the documentation for the NBA Career Longevity Prediction model. This project is developed individually by **Aryan Goel** for UTS Coursework (Assessment Task 1 - Advanced Machine Learning).

---

## Description
This project aims to build a machine learning model to predict whether a U.S. college basketball player will go on to have a sustained career in the NBA. 

A player is considered successful (achieved longevity) if they:
* **Play at least 3 seasons in the NBA.**

The predictive pipeline is built using advanced basketball analytics, sequential career-level feature engineering, a 3-model ensemble (LightGBM, XGBoost, CatBoost), and custom player-level post-processing.

---

## Core Modeling Pipeline
Our best approach uses:
1. **Deduplication**: Cleaned 3,481 exact duplicate records from the raw training dataset.
2. **Stratified Group K-Fold Cross-Validation**: Folds are grouped by player ID (`pid`) to ensure player records do not leak between training and validation.
3. **Advanced Feature Engineering**: Incorporates per-minute productivity, power conference classification, high school recruiting status, and chronological cumulative progression.
4. **Three-Model Blend**: Combines LightGBM (40%), XGBoost (30%), and CatBoost (30%).
5. **Player-level Average Post-Processing**: Computes the average prediction across all college seasons for a player ID to stabilize predictions.

---

## Commands and Execution
The pipeline is fully automated and can be executed via the CLI:
* **Data Preparation**: `python -m AT1_AML.dataset`
* **Model Training & Evaluation**: `python -m AT1_AML.modeling.train`
* **Test Suite**: `python -m pytest tests`
