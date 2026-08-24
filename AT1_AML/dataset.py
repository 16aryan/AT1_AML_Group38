from pathlib import Path

from loguru import logger
from nba_prep.cleaning import calculate_age, clean_height, clean_year
from nba_prep.features import (
    calculate_advanced_recruit_and_conference_features,
    calculate_ast_tov_ratio,
    calculate_cumulative_features,
    calculate_defensive_impact,
    calculate_scoring_efficiency,
    calculate_true_shooting_percentage,
)
import numpy as np
import pandas as pd
import typer

from AT1_AML.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies initial data cleaning steps to the raw dataframe.
    """
    processed = df.copy()
    
    logger.info("Cleaning height and year columns...")
    processed["ht_inches"] = processed["ht"].apply(clean_height)
    processed["yr_clean"] = processed["yr"].apply(clean_year)
    
    logger.info("Calculating player age during the season...")
    # Clean DOB to avoid float formats if any, then calculate age
    processed["age"] = processed.apply(
        lambda r: calculate_age(r["year"], r["dob"]), axis=1
    )
    
    return processed

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates advanced basketball metrics and cumulative player career features.
    """
    processed = df.copy()
    
    logger.info("Calculating advanced basketball features...")
    
    # Estimate turnovers from ast and ast/tov
    tov_est = np.where(
        (processed["ast/tov"] > 0) & (processed["ast/tov"].notna()),
        processed["ast"] / processed["ast/tov"],
        0.0
    )
    processed["ast_tov_ratio_clean"] = calculate_ast_tov_ratio(processed["ast"], tov_est)
    
    # Calculate True Shooting Percentage
    fga = processed["twoPA"] + processed["TPA"]
    processed["true_shooting_per_clean"] = calculate_true_shooting_percentage(
        processed["pts"], fga, processed["FTA"]
    )
    
    # Calculate Defensive Impact Score
    processed["defensive_impact_score"] = calculate_defensive_impact(
        processed["stl"], processed["blk"], processed["dreb"]
    )
    
    # Calculate Scoring Efficiency
    processed["scoring_efficiency_score"] = calculate_scoring_efficiency(
        processed["pts"], processed["Min_per"]
    )
    
    # Calculate Usage-Offensive Rating Interaction
    processed["usage_ortg_interaction"] = processed["usg"] * processed["ORtg"]
    
    # Compute cumulative player career features
    logger.info("Sorting and computing cumulative player career features...")
    processed = processed.sort_values(by=["pid", "year"])
    cum_feats = calculate_cumulative_features(processed)
    processed = pd.concat([processed, cum_feats], axis=1)
    
    # Compute advanced recruit and conference features
    logger.info("Computing advanced recruit and conference features...")
    adv_feats = calculate_advanced_recruit_and_conference_features(processed)
    processed = pd.concat([processed, adv_feats], axis=1)
    
    # Compute interaction features
    logger.info("Computing advanced interaction features...")
    processed["bpm_age_interaction"] = processed["bpm"] * (25.0 - processed["age"])
    processed["conf_bpm_interaction"] = processed["bpm"] * processed["is_major_conf"]
    processed["usage_bpm_interaction"] = processed["usg"] * processed["bpm"]
    processed["scouting_score"] = (151.0 - processed["rec_rank_filled"]) * processed["max_bpm_to_date"]
    
    return processed

@app.command()
def main(
    train_input: Path = RAW_DATA_DIR / "train.csv",
    test_input: Path = RAW_DATA_DIR / "test.csv",
    train_interim_output: Path = INTERIM_DATA_DIR / "train_cleaned.csv",
    test_interim_output: Path = INTERIM_DATA_DIR / "test_cleaned.csv",
    train_output: Path = PROCESSED_DATA_DIR / "train_processed.csv",
    test_output: Path = PROCESSED_DATA_DIR / "test_processed.csv",
):
    logger.info("Loading raw datasets...")
    train = pd.read_csv(train_input, low_memory=False)
    test = pd.read_csv(test_input, low_memory=False)
    
    logger.info(f"Original train size: {len(train)}")
    train = train.drop_duplicates()
    logger.info(f"Train size after dropping duplicates: {len(train)}")
    
    # Add tracking columns to split and restore original order
    train["is_train"] = 1
    test["is_train"] = 0
    train["orig_index"] = train.index
    test["orig_index"] = test.index
    
    combined = pd.concat([train, test], axis=0, ignore_index=True)
    
    # Phase 1: Clean Data
    logger.info("Phase 1: Cleaning combined dataset...")
    combined_cleaned = clean_dataframe(combined)
    
    # Split cleaned data back and save to interim folder
    logger.info("Splitting cleaned datasets back to save to interim...")
    train_cleaned = combined_cleaned[combined_cleaned["is_train"] == 1].copy().sort_values(by="orig_index")
    test_cleaned = combined_cleaned[combined_cleaned["is_train"] == 0].copy().sort_values(by="orig_index")
    
    train_cleaned = train_cleaned.drop(columns=["is_train", "orig_index"])
    test_cleaned = test_cleaned.drop(columns=["is_train", "orig_index", "label"], errors="ignore")
    
    logger.info("Saving interim datasets...")
    train_cleaned.to_csv(train_interim_output, index=False)
    test_cleaned.to_csv(test_interim_output, index=False)
    
    # Phase 2: Feature Engineering
    logger.info("Phase 2: Running feature engineering on interim data...")
    combined_processed = engineer_features(combined_cleaned)
    
    logger.info("Splitting processed datasets back...")
    train_processed = combined_processed[combined_processed["is_train"] == 1].copy()
    test_processed = combined_processed[combined_processed["is_train"] == 0].copy()
    
    # Restore original sorting
    train_processed = train_processed.sort_values(by="orig_index")
    test_processed = test_processed.sort_values(by="orig_index")
    
    # Drop helper columns
    drop_cols = ["is_train", "orig_index"]
    train_processed = train_processed.drop(columns=drop_cols)
    test_processed = test_processed.drop(columns=drop_cols + ["label"], errors="ignore")
    
    logger.info("Saving processed datasets...")
    train_processed.to_csv(train_output, index=False)
    test_processed.to_csv(test_output, index=False)
    
    logger.success("Preprocessing and feature engineering complete.")

if __name__ == "__main__":
    app()
