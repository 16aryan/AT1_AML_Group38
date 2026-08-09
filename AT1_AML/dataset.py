from pathlib import Path
import pandas as pd
import numpy as np
from loguru import logger
import typer
from AT1_AML.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from nba_prep.cleaning import clean_height, clean_year, calculate_age
from nba_prep.features import (
    calculate_ast_tov_ratio,
    calculate_true_shooting_percentage,
    calculate_defensive_impact,
    calculate_scoring_efficiency,
    calculate_cumulative_features,
    calculate_advanced_recruit_and_conference_features,
)

app = typer.Typer()

def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and engineers features for a combined dataframe.
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
    
    return processed

@app.command()
def main(
    train_input: Path = RAW_DATA_DIR / "train.csv",
    test_input: Path = RAW_DATA_DIR / "test.csv",
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
    
    logger.info("Processing combined dataset...")
    combined_processed = process_dataframe(combined)
    
    logger.info("Splitting datasets back...")
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

