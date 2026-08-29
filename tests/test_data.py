import pandas as pd
from pathlib import Path
from AT1_AML.config import PROCESSED_DATA_DIR

def test_processed_data_exists():
    train_path = PROCESSED_DATA_DIR / "train_processed.csv"
    test_path = PROCESSED_DATA_DIR / "test_processed.csv"
    
    assert train_path.exists(), "Processed training data does not exist"
    assert test_path.exists(), "Processed test data does not exist"
    
    train = pd.read_csv(train_path, nrows=5)
    test = pd.read_csv(test_path, nrows=5)
    
    # Assert expected column counts (including all engineered interaction and composite metrics)
    assert train.shape[1] == 117, f"Expected 117 columns, got {train.shape[1]}"
    assert test.shape[1] == 116, f"Expected 116 columns, got {test.shape[1]}"
