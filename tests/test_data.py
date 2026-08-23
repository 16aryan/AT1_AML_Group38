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
    
    # Assert expected column counts
    assert train.shape[1] == 113, f"Expected 113 columns, got {train.shape[1]}"
    assert test.shape[1] == 112, f"Expected 112 columns, got {test.shape[1]}"
