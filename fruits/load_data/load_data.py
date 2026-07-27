import pandas as pd
from pathlib import Path
def load_data():
    """
    Load the fruit data from the CSV file.

    Returns:
        pd.DataFrame: A DataFrame containing the fruit data.
    """

    DATASET_PATH = Path(r"C:\Users\PC\.cache\kagglehub\datasets\moltean\fruits\versions\99\fruits-360_100x100/fruits-360")
    TRAIN_PATH = DATASET_PATH / "Training"
    TEST_PATH = DATASET_PATH / "Test"
    print("Dataset Path: ", DATASET_PATH.exists())
    print("Training Path: ", TRAIN_PATH.exists())
    print("Test Path: ", TEST_PATH.exists())
    return TRAIN_PATH, TEST_PATH

