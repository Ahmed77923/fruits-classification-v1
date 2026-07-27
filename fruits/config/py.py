from pathlib import Path
import torch
DATASET_PATH = Path(r"C:\Users\moaze\Downloads\fruits-360_100x100\fruits-360_100x100/fruits-360")
TRAIN_PATH = DATASET_PATH / "Training"
TEST_PATH = DATASET_PATH / "Test"
print("Dataset Path: ", DATASET_PATH.exists())
print("Training Path: ", TRAIN_PATH.exists())
print("Test Path: ", TEST_PATH.exists())