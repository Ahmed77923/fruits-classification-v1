from pathlib import Path
import torch

# ==========================================================
# Paths
# ==========================================================
IMAGE_SIZE = (100, 100)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEFAULT_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


class Config:
    """Configuration class for the fruit classification project."""
    DATASET_PATH = Path(r"C:\Users\PC\.cache\kagglehub\datasets\moltean\fruits\versions\99\fruits-360_100x100/fruits-360")
    TRAIN_PATH = DATASET_PATH / "Training"
    TEST_PATH = DATASET_PATH / "Test"
    # print("Dataset Path: ", DATASET_PATH.exists())
    # print("Training Path: ", TRAIN_PATH.exists())
    # print("Test Path: ", TEST_PATH.exists())

    # ==========================================================
    # Training
    # ==========================================================

    BATCH_SIZE = 32
    EPOCHS = 15
    LEARNING_RATE = 1e-3

    # ==========================================================
    # Model
    # ==========================================================

    IMAGE_SIZE = (128, 128)
    NUM_WORKERS = 2

    # ==========================================================
    # Device
    # ==========================================================

    DEVICE = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    