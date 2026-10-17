from pathlib import Path

# Project root (one folder up from src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Paths
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "images" / "Images"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

# Hyperparameters
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
LR = 0.001
SEED = 42
NUM_WORKERS = 4

# Modelling
NUM_CLASSES = 120