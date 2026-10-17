from pathlib import Path
import os
import random
import numpy as np
import torch
import torch.nn as nn
from stanford_dogs.config import SEED

def seed_everything(seed: int = SEED) -> None:
    """
    Sets random seeds across Python, NumPy, PyTorch, and CUDA
    to ensure full experiment reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic cuDNN algorithms
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False