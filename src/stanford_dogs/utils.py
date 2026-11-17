import os
import sys
import logging
import random
import numpy as np
import torch
from typing import List
from pathlib import Path
def seed_everything(seed: int) -> None:
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

def configure_logging(level = logging.INFO, logfile_path: Path | None = None):
    # using sys.stdout as it's better than sys.stderr for cloud log aggregators
    handlers: List[logging.Handler]  = [logging.StreamHandler(sys.stdout)]

    if logfile_path is not None:
        logfile_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(filename= logfile_path, mode= 'a', encoding='utf-8')
        handlers.append(file_handler)

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=handlers,
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

    # Suppress verbose third-party loggers
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)