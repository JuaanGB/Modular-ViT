import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = False):
    """
    Fija semillas para reproducibilidad en PyTorch, NumPy y Python.

    Args:
        seed (int): semilla base.
        deterministic (bool): si True, fuerza comportamiento determinista en CUDA
                              (más lento pero reproducible).
    """

    # Python
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch GPU
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Evitar nondeterminismo en cuDNN
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True

    # Variables de entorno (importante para dataloaders)
    os.environ["PYTHONHASHSEED"] = str(seed)

    print(f"[Reproducibility] Seed fijada a {seed} | deterministic={deterministic}")