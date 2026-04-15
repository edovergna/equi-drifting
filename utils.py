import random

import numpy as np
import torch


# Training settings
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Parse a device string and return a torch.device."""

    if torch.cuda.is_available():
        print("CUDA is available. Using GPU.")
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("MPS is available. Using Apple Silicon GPU.")
        device = torch.device("mps")
    else:
        print("GPU is not available. Using CPU.")
        device = torch.device("cpu")

    return device
