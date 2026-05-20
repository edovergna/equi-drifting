import random

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def initialize_training_config(args):
    set_seed(args.seed)
    torch.set_float32_matmul_precision("medium")
    precision = set_precision(args.precision)
    pl.seed_everything(args.seed, workers=True)
    device = get_device()
    deterministic = bool(args.deterministic)
    benchmark = torch.cuda.is_available() and not deterministic
    print(f"Deterministic: {deterministic}\nBenchmark: {benchmark}")
    return device, precision, deterministic, benchmark


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Return the best available torch.device (CUDA, then MPS, else CPU)."""

    if torch.cuda.is_available():
        print("CUDA is available. Using GPU.")
        device = torch.device("cuda")
    # elif torch.backends.mps.is_available():
    #     print("MPS is available. Using Apple Silicon GPU.")
    #     device = torch.device("mps")
    else:
        print("GPU is not available. Using CPU.")
        device = torch.device("cpu")

    return device


def set_precision(precision: str) -> str:
    """Determine the appropriate precision setting based on user input and hardware capabilities."""
    if precision == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return "bf16-mixed"
        elif torch.cuda.is_available():
            return "16-mixed"
        else:
            return "32-true"
    else:
        return precision
