"""
Shared utility helpers.

Add common functions here so every team member can import them
without duplicating code across files.
"""

import random
import numpy as np


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def moving_average(values, window: int = 10):
    """Return the moving average of *values* with the given *window* size."""
    weights = np.ones(window) / window
    return np.convolve(values, weights, mode="valid")
