"""Device selection helper."""
from __future__ import annotations

import torch

__all__ = ["get_device"]


def get_device(preferred: str) -> torch.device:
    """
    Return the compute device based on availability and preference.

    Args:
        preferred (str): 'cuda' or 'cpu'.

    Returns:
        torch.device: Selected device.
    """
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
