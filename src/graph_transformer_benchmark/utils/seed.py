"""Seed management and determinism switches."""
from __future__ import annotations

import random
from typing import Final

import numpy as np
import torch

__all__ = ["set_seed", "configure_determinism", "worker_init_fn"]

_BACKENDS_SET: Final[dict[str, bool]] = {
    "cudnn": torch.backends.cudnn.is_available(),
}
_GLOBAL_SEED = 0


def set_seed(seed: int) -> None:
    """
    Seed all relevant RNGs for reproducibility.

    Args:
        seed (int): The seed to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Make CuDNN deterministic (may slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def configure_determinism(seed: int, cuda: bool) -> None:
    """Seed Python, NumPy and PyTorch PRNGs and enforce determinism.

    The helper toggles :pyfunc:`torch.use_deterministic_algorithms` and, when
    CUDA is available, disables cuDNN autotune so that identical seeds yield
    identical results across runs.

    Parameters
    ----------
    seed:
        Global random seed.
    cuda:
        ``True`` if at least one CUDA device is visible.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.use_deterministic_algorithms(True, warn_only=False)

    if cuda and _BACKENDS_SET["cudnn"]:
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id: int) -> None:
    """Set the seed for each worker in a DataLoader.

    Args:
        worker_id (int): The ID of the worker.
    """
    worker_seed = _GLOBAL_SEED + worker_id
    torch.manual_seed(worker_seed)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
