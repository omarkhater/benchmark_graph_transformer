#!/usr/bin/env python
"""
Utility functions for MLflow setup and config logging.
"""
import random
from pathlib import Path
from typing import Any, Dict

import mlflow
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf


def init_mlflow(cfg: DictConfig) -> None:
    """
    Initialize MLflow using tracking URI and experiment name.

    Args:
        cfg (DictConfig): Contains `training.mlflow` settings.
    """
    mlflow.set_tracking_uri(cfg.training.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.training.mlflow.experiment_name)


def _flatten_cfg(section: DictConfig | Dict[str, Any],
                 prefix: str = "") -> Dict[str, Any]:
    """
    Recursively flatten a DictConfig (or plain dict).

    Each leaf key becomes "<prefix>key[.subkey...]" so that
    all parameter names are globally unique.
    """
    flat = {}
    for k, v in section.items():
        full_key = f"{prefix}{k}"
        if isinstance(v, (DictConfig, dict)):
            flat.update(_flatten_cfg(v, prefix=f"{full_key}."))
        else:
            if hasattr(v, "item"):
                v = v.item()
            flat[full_key] = v
    return flat


def log_config(cfg: DictConfig) -> None:
    """
    Log *all* run‑time configuration parameters to MLflow with
    unique keys and store the merged config file as an artifact.

    Args:
        cfg (DictConfig): Full merged run configuration.
    """
    # 1. Build a single flattened param dict
    params = {}
    for section_name in ("data", "model", "training"):
        if section_name in cfg:
            params.update(_flatten_cfg(cfg[section_name],
                                       prefix=f"{section_name}."))

    mlflow.log_params(params)

    config_path = Path("config.yaml")
    OmegaConf.save(config=cfg, f=config_path)
    mlflow.log_artifact(str(config_path))


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


def log_health_metrics(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
) -> None:
    """
    Log gradient norms, weight norms, and learning rates.

    Args:
        model (nn.Module): Trained model.
        optimizer (Optimizer): Optimizer.
        epoch (int): Current epoch number.
    """
    grad_norms: Dict[str, float] = {}
    weight_norms: Dict[str, float] = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms[f"grad_norm/{name}"] = param.grad.norm().item()
        weight_norms[f"weight_norm/{name}"] = param.data.norm().item()

    mlflow.log_metrics(grad_norms, step=epoch)
    mlflow.log_metrics(weight_norms, step=epoch)
    for idx, group in enumerate(optimizer.param_groups):
        lr = group["lr"]
        mlflow.log_metric(f"lr/group_{idx}", lr, step=epoch)
