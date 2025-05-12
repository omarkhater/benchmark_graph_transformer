"""MLflow initialisation and configuration logging helpers."""
from __future__ import annotations

from pathlib import Path

import mlflow
import torch
import torch.nn as nn
import torch.optim
from omegaconf import DictConfig, OmegaConf

from .config_utils import flatten_cfg

__all__ = [
    "init_mlflow",
    "log_config",
    "log_health_metrics",
    ]


def init_mlflow(cfg: DictConfig) -> None:
    """
    Initialize MLflow using tracking URI and experiment name.

    Args:
        cfg (DictConfig): Contains `training.mlflow` settings.
    """
    mlflow.set_tracking_uri(cfg.training.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.training.mlflow.experiment_name)


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
            params.update(flatten_cfg(
                cfg[section_name],
                prefix=f"{section_name}.")
            )

    mlflow.log_params(params)

    config_path = Path("config.yaml")
    OmegaConf.save(config=cfg, f=config_path)
    mlflow.log_artifact(str(config_path))


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
    grad_norms: dict[str, float] = {}
    weight_norms: dict[str, float] = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms[f"grad_norm/{name}"] = param.grad.norm().item()
        weight_norms[f"weight_norm/{name}"] = param.data.norm().item()

    mlflow.log_metrics(grad_norms, step=epoch)
    mlflow.log_metrics(weight_norms, step=epoch)
    for idx, group in enumerate(optimizer.param_groups):
        lr = group["lr"]
        mlflow.log_metric(f"lr/group_{idx}", lr, step=epoch)
