#!/usr/bin/env python
"""
Utility functions for MLflow setup and config logging.
"""
from pathlib import Path

import mlflow
from omegaconf import DictConfig, OmegaConf


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
    Log config parameters to MLflow and save merged config file.

    Args:
        cfg (DictConfig): Full merged run configuration.
    """
    mlflow.log_params(dict(cfg.data))
    mlflow.log_params(dict(cfg.model))
    mlflow.log_params(dict(cfg.training))

    config_path = Path("config.yaml")
    OmegaConf.save(config=cfg, f=config_path.as_posix())
    mlflow.log_artifact(config_path.as_posix())
