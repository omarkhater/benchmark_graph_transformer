#!/usr/bin/env python
"""
Training pipeline for GraphTransformer benchmarks with reproducible seeding
and health-metrics logging.
"""
import random
from typing import Optional

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from graph_transformer_benchmark.data import build_dataloaders, enrich_batch
from graph_transformer_benchmark.evaluate import evaluate_accuracy
from graph_transformer_benchmark.model import build_model
from graph_transformer_benchmark.utils import (
    get_device,
    init_mlflow,
    log_config,
    log_health_metrics,
    set_seed,
)

_GLOBAL_SEED: int = 0


def worker_init_fn(worker_id: int) -> None:
    """
    Initialize each DataLoader worker with a reproducible seed.

    Args:
        worker_id (int): Worker index.
    """
    seed = _GLOBAL_SEED + worker_id
    random.seed(seed)
    np.random.seed(seed)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cfg: DictConfig,
    epoch: int,
) -> float:
    """
    Train the model for one epoch.

    Args:
        model (nn.Module): Model to train.
        loader (DataLoader): Training data loader.
        optimizer (Optimizer): Optimizer instance.
        device (torch.device): Compute device.
        cfg (DictConfig): Data configuration.
        epoch (int): Current epoch number.

    Returns:
        float: Average loss.
    """
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc=f"Train Epoch {epoch}", leave=False):
        batch = batch.to(device)
        batch = enrich_batch(batch, cfg.data)
        optimizer.zero_grad()
        outputs = model(batch)
        loss = F.cross_entropy(outputs, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


def run_training(cfg: DictConfig) -> None:
    """
    Execute training and evaluation, logging metrics to MLflow.

    Args:
        cfg (DictConfig): Configuration for data, model, training.
    """
    set_seed(cfg.training.seed)
    global _GLOBAL_SEED
    _GLOBAL_SEED = cfg.training.seed

    init_mlflow(cfg)
    run_name: Optional[str] = cfg.training.mlflow.run_name
    description: Optional[str] = cfg.training.mlflow.description

    with mlflow.start_run(run_name=run_name):
        if description:
            mlflow.set_tag("mlflow.note.content", description)
        log_config(cfg)

        generator = torch.Generator().manual_seed(cfg.training.seed)
        train_loader, test_loader = build_dataloaders(
            cfg, generator=generator, worker_init_fn=worker_init_fn
        )

        device = get_device(cfg.training.device)
        num_feat = train_loader.dataset.num_node_features
        num_cls = train_loader.dataset.num_classes
        model = build_model(cfg.model, num_feat, num_cls).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)

        for epoch in trange(
            1, cfg.training.epochs + 1, desc="Epochs", unit="epoch"
        ):
            loss = train_one_epoch(
                model, train_loader, optimizer, device, cfg, epoch
            )
            mlflow.log_metric("train_loss", loss, step=epoch)

            val_acc = evaluate_accuracy(
                model, test_loader, device, cfg
            )
            mlflow.log_metric("val_acc", val_acc, step=epoch)

            log_health_metrics(model, optimizer, epoch)

        test_acc = evaluate_accuracy(
            model, test_loader, device, cfg
        )
        mlflow.log_metric("test_acc", test_acc)

        if cfg.training.mlflow.log_artifacts:
            torch.save(model.state_dict(), "model.pth")
            mlflow.log_artifact("model.pth")
