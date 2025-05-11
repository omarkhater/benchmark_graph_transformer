#!/usr/bin/env python
"""
Training pipeline for GraphTransformer benchmarks with reproducible seeding
and health-metrics logging.
"""
import logging
import random

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from graph_transformer_benchmark.data import build_dataloaders
from graph_transformer_benchmark.evaluate import evaluate
from graph_transformer_benchmark.graph_models import build_model
from graph_transformer_benchmark.utils import (
    BatchEnrichedModel,
    build_run_name,
    get_device,
    init_mlflow,
    log_config,
    log_dataset_stats,
    log_health_metrics,
    set_seed,
)

logging.basicConfig(level=logging.INFO)

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


def _infer_num_node_features(loader: DataLoader) -> int:
    """
    Infer the number of input features per node from a DataLoader.

    Tries, in order:
    1. loader.dataset.num_node_features
    2. loader.dataset.dataset.num_node_features (if Subset)
    3. batch.x.size(-1) from first batch
    """
    ds = loader.dataset
    if hasattr(ds, "num_node_features"):
        return ds.num_node_features  # type: ignore[attr-defined]
    ds_under = getattr(ds, "dataset", None)
    if hasattr(ds_under, "num_node_features"):
        return ds_under.num_node_features  # type: ignore[attr-defined]
    first_batch = next(iter(loader))
    return int(first_batch.x.size(-1))


def _infer_num_classes(loader: DataLoader) -> int:
    """
    Infer the number of target classes from a DataLoader.

    Tries, in order:
    1. loader.dataset.num_classes
    2. loader.dataset.dataset.num_classes (if Subset)
    3. y.max().item() + 1 from first batch
    """
    ds = loader.dataset
    if hasattr(ds, "num_classes"):
        return ds.num_classes  # type: ignore[attr-defined]
    ds_under = getattr(ds, "dataset", None)
    if hasattr(ds_under, "num_classes"):
        return ds_under.num_classes  # type: ignore[attr-defined]
    first_batch = next(iter(loader))
    y = first_batch.y
    if y.dim() > 1:
        # e.g. one-hot or multi-task
        return int(y.size(-1))
    return int(y.max().item() + 1)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> float:
    """
    Train the model for one epoch.

    Args:
        model (nn.Module): Model to train.
        loader (DataLoader): Training data loader.
        optimizer (Optimizer): Optimizer instance.
        device (torch.device): Compute device.
        epoch (int): Current epoch number.

    Returns:
        float: Average loss.
    """
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc=f"Train Epoch {epoch}", leave=False):
        batch = batch.to(device)
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
    run_name = cfg.model.training.mlflow.run_name or build_run_name(cfg)
    description = cfg.model.training.mlflow.description
    train_accuracy = []
    val_accuracy = []
    with mlflow.start_run(run_name=run_name, nested=True):
        if description:
            mlflow.set_tag("mlflow.note.content", description)
        log_config(cfg)
        generator = torch.Generator().manual_seed(cfg.training.seed)
        train_loader, val_loader, test_loader = build_dataloaders(
            cfg, generator=generator, worker_init_fn=worker_init_fn
        )
        log_dataset_stats(train_loader, "train", log_to_mlflow=True)
        log_dataset_stats(val_loader, "val",   log_to_mlflow=True)
        log_dataset_stats(test_loader, "test", log_to_mlflow=True)

        device = get_device(cfg.training.device)
        num_feat = _infer_num_node_features(train_loader)
        num_cls = _infer_num_classes(train_loader)
        model = build_model(cfg.model, num_feat, num_cls).to(device)
        model = BatchEnrichedModel(model, cfg.model)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)

        for epoch in trange(
            1, cfg.training.epochs + 1, desc="Epochs", unit="epoch"
        ):
            loss = train_one_epoch(
                model, train_loader, optimizer, device, epoch
            )
            mlflow.log_metric("train_loss", loss, step=epoch)
            logging.info(f"Evaluating epoch {epoch}...")
            train_acc = evaluate(
                model, train_loader, device, cfg
            )
            mlflow.log_metric("train_acc", train_acc, step=epoch)
            val_acc = evaluate(
                model, val_loader, device, cfg
            )
            mlflow.log_metric("val_acc", val_acc, step=epoch)
            logging.info(
                f"Epoch {epoch} - Train Acc: {train_acc:.4f}"
                + f", Val Acc: {val_acc:.4f}")

            log_health_metrics(model, optimizer, epoch)
            train_accuracy.append(train_acc)
            val_accuracy.append(val_acc)

        test_acc = evaluate(
            model, test_loader, device, cfg
        )
        mlflow.log_metric("avg_train_acc", np.mean(train_accuracy))
        mlflow.log_metric("avg_val_acc", np.mean(val_accuracy))
        mlflow.log_metric("test_acc", test_acc)

        if cfg.training.mlflow.log_artifacts:
            torch.save(model.state_dict(), "model.pth")
            mlflow.log_artifact("model.pth")

        return val_acc
