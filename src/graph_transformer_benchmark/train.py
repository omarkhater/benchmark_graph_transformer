#!/usr/bin/env python
"""
Training pipeline for GraphTransformer benchmarks with health metrics logging.
"""
from typing import Dict, Optional

import mlflow
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from graph_transformer_benchmark.data import build_dataloaders, enrich_batch
from graph_transformer_benchmark.model import build_model
from graph_transformer_benchmark.utils import init_mlflow, log_config


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


def evaluate_accuracy(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg: DictConfig,
) -> float:
    """
    Compute classification accuracy over a dataset.

    Args:
        model (nn.Module): Trained model.
        loader (DataLoader): Evaluation data loader.
        device (torch.device): Compute device.
        cfg (DictConfig): Data configuration.

    Returns:
        float: Classification accuracy.
    """
    model.eval()
    correct, total = 0, len(loader.dataset)
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            batch = batch.to(device)
            batch = enrich_batch(batch, cfg.data)
            preds = model(batch).argmax(dim=-1)
            correct += int((preds == batch.y).sum())
    return correct / total


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


def run_training(cfg: DictConfig) -> None:
    """
    Execute training and evaluation, logging metrics to MLflow.

    Logs loss, accuracy, health metrics, and model artifacts.

    Args:
        cfg (DictConfig): Configuration for data, model, training.
    """
    init_mlflow(cfg)
    run_name: Optional[str] = cfg.training.mlflow.run_name
    description: Optional[str] = cfg.training.mlflow.description

    with mlflow.start_run(run_name=run_name):
        if description:
            mlflow.set_tag("mlflow.note.content", description)
        log_config(cfg)

        train_loader, test_loader = build_dataloaders(cfg)
        device = get_device(cfg.training.device)
        num_feat = train_loader.dataset.num_node_features
        num_cls = train_loader.dataset.num_classes
        model = build_model(cfg.model, num_feat, num_cls).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)

        for epoch in trange(
                1,
                cfg.training.epochs + 1,
                desc="Epochs",
                unit="epoch",
        ):
            loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                device,
                cfg,
                epoch,
            )
            mlflow.log_metric("train_loss", loss, step=epoch)

            accuracy = evaluate_accuracy(
                model,
                test_loader,
                device,
                cfg,
            )
            mlflow.log_metric("val_acc", accuracy, step=epoch)

            log_health_metrics(model, optimizer, epoch)

        final_acc = evaluate_accuracy(
            model,
            test_loader,
            device,
            cfg,
        )
        mlflow.log_metric("test_acc", final_acc)

        if cfg.training.mlflow.log_artifacts:
            torch.save(model.state_dict(), "model.pth")
            mlflow.log_artifact("model.pth")
