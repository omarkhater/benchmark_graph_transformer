"""GraphTransformer‑specific :class:`BaseTrainer` implementation."""
from __future__ import annotations

import copy
from typing import Any

import mlflow
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from graph_transformer_benchmark.evaluation import evaluate
from graph_transformer_benchmark.training.base import BaseTrainer
from graph_transformer_benchmark.utils import infer_task_and_loss


class GraphTransformerTrainer(BaseTrainer):
    """Wrap a GraphTransformer model with the generic *BaseTrainer* logic.

    The subclass defines the forward–loss–backward routine for training and
    plugs evaluation metrics into *BaseTrainer* hooks so that MLflow receives
    per‑epoch results without cluttering the core loop.
    """

    def __init__(
        self,
        cfg: DictConfig,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
        num_epochs: int = 100,
        val_frequency: int = 5,
        patience: int = 5,
    ) -> None:
        super().__init__(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            device=device,
            scheduler=scheduler,
            num_epochs=num_epochs,
            val_frequency=val_frequency,
            patience=patience,
        )
        self.best_loss = float("inf")
        self.best_epoch = 0
        self.best_state = copy.deepcopy(self.model.state_dict())
        self.train_losses = []
        self.val_losses = []
        self.all_val_metrics = []
        self.num_epochs = num_epochs
        self.current_epoch = 0
        self.task_kind, self.criterion = infer_task_and_loss(train_loader)
        self.cfg = cfg
        self.test_loader = test_loader

    def _gather_tensors(self, batch: Any) -> Any:
        return batch

    def calculate_loss(self, batch: Any) -> dict[str, torch.Tensor]:
        data = batch.to(self.device)
        logits = self.model(data)

        if isinstance(self.criterion, nn.BCEWithLogitsLoss):
            targets = data.y.float()
        else:
            targets = data.y

        loss = self.criterion(logits, targets)
        return {"total_loss": loss, "raw_losses": {"total_loss": loss}}

    def train_epoch(self) -> dict[str, float]:
        self.model.train()
        running = 0.0
        for batch in self.train_loader:
            loss = self.calculate_loss(batch)["total_loss"]
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            running += loss.item() * batch.num_graphs
        avg = running / len(self.train_loader.dataset)
        return {"total_loss": avg}

    def validate_batch(self, batch: Any) -> dict[str, torch.Tensor]:
        self.model.eval()
        with torch.no_grad():
            return self.calculate_loss(batch)

    def after_validation(self, current_epoch: int) -> None:
        score, metric = evaluate(
            self.model,
            self.val_loader,
            self.device,
            self.cfg
        )
        self.all_val_metrics.append(score)
        mlflow.log_metric(f"val_{metric}", score, step=current_epoch-1)

    def after_training(self) -> None:
        train_score, metric = evaluate(
            self.model,
            self.train_loader,
            self.device,
            self.cfg,
        )
        mlflow.log_metric(f"train_{metric}", train_score)

        test_score, _ = evaluate(
            self.model,
            self.test_loader,
            self.device,
            self.cfg,
        )
        mlflow.log_metric(f"test_{metric}", test_score)
        mlflow.log_metric("best_val_loss", self.best_loss)
        mlflow.log_metric("best_val_epoch", self.best_epoch)
        torch.save(self.best_state, "best_model.pth")
        mlflow.log_artifact("best_model.pth")
        mlflow.pytorch.log_model(self.model, "model")
