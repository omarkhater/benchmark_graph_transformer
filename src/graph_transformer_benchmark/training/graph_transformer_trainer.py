"""GraphTransformer‑specific :class:`BaseTrainer` implementation."""
from __future__ import annotations

import copy
import logging
import os
import traceback
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
    higher_is_better = {
        "accuracy",
        "macro_f1",
        "roc_auc",
        "pr_auc",
        "mcc",
        "f1",
        "precision",
        "recall",
        "r2",
        }
    lower_is_better = {"loss", "mse", "rmse", "mae"}

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
        save_dir: str | None = None,
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
        self.save_dir = save_dir

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
        val_metrics = evaluate(
            self.model,
            self.val_loader,
            self.device,
            self.cfg
        )
        self.all_val_metrics.append(val_metrics)
        self._log_evaluation_metrics(val_metrics, prefix="val")

    def after_training(self) -> None:
        train_metrics = evaluate(
            self.model,
            self.train_loader,
            self.device,
            self.cfg,
        )
        self._log_evaluation_metrics(train_metrics, prefix="train")

        if self.save_dir:
            local_path = f"{self.save_dir}/model/best_model.pth"
        else:
            local_path = None
        self.save_pytorch_model(self.best_state, local_path)
        if mlflow.active_run():
            mlflow.log_artifact(local_path)
            mlflow.log_metric("best_val_loss", self.best_loss)
            mlflow.log_metric("best_val_epoch", self.best_epoch)
            mlflow.pytorch.log_model(self.model, "model")

        test_metrics = evaluate(
            self.model,
            self.test_loader,
            self.device,
            self.cfg,
        )
        self._log_evaluation_metrics(test_metrics, prefix="test")
        mlflow.log_metric("best_val_loss", self.best_loss)
        mlflow.log_metric("best_val_epoch", self.best_epoch)
        torch.save(self.best_state, "best_model.pth")
        mlflow.log_artifact("best_model.pth")
        mlflow.pytorch.log_model(self.model, "model")

    @staticmethod
    def _log_evaluation_metrics(metrics: dict, prefix: str = "val") -> None:
        """
        Log evaluation metrics to mlflow.

        Args:
            metrics (dict): Dictionary of evaluation metrics.
            prefix (str): Prefix for the metric names.
        """
        for name, val in metrics.items():
            if name in GraphTransformerTrainer.higher_is_better:
                key = f"{prefix}/higher_is_better/{name}"
            elif name in GraphTransformerTrainer.lower_is_better:
                key = f"{prefix}/lower_is_better/{name}"
            else:
                key = f"{prefix}/{name}"
            mlflow.log_metric(key, float(val))

    @staticmethod
    def save_pytorch_model(model: nn.Module, path: str) -> None:
        """
        Save the PyTorch model to a file.

        Args:
            model (nn.Module): The PyTorch model to save.
            path (str): Path to save the model.
        """
        try:
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(model, path)
            logging.info(f"Model saved to {path}")
        except Exception:
            logging.info(f"Failed to save model to {path} with error\n")
            logging.info(f"{traceback.format_exc()}")
