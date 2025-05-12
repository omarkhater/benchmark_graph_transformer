"""Prediction collection utilities for model evaluation."""

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader


@dataclass
class BatchPrediction:
    """Single batch prediction result.

    Attributes
    ----------
    labels : Tensor
        Ground truth labels for the batch
    logits : Tensor
        Raw model outputs for the batch
    """
    labels: Tensor
    logits: Tensor

    def to_numpy(self) -> tuple[np.ndarray, np.ndarray]:
        """Convert tensors to numpy arrays."""
        return (
            self.labels.cpu().numpy(),
            self.logits.detach().cpu().numpy()
        )


def collect_batch_prediction(
    model: nn.Module,
    batch: Batch,
    device: torch.device,
) -> BatchPrediction:
    """Extract predictions for a single batch."""
    batch = batch.to(device)
    logits = model(batch)

    if logits.ndim > 2:
        logits = logits.reshape(-1, logits.size(-1))
    elif logits.ndim == 2 and batch.y.ndim == 1:
        logits = logits.squeeze(1)

    labels = batch.y.reshape(-1) if batch.y.ndim > 1 else batch.y
    return BatchPrediction(labels=labels, logits=logits)


def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect all model predictions and ground truth labels."""
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in loader:
            pred = collect_batch_prediction(model, batch, device)
            predictions.append(pred)

    labels, logits = zip(*(p.to_numpy() for p in predictions))
    return np.concatenate(labels), np.concatenate(logits)
