"""Prediction collection utilities for model evaluation."""

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader


def _will_labels_be_squeezed(y: Tensor) -> bool:
    """Check if labels will be squeezed to 1D by _reshape_labels."""
    return y.ndim == 1 or (y.ndim == 2 and y.size(1) == 1)


def _is_binary_logits(logits: Tensor) -> bool:
    """Check if logits represent binary classification output."""
    return logits.ndim == 2 and logits.size(1) == 1


def _reshape_logits(logits: Tensor, y: Tensor) -> Tensor:
    """
    Match logits’ first dimension to `y` while preserving class dimension.

    Rules
    -----
    * For binary classification:
        - If labels are 2D (node-level, shape (N, 1)), keep logits as (N, 1)
        - If labels are 1D (graph-level, shape (N,)), squeeze logits to (N,)
    * For multiclass, always (N, C)

    Parameters
    ----------
    logits : Tensor
        Raw model outputs (logits) for the batch.
    y : Tensor
        Ground truth labels for the batch.

    Returns
    -------
    Tensor
        Logits reshaped to match the label format.
    """
    # For binary classification, squeeze logits if labels will be 1D
    if _is_binary_logits(logits) and _will_labels_be_squeezed(y):
        logits = logits.squeeze(1)

    return logits


def _reshape_labels(y: Tensor) -> Tensor:
    """
    Make label tensor compatible with reshaped logits.

    * `(N, 1)` → `(N,)`
    * `(N, L)` → keep as-is
    *  1-D `(N,)` → keep as-is

    Parameters
    ----------
    y : Tensor
        Ground truth labels for the batch.

    Returns
    -------
    Tensor
        Labels reshaped to match the logits format.
    """
    if y.ndim == 2 and y.size(1) == 1:
        return y.squeeze(1)
    return y


@dataclass
class BatchPrediction:
    """
    Single batch prediction result.

    Attributes
    ----------
    labels : Tensor
        Ground truth labels for the batch.
    logits : Tensor
        Raw model outputs for the batch.
    """
    labels: Tensor
    logits: Tensor

    def to_numpy(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert tensors to numpy arrays.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Tuple of (labels, logits) as numpy arrays.
        """
        return (
            self.labels.cpu().numpy(),
            self.logits.detach().cpu().numpy()
        )


def collect_batch_prediction(
    model: nn.Module,
    batch: Batch,
    device: torch.device,
) -> BatchPrediction:
    """
    Extract predictions for a single batch.

    Parameters
    ----------
    model : nn.Module
        The model to use for prediction.
    batch : Batch
        The batch of data to predict on.
    device : torch.device
        The device to run the model on.

    Returns
    -------
    BatchPrediction
        The batch prediction result containing labels and logits.
    """
    batch = batch.to(device)
    logits = model(batch)
    logits = _reshape_logits(logits, batch.y)
    return BatchPrediction(labels=_reshape_labels(batch.y), logits=logits)


def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Collect all model predictions and ground truth labels.

    Parameters
    ----------
    model : nn.Module
        The model to use for prediction.
    loader : DataLoader
        The data loader providing batches.
    device : torch.device
        The device to run the model on.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Tuple of (labels, logits) for the entire dataset.
    """
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in loader:
            pred = collect_batch_prediction(model, batch, device)
            predictions.append(pred)
    labels, logits = zip(*(p.to_numpy() for p in predictions))
    return np.concatenate(labels, axis=0), np.concatenate(logits, axis=0)
