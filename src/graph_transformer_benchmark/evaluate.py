from typing import Dict

import numpy as np
import torch
from ogb.graphproppred import Evaluator as GraphEvaluator
from ogb.nodeproppred import Evaluator as NodeEvaluator
from omegaconf import DictConfig
from torch import Tensor
from torch_geometric.loader import DataLoader


def _evaluate_graph_level(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    dataset_name: str,
) -> float:
    """Compute ROC-AUC on an OGB graph-level dataset.

    Args:
        model: A graph classification model.
        loader: DataLoader yielding batched graph data.
        device: Device on which to perform inference.
        dataset_name: Name of the OGB graph dataset (e.g., "ogbg-molhiv").

    Returns:
        The ROC-AUC score.

    Raises:
        RuntimeError: If the OGB graph evaluator cannot compute the metric.
    """
    evaluator = GraphEvaluator(name=dataset_name)
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits: Tensor = model(batch)
            true = batch.y.cpu().numpy()
            pred = logits.cpu().numpy()
            all_true.append(true)
            all_pred.append(pred)
    y_true = np.concatenate(all_true, axis=0)
    y_pred = np.concatenate(all_pred, axis=0)
    result: Dict[str, float] = evaluator.eval(
        {"y_true": y_true, "y_pred": y_pred})
    return result["rocauc"]


def _evaluate_node_level(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    dataset_name: str,
) -> float:
    """Compute accuracy on an OGB node-level dataset.

    Args:
        model: A node classification model.
        loader: DataLoader yielding the full graph as a single batch.
        device: Device on which to perform inference.
        dataset_name: Name of the OGB node dataset (e.g., "ogbn-arxiv").

    Returns:
        The classification accuracy.

    Raises:
        RuntimeError: If the OGB node evaluator cannot compute the metric.
    """
    evaluator = NodeEvaluator(name=dataset_name)
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits: Tensor = model(batch)
            logits = logits.view(-1, logits.size(-1))
            preds = logits.argmax(dim=-1).cpu().numpy()
            labels = batch.y.view(-1).cpu().numpy()
            all_true.append(labels)
            all_pred.append(preds)
    y_true = np.concatenate(all_true, axis=0)
    y_pred = np.concatenate(all_pred, axis=0)
    result: Dict[str, float] = evaluator.eval(
        {"y_true": y_true, "y_pred": y_pred})
    return result["acc"]


def _evaluate_generic(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Compute plain classification accuracy for non-OGB datasets.

    Args:
        model: A classification model.
        loader: DataLoader yielding batched data with `.y` labels.
        device: Device on which to perform inference.

    Returns:
        The overall classification accuracy.
    """
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits: Tensor = model(batch)
            preds = logits.argmax(dim=-1)
            labels = batch.y.view(-1)
            correct += int((preds == labels).sum().item())
            total += labels.size(0)
    return correct / total if total > 0 else 0.0


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg: DictConfig,
) -> float:
    """Dispatch to the correct evaluation routine based on dataset.

    Args:
        model: The model to evaluate.
        loader: DataLoader for validation or test data.
        device: Device for model inference.
        cfg: Configuration containing `cfg.data.dataset`.

    Returns:
        The chosen performance metric:
            ROC-AUC,
            OGB node accuracy,
            or generic accuracy.

    Raises:
        ValueError: If `cfg.data.dataset` does not match any supported pattern.
    """
    name = cfg.data.dataset.lower()
    if name.startswith("ogbg-"):
        return _evaluate_graph_level(model, loader, device, name)
    if name.startswith("ogbn-"):
        return _evaluate_node_level(model, loader, device, name)
    if isinstance(loader, DataLoader):
        return _evaluate_generic(model, loader, device)
    raise ValueError(f"Cannot evaluate dataset '{cfg.data.dataset}'.")
