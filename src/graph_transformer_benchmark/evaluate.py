import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from graph_transformer_benchmark.data import enrich_batch


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
