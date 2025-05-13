"""Compute and log basic graph dataset statistics."""
from __future__ import annotations

import logging
import statistics
from collections import Counter

import mlflow
from torch_geometric.loader import DataLoader

__all__ = [
    "log_dataset_stats",
    "infer_num_node_features",
    "infer_num_classes",
    ]


def log_dataset_stats(
    loader: DataLoader,
    split_name: str = "train",
    *,
    log_to_mlflow: bool = False,
) -> None:
    """Log basic structural statistics of a graph DataLoader.

    Computes per-graph node / edge counts and prints:
      • #Graphs
      • Avg / Min / Max #Nodes
      • Avg / Min / Max #Edges
      • Mode node-count (helpful to spot fixed-size batches)

    Args
    ----
    loader:
        PyG‐style ``DataLoader`` that yields ``torch_geometric.data.Data``.
        The underlying ``dataset`` may be a ``Subset`` or any object that
        supports ``__len__`` and ``__getitem__``.
    split_name:
        Friendly name that appears in the log line (e.g. ``"train"``).
    log_to_mlflow:
        If *True*, the numbers are also recorded as MLflow parameters with
        keys like ``train_num_graphs`` and ``val_avg_nodes``.
    """
    ds = loader.dataset
    num_graphs = len(ds)

    # Gather node/edge counts -------------------------------------------------
    n_nodes: list[int] = []
    n_edges: list[int] = []
    for i in range(num_graphs):
        g = ds[i]
        # Handle different ways to get number of nodes
        if hasattr(g, 'num_nodes'):
            num_nodes = int(g.num_nodes)
        elif hasattr(g, 'x'):
            num_nodes = int(g.x.size(0))
        else:
            num_nodes = int(g)  # For some samplers that return node indices

        # Handle different ways to get number of edges
        if hasattr(g, 'edge_index'):
            num_edges = int(g.edge_index.size(1))
        else:
            num_edges = 0  # Default if no edge information available

        n_nodes.append(num_nodes)
        n_edges.append(num_edges)

    def _basic_stats(vals: list[int]) -> tuple[float, int, int]:
        return (statistics.mean(vals), min(vals), max(vals))

    avg_n, min_n, max_n = _basic_stats(n_nodes)
    avg_e, min_e, max_e = _basic_stats(n_edges)
    mode_n = Counter(n_nodes).most_common(1)[0][0]

    msg = (
        f"[{split_name.upper():5}] "
        f"#Graphs={num_graphs:<5d} "
        f"Nodes (avg/min/max)={avg_n:.1f}/{min_n}/{max_n} "
        f"Edges (avg/min/max)={avg_e:.1f}/{min_e}/{max_e} "
        f"Mode Nodes={mode_n}"
    )
    logging.info(msg)

    if log_to_mlflow:  # ── optional experiment tracking ────────────────────
        prefix = f"{split_name}_"
        mlflow.log_params(
            {
                f"{prefix}num_graphs": num_graphs,
                f"{prefix}avg_nodes": round(avg_n, 2),
                f"{prefix}min_nodes": min_n,
                f"{prefix}max_nodes": max_n,
                f"{prefix}avg_edges": round(avg_e, 2),
                f"{prefix}min_edges": min_e,
                f"{prefix}max_edges": max_e,
            }
        )


def infer_num_node_features(loader: DataLoader) -> int:
    """Return the input feature dimension inferred from ``loader``."""
    dataset = loader.dataset
    if hasattr(dataset, "num_node_features"):
        return int(dataset.num_node_features)  # type: ignore[attr-defined]
    parent = getattr(dataset, "dataset", None)
    if hasattr(parent, "num_node_features"):
        return int(parent.num_node_features)  # type: ignore[attr-defined]
    batch = next(iter(loader))
    return int(batch.x.size(-1))


def infer_num_classes(loader: DataLoader) -> int:
    """Return the number of target classes inferred from ``loader``."""
    dataset = loader.dataset
    if hasattr(dataset, "num_classes"):
        return int(dataset.num_classes)  # type: ignore[attr-defined]
    parent = getattr(dataset, "dataset", None)
    if hasattr(parent, "num_classes"):
        return int(parent.num_classes)  # type: ignore[attr-defined]
    batch = next(iter(loader))
    labels = batch.y
    return int(
        labels.size(-1) if labels.dim() > 1 else labels.max().item() + 1)
