from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch_geometric.data import Batch, Dataset
from torch_geometric.utils import degree


def is_node_level_graph(ds: Dataset) -> bool:
    """Check if the dataset is a single-graph node-level dataset.

    Parameters
    ----------
    ds : Dataset
        The dataset to be checked.
    Returns
    -------
    bool
        True if the dataset is a single-graph node-level dataset,
        False otherwise.
    Notes
    -----
    The function checks if the dataset contains only one graph by
    verifying that the length of the dataset is 1.  It also checks
    if the graph has train, validation, and test masks by checking
    if the graph has the attributes `train_mask`, `val_mask`, and
    `test_mask`.  These masks are used to indicate which nodes in
    the graph are used for training, validation, and testing,
    respectively.  If the dataset contains only one graph and
    the graph has these masks, the function returns True.  If the
    dataset contains more than one graph or if the graph does not
    have these masks, the function returns False.
    """
    return len(ds) == 1 and all(
        hasattr(ds[0], k) for k in ("train_mask", "val_mask", "test_mask")
    )


def split_from_masks(data) -> dict[str, torch.Tensor]:
    """Turn boolean node masks into index tensors.

    Parameters
    ----------
    data : Data
        The data object containing the node masks.
    Returns
    -------
    dict[str, torch.Tensor]
        A dictionary containing the indices of the nodes in the
        training, validation, and test sets.  The keys are "train",
        "valid", and "test", and the values are tensors containing
        the indices of the nodes in each set.

    """
    return {
        "train": data.train_mask.nonzero(as_tuple=False).view(-1),
        "valid": data.val_mask.nonzero(as_tuple=False).view(-1),
        "test": data.test_mask.nonzero(as_tuple=False).view(-1),
    }


def ensure_node_features(batch: Batch, feature_dim: int = 1) -> Batch:
    """
    Ensure that the batch has node features, creating synthetic ones if needed.

    Parameters
    ----------
    batch : Batch
        The input batch that may have None node features.
    feature_dim : int, default=1
        The dimension of synthetic node features to create if x is None.

    Returns
    -------
    Batch
        The batch with guaranteed node features.

    Notes
    -----
    This function handles datasets like QM7b where node features (batch.x)
    are None. It creates a tensor of ones as synthetic node features to allow
    GraphTransformer models to function properly. The synthetic features have
    the same device and dtype as the edge_index.
    """
    if batch.x is None:
        num_nodes = batch.num_nodes
        device = batch.edge_index.device
        dtype = batch.edge_index.dtype
        # Create synthetic node features as ones
        batch.x = torch.ones(
            (num_nodes, feature_dim),
            device=device,
            dtype=dtype
        )
    return batch


def enrich_batch(batch: Batch, cfg: Mapping[str, Any]) -> Batch:
    """
    Enrich the batch with additional features based on the
    configuration settings.
    ----------
    batch : Batch
        The input batch to be enriched.
    cfg : Mapping[str, Any]
        The configuration settings that determine which
        features to add to the batch.
    Returns
    -------
    Batch
        The enriched batch with additional features.
    Notes
    -----
    This function adds various positional encodings and
    distance encodings to the batch based on the configuration
    settings.  The added features include:
    - out_degree: The out-degree of each node in the graph.
    - in_degree: The in-degree of each node in the graph.
    - eig_pos_emb: A random normal tensor for eigenvalue
      positional encoding.
    - svd_pos_emb: A random normal tensor for SVD positional
      encoding.
    - spatial_pos: A zero tensor for spatial positional
        encoding.
    - edge_dist: A zero tensor for edge distance encoding.
    - hop_dist: A zero tensor for hop distance encoding.

    The function also sets the shape of the tensors based on the
    number of nodes in the batch.  The tensors are created with
    the same device and data type as the input batch.

    If batch.x is None (e.g., QM7b dataset), synthetic node features
    are created to ensure compatibility with GraphTransformer models.
    """
    # Ensure node features exist for model compatibility
    encoder_cfg = cfg.get("encoder_cfg", {})
    batch = ensure_node_features(batch, feature_dim=1)

    batch = _enrich_batch_for_positional_encoders(
        batch,
        encoder_cfg.get("positional", {})
    )

    batch = _enrich_batch_for_attention_biases(
        batch,
        encoder_cfg.get("bias", {})
    )

    return batch


def _enrich_batch_for_positional_encoders(
    batch: Batch,
    positional_cfg: Mapping[str, Any]
) -> Batch:
    """
    Add various positional encodings to the batch based on cfg.
    ----------
    batch : Batch
        The input batch to be enriched.
    positional_cfg : Mapping[str, Any]
        The positional configuration dict.
    Returns
    -------
    Batch
        The batch with added positional attributes:
        - out_degree, in_degree
        - eig_pos_emb
        - svd_pos_emb
    """
    num_nodes = batch.num_nodes
    device = batch.x.device
    dtype = batch.x.dtype

    degree_cfg = positional_cfg.get("degree", {})
    if degree_cfg.get("enabled", False):
        row, col = batch.edge_index
        batch.out_degree = degree(row, num_nodes, dtype=dtype)
        batch.in_degree = degree(col, num_nodes, dtype=dtype)

    eig_cfg = positional_cfg.get("eig", {})
    if eig_cfg.get("enabled", False):
        dim = int(eig_cfg.get("num_eigenc"))
        batch.eig_pos_emb = torch.empty(
            (num_nodes, dim), device=device, dtype=dtype
        )
    svd_cfg = positional_cfg.get("svd", {})
    if svd_cfg.get("enabled", False):
        r = int(svd_cfg.get("num_svdenc", 0))
        batch.svd_pos_emb = torch.empty(
            (num_nodes, 2 * r), device=device, dtype=dtype
        )

    return batch


def _enrich_batch_for_attention_biases(
    batch: Batch,
    bias_cfg: Mapping[str, Any]
) -> Batch:
    """
    Add attention bias distance matrices to the batch based on cfg.
    ----------
    batch : Batch
        The input batch to be enriched.
    bias_cfg : Mapping[str, Any]
        The bias configuration dict.
    Returns
    -------
    Batch
        The batch with added bias attributes:
        - spatial_pos
        - edge_dist
        - hop_dist
    """

    num_nodes = batch.num_nodes
    device = batch.edge_index.device
    dtype = batch.edge_index.dtype

    spatial_cfg = bias_cfg.get("spatial", {})
    if spatial_cfg.get("enabled", False):
        batch.spatial_pos = make_square_matrix(num_nodes, device, dtype)

    edge_cfg = bias_cfg.get("edge", {})
    if edge_cfg.get("enabled", False):
        batch.edge_dist = make_square_matrix(num_nodes, device, dtype)

    hop_cfg = bias_cfg.get("hop", {})
    if hop_cfg.get("enabled", False):
        batch.hop_dist = make_square_matrix(num_nodes, device, dtype)

    return batch


def make_square_matrix(
        size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32
        ) -> torch.Tensor:
    """
    Create a square matrix of zeros with the given size.
    """
    return torch.zeros((size, size), dtype=dtype, device=device)
