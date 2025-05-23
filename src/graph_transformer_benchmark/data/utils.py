from __future__ import annotations

import torch
from omegaconf import DictConfig
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


def enrich_batch(batch: Batch, cfg: DictConfig) -> Batch:
    """
    Enrich the batch with additional features based on the
    configuration settings.
    ----------
    batch : Batch
        The input batch to be enriched.
    cfg : DictConfig
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
    Parameters
    """
    num_nodes = batch.x.size(0)

    if getattr(cfg, "with_degree_enc", False):
        row, col = batch.edge_index
        batch.out_degree = degree(row, num_nodes, dtype=torch.long)
        batch.in_degree = degree(col, num_nodes, dtype=torch.long)

    if getattr(cfg, "with_eig_enc", False):
        dim = int(getattr(cfg, "num_eigenc", 0))
        batch.eig_pos_emb = batch.x.new_empty((num_nodes, dim)).normal_()

    if getattr(cfg, "with_svd_enc", False):
        r = int(getattr(cfg, "num_svdenc", 0))
        batch.svd_pos_emb = batch.x.new_empty((num_nodes, 2 * r)).normal_()

    bias_shape = (num_nodes, num_nodes)
    if getattr(cfg, "with_spatial_bias", False):
        batch.spatial_pos = torch.zeros(bias_shape, dtype=torch.long)
    if getattr(cfg, "with_edge_bias", False):
        batch.edge_dist = torch.zeros(bias_shape, dtype=torch.long)
    if getattr(cfg, "with_hop_bias", False):
        batch.hop_dist = torch.zeros(bias_shape, dtype=torch.long)

    return batch
