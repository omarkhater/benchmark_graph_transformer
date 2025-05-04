"""
Factory functions to build GNN classifiers.
"""
from typing import Callable

import torch.nn as nn
from torch_geometric.nn import GINConv

from .base import BaseGNN
from .node import NodeGNN


def build_gnn_classifier(
    conv_cls: Callable[..., nn.Module],
    in_channels: int,
    hidden_dim: int,
    num_classes: int,
    use_batch_norm: bool,
    use_residual: bool,
) -> BaseGNN:
    """
    Build a graph-level GNN classifier with one convolutional layer.
    """
    conv = conv_cls(in_channels, hidden_dim)
    return BaseGNN(
        conv,
        in_channels,
        hidden_dim,
        num_classes,
        use_batch_norm,
        use_residual,
    )


def build_node_classifier(
    conv_cls: Callable[..., nn.Module],
    in_channels: int,
    hidden_dim: int,
    num_classes: int,
    use_batch_norm: bool,
    use_residual: bool,
) -> NodeGNN:
    """
    Build a node-level GNN classifier with one convolutional layer.
    """
    conv = conv_cls(in_channels, hidden_dim)
    return NodeGNN(
        conv,
        in_channels,
        hidden_dim,
        num_classes,
        use_batch_norm,
        use_residual,
    )


def build_gin_classifier(
    in_channels: int,
    hidden_dim: int,
    num_classes: int,
    use_batch_norm: bool,
    use_residual: bool,
) -> BaseGNN:
    """
    Build a graph-level GIN classifier with two MLP layers.
    """
    mlp = nn.Sequential(
        nn.Linear(in_channels, hidden_dim),
        nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity(),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity(),
        nn.ReLU(),
    )
    gin_conv = GINConv(mlp)
    return BaseGNN(
        gin_conv,
        in_channels,
        hidden_dim,
        num_classes,
        use_batch_norm,
        use_residual,
    )
