#!/usr/bin/env python
"""
Model definitions for GraphTransformer Benchmark.
"""
from typing import Callable

import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor
from torch_geometric.contrib.nn.bias import (
    GraphAttnEdgeBias,
    GraphAttnHopBias,
    GraphAttnSpatialBias,
)
from torch_geometric.contrib.nn.models import GraphTransformer
from torch_geometric.contrib.nn.positional import (
    DegreeEncoder,
    EigEncoder,
    SVDEncoder,
)
from torch_geometric.data import Data
from torch_geometric.nn import (
    GATConv,
    GCNConv,
    GINConv,
    SAGEConv,
    global_mean_pool,
)


class BaseGNN(nn.Module):
    """Base graph‐level GNN classifier with optional BatchNorm & residual."""

    def __init__(
        self,
        conv: nn.Module,
        in_channels: int,
        hidden_dim: int,
        num_classes: int,
        use_batch_norm: bool,
        use_residual: bool,
    ) -> None:
        """
        Args:
            conv: Instantiated GNN convolution module.
            in_channels: Dimensionality of input node features.
            hidden_dim: Dimensionality of convolution output.
            num_classes: Number of graph‐level target classes.
            use_batch_norm: If True, applies BatchNorm1d after conv.
            use_residual: If True, adds a skip connection from input to conv.
        """
        super().__init__()
        self.conv = conv
        if use_batch_norm:
            self.bn = nn.BatchNorm1d(hidden_dim)
        else:
            self.bn = nn.Identity()
        if use_residual:
            if in_channels != hidden_dim:
                self.res_proj = nn.Linear(in_channels, hidden_dim)
            else:
                self.res_proj = nn.Identity()
        else:
            self.res_proj = None

        self.lin = nn.Linear(hidden_dim, num_classes)
        self.use_residual = use_residual

    def forward(self, data: Data) -> Tensor:
        """
        Args:
            data: A torch_geometric.data.Data object with attributes:
                - x: [N, in_channels] node features
                - edge_index: [2, E] graph connectivity
                - batch: [N] batch assignment vector
        Returns:
            Tensor of shape [num_graphs, num_classes], raw logits.
        """
        x0 = data.x
        x = self.conv(x0, data.edge_index)
        x = self.bn(x)

        if self.use_residual and self.res_proj is not None:
            x = x + self.res_proj(x0)

        x = global_mean_pool(x, data.batch)
        return self.lin(x)


def build_gnn_classifier(
    conv_cls: Callable[..., nn.Module],
    in_channels: int,
    hidden_dim: int,
    num_classes: int,
    use_batch_norm: bool,
    use_residual: bool,
) -> BaseGNN:
    """
    Build a single‐layer GNN classifier.

    Args:
        conv_cls: GCNConv, SAGEConv, or GATConv class.
        in_channels: Input feature dimension.
        hidden_dim: Hidden feature dimension.
        num_classes: Number of graph‐level classes.
        use_batch_norm: Whether to apply batch normalization.
        use_residual: Whether to apply residual connection.
    """
    conv = conv_cls(in_channels, hidden_dim)
    return BaseGNN(
        conv,
        in_channels,
        hidden_dim,
        num_classes,
        use_batch_norm,
        use_residual
    )


def build_gin_classifier(
    in_channels: int,
    hidden_dim: int,
    num_classes: int,
    use_batch_norm: bool,
    use_residual: bool,
) -> BaseGNN:
    """
    Build a 2‐layer GIN classifier.

    Args:
        in_channels: Input feature dimension.
        hidden_dim: Hidden feature dimension.
        num_classes: Number of graph‐level classes.
        use_batch_norm: Whether to apply batch normalization in MLP after conv.
        use_residual: Whether to apply residual connection.
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
        use_residual
    )


def _build_graph_transformer(
        cfg_model: DictConfig,
        num_features: int,
        num_classes: int
        ) -> GraphTransformer:
    """Build a GraphTransformer with bias and positional encoders."""
    encoder = nn.Sequential(
        nn.Linear(num_features, cfg_model.hidden_dim),
        nn.ReLU(),
        nn.LayerNorm(cfg_model.hidden_dim),
        nn.Dropout(cfg_model.dropout),
    )
    bias_providers = []
    if cfg_model.with_spatial_bias:
        bias_providers.append(
            GraphAttnSpatialBias(
                num_heads=cfg_model.num_heads,
                num_spatial=cfg_model.num_spatial,
                use_super_node=cfg_model.use_super_node,
            )
        )
    if cfg_model.with_edge_bias:
        bias_providers.append(
            GraphAttnEdgeBias(
                num_heads=cfg_model.num_heads,
                num_edges=cfg_model.num_edges,
                use_super_node=cfg_model.use_super_node,
            )
        )
    if cfg_model.with_hop_bias:
        bias_providers.append(
            GraphAttnHopBias(
                num_heads=cfg_model.num_heads,
                num_hops=cfg_model.num_hops,
                use_super_node=cfg_model.use_super_node,
            )
        )

    pos_encoders = []
    if cfg_model.with_degree_enc:
        pos_encoders.append(
            DegreeEncoder(
                num_in=cfg_model.max_degree,
                num_out=cfg_model.max_degree,
                hidden_dim=cfg_model.hidden_dim,
            )
        )
    if cfg_model.with_eig_enc:
        pos_encoders.append(
            EigEncoder(
                num_eigvec=cfg_model.num_eigenc,
                hidden_dim=cfg_model.hidden_dim,
            )
        )
    if cfg_model.with_svd_enc:
        pos_encoders.append(
            SVDEncoder(
                r=cfg_model.num_svdenc,
                hidden_dim=cfg_model.hidden_dim,
            )
        )
    gnn_block = None
    if cfg_model.gnn_conv_type:
        conv_map = {"gcn": GCNConv, "sage": SAGEConv, "gat": GATConv}
        ConvClass = conv_map[cfg_model.gnn_conv_type]
        conv_layer = ConvClass(cfg_model.hidden_dim, cfg_model.hidden_dim)

        def hook(data, x):  # noqa: E731
            return conv_layer(x, data.edge_index)

        gnn_block = hook

    return GraphTransformer(
        hidden_dim=cfg_model.hidden_dim,
        num_class=num_classes,
        use_super_node=cfg_model.use_super_node,
        node_feature_encoder=encoder,
        num_encoder_layers=cfg_model.num_layers,
        num_heads=cfg_model.num_heads,
        dropout=cfg_model.dropout,
        ffn_hidden_dim=cfg_model.ffn_hidden_dim,
        activation=cfg_model.activation,
        attn_bias_providers=bias_providers,
        positional_encoders=pos_encoders,
        gnn_block=gnn_block,
        gnn_position=cfg_model.gnn_position,
    )


def build_model(
    cfg_model: DictConfig, num_features: int, num_classes: int
) -> nn.Module:
    """
    Instantiate the requested model variant.

    Supports GraphTransformer, GCN, SAGE, GAT, and GIN.
    """
    t = cfg_model.type.lower()
    if t == "graphtransformer":
        return _build_graph_transformer(cfg_model, num_features, num_classes)

    use_bn = bool(getattr(cfg_model, "use_batch_norm", False))
    use_res = bool(getattr(cfg_model, "use_residual", False))

    if t == "gcn":
        return build_gnn_classifier(
            GCNConv, num_features, cfg_model.hidden_dim,
            num_classes, use_bn, use_res
        )
    if t == "sage":
        return build_gnn_classifier(
            SAGEConv, num_features, cfg_model.hidden_dim,
            num_classes, use_bn, use_res
        )
    if t == "gat":
        return build_gnn_classifier(
            GATConv, num_features, cfg_model.hidden_dim,
            num_classes, use_bn, use_res
        )
    if t == "gin":
        return build_gin_classifier(
            num_features, cfg_model.hidden_dim,
            num_classes, use_bn, use_res
        )

    raise ValueError(f"Unsupported model type: {cfg_model.type}")
