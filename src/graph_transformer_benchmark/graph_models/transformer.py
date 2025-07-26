"""
GraphTransformer builders and node-adapted subclass.
"""
from typing import Callable

import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor
from torch.nn import ModuleList
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
from torch_geometric.nn import global_mean_pool


def build_graph_transformer(
    cfg: dict,
    num_features: int,
    out_channels: int,
) -> GraphTransformer:
    """
    Build a graph-level GraphTransformer with configurable biases and encoders.

    Args:
        cfg (dict): Model hyperparameters and flags.
        num_features (int): Dimensionality of input node features.
        out (int): Number of graph classes.

    Returns:
        GraphTransformer: Instantiated transformer model.
    """
    hidden_dim = cfg.get("hidden_dim", num_features)
    encoder_cfg = cfg.get("encoder_cfg", {})
    gnn_cfg = cfg.get("gnn_cfg", {})
    encoder = nn.Sequential(
        nn.Linear(num_features, hidden_dim),
        nn.ReLU(),
        nn.LayerNorm(hidden_dim),
        nn.Dropout(encoder_cfg.get("dropout", 0.0)),
    )
    biases = []
    if cfg.get("with_spatial_bias", None):
        biases.append(
            GraphAttnSpatialBias(
                num_heads=encoder_cfg.get("num_heads", 1),
                num_spatial=cfg.get("num_spatial", 0),
                use_super_node=encoder_cfg.get("use_super_node", False),
            )
        )
    if cfg.get("with_edge_bias", None):
        biases.append(
            GraphAttnEdgeBias(
                num_heads=encoder_cfg.get("num_heads", 1),
                num_edges=cfg.get("num_edges", 0),
                use_super_node=encoder_cfg.get("use_super_node", False),
            )
        )
    if cfg.get("with_hop_bias", None):
        biases.append(
            GraphAttnHopBias(
                num_heads=encoder_cfg.get("num_heads", 1),
                num_hops=cfg.get("num_hops", 0),
                use_super_node=encoder_cfg.get("use_super_node", False),
            )
        )

    encoders = []
    if cfg.get("with_degree_enc", None):
        encoders.append(
            DegreeEncoder(
                cfg.get("max_degree", 0),
                cfg.get("max_degree", 0),
                hidden_dim
                )
            )
    if cfg.get("with_eig_enc", None):
        encoders.append(
            EigEncoder(
                cfg.get("num_eigenc", 0),
                hidden_dim
            )
        )
    if cfg.get("with_svd_enc", None):
        encoders.append(
            SVDEncoder(
                cfg.get("num_svdenc", 0),
                hidden_dim
                )
        )

    gnn_block = None
    if gnn_cfg.get("gnn_conv_type", None) is not None:
        from torch_geometric.nn import GATConv, GCNConv, SAGEConv
        conv_map = {"gcn": GCNConv, "sage": SAGEConv, "gat": GATConv}
        gnn_conv_type = gnn_cfg.get("gnn_conv_type", "gcn").lower()
        layer = conv_map[gnn_conv_type](hidden_dim, hidden_dim)

        def _hook(data: Data, x: Tensor) -> Tensor:
            return layer.to(x.device)(x, data.edge_index)

        gnn_block = _hook

    encoder_cfg.update({
        "attn_bias_providers": ModuleList(biases),
        "positional_encoders": ModuleList(encoders),
        "node_feature_encoder": encoder,
    })
    gnn_cfg.update({
        "gnn_block": gnn_block,
    })
    return GraphTransformer(
        hidden_dim=hidden_dim,
        out_channels=out_channels,
        encoder_cfg=encoder_cfg,
        gnn_cfg=gnn_cfg,
        cache_masks=cfg.get("cache_masks", False),
        cast_bias=cfg.get("cast_bias", False),
    )


class GraphLevelGraphTransformer(GraphTransformer):
    """
    GraphTransformer for graph-level classification.

    Inherits from GraphTransformer and uses global pooling.
    """

    def __init__(
            self,
            *,
            hidden_dim: int,
            encoder_cfg: DictConfig,
            gnn_cfg: DictConfig,
            out_channels: int,
            pool_function: Callable = global_mean_pool,
    ) -> None:
        """
        Initialize GraphLevelGraphTransformer.

        It inherits from GraphTransformer and adds a linear head
        for graph-level classification.
        It sets the out_channels to None to avoid building the models with
        projection layer before the global pooling step.

        Args:
            hidden_dim (int): Dimensionality of hidden representations.
            encoder_cfg (DictConfig): Configuration for the encoder.
            gnn_cfg (DictConfig): Configuration for the GNN block.
            out_channels (int): Number of output classes.
            pool_function (Callable): Function to pool node features to
            graph level.
        """
        super().__init__(
            hidden_dim=hidden_dim,
            encoder_cfg=encoder_cfg,
            gnn_cfg=gnn_cfg,
            out_channels=None,
        )
        self.graph_head = nn.Linear(
            hidden_dim, out_channels
        )
        self.pool_function = pool_function

    def forward(self, data: Data) -> Tensor:
        """
        Forward pass producing graph-level logits.

        Args:
            data (Data): Batched graph data with attributes:
                - x: [N, in_channels]
                - edge_index: [2, E]
                - batch: [N] graph indices

        Returns:
            Tensor: [num_graphs, out_channels] graph logits.
        """
        x = super().forward(data)  # [N, hidden_dim]
        g = self.pool_function(x, data.batch)  # [num_graphs, hidden_dim]
        return self.graph_head(g)  # [num_graphs, out_channels]
