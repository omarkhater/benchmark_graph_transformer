from typing import Literal

import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool

from .backbones import NodeBackbone


class GraphModel(nn.Module):
    """Base class for GNNs and GraphTransformers.
    Args:
        backbone (NodeBackbone): Backbone for node-level GNNs.
        task (Literal["node", "graph"]): Task type, either "node" or "graph".
        objective (Literal["classification", "regression"]): Objective type.
        out_channels (int): Number of output channels.
        use_batch_norm (bool): Whether to use batch normalization.
        use_residual (bool): Whether to use residual connections.
    """
    def __init__(
            self,
            backbone: NodeBackbone,
            task: Literal["node", "graph"],
            objective: Literal["classification", "regression"],
            out_channels: int,
            use_batch_norm: bool,
            use_residual: bool
            ) -> None:
        super().__init__()
        self.backbone = backbone
        if use_batch_norm:
            self.bn = nn.BatchNorm1d(backbone.hidden_dim)
        else:
            self.bn = nn.Identity()
        self.res_proj = (nn.Linear(backbone.in_channels, backbone.hidden_dim)
                         if use_residual else None)
        self.pool = (global_mean_pool if task == "graph"
                     else (lambda x, b: x))
        head_dim = out_channels if objective == "classification" else 1
        self.head = nn.Linear(backbone.hidden_dim, head_dim)

    def forward(self, data: Data) -> Tensor:
        """
        Forward pass through the model.
        Args:
            data (Data): Input data containing node features and edge indices.
        Returns:
            Tensor: Output tensor of shape [N, out_channels] for node tasks
            or [B, out_channels] for graph tasks.
        """
        x0 = data.x
        h = self.backbone(data)
        h = self.bn(h)
        if self.res_proj is not None:
            h = h + self.res_proj(x0)
        h = self.pool(h, data.batch)
        return self.head(h)
