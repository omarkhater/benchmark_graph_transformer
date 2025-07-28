import inspect
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Tuple
from collections.abc import Mapping
import torch.nn as nn
from torch import Tensor
from torch.nn import (
    Dropout,
    LayerNorm,
    Linear,
    ModuleList,
    ReLU,
    Sequential,
)
from torch_geometric.contrib.nn.bias import (
    BaseBiasProvider,
    GraphAttnEdgeBias,
    GraphAttnHopBias,
    GraphAttnSpatialBias,
)
from torch_geometric.contrib.nn.models import GraphTransformer
from torch_geometric.contrib.nn.positional import (
    BasePositionalEncoder,
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
)


class NodeBackbone(ABC, nn.Module):
    """Base class for node-level GNN backbones."""
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, data: Data) -> Tensor:
        """Return node embeddings of shape [N, hidden_dim]."""
        ...


class ConvBackbone(NodeBackbone):
    def __init__(
            self,
            conv_cls: Callable[..., nn.Module],
            in_channels: int,
            hidden_dim: int
            ) -> None:
        """
        Initialize a convolutional backbone.
        Args
            conv_cls: Convolutional layer class (e.g., GCNConv, GATConv).
            in_channels: Number of input features per node.
            hidden_dim: Output dimension of the convolutional layer.
        """
        super().__init__()
        if conv_cls is GINConv:
            mlp = nn.Sequential(
                Linear(in_channels, hidden_dim),
                ReLU(),
                Linear(hidden_dim, hidden_dim),
            )
            self.conv = GINConv(mlp)
        else:
            self.conv = conv_cls(in_channels, hidden_dim)
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim

    def forward(self, data: Data) -> Tensor:
        return self.conv(data.x, data.edge_index)


class TransformerBackbone(nn.Module):
    """
    Backbone that applies GraphTransformer as a node-level feature extractor.

    Args:
        cfg: transformer hyperparameters dict
        in_channels: input feature dimension
        hidden_dim: transformer embedding dimension
    """
    def __init__(
            self,
            cfg: Dict[str, Any],
            in_channels: int,
            hidden_dim: int
            ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        enc_cfg = cfg.get("encoder_cfg", {})
        gnn_cfg = cfg.get("gnn_cfg", {})
        self._validate_encoder_cfg(enc_cfg)
        self._validate_gnn_cfg(gnn_cfg)
        encoder_cfg = self._make_encoder_cfg(enc_cfg)
        gnn_cfg = self._make_gnn_cfg(gnn_cfg)
        self.model = GraphTransformer(
            hidden_dim=hidden_dim,
            out_channels=None,
            encoder_cfg=encoder_cfg,
            gnn_cfg=gnn_cfg,
            cache_masks=cfg.get("cache_masks", False),
            cast_bias=cfg.get("cast_bias", False)
        )

    def forward(self, data: Data) -> Tensor:
        """
        Forward pass through the transformer backbone.
        Args:
            data (Data): Batched graph data with attributes:
                - x: [N, in_channels]
                - edge_index: [2, E]
        Returns:
            Tensor: Node embeddings of shape [N, hidden_dim].
        """
        return self.model(data)

    def _validate_encoder_cfg(self, enc_cfg: Mapping[str, Any]) -> None:
        """Validate the encoder configuration."""
        if not isinstance(enc_cfg, Mapping):
            raise ValueError(
                f"`encoder_cfg` must be a mapping. Passed: {type(enc_cfg)}"
            )

        self._validate_dropout(enc_cfg.get("dropout"))
        self._validate_heads_and_supernode(enc_cfg)
        self._validate_section(
            name="bias",
            cfg=enc_cfg.get("bias", {}),
            schema={
                "spatial": ("num_spatial",),
                "edge": ("num_edges",),
                "hop":  ("num_hops",),
            }
        )
        self._validate_section(
            name="positional",
            cfg=enc_cfg.get("positional", {}),
            schema={
                "degree": ("max_degree",),
                "eig": ("num_eigvec",),
                "svd": ("num_svdenc",),
            }
        )

    def _validate_dropout(self, dropout: Any) -> None:
        """Validate the dropout configuration."""
        if not isinstance(dropout, float) or not (0.0 <= dropout < 1.0):
            raise ValueError(
                f"`dropout` must be a float in [0.0, 1.0). Passed: {dropout}"
            )

    def _validate_heads_and_supernode(self, enc_cfg: Dict[str, Any]) -> None:
        h = enc_cfg.get("num_heads")
        if not isinstance(h, int) or h <= 0:
            raise ValueError(f"`num_heads` must be positive int, got {h!r}")
        su = enc_cfg.get("use_super_node")
        if not isinstance(su, bool):
            raise TypeError(f"`use_super_node` must be bool, got {type(su)}")

    def _validate_section(
        self,
        name: str,
        cfg: Any,
        schema: Dict[str, Tuple[str, ...]]
    ) -> None:
        """
        Generic section checker for `bias` or `positional`.
        """
        self._assert_mapping(name, cfg)

        for key, params in cfg.items():
            self._assert_in_schema(name, key, schema)
            self._assert_mapping(f"{name}.{key}", params)
            self._assert_bool(f"{name}.{key}.enabled", params.get("enabled"))
            if params["enabled"]:
                self._assert_required(f"{name}.{key}", params, schema[key])

    def _assert_mapping(self, where: str, obj: Mapping[str, Any]) -> None:
        if not isinstance(obj, Mapping):
            raise TypeError(f"`{where}` must be a mapping, got {type(obj)}")

    def _assert_in_schema(
        self,
        section: str,
        key: str,
        schema: Dict[str, Any]
    ) -> None:
        if key not in schema:
            allowed = ", ".join(schema)
            raise ValueError(
                f"Unsupported {section} provider {key!r}; allowed: {allowed}")

    def _assert_bool(self, where: str, val: Any) -> None:
        if not isinstance(val, bool):
            raise ValueError(f"`{where}` must be a bool")

    def _assert_required(
        self,
        where: str,
        params: Dict[str, Any],
        required: Tuple[str, ...]
    ) -> None:
        missing = [r for r in required if r not in params]
        if missing:
            rlist = ", ".join(missing)
            raise ValueError(f"`{where}` missing required keys: {rlist}")

    def _validate_gnn_cfg(self, gnn_cfg: Mapping[str, Any]) -> None:
        """Validate the GNN configuration."""
        if not isinstance(gnn_cfg, Mapping):
            raise ValueError(
                f"`gnn_cfg` must be a mapping. Passed: {type(gnn_cfg)}")

        conv = gnn_cfg.get("gnn_conv_type")
        supported_convs = {"gcn", "sage", "gat", "gin"}
        supported_gnn_positions = {"pre", "post", "parallel"}
        if conv and conv.lower() not in supported_convs:
            raise ValueError(
                f"Unsupported GNN conv type: {conv}. "
                f"Supported types: {supported_convs}"
            )
        if conv and gnn_cfg.get("gnn_position") is None:
            raise ValueError(
                "`gnn_position` must be specified when `gnn_conv_type` is set"
            )
        if gnn_cfg.get("gnn_position") not in supported_gnn_positions:
            raise ValueError(
                f"Unsupported `gnn_position`: {gnn_cfg['gnn_position']}. "
                f"Supported positions: {supported_gnn_positions}"
            )

    def _make_encoder_cfg(self, enc_cfg: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create the encoder configuration.

        Args:
            enc_cfg: Encoder hyperparameters dict
        Returns:
            Dict[str, Any]: Encoder configuration with node feature encoder,
            attention bias providers, and positional encoders.
        """
        node_encoder = Sequential(
            Linear(self.in_channels, self.hidden_dim),
            ReLU(),
            LayerNorm(self.hidden_dim),
            Dropout(enc_cfg.get("dropout")),
        )
        providers = self._get_bias_providers(enc_cfg)
        positional = self._make_positional_encoders(enc_cfg)
        clean = {
            k: v for k, v in enc_cfg.items() if k not in ("bias", "positional")
        }
        clean.update(
            node_feature_encoder=node_encoder,
            attn_bias_providers=ModuleList(providers),
            positional_encoders=ModuleList(positional)
        )
        return clean

    def _get_bias_providers(
            self, enc_cfg: Dict[str, Any]) -> List[BaseBiasProvider]:
        """Create a list of bias providers based on the configuration.

        Args:
            enc_cfg: Encoder hyperparameters dict
        Returns:
            List[nn.Module]: List of bias providers.
        """
        providers = []
        bias_cfg = enc_cfg.get("bias", {})
        needed_keys = ["num_heads", "use_super_node"]

        any_enabled = any(
            params.get("enabled", False) for params in bias_cfg.values())

        if any_enabled:
            missing = [k for k in needed_keys if k not in enc_cfg]
            if missing:
                raise ValueError(
                    f"Missing {missing} when bias providers enabled "
                    f"({list(bias_cfg.keys())})"
                )

        bias_map = {
            "spatial": (GraphAttnSpatialBias, "num_spatial"),
            "edge": (GraphAttnEdgeBias, "num_edges"),
            "hop": (GraphAttnHopBias, "num_hops"),
        }

        for name, (cls, attr) in bias_map.items():
            params = bias_cfg.get(name, {})
            if not params.get("enabled", False):
                continue
            if attr not in params:
                raise ValueError(
                    f"Missing '{attr}' for bias provider '{name}'"
                )
            providers.append(cls(
                num_heads=enc_cfg["num_heads"],
                **{attr: params[attr]},
                use_super_node=enc_cfg["use_super_node"],
            ))
        return providers

    def _make_positional_encoders(
            self, enc_cfg: Dict[str, Any]) -> List[BasePositionalEncoder]:
        """
        Create positional encoders based on the configuration.

        Args:
            enc_cfg: Encoder hyperparameters dict
        Returns:
            List[nn.Module]: List of positional encoders.
        """
        positional: List[BasePositionalEncoder] = []
        pos_cfg = enc_cfg.get("positional", {})

        positional_map = {
            "degree": (DegreeEncoder, "max_degree"),
            "eig": (EigEncoder, "num_eigenc"),
            "svd": (SVDEncoder, "num_svdenc"),
        }

        for name, (cls, param_key) in positional_map.items():
            params = pos_cfg.get(name, {})
            if not params.get("enabled", False):
                continue
            if param_key not in params:
                raise ValueError(
                    f"Missing '{param_key}' for positional encoder '{name}'"
                )

            if cls is DegreeEncoder:
                max_deg = params[param_key]
                positional.append(cls(max_deg, max_deg, self.hidden_dim))
            else:
                positional.append(cls(params[param_key], self.hidden_dim))

        return positional

    def _make_gnn_cfg(self, gnn_cfg: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create the GNN configuration, adding a gnn block if specified.
        Args:
            gnn_cfg: GNN hyperparameters dict
        Returns:
            Dict[str, Any]: GNN configuration with optional GNN block.
        """
        conv_type = gnn_cfg.get("gnn_conv_type")
        if not conv_type:
            return gnn_cfg
        conv_map = {
            "gcn": GCNConv,
            "sage": SAGEConv,
            "gat": GATConv,
            "gin": GINConv
        }
        block = None
        cls = conv_map.get(conv_type.lower())
        if cls is None:
            raise ValueError(f"Unsupported GNN conv type: {conv_type}")
        layer = cls(self.hidden_dim, self.hidden_dim)
        block = _GNNHook(layer)

        return {**gnn_cfg, "gnn_block": block}


class _GNNHook(nn.Module):
    """
    Callable that wraps a conv layer,
    automatically forwarding extra Data attrs.
    """
    def __init__(self, layer: nn.Module):
        super().__init__()
        self.layer = layer
        sig = inspect.signature(layer.forward)
        # skip 'self' and 'x', 'edge_index'
        self.extra_keys = list(sig.parameters.keys())[3:]

    def __call__(self, data: Data, x: Tensor) -> Tensor:
        kwargs = {
            key: getattr(data, key)
            for key in self.extra_keys
            if hasattr(data, key)
        }
        return self.layer(x, data.edge_index, **kwargs)
