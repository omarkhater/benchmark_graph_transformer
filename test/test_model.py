import pytest
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor
from torch_geometric.contrib.nn.models import GraphTransformer
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GCNConv, GINConv

from graph_transformer_benchmark.model import (
    BaseGNN,
    _build_graph_transformer,
    build_gin_classifier,
    build_gnn_classifier,
    build_model,
)


def test_basegnn_forward_without_bn_or_residual(
    simple_graph: Data,
) -> None:
    """BaseGNN with no batchnorm or residual returns correct logits."""
    # Set up a conv that preserves input dim
    conv = GCNConv(2, 2)
    model = BaseGNN(
        conv=conv,
        in_channels=2,
        hidden_dim=2,
        num_classes=3,
        use_batch_norm=False,
        use_residual=False,
    )
    simple_graph.y = torch.tensor([1, 0])
    simple_graph.batch = torch.zeros(2, dtype=torch.long)
    out = model(simple_graph)
    assert isinstance(out, Tensor)
    assert out.shape == (1, 3)


@pytest.mark.parametrize(
    "use_bn,use_res",
    [(True, False), (False, True), (True, True)],
)
def test_basegnn_forward_with_bn_and_or_residual(
    simple_graph: Data,
    use_bn: bool,
    use_res: bool,
) -> None:
    """BaseGNN handles batchnorm and residual projection correctly."""
    # Make input dim != hidden dim to trigger projection
    mlp = nn.Sequential(nn.Linear(2, 4), nn.ReLU())
    conv = GINConv(mlp)
    model = BaseGNN(
        conv=conv,
        in_channels=2,
        hidden_dim=4,
        num_classes=2,
        use_batch_norm=use_bn,
        use_residual=use_res,
    )
    simple_graph.y = torch.tensor([0, 1])
    simple_graph.batch = torch.zeros(2, dtype=torch.long)
    out = model(simple_graph)
    assert out.shape == (1, 2)


def test_build_gnn_classifier_returns_basegnn() -> None:
    """build_gnn_classifier constructs a BaseGNN with correct conv type."""
    cls = build_gnn_classifier(
        conv_cls=GCNConv,
        in_channels=5,
        hidden_dim=7,
        num_classes=4,
        use_batch_norm=True,
        use_residual=True,
    )
    assert isinstance(cls, BaseGNN)
    assert isinstance(cls.conv, GCNConv)
    assert cls.lin.out_features == 4


def test_build_gin_classifier_returns_basegnn() -> None:
    """build_gin_classifier constructs a BaseGNN with GINConv inside."""
    cls = build_gin_classifier(
        in_channels=3,
        hidden_dim=6,
        num_classes=5,
        use_batch_norm=False,
        use_residual=False,
    )
    assert isinstance(cls, BaseGNN)
    assert isinstance(cls.conv, GINConv)
    assert cls.lin.out_features == 5


def test_graph_transformer_builder_and_forward(
    cfg_transformer: DictConfig,
    graph_batch: Batch,
) -> None:
    """_build_graph_transformer returns a working GraphTransformer."""
    model = _build_graph_transformer(
        cfg_model=cfg_transformer,
        num_features=2,
        num_classes=3,
    )
    assert isinstance(model, GraphTransformer)
    graph_batch.y = torch.tensor([0, 1])
    out = model(graph_batch)
    assert out.shape == (2, 3)


@pytest.mark.parametrize(
    "model_type,expected_class",
    [
        ("GCN", BaseGNN),
        ("sage", BaseGNN),
        ("gat", BaseGNN),
        ("gin", BaseGNN),
        ("graphtransformer", GraphTransformer),
    ],
)
def test_build_model_dispatches_correctly(
    cfg_transformer: DictConfig,
    model_type: str,
    expected_class: type,
) -> None:
    """build_model returns the right class for each model type."""
    cfg = cfg_transformer.copy()
    cfg.type = model_type
    model = build_model(cfg, num_features=4, num_classes=2)
    assert isinstance(model, expected_class)
