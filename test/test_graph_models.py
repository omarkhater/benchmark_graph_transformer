import pytest
import torch

from src.graph_transformer_benchmark.graph_models import build_model


@pytest.mark.parametrize(
        "model_type",
        [
            "gcn",
            "gat",
            "gin",
            "sage",
        ]
    )
def test_node_classification_shapes(
    model_type,
    cfg_node,
    node_loader,
):
    """
    Ensure node classification models return correct shapes.
    """
    data = next(iter(node_loader))
    num_nodes = data.x.size(0)
    num_classes = int(data.y.unique().numel())
    cfg = cfg_node.copy()
    cfg.update(
        type=model_type,
        task="node",
        objective="classification"
    )
    model = build_model(
        cfg, num_features=data.x.size(1), out_channels=num_classes)
    out = model(data)
    assert out.shape == (num_nodes, num_classes)
    assert out.dtype == torch.float32
    assert not torch.allclose(out, out[0].expand_as(out))


@pytest.mark.parametrize(
        "model_type", [
            "gcn",
            "gat",
            "gin",
            "sage"
        ]
    )
def test_graph_classification_shapes(
    model_type,
    cfg_graph,
    graph_batch,
):
    data = graph_batch
    num_graphs = int(data.y.size(0))
    num_classes = int(data.y.unique().numel())

    cfg = cfg_graph.copy()
    cfg.update(
        type=model_type,
        task="graph",
        objective="classification"
    )

    model = build_model(
        cfg, num_features=data.x.size(1), out_channels=num_classes)
    out = model(data)

    assert out.shape == (num_graphs, num_classes)
    assert out.dtype == torch.float32


@pytest.mark.parametrize(
        "model_type",
        [
            "gcn", "gat", "gin", "sage"
        ]
    )
def test_graph_regression_shapes(
    model_type,
    cfg_graph,
    graph_batch,
):
    data = graph_batch
    num_graphs = data.y.size(0)

    cfg = cfg_graph.copy()
    cfg.update(
        type=model_type,
        task="graph",
        objective="regression"
    )

    model = build_model(cfg, num_features=data.x.size(1), out_channels=1)
    out = model(data)

    assert out.shape == (num_graphs, 1)
    assert out.dtype == torch.float32
