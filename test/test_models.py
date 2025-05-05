"""
PyTest suite for graph_models package.

This module exercises graph- and node-level models, factory dispatch,
and GraphTransformer variants, leveraging shared fixtures
for comprehensive coverage.
"""
import pytest
from torch import nn
from torch_geometric.nn import GATConv, GCNConv, SAGEConv

from src.graph_transformer_benchmark.graph_models import (
    BaseGNN,
    NodeGNN,
    NodeGraphTransformer,
    build_gin_classifier,
    build_gnn_classifier,
    build_graph_transformer,
    build_model,
    build_node_classifier,
)


def test_build_gnn_classifier_returns_base_gnn() -> None:
    model = build_gnn_classifier(
        GCNConv, in_channels=3, hidden_dim=4, num_classes=2,
        use_batch_norm=False, use_residual=False
    )
    assert isinstance(model, BaseGNN)


def test_build_node_classifier_returns_node_gnn() -> None:
    model = build_node_classifier(
        SAGEConv, in_channels=3, hidden_dim=5, num_classes=3,
        use_batch_norm=True, use_residual=True
    )
    assert isinstance(model, NodeGNN)


def test_build_gin_classifier_returns_base_gnn() -> None:
    model = build_gin_classifier(
        in_channels=3, hidden_dim=6, num_classes=4,
        use_batch_norm=True, use_residual=False
    )
    assert isinstance(model, BaseGNN)


def test_graphtransformer_via_factory(
    cfg_transformer, simple_batch
) -> None:
    """
    Ensure factory builds GraphTransformer in graph mode correctly.
    """
    cfg = cfg_transformer.copy()
    cfg.task = "graph"
    feat_dim = simple_batch.x.size(1)
    model = build_model(cfg, num_features=feat_dim, num_classes=2)
    out = model(simple_batch)
    assert out.shape == (2, 2)


def test_node_graphtransformer_direct(
    cfg_transformer, simple_graph
) -> None:
    """
    Directly construct and run NodeGraphTransformer.
    """
    cfg = cfg_transformer.copy()
    feat_dim = simple_graph.x.size(1)
    base = build_graph_transformer(cfg, num_features=feat_dim, num_classes=3)
    ffn_dim = cfg.ffn_hidden_dim or cfg.hidden_dim
    ngt = NodeGraphTransformer(
        hidden_dim=cfg.hidden_dim,
        num_node_classes=3,
        use_super_node=cfg.use_super_node,
        node_feature_encoder=base.node_feature_encoder,
        num_encoder_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        dropout=cfg.dropout,
        ffn_hidden_dim=ffn_dim,
        activation=cfg.activation,
        attn_bias_providers=base.attn_bias_providers or [],
        positional_encoders=base.positional_encoders or [],
        gnn_block=base.gnn_block,
        gnn_position=cfg.gnn_position,
    )
    out = ngt(simple_graph)
    assert out.shape == (simple_graph.x.size(0), 3)


def test_graphtransformer_node_mode_factory(
    cfg_transformer, simple_graph
) -> None:
    """
    Ensure factory builds NodeGraphTransformer via build_model.
    """
    cfg = cfg_transformer.copy()
    cfg.task = "node"
    cfg.type = "graphtransformer"
    feat_dim = simple_graph.x.size(1)
    model = build_model(cfg, num_features=feat_dim, num_classes=3)
    out = model(simple_graph)
    assert out.shape == (simple_graph.x.size(0), 3)


def test_batchnorm_and_residual_flags() -> None:
    """
    Verify BatchNorm and residual projection presence.
    """
    bn = build_gnn_classifier(
        GATConv, 2, 2, 2, use_batch_norm=True, use_residual=False
    )
    assert isinstance(bn.bn, nn.BatchNorm1d)
    res = build_gnn_classifier(
        GCNConv, 2, 2, 2, use_batch_norm=False, use_residual=True
    )
    assert getattr(res, "res_proj", None) is not None


def test_factory_graph_and_node_tasks(
    cfg_graph, cfg_node, graph_batch, node_loader
) -> None:
    """
    Parametrized test for both graph and node tasks across conv types.
    """
    from omegaconf import OmegaConf
    default_cfg = OmegaConf.create({
        "hidden_dim": 4,
        "use_batch_norm": False,
        "use_residual": False,
    })
    for task, cfg in [("graph", cfg_graph), ("node", cfg_node)]:
        for conv in ["gcn", "sage", "gat", "gin"]:
            cfg2 = OmegaConf.merge(default_cfg, cfg)
            cfg2.type = conv
            cfg2.task = task
            data = graph_batch if task == "graph" else next(iter(node_loader))
            feat_dim = data.x.size(1)
            model = build_model(cfg2, num_features=feat_dim, num_classes=2)
            if task == "graph":
                out = model(graph_batch)
                assert out.shape == (graph_batch.y.size(0), 2)
            else:
                data = next(iter(node_loader))
                out = model(data)
                assert out.shape == (
                    data.x.size(0), len(data.y.view(-1).unique())
                )


def test_generic_loader_support(
    cfg_generic, generic_loader
) -> None:
    """
    Validate build_model output shape using generic_loader fixture.
    """
    cfg = cfg_generic.copy()
    cfg.type = "gcn"
    cfg.task = "graph"
    num_cls = len(next(iter(generic_loader)).y.view(-1).unique())
    model = build_model(
        cfg,
        num_features=next(iter(generic_loader)).x.size(1),
        num_classes=num_cls,
    )
    batch = next(iter(generic_loader))
    out = model(batch)
    assert out.size(0) == batch.y.view(-1).size(0)


def test_invalid_config_raises(cfg_unsupported) -> None:
    """
    Ensure unsupported type/task combos raise ValueError.
    """
    with pytest.raises(ValueError):
        cfg = cfg_unsupported.copy()
        cfg.type = "unknown"
        cfg.task = "graph"
        build_model(cfg, num_features=1, num_classes=1)
