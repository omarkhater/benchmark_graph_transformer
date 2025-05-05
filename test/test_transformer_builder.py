import pytest
import torch
from torch_geometric.data import Batch

from graph_transformer_benchmark.graph_models.transformer import (
    build_graph_transformer,
)

# ==== 1) Bias providers ====


@pytest.mark.parametrize("flag,expected_cls", [
    ("with_spatial_bias", "GraphAttnSpatialBias"),
    ("with_edge_bias",    "GraphAttnEdgeBias"),
    ("with_hop_bias",     "GraphAttnHopBias"),
])
def test_adds_exactly_one_bias_provider(cfg_transformer, flag, expected_cls):
    cfg = cfg_transformer.copy()
    setattr(cfg, flag, True)
    model = build_graph_transformer(cfg, num_features=4, num_classes=2)

    providers = [p.__class__.__name__ for p in model.attn_bias_providers]
    assert providers == [expected_cls]

# ==== 2) Positional encoders ====


@pytest.mark.parametrize("flag,expected_cls,param_name,param_val", [
    ("with_degree_enc", "DegreeEncoder", "max_degree", 5),
    ("with_eig_enc",    "EigEncoder",    "num_eigenc",  3),
    ("with_svd_enc",    "SVDEncoder",    "num_svdenc",  4),
])
def test_positional_encoders(
        cfg_transformer, flag, expected_cls, param_name, param_val):
    cfg = cfg_transformer.copy()
    setattr(cfg, flag, True)
    setattr(cfg, param_name, param_val)
    model = build_graph_transformer(cfg, num_features=4, num_classes=3)

    names = [enc.__class__.__name__ for enc in model.positional_encoders]
    assert expected_cls in names

# ==== 3) GNN hook in all positions ====


@pytest.mark.parametrize("position", ["pre", "post", "parallel"])
def test_gnn_hook_positions(cfg_transformer, simple_graph, position):
    cfg = cfg_transformer.copy()
    cfg.gnn_conv_type = "gcn"
    cfg.gnn_position = position

    model = build_graph_transformer(
        cfg, num_features=simple_graph.x.size(1), num_classes=5)
    batch = Batch.from_data_list([simple_graph, simple_graph])

    # smoke test the hook exists and is callable
    assert callable(model.gnn_block)

    # forward must return [batch_size, num_classes]
    out = model(batch)
    assert out.shape == (2, 5)
    assert torch.isfinite(out).all()

# ==== 4) Super‐node readout ====


def test_use_super_node_changes_readout(cfg_transformer, simple_graph):
    cfg = cfg_transformer.copy()
    cfg.use_super_node = True
    feat_dim = simple_graph.x.size(1)
    # for super_node, the model inserts a CLS token per graph
    model = build_graph_transformer(cfg, num_features=feat_dim, num_classes=2)
    batch = Batch.from_data_list([simple_graph]*3)

    out = model(batch)
    # still returns (#graphs × num_classes)
    assert out.shape == (3, 2)

# ==== 5) Zero‐layer transformer ====


def test_zero_layers_builds_single_layer(cfg_transformer, simple_graph):
    cfg = cfg_transformer.copy()
    cfg.num_layers = 0
    feat_dim = simple_graph.x.size(1)
    model = build_graph_transformer(cfg, num_features=feat_dim, num_classes=3)
    # ensure .encoder is a single layer when num_layers=0
    import torch_geometric.contrib.nn.layers.transformer as T
    assert isinstance(model.encoder, T.GraphTransformerEncoderLayer)

    out = model(Batch.from_data_list([simple_graph]))
    assert out.shape == (1, 3)

# ==== 6) Invalid configs raise ====


def test_invalid_gnn_position_raises(cfg_transformer):
    cfg = cfg_transformer.copy()
    cfg.gnn_conv_type = "gat"
    cfg.gnn_position = "not_a_position"
    with pytest.raises(ValueError):
        build_graph_transformer(cfg, num_features=5, num_classes=2)


def test_invalid_activation_raises(cfg_transformer):
    cfg = cfg_transformer.copy()
    cfg.activation = "no_such_act"
    with pytest.raises(ValueError):
        build_graph_transformer(cfg, num_features=5, num_classes=2)
