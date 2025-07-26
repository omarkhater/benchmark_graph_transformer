import pytest

from graph_transformer_benchmark.graph_models.transformer import (
    build_graph_transformer,
)

# ==== Bias providers ====


@pytest.mark.parametrize("flag,expected_cls", [
    ("with_spatial_bias", "GraphAttnSpatialBias"),
    ("with_edge_bias",    "GraphAttnEdgeBias"),
    ("with_hop_bias",     "GraphAttnHopBias"),
])
def test_adds_exactly_one_bias_provider(cfg_transformer, flag, expected_cls):
    cfg = cfg_transformer.copy()
    cfg.update(
        {
            flag: True,
        }
    )
    model = build_graph_transformer(cfg, num_features=4, out_channels=2)

    providers = [p.__class__.__name__ for p in model.attn_bias_providers]
    assert providers == [expected_cls]

# ==== Positional encoders ====


@pytest.mark.parametrize("flag,expected_cls,param_name,param_val", [
    ("with_degree_enc", "DegreeEncoder", "max_degree", 5),
    ("with_eig_enc",    "EigEncoder",    "num_eigenc",  3),
    ("with_svd_enc",    "SVDEncoder",    "num_svdenc",  4),
])
def test_positional_encoders(
        cfg_transformer, flag, expected_cls, param_name, param_val):
    cfg = cfg_transformer.copy()
    cfg.update(
        {
            flag: True,
            param_name: param_val,
        }
    )
    model = build_graph_transformer(cfg, num_features=4, out_channels=3)

    names = [enc.__class__.__name__ for enc in model.positional_encoders]
    assert expected_cls in names

# ==== GNN hook in all positions ====


@pytest.mark.parametrize("position", ["pre", "post", "parallel"])
def test_gnn_hook_positions(cfg_transformer, simple_graph, position):
    cfg = cfg_transformer.copy()
    cfg.get("gnn_cfg", {}).update(
        {
            "gnn_conv_type": "gcn",
            "gnn_position": position,
        }
    )
    out_channels = 5

    model = build_graph_transformer(
        cfg,
        num_features=simple_graph.x.size(1),
        out_channels=out_channels
    )
    assert callable(model.gnn_block)
    assert model.gnn_position == position


# ==== Zero‐layer transformer ====


def test_zero_layers_builds_single_layer(cfg_transformer, simple_graph):
    cfg = cfg_transformer.copy()
    encoder_cfg = cfg.get("encoder_cfg", {})
    encoder_cfg.update({"num_encoder_layers": 0})
    feat_dim = simple_graph.x.size(1)
    model = build_graph_transformer(cfg, num_features=feat_dim, out_channels=3)
    # ensure .encoder is a single layer when num_layers=0
    import torch_geometric.contrib.nn.layers.transformer as T
    assert isinstance(model.encoder, T.GraphTransformerEncoderLayer)


# ==== Invalid configs raise ====


def test_invalid_gnn_position_raises(cfg_transformer):
    cfg = cfg_transformer.copy()
    gnn_cfg = cfg.get("gnn_cfg", {})
    gnn_cfg.update(
        {
            "gnn_conv_type": "gat",  # valid
            "gnn_position": "not_a_position",  # invalid
        }
    )

    with pytest.raises(ValueError):
        build_graph_transformer(cfg, num_features=5, out_channels=2)


def test_invalid_activation_raises(cfg_transformer):
    cfg = cfg_transformer.copy()
    enocder_cfg = cfg.get("encoder_cfg", {})
    enocder_cfg.update(
        {
            "activation": "no_such_act",  # invalid
        }
    )
    with pytest.raises(ValueError):
        build_graph_transformer(cfg, num_features=5, out_channels=2)
