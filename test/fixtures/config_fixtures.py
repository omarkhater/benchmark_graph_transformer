"""Config fixtures."""
import pytest
from omegaconf import DictConfig, OmegaConf


@pytest.fixture(
    params=[
        pytest.param("minimal", id="minimal"),
        pytest.param("bias_only", id="bias_only"),
        pytest.param("pos_only", id="positional_only"),
        pytest.param("gnn_only", id="gnn_only"),
        pytest.param("all_features", id="all_features"),
    ]
)
def cfg_transformer(request) -> dict:
    """
    Parametrized configs for GraphTransformer builder:
      - minimal:    no bias, no positional‐encoders, no gnn
      - bias_only:  only attention‐bias providers enabled
      - pos_only:   only positional encoders enabled
      - gnn_only:   only gnn hook enabled (uses GCNConv)
      - all_features: all of the above
    """
    # base encoder config
    encoder_cfg = {
        "num_encoder_layers": 2,
        "num_heads": 4,
        "ffn_hidden_dim": None,
        "activation": "relu",
        "use_super_node": False,
        "node_feature_encoder": None,
        "dropout": 0.1,
        # start all disabled
        "bias": {
            "spatial": {"enabled": False, "num_spatial": 4},
            "edge": {"enabled": False, "num_edges": 3},
            "hop": {"enabled": False, "num_hops": 2},
        },
        "positional": {
            "degree": {"enabled": False, "max_degree": 3},
        },
    }

    # base gnn config
    gnn_cfg = {
        "gnn_position": "pre",
        "gnn_conv_type": None,
    }

    mode = request.param
    if mode in ("bias_only", "all_features"):
        for p in encoder_cfg["bias"].values():
            p["enabled"] = True

    if mode in ("positional_only", "all_features"):
        for p in encoder_cfg["positional"].values():
            p["enabled"] = True

    if mode in ("gnn_only", "all_features"):
        # flip on a real GNN block so we hit that codepath
        gnn_cfg["gnn_conv_type"] = "gcn"
        # leave gnn_position="pre" by default

    return {
        "type": "GraphTransformer",
        "hidden_dim":  16,
        "cache_masks": False,
        "cast_bias":   False,
        "encoder_cfg": encoder_cfg,
        "gnn_cfg": gnn_cfg,
    }


@pytest.fixture()
def cfg_graph() -> dict[str, any]:
    """
    Base config for any graph‐level test.
    Tests will override type, task, and objective.
    """
    return {
        "hidden_dim": 8,
        "use_batch_norm": False,
        "use_residual": False,
    }


@pytest.fixture
def cfg_node() -> dict[str, any]:
    """
    Base config for any node‐level test.
    Tests will override type, task, and objective.
    """
    return {
        "hidden_dim": 8,
        "use_batch_norm": False,
        "use_residual": False,
    }


@pytest.fixture
def cfg_generic() -> dict[str, any]:
    """
    Minimal config for 'generic_loader' tests.
    """
    return {
        "hidden_dim": 8,
        "use_batch_norm": False,
        "use_residual": False,
    }


@pytest.fixture
def cfg_unsupported() -> dict[str, any]:
    """
    Minimal config for testing invalid‐type errors.
    """
    return {
        "hidden_dim": 8,
        "use_batch_norm": False,
        "use_residual": False,
    }


@pytest.fixture
def cfg_data() -> DictConfig:
    """DictConfig for minimal train_one_epoch configuration."""
    return OmegaConf.create({"data": {}})
