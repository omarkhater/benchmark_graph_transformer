"""Config fixtures."""
import pytest
from omegaconf import DictConfig, OmegaConf


@pytest.fixture
def cfg_transformer() -> dict:
    """Minimal DictConfig for GraphTransformer builder."""

    encode_cfg = {
        "num_encoder_layers": 2,
        "num_heads": 2,
        "dropout": 0.0,
        "ffn_hidden_dim": None,
        "activation": "relu",
        "use_super_node": False,
        "attn_bias_providers": (),
        "positional_encoders": (),
        "node_feature_encoder": None,
    }
    gnn_confg = {
        "gnn_position": "pre",
        "gnn_conv_type": None
    }
    return {
        "type": "GraphTransformer",
        "encoder_cfg": encode_cfg,
        "gnn_cfg": gnn_confg,
        "hidden_dim": 8,
        "with_spatial_bias": False,
        "with_edge_bias": False,
        "with_hop_bias": False,
        "with_degree_enc": False,
        "with_eig_enc": False,
        "with_svd_enc": False,
        "gnn_conv_type": None,
        "max_degree": 0,
        "num_spatial": 0,
        "num_edges": 0,
        "num_hops": 0,
        "num_eigenc": 0,
        "num_svdenc": 0,
    }


@pytest.fixture
def cfg_graph() -> DictConfig:
    """DictConfig for an OGB graph‐level dataset."""
    return OmegaConf.create({"data": {"dataset": "ogbg-molhiv"}})


@pytest.fixture
def cfg_node() -> DictConfig:
    """DictConfig for an OGB node‐level dataset."""
    return OmegaConf.create({"data": {"dataset": "ogbn-arxiv"}})


@pytest.fixture
def cfg_generic() -> DictConfig:
    """DictConfig for a generic graph dataset."""
    return OmegaConf.create({"data": {"dataset": "MUTAG"}})


@pytest.fixture
def cfg_unsupported() -> DictConfig:
    """DictConfig for an unsupported dataset name."""
    return OmegaConf.create({"data": {"dataset": "unknown"}})


@pytest.fixture
def cfg_data() -> DictConfig:
    """DictConfig for minimal train_one_epoch configuration."""
    return OmegaConf.create({"data": {}})
