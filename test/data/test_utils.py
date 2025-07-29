"""Unit tests for utils module."""
import pytest
from torch_geometric.data import Dataset

from graph_transformer_benchmark.data import utils


def test_is_node_level_graph(one_graph_dataset):
    """Test node-level graph detection functionality."""
    assert utils.is_node_level_graph(one_graph_dataset)
    many = [one_graph_dataset[0] for _ in range(3)]
    many_ds = type(
        "Many",
        (Dataset,),
        {"__len__": lambda self: 3, "__getitem__": lambda s, i: many[i]}
    )()
    assert not utils.is_node_level_graph(many_ds)


class TestEnrichBatchFlags:
    # ── shared constants ────────────────────────────────────────────
    _POS_ATTRS = ["in_degree", "out_degree", "eig_pos_emb", "svd_pos_emb"]
    _BIAS_ATTRS = ["spatial_pos", "edge_dist", "hop_dist"]

    _POS_MODE_PARAMS = [
        ("minimal", False),
        ("bias_only", False),
        ("positional_only", True),
        ("gnn_only", False),
        ("all_features", True),
    ]

    _BIAS_MODE_PARAMS = [
        ("minimal", False),
        ("positional_only", False),
        ("gnn_only", False),
        ("bias_only", True),
        ("all_features", True),
    ]

    _REQUESTED_BATCHES = [
        "simple_batch",
        "graph_batch",
        "generic_batch",
        "masked_node_batch",
        "cora_style_batch",
        "regression_none_x_batch"
    ]

    @pytest.mark.parametrize(
        "cfg_transformer,pos_on",
        _POS_MODE_PARAMS,
        indirect=["cfg_transformer"],
    )
    @pytest.mark.parametrize(
        "data_batch",
        _REQUESTED_BATCHES,
        indirect=True,
    )
    def test_positional_flags(self, cfg_transformer, data_batch, pos_on):
        """Check positional-encoder attributes."""
        enriched = utils.enrich_batch(data_batch, cfg_transformer)
        for attr in self._POS_ATTRS:
            assert hasattr(enriched, attr) is pos_on, f"{attr=} mismatch"

    @pytest.mark.parametrize(
        "cfg_transformer,bias_on",
        _BIAS_MODE_PARAMS,
        indirect=["cfg_transformer"],
    )
    @pytest.mark.parametrize(
        "data_batch",
        _REQUESTED_BATCHES,
        indirect=True,
    )
    def test_bias_flags(self, cfg_transformer, data_batch, bias_on):
        """Check attention-bias attributes."""
        enriched = utils.enrich_batch(data_batch, cfg_transformer)
        for attr in self._BIAS_ATTRS:
            assert hasattr(enriched, attr) is bias_on, f"{attr=} mismatch"
