import pytest
import torch

from src.graph_transformer_benchmark.graph_models.factory import build_model

MODES = ["minimal", "bias_only", "positional_only", "gnn_only", "all_features"]

# ── Helpers ──────────────────────────────────────────────────────────────────


@pytest.fixture(params=["graph_binary", "graph_multiclass"])
def graph_cls_loader(graph_classification_suite, request):
    """Parametrized graph‐classification loader + expected_classes."""
    loader = graph_classification_suite[request.param]
    return loader


@pytest.fixture(
        params=["single_target", "multi_target", "edge_attr", "no_features"])
def graph_reg_loader(graph_regression_suite, request):
    """Parametrized graph‐regression loader."""
    loader = graph_regression_suite[request.param]
    return loader

# ── Test Graph Transformer Modes ────────────────────────────────────────────


@pytest.mark.parametrize("cfg_transformer", MODES, indirect=True)
class TestGraphTransformerModes:
    """
    Test GraphTransformer model with various configurations.
    Each mode corresponds to a different set of enabled features.
    - `minimal`:    no bias, no positional encoders, no GNN
    - `bias_only`:  only attention bias providers enabled
    - `positional_only`:   only positional encoders enabled
    - `gnn_only`:   only GNN hook enabled (uses GCNConv)
    - `all_features`: all of the above
    """

    def test_node_classification(self, cfg_transformer, node_loader):
        """
        Ensure node classification models return correct shapes.
        """
        data = next(iter(node_loader))
        N, C = data.x.size(0), int(data.y.unique().numel())

        cfg = cfg_transformer.copy()
        cfg.update(type="graphtransformer",
                   task="node",
                   objective="classification")
        model = build_model(cfg,
                            num_features=data.x.size(1),
                            out_channels=C)
        out = model(data)

        assert out.shape == (N, C)
        assert out.dtype == torch.float32
        # sanity: not all rows identical
        assert not torch.allclose(out, out[0].expand_as(out))

    def test_node_regression(self, cfg_transformer, regression_loader):
        """
        Ensure node regression models return correct shapes.
        """
        data = next(iter(regression_loader))
        N = data.x.size(0)

        cfg = cfg_transformer.copy()
        cfg.update(type="graphtransformer",
                   task="node",
                   objective="regression")
        model = build_model(cfg,
                            num_features=data.x.size(1),
                            out_channels=1)
        out = model(data)

        assert out.shape == (N, 1)
        assert out.dtype == torch.float32

    def test_graph_classification(self, cfg_transformer, graph_cls_loader):
        """
        Ensure graph classification models return correct shapes.
        """
        data = next(iter(graph_cls_loader))
        B = data.y.size(0)
        C = graph_cls_loader.expected_classes

        cfg = cfg_transformer.copy()
        cfg.update(type="graphtransformer",
                   task="graph",
                   objective="classification")
        model = build_model(
            cfg,
            num_features=(0 if data.x is None else data.x.size(1)),
            out_channels=C,
        )
        out = model(data)

        assert out.shape == (B, C)
        assert out.dtype == torch.float32

    def test_graph_regression(self, cfg_transformer, graph_reg_loader):
        """
        Ensure graph regression models return correct shapes.
        """
        data = next(iter(graph_reg_loader))
        B = data.y.size(0)

        cfg = cfg_transformer.copy()
        cfg.update(type="graphtransformer",
                   task="graph",
                   objective="regression")
        model = build_model(
            cfg,
            num_features=(0 if data.x is None else data.x.size(1)),
            out_channels=1,
        )
        out = model(data)

        assert out.shape == (B, 1)
        assert out.dtype == torch.float32

# ── Negative-path tests ──────────────────────────────────────────────────────


class TestGraphTransformerInvalidConfigs:
    """
    Test invalid configurations for GraphTransformer models.
    These tests ensure that the model factory raises appropriate errors
    when given incorrect configurations.

    """

    def test_invalid_num_heads(self, cfg_transformer):
        # every mode should reject num_heads ≤ 0
        cfg = cfg_transformer.copy()
        cfg["encoder_cfg"]["num_heads"] = 0
        cfg.update(
            type="graphtransformer", task="node", objective="classification"
        )
        with pytest.raises(ValueError, match="num_heads"):
            build_model(cfg, num_features=4, out_channels=2)

    def test_invalid_dropout_type(self, cfg_transformer):
        # dropout must be float
        cfg = cfg_transformer.copy()
        cfg["encoder_cfg"]["dropout"] = "0.5"
        cfg.update(
            type="graphtransformer", task="node", objective="regression"
        )
        with pytest.raises(ValueError, match="dropout"):
            build_model(cfg, num_features=4, out_channels=1)

    def test_mismatched_bias_provider(self, cfg_transformer):
        # inject something that's not a BaseBiasProvider
        cfg = cfg_transformer.copy()
        # force the builder to pick up a bad provider list
        bias = cfg["encoder_cfg"]["bias"]
        bias["edge"]["enabled"] = True
        bias["edge"].pop("num_edges", None)
        cfg.update(
            type="graphtransformer", task="graph", objective="classification"
        )
        with pytest.raises(ValueError, match=r"bias\.edge.*num_edges"):
            build_model(cfg, num_features=4, out_channels=2)

    def test_bad_positional_encoder(self, cfg_transformer):

        cfg = cfg_transformer.copy()
        pos = cfg["encoder_cfg"]["positional"]
        pos["degree"]["enabled"] = True
        pos["degree"].pop("max_degree", None)
        cfg.update(
            type="graphtransformer", task="graph", objective="regression"
        )
        with pytest.raises(
                ValueError, match=r"positional\.degree.*max_degree"):
            build_model(cfg, num_features=4, out_channels=1)
