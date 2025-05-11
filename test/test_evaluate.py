# tests/test_evaluate.py

import pytest
import torch

from graph_transformer_benchmark import evaluate as eval_mod


def test_evaluate_graph_level(
    dummy_model, graph_loader, cfg_graph
) -> None:
    """Verify graph-level ROC-AUC via private and public APIs."""
    direct, _ = eval_mod._evaluate_graph_level(
        dummy_model, graph_loader, torch.device("cpu"), "ogbg-molhiv"
    )
    assert pytest.approx(0.75) == direct

    dispatched, _ = eval_mod.evaluate(
        dummy_model, graph_loader, torch.device("cpu"), cfg_graph
    )
    assert dispatched == direct


def test_evaluate_node_level(
    dummy_model, node_loader, cfg_node
) -> None:
    """Verify node-level accuracy via private and public APIs."""
    direct, _ = eval_mod._evaluate_node_level(
        dummy_model, node_loader, torch.device("cpu"), "ogbn-arxiv"
    )
    assert pytest.approx(0.5) == direct

    dispatched, _ = eval_mod.evaluate(
        dummy_model, node_loader, torch.device("cpu"), cfg_node
    )
    assert dispatched == direct


def test_evaluate_generic_accuracy(
    dummy_model, generic_loader, cfg_generic
) -> None:
    """Verify generic accuracy via private and public APIs."""
    direct, _ = eval_mod._evaluate_generic(
        dummy_model, generic_loader, torch.device("cpu")
    )
    assert pytest.approx(1.0) == direct

    dispatched, _ = eval_mod.evaluate(
        dummy_model, generic_loader, torch.device("cpu"), cfg_generic
    )
    assert dispatched == direct


def test_evaluate_unsupported_fallback(
    dummy_model, graph_loader, cfg_unsupported
) -> None:
    """Unsupported dataset should fallback to generic evaluation."""
    metric, _ = eval_mod.evaluate(
        dummy_model, graph_loader, torch.device("cpu"), cfg_unsupported
    )
    assert pytest.approx(1.0) == metric
