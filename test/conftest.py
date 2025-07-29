"""Test configuration and shared fixtures.

This module re-exports all fixtures from their respective modules to provide
a single import point for test fixtures.
"""
import random

import numpy as np
import pytest
import torch

from .fixtures.config_fixtures import (
    cfg_data,
    cfg_generic,
    cfg_graph,
    cfg_node,
    cfg_transformer,
    cfg_unsupported,
)
from .fixtures.data_fixtures import (
    binary_graph_dataset,
    binary_node_data,
    cora_style_batch,
    cora_style_loader,
    data_batch,
    data_manager,
    generic_batch,
    generic_loader,
    graph_batch,
    graph_classification_suite,
    graph_loader,
    graph_reg_multi_target,
    graph_reg_single_target,
    graph_regression_suite,
    make_graph_dataset,
    make_node_data,
    masked_node_batch,
    masked_node_loader,
    multiclass_graph_dataset,
    multiclass_node_data,
    node_loader,
    node_reg_multi_target,
    node_reg_single_target,
    node_regression_batch,
    node_regression_loader,
    regression_loader,
    regression_none_x_batch,
    regression_none_x_loader,
    simple_graph,
)
from .fixtures.dataset_fixtures import (
    cleanup_all,
    cora_node_dataloaders,
    cora_node_degree_regression_dataloaders,
    dataset_info,
    generic_cfg_and_cls,
    mutag_graph_dataloaders,
    ogb_graph_dataset,
    ogb_node_dataset,
    zinc_graph_regression_dataloaders,
)
from .fixtures.model_fixtures import (
    dummy_model,
    ensure_model_has_parameter,
    graph_classifier_model,
    node_classifier_model,
    regressor_model,
)
from .fixtures.training_fixtures import (
    base_training_config,
    device,
    disable_mlflow,
    optimizer,
    patch_training_dependencies,
)

__all__ = [
    # Model fixtures
    'dummy_model',
    'ensure_model_has_parameter',
    'node_classifier_model',
    'graph_classifier_model',
    'regressor_model',

    # Data fixtures
    'graph_loader',
    'node_loader',
    'generic_loader',
    'regression_loader',
    'regression_none_x_loader',
    'make_node_data',
    'make_graph_dataset',
    'binary_node_data',
    'multiclass_node_data',
    'simple_graph',
    'graph_batch',
    'node_reg_single_target',
    'node_reg_multi_target',
    'binary_graph_dataset',
    'multiclass_graph_dataset',
    'graph_reg_single_target',
    'graph_reg_multi_target',
    'data_manager',
    'graph_regression_suite',
    'graph_classification_suite',
    'masked_node_loader',
    'cora_style_loader',
    'data_batch',
    'generic_batch',
    'masked_node_batch',
    'cora_style_batch',
    'regression_none_x_batch',
    'node_regression_loader',
    'node_regression_batch',

    # Dataset fixtures
    'cleanup_all',
    'dataset_info',
    'generic_cfg_and_cls',
    'ogb_graph_dataset',
    'ogb_node_dataset',
    'mutag_graph_dataloaders',
    'cora_node_dataloaders',
    'zinc_graph_regression_dataloaders',
    'cora_node_degree_regression_dataloaders',

    # Config fixtures
    'cfg_transformer',
    'cfg_graph',
    'cfg_node',
    'cfg_generic',
    'cfg_unsupported',
    'cfg_data',

    # Training fixtures
    'device',
    'optimizer',
    'patch_training_dependencies',
    'disable_mlflow',
    'base_training_config'
]


@pytest.fixture(autouse=True)
def deterministic_seed():
    """
    Seeding before every single test run so that
    torch.randn, torch.randint, torch.randperm, Python random, and numpy
    all produce the exact same “random” draws each time.
    """
    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    return seed
