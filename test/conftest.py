"""Test configuration and shared fixtures.

This module re-exports all fixtures from their respective modules to provide
a single import point for test fixtures.
"""

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
    generic_loader,
    graph_batch,
    graph_loader,
    graph_reg_multi_target,
    graph_reg_single_target,
    make_graph_dataset,
    make_node_data,
    multiclass_graph_dataset,
    multiclass_node_data,
    node_loader,
    node_reg_multi_target,
    node_reg_single_target,
    regression_loader,
    regression_none_x_loader,
    simple_batch,
    simple_graph,
)
from .fixtures.dataset_fixtures import (
    cleanup_all,
    dataset_info,
    generic_cfg_and_cls,
    ogb_graph_dataset,
    ogb_node_dataset,
)
from .fixtures.model_fixtures import (
    dummy_model,
    ensure_model_has_parameter,
)
from .fixtures.training_fixtures import (
    device,
    optimizer,
    patch_training_dependencies,
)

__all__ = [
    # Model fixtures
    'dummy_model',
    'ensure_model_has_parameter',

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
    'simple_batch',
    'graph_batch',
    'node_reg_single_target',
    'node_reg_multi_target',
    'binary_graph_dataset',
    'multiclass_graph_dataset',
    'graph_reg_single_target',
    'graph_reg_multi_target',

    # Dataset fixtures
    'cleanup_all',
    'dataset_info',
    'generic_cfg_and_cls',
    'ogb_graph_dataset',
    'ogb_node_dataset',

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
]
