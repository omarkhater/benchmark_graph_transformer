"""Type definitions for evaluation framework."""

from enum import Enum, auto


class TaskType(Enum):
    """Task types supported by the evaluation framework.

    Attributes
    ----------
    GRAPH_CLASSIFICATION : auto
        Graph-level classification tasks
    NODE_CLASSIFICATION : auto
        Node-level classification tasks
    GRAPH_REGRESSION : auto
        Graph-level regression tasks (single value per graph)
    NODE_REGRESSION : auto
        Node-level regression tasks (one value per node)
    """
    GRAPH_CLASSIFICATION = auto()
    NODE_CLASSIFICATION = auto()
    GRAPH_REGRESSION = auto()
    NODE_REGRESSION = auto()
