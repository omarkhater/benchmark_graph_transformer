"""Classification metric computation for different dataset types."""

from typing import Dict

import numpy as np
from ogb.graphproppred import Evaluator as GraphEvaluator
from ogb.nodeproppred import Evaluator as NodeEvaluator
from sklearn.metrics import f1_score


def compute_generic_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    is_multiclass: bool = True,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute standard classification metrics."""
    if is_multiclass and y_pred.ndim > 1:
        preds = y_pred.argmax(axis=-1)
    else:
        preds = (y_pred >= threshold).astype(int)

    return {
        "accuracy": float((preds == y_true).mean()),
        "macro_f1": float(
            f1_score(y_true, preds, average="macro", zero_division=0)
        )
    }


def compute_ogb_graph_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dataset_name: str,
) -> Dict[str, float]:
    """Compute OGB graph classification metrics."""
    evaluator = GraphEvaluator(name=dataset_name)
    result = evaluator.eval({"y_true": y_true, "y_pred": y_pred})

    return {
        "accuracy": result.get("acc", 0.0),
        "rocauc": result["rocauc"],
        "macro_f1": result.get("macro_f1", 0.0)
    }


def compute_ogb_node_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dataset_name: str,
) -> Dict[str, float]:
    """Compute OGB node classification metrics."""
    evaluator = NodeEvaluator(name=dataset_name)
    preds = y_pred.argmax(axis=-1) if y_pred.ndim > 1 else y_pred
    result = evaluator.eval({"y_true": y_true, "y_pred": preds})

    return {
        "accuracy": result["acc"],
        "macro_f1": result.get("macro_f1", 0.0)
    }
