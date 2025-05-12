"""Generic and OGB-specific classification metric implementations.

This module provides detailed classification metrics for both standard and
OGB-specific evaluation scenarios. It handles binary, multi-label, and
multi-class settings with configurable thresholds and top-k computations.

The main functions are designed to work with NumPy arrays and return
dictionaries of scalar metrics that can be logged during training.

Functions
---------
compute_generic_classification : Standard classification metrics
compute_ogb_graph_metrics : OGB graph-level metrics
compute_ogb_node_metrics : OGB node-level metrics

Notes
-----
All functions expect NumPy arrays as input. For PyTorch integration,
see the metrics module.
"""

from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
from ogb.graphproppred import Evaluator as GraphEvaluator
from ogb.nodeproppred import Evaluator as NodeEvaluator
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    top_k_accuracy_score,
)

MetricDict = Dict[str, float]
Array = np.ndarray


# --------------------------------------------------------------------------- #
# Small utility helpers
# --------------------------------------------------------------------------- #
def _safe_call(func, *args, **kwargs) -> float | None:
    """Execute *func* and swallow any exception, returning ``None`` instead.

    This is useful for metrics that may fail under certain conditions,
    such as AUROC with constant predictions.
    Parameters
    ----------
    func : callable
        The function to call.
    *args : tuple
        Positional arguments to pass to the function.
    **kwargs : dict
        Keyword arguments to pass to the function.
    Returns
    -------
    float or None
        The result of the function call, or None if an exception occurred.
    Notes
    -----
    This function is a wrapper around the original function to handle
    exceptions gracefully. It is particularly useful for metrics that
    may fail under certain conditions, such as AUROC with constant
    predictions.
    """
    try:
        return float(func(*args, **kwargs))  # type: ignore[arg-type]
    except Exception:  # pylint: disable=broad-except
        return None


def _discrete_preds(
    y_pred: Array,
    *,
    is_multiclass: bool,
    threshold: float,
) -> Array:
    """Convert probability/logit predictions into discrete class labels.

    Parameters
    ----------
    y_pred : Array
        Raw predictions (logits or probabilities).
        Shape (n_samples,) for binary classification or
        (n_samples, n_classes) for multi-class/multi-label.
    is_multiclass : bool
        If True, treats as multi-class with single label per sample.
    threshold : float
        Decision threshold for binary/multi-label predictions.
    Returns
    -------
    Array
        Discrete class labels. Shape (n_samples,) for binary classification
        or (n_samples, n_labels) for multi-label classification.
    Notes
    -----
    This function is used to convert raw predictions (logits or
    probabilities) into discrete class labels based on a specified
    threshold. It handles both binary and multi-class/multi-label
    scenarios. For multi-class, it uses the argmax to select the
    predicted class.
    """
    if is_multiclass and y_pred.ndim > 1:
        return y_pred.argmax(axis=-1)
    return (y_pred >= threshold).astype(int)


def _prob_metrics(
    y_true: Array,
    y_pred: Array,
    *,
    is_multiclass: bool,
) -> MetricDict:
    """Compute AUROC and AUPRC for binary or multi-label classification.
    Parameters
    ----------
    y_true : Array
        Ground truth labels. For multi-class, shape (n_samples,).
        For multi-label, shape (n_samples, n_labels).
    y_pred : Array
        Raw predictions (logits or probabilities).
        Shape (n_samples,) for binary classification or
        (n_samples, n_classes) for multi-class/multi-label.
    is_multiclass : bool
        If True, treats as multi-class with single label per sample.
    Returns
    -------
    MetricDict
        Dictionary containing AUROC and AUPRC metrics.
    Notes
    -----
    This function computes AUROC and AUPRC metrics for binary or
    multi-label classification tasks. It handles both binary and
    multi-class scenarios, and it can compute macro-averaged AUROC
    for multi-class tasks. The function uses the `roc_auc_score`
    and `average_precision_score` functions from scikit-learn to
    calculate the metrics. It also handles exceptions gracefully
    using the `_safe_call` function to avoid crashes in case of
    constant predictions or other issues."""
    out: MetricDict = {}
    is_binary = (not is_multiclass) or np.unique(y_true).size == 2
    if is_binary:
        auroc = _safe_call(roc_auc_score, y_true, y_pred)
        auprc = _safe_call(average_precision_score, y_true, y_pred)
        if auroc is not None:
            out["auroc"] = auroc
        if auprc is not None:
            out["auprc"] = auprc
        return out

    # Multi-class OvR AUROC
    if y_pred.ndim > 1:
        macro_auroc = _safe_call(
            roc_auc_score,
            y_true,
            y_pred,
            multi_class="ovr",
            average="macro",
        )
        if macro_auroc is not None:
            out["auroc_macro_ovr"] = macro_auroc
    return out


def _topk_metrics(
    y_true: Array,
    y_pred: Array,
    ks: Iterable[int],
) -> MetricDict:
    """Compute *top-k* accuracies for the given ``k`` values.

    Parameters
    ----------
    y_true : Array
        Ground truth labels. For multi-class, shape (n_samples,).
        For multi-label, shape (n_samples, n_labels).
    y_pred : Array
        Raw predictions (logits or probabilities).
        Shape (n_samples,) for binary classification or
        (n_samples, n_classes) for multi-class/multi-label.
    ks : iterable of int
        K values for computing top-k accuracy in multi-class case.
    Returns
    -------
    MetricDict
        Dictionary containing top-k accuracy metrics.
    Notes
    -----
    This function computes top-k accuracy metrics for multi-class
    classification tasks. It handles both binary and multi-class
    scenarios, and it can compute top-k accuracy for multiple
    values of k. The function uses the `top_k_accuracy_score`
    function from scikit-learn to calculate the metrics. It also
    handles exceptions gracefully using the `_safe_call` function
    to avoid crashes in case of constant predictions or other issues.
    """
    if y_pred.ndim <= 1:
        return {}

    num_classes = y_pred.shape[1]
    out: MetricDict = {}
    for k in ks:
        if k > num_classes:
            continue
        score = _safe_call(top_k_accuracy_score, y_true, y_pred, k=k)
        if score is not None:
            out[f"top_{k}_accuracy"] = score
    return out


# --------------------------------------------------------------------------- #
# Public, framework-agnostic API
# --------------------------------------------------------------------------- #
def compute_generic_classification(
    y_true: Array,
    y_pred: Array,
    *,
    is_multiclass: bool = True,
    threshold: float = 0.5,
    topk: tuple[int, ...] | Iterable[int] = (1, 3, 5),
) -> MetricDict:
    """Compute comprehensive classification metrics.

    Supports binary, multi-label, and multi-class scenarios with
    configurable decision thresholds and top-k accuracy computation.

    Parameters
    ----------
    y_true : Array
        Ground truth labels. For multi-class, shape (n_samples,).
        For multi-label, shape (n_samples, n_labels).
    y_pred : Array
        Raw predictions (logits or probabilities).
        Shape (n_samples,) for binary classification or
        (n_samples, n_classes) for multi-class/multi-label.
    is_multiclass : bool, default=True
        If True, treats as multi-class with single label per sample.
        If False, treats as binary or multi-label classification.
    threshold : float, default=0.5
        Decision threshold for binary/multi-label predictions.
    topk : tuple or list of int, default=(1, 3, 5)
        K values for computing top-k accuracy in multi-class case.

    Returns
    -------
    MetricDict
        Dictionary containing metrics:
            - accuracy: Overall classification accuracy
            - balanced_accuracy: Balanced accuracy for imbalanced data
            - macro_precision: Macro-averaged precision
            - macro_recall: Macro-averaged recall
            - macro_f1: Macro-averaged F1 score
            - weighted_f1: Sample-weighted F1 score
            - auroc: Area under ROC curve (binary/multilabel only)
            - auprc: Area under PR curve (binary/multilabel only)
            - top_k_accuracy: For k in topk (multi-class only)

    Examples
    --------
    >>> # Binary classification
    >>> y_true = np.array([0, 1, 0, 1])
    >>> y_pred = np.array([0.1, 0.9, 0.2, 0.8])
    >>> metrics = compute_generic_classification(
    ...     y_true, y_pred, is_multiclass=False
    ... )

    >>> # Multi-class classification
    >>> y_true = np.array([0, 1, 2])
    >>> y_pred = np.array([
    ...     [0.8, 0.1, 0.1],
    ...     [0.1, 0.7, 0.2],
    ...     [0.1, 0.2, 0.7]
    ... ])
    >>> metrics = compute_generic_classification(y_true, y_pred)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    preds = _discrete_preds(
        y_pred, is_multiclass=is_multiclass, threshold=threshold
    )

    metrics: MetricDict = {
        "accuracy": _safe_call(accuracy_score, y_true, preds) or 0.0,
        "balanced_accuracy": _safe_call(
            balanced_accuracy_score, y_true, preds
        )
        or 0.0,
        "macro_precision": _safe_call(
            precision_score, y_true, preds, average="macro", zero_division=0
        )
        or 0.0,
        "macro_recall": _safe_call(
            recall_score, y_true, preds, average="macro", zero_division=0
        )
        or 0.0,
        "macro_f1": _safe_call(
            f1_score, y_true, preds, average="macro", zero_division=0
        )
        or 0.0,
        "weighted_f1": _safe_call(
            f1_score, y_true, preds, average="weighted", zero_division=0
        )
        or 0.0,
    }

    metrics.update(_prob_metrics(y_true, y_pred, is_multiclass=is_multiclass))
    metrics.update(_topk_metrics(y_true, y_pred, ks=topk))
    return metrics


# --------------------------------------------------------------------------- #
# Thin OGB wrappers (add official leaderboard scores)
# --------------------------------------------------------------------------- #
def compute_ogb_graph_metrics(
    y_true: Array,
    y_pred: Array,
    dataset_name: str,
) -> MetricDict:
    """Merge OGB graph evaluator output with generic diagnostics.

    Parameters
    ----------
    y_true : Array
        Ground truth labels. For multi-class, shape (n_samples,).
        For multi-label, shape (n_samples, n_labels).
    y_pred : Array
        Raw predictions (logits or probabilities).
        Shape (n_samples,) for binary classification or
        (n_samples, n_classes) for multi-class/multi-label.
    dataset_name : str
        Name of the OGB dataset (e.g., 'ogbg-molhiv').
    Returns
    -------
    MetricDict
        Dictionary containing metrics:
            - accuracy: Overall classification accuracy
            - rocauc: Area under ROC curve (binary/multilabel only)
            - macro_f1: Macro-averaged F1 score
            - acc: Accuracy from OGB evaluator
            - rocauc: ROC AUC from OGB evaluator
            - macro_f1: Macro F1 from OGB evaluator
    Notes
    -----
    This function computes OGB-specific metrics for graph-level
    classification tasks. It merges the official OGB evaluator
    output with standard metrics. The function handles both binary
    and multi-class scenarios, and it can compute macro-averaged
    AUROC for multi-class tasks. The function uses the `GraphEvaluator`
    class from OGB to calculate the metrics. It also handles
    exceptions gracefully using the `_safe_call` function to avoid
    crashes in case of constant predictions or other issues.

    """
    evaluator = GraphEvaluator(name=dataset_name)
    ogb_scores = evaluator.eval({"y_true": y_true, "y_pred": y_pred})

    generic = compute_generic_classification(
        y_true, y_pred, is_multiclass=True
    )
    return {**generic, **ogb_scores}


def compute_ogb_node_metrics(
    y_true: Array,
    y_pred: Array,
    dataset_name: str,
) -> MetricDict:
    """Merge OGB node evaluator output with generic diagnostics.

    Parameters
    ----------
    y_true : Array
        Ground truth labels. For multi-class, shape (n_samples,).
        For multi-label, shape (n_samples, n_labels).
    y_pred : Array
        Raw predictions (logits or probabilities).
        Shape (n_samples,) for binary classification or
        (n_samples, n_classes) for multi-class/multi-label.
    dataset_name : str
        Name of the OGB dataset (e.g., 'ogbn-arxiv').
    Returns
    -------
    MetricDict
        Dictionary containing metrics:
            - accuracy: Overall classification accuracy
            - macro_f1: Macro-averaged F1 score
            - acc: Accuracy from OGB evaluator
            - macro_f1: Macro F1 from OGB evaluator

    Notes
    -----
    This function computes OGB-specific metrics for node-level
    classification tasks. It merges the official OGB evaluator
    output with standard metrics. The function handles both binary
    and multi-class scenarios, and it can compute macro-averaged
    AUROC for multi-class tasks. The function uses the `NodeEvaluator`
    class from OGB to calculate the metrics. It also handles
    exceptions gracefully using the `_safe_call` function to avoid
    crashes in case of constant predictions or other issues.
    """
    evaluator = NodeEvaluator(name=dataset_name)
    preds = y_pred.argmax(axis=-1) if y_pred.ndim > 1 else y_pred
    ogb_scores = evaluator.eval({"y_true": y_true, "y_pred": preds})

    generic = compute_generic_classification(
        y_true, y_pred, is_multiclass=True
    )
    return {**generic, **ogb_scores}
