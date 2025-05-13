"""
Generic and OGB-specific classification metric implementations.

The module supports **binary**, **multi-label**, and **multi-class**
settings and can be reused by both custom datasets and OGB leader-board
evaluations.
"""

from __future__ import annotations

import logging
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
# Utility helpers
# --------------------------------------------------------------------------- #


def _safe_call(func, *args, **kwargs) -> float | None:
    """Run function with given arguments and handle exceptions gracefully.

    Parameters
    ----------
    func : callable
        Function to execute
    *args : tuple
        Positional arguments to pass to func
    **kwargs : dict
        Keyword arguments to pass to func

    Returns
    -------
    float or None
        Function result cast to float, or None if execution failed
    """
    try:
        return float(func(*args, **kwargs))  # type: ignore[arg-type]
    except Exception:  # pylint: disable=broad-except
        logging.debug(
            "Metric %s failed with args=%s kwargs=%s",
            func.__name__,
            args,
            kwargs,
            exc_info=True,
        )
        return None


def _discrete_preds(
    y_pred: Array,
    *,
    is_multiclass: bool,
    threshold: float,
) -> Array:
    """Convert continuous predictions to discrete class labels.

    Parameters
    ----------
    y_pred : ndarray
        Predicted probabilities or logits
    is_multiclass : bool
        If True, use argmax for predictions. If False, use thresholding
    threshold : float
        Decision threshold for binary/multilabel predictions

    Returns
    -------
    ndarray
        Discrete class predictions
    """
    if is_multiclass and y_pred.ndim > 1:
        return y_pred.argmax(axis=-1)
    # binary or multi-label → threshold
    return (y_pred >= threshold).astype(int)


def _prob_metrics(
    y_true: Array,
    y_pred: Array,
    *,
    is_multiclass: bool,
) -> MetricDict:
    """Compute probability-based metrics like AUROC and AUPRC.

    Parameters
    ----------
    y_true : ndarray
        Ground truth labels
    y_pred : ndarray
        Predicted probabilities or logits
    is_multiclass : bool
        If True, compute multiclass metrics. If False, binary metrics

    Returns
    -------
    dict
        Dictionary containing AUROC and AUPRC scores
    """
    out: MetricDict = {}
    is_binary = (not is_multiclass) or np.unique(y_true).size == 2

    if is_binary:
        # Handle both single-column and two-column binary predictions
        if y_pred.ndim > 1 and y_pred.shape[1] > 1:
            y_score = y_pred[:, 1]  # Two-column case: use positive class score
        else:
            y_score = y_pred.ravel()  # Single-column case: use raw predictions
        out["auroc"] = (
            _safe_call(roc_auc_score, y_true.ravel(), y_score) or 0.0
        )
        out["auprc"] = (
            _safe_call(average_precision_score, y_true.ravel(), y_score) or 0.0
        )
        return out

    # one-vs-rest AUROC for multi-class
    if y_pred.ndim > 1:
        auc = _safe_call(
            roc_auc_score,
            y_true,
            y_pred,
            multi_class="ovr",
            average="macro",
        )
        if auc is not None:
            out["auroc_macro_ovr"] = auc
    return out


def _topk_metrics(
    y_true: Array,
    y_pred: Array,
    ks: Iterable[int],
) -> MetricDict:
    """Compute top-k accuracy scores.

    Parameters
    ----------
    y_true : ndarray
        Ground truth labels
    y_pred : ndarray
        Predicted probabilities or logits
    ks : iterable of int
        Values of k for which to compute accuracy

    Returns
    -------
    dict
        Dictionary containing top-k accuracy scores
    """
    if y_pred.ndim <= 1:
        return {}

    n_cls = y_pred.shape[1]
    out: MetricDict = {}
    for k in ks:
        if k >= n_cls:
            continue
        acc = _safe_call(top_k_accuracy_score, y_true, y_pred, k=k)
        if acc is not None:
            out[f"top_{k}_accuracy"] = acc
    return out


def _core_metrics(y_true: Array, preds: Array) -> MetricDict:
    """Compute core classification metrics.

    Parameters
    ----------
    y_true : ndarray
        Ground truth labels
    preds : ndarray
        Predicted class labels

    Returns
    -------
    dict
        Dictionary containing accuracy, F1 and other core metrics
    """
    return {
        "accuracy": _safe_call(accuracy_score, y_true, preds) or 0.0,
        "balanced_accuracy": _safe_call(
            balanced_accuracy_score, y_true, preds
        )
        or 0.0,
        "macro_precision": _safe_call(
            precision_score,
            y_true,
            preds,
            average="macro",
            zero_division=0,
        )
        or 0.0,
        "macro_recall": _safe_call(
            recall_score,
            y_true,
            preds,
            average="macro",
            zero_division=0,
        )
        or 0.0,
        "macro_f1": _safe_call(
            f1_score,
            y_true,
            preds,
            average="macro",
            zero_division=0,
        )
        or 0.0,
        "weighted_f1": _safe_call(
            f1_score,
            y_true,
            preds,
            average="weighted",
            zero_division=0,
        )
        or 0.0,
    }


# --------------------------------------------------------------------------- #
# Public API
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

    Parameters
    ----------
    y_true : ndarray
        Ground truth labels
    y_pred : ndarray
        Model predictions (probabilities or logits)
    is_multiclass : bool, optional
        If True, treat as multiclass problem, by default True
    threshold : float, optional
        Decision threshold for binary predictions, by default 0.5
    topk : tuple or iterable of int, optional
        K values for top-k accuracy computation, by default (1,3,5)

    Returns
    -------
    dict
        Dictionary containing various classification metrics

    Examples
    --------
    >>> y_true = np.array([0, 1, 2])
    >>> y_pred = np.array([[0.8, 0.1, 0.1],
    ...                    [0.1, 0.7, 0.2],
    ...                    [0.1, 0.2, 0.7]])
    >>> metrics = compute_generic_classification(y_true, y_pred)
    >>> metrics['accuracy']
    1.0
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    preds = _discrete_preds(
        y_pred, is_multiclass=is_multiclass, threshold=threshold
    )

    metrics = _core_metrics(y_true, preds)

    metrics.update(_prob_metrics(y_true, y_pred, is_multiclass=is_multiclass))

    # top-k makes sense only for multi-class logits
    if is_multiclass:
        metrics.update(_topk_metrics(y_true, y_pred, ks=topk))

    return metrics


# --------------------------------------------------------------------------- #
# OGB wrappers ­– unchanged public behaviour
# --------------------------------------------------------------------------- #
def compute_ogb_graph_metrics(
    y_true: Array,
    y_pred: Array,
    dataset_name: str,
) -> MetricDict:
    """Compute OGB graph classification metrics.

    Parameters
    ----------
    y_true : ndarray
        Ground truth labels
    y_pred : ndarray
        Model predictions
    dataset_name : str
        Name of OGB dataset (e.g. 'ogbg-molhiv')

    Returns
    -------
    dict
        Combined OGB and generic classification metrics
    """
    evaluator = GraphEvaluator(name=dataset_name)
    ogb_scores = _modify_ogb_metrics(
        evaluator.eval({"y_true": y_true, "y_pred": y_pred})
    )
    generic = compute_generic_classification(
        y_true, y_pred, is_multiclass=True
    )
    return {**generic, **ogb_scores}


def compute_ogb_node_metrics(
    y_true: Array,
    y_pred: Array,
    dataset_name: str,
) -> MetricDict:
    """Compute OGB node classification metrics.

    Parameters
    ----------
    y_true : ndarray
        Ground truth labels
    y_pred : ndarray
        Model predictions
    dataset_name : str
        Name of OGB dataset (e.g. 'ogbn-arxiv')

    Returns
    -------
    dict
        Combined OGB and generic classification metrics
    """
    evaluator = NodeEvaluator(name=dataset_name)
    preds = y_pred.argmax(axis=-1) if y_pred.ndim > 1 else y_pred
    ogb_scores = _modify_ogb_metrics(
        evaluator.eval({"y_true": y_true, "y_pred": preds})
    )
    generic = compute_generic_classification(
        y_true, y_pred, is_multiclass=True
    )
    return {**generic, **ogb_scores}


# --------------------------------------------------------------------------- #
# Misc
# --------------------------------------------------------------------------- #
def _modify_ogb_metrics(ogb_scores: MetricDict) -> MetricDict:
    """Standardize OGB metric names and ensure default values.

    Parameters
    ----------
    ogb_scores : dict
        Original metrics from OGB evaluator

    Returns
    -------
    dict
        Modified metrics with standardized names and defaults
    """
    if "acc" in ogb_scores:
        ogb_scores["accuracy"] = ogb_scores.pop("acc")

    ogb_scores.setdefault("rocauc", 0.0)
    ogb_scores.setdefault("macro_f1", 0.0)
    ogb_scores.setdefault("accuracy", 0.0)
    return ogb_scores
