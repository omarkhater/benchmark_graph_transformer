from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from ogb.graphproppred import Evaluator as GraphEvaluator
from ogb.nodeproppred import Evaluator as NodeEvaluator
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader

from .classification_metrics import (
    compute_generic_classification,
)
from .metrics import compute_regression_metrics
from .predictors import collect_predictions
from .task_detection import detect_task_type, is_multiclass_task
from .types import TaskType

MetricDict = Dict[str, float]
Array = np.ndarray


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg: DictConfig,
) -> MetricDict:
    """Run a *single* forward-pass over ``loader`` and compute task-aware
    metrics.

    The function:
    1. **Detects the task type** (graph/node × regression/classification)
       by inspecting the first batch.
    2. **Collects predictions** in *one* inference sweep (no duplicate
       model calls).
    3. **Dispatches** to the appropriate metric routine, handling
       Open Graph Benchmark (OGB) datasets specially so that the output
       includes the official leaderboard numbers *and* the generic
       diagnostics used elsewhere in the codebase.

    Parameters
    ----------
    model
        PyTorch module in evaluation mode.  The caller is responsible for
        putting the model into ``eval()`` and for disabling gradient
        tracking if desired; this function adds an extra guard via the
        ``torch.no_grad`` context inside :func:`collect_predictions`.
    loader
        A :class:`torch_geometric.loader.DataLoader` yielding
        :class:`torch_geometric.data.Data` or :class:`…Batch` objects.
    device
        The device on which inference is executed.
    cfg
        Hydra/OmegaConf configuration.  Only ``cfg.data.dataset`` is
        accessed.

    Returns
    -------
    Dict[str, float]
        A flat mapping from metric name to scalar value.  Key sets differ
        by task:

        * **Regression** – ``{"mse", "rmse", "mae", "r2"}``
        * **Classification (custom)** – accuracy, F-scores, AUROC/AUPRC,
          top-k accuracy, etc.
        * **Classification (OGB)** – everything above **plus** the
          ``Evaluator``’s official scores (e.g. ``acc``, ``rocauc``).

    Notes
    -----
    * Only one forward pass is performed regardless of dataset type;
      OGB helpers are called with *pre-computed* predictions to avoid
      doubled compute and dropout nondeterminism.
    * OGB node tasks expect *discrete* predictions, so ``argmax`` is
      applied when ``y_pred`` is 2-D.
    """
    if not isinstance(loader, DataLoader):
        raise TypeError(f"Unsupported loader type: {type(loader)}")

    # ------------------------------------------------------------------ #
    # 1. Task detection (cheap – only looks at first batch)
    # ------------------------------------------------------------------ #
    task = detect_task_type(loader)
    dataset_name: str = str(cfg.data.dataset)

    # ------------------------------------------------------------------ #
    # 2. Single inference sweep
    # ------------------------------------------------------------------ #
    y_true, y_pred = collect_predictions(model, loader, device)

    # ------------------------------------------------------------------ #
    # 3. Task-specific metric selection
    # ------------------------------------------------------------------ #
    # -------- Regression ------------------------------------------------
    if task in {TaskType.GRAPH_REGRESSION, TaskType.NODE_REGRESSION}:
        return compute_regression_metrics(y_true, y_pred)

    # -------- Classification -------------------------------------------
    is_multiclass = is_multiclass_task(loader)

    # --- OGB datasets ---------------------------------------------------
    if dataset_name.startswith(("ogbg", "ogbn")):
        if task == TaskType.GRAPH_CLASSIFICATION:
            evaluator = GraphEvaluator(name=dataset_name)
            ogb_scores = evaluator.eval(
                {"y_true": y_true, "y_pred": y_pred}
            )
        else:
            evaluator = NodeEvaluator(name=dataset_name)
            preds = (
                y_pred.argmax(axis=-1) if y_pred.ndim > 1 else y_pred
            )
            ogb_scores = evaluator.eval(
                {"y_true": y_true, "y_pred": preds}
            )

        generic = compute_generic_classification(
            y_true, y_pred, is_multiclass=is_multiclass
        )
        # Merge, preferring OGB keys on collision
        return {**generic, **ogb_scores}

    # --- Custom datasets -----------------------------------------------
    return compute_generic_classification(
        y_true, y_pred, is_multiclass=is_multiclass
    )
