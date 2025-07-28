from typing import Any, Dict

from omegaconf import DictConfig

from graph_transformer_benchmark.evaluation import TaskType

__all__ = ["flatten_cfg", "update_training_pipeline_config"]


def flatten_cfg(
        section: DictConfig | Dict[str, Any],
        prefix: str = ""
        ) -> Dict[str, Any]:
    """
    Recursively flatten a DictConfig (or plain dict).

    Each leaf key becomes "<prefix>key[.subkey...]" so that
    all parameter names are globally unique.
    """
    flat = {}
    for k, v in section.items():
        full_key = f"{prefix}{k}"
        if isinstance(v, (DictConfig, dict)):
            flat.update(flatten_cfg(v, prefix=f"{full_key}."))
        else:
            if hasattr(v, "item"):
                v = v.item()
            flat[full_key] = v
    return flat


def update_training_pipeline_config(cfg: dict, task_type: TaskType) -> None:
    """
    Mutate `cfg` so that:
      - cfg["model"]["task"]      ∈ {"node","graph"}
      - cfg["model"]["objective"] ∈ {"classification","regression"}

    Parameters
    ----------
    cfg : dict
        Hydra configuration dictionary. It should contain a "model" key.
    task_type : TaskType
        The task type to update the configuration for.
    Raises
    ------
    ValueError
        If the task type is not recognized.
    """
    model_cfg = cfg.setdefault("model", {})

    if task_type is TaskType.GRAPH_CLASSIFICATION:
        model_cfg.update(task="graph", objective="classification")
    elif task_type is TaskType.GRAPH_REGRESSION:
        model_cfg.update(task="graph", objective="regression")
    elif task_type is TaskType.NODE_CLASSIFICATION:
        model_cfg.update(task="node", objective="classification")
    elif task_type is TaskType.NODE_REGRESSION:
        model_cfg.update(task="node", objective="regression")
    else:
        raise ValueError(f"Unrecognized TaskType: {task_type!r}")
