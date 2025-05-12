from typing import Any, Dict

from omegaconf import DictConfig

__all__ = ["flatten_cfg"]


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
