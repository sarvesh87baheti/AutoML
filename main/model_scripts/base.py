from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.pipeline import Pipeline
from inspect import signature, isfunction
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class ModelSpec:
    name: str
    supported_problem_types: Tuple[str, ...]

# def train_model(
#     X_train: np.ndarray,
#     y_train: np.ndarray,
#     X_val: Optional[np.ndarray] = None,
#     y_val: Optional[np.ndarray] = None,
#     save_path: Optional[Path] = None,
#     scale: bool = True,
#     **kwargs,
# ) -> Tuple[Pipeline, Dict[str, Dict[str, float]], Dict[str, Any]]:
#     """
#     Train a model on (X_train, y_train), optionally validate on (X_val, y_val).

#     Returns:
#         - model_pipeline (Pipeline): fitted model (possibly with preprocessing)
#         - metrics (dict[str, float]): evaluation metrics (mse, r2, etc.)
#         - metadata (dict[str, Any]): extra info (training time, params, etc.)
#     """


def validate_module(mod) -> Tuple[bool, str]:
    """Validate that a module implements the required model-script contract.

    Checks for:
      - MODEL_NAME (str)
      - SUPPORTED_PROBLEM_TYPES (iterable)
      - train_model callable with at least two params (X_train, y_train)

    Returns (is_valid, reason)."""
    # require MODEL_NAME and SUPPORTED_PROBLEM_TYPES
    if not hasattr(mod, "MODEL_NAME"):
        return False, "missing MODEL_NAME"
    if not hasattr(mod, "SUPPORTED_PROBLEM_TYPES"):
        return False, "missing SUPPORTED_PROBLEM_TYPES"

    # require a class named Model that subclasses ModelScript and implements train_model
    if not hasattr(mod, "Model"):
        return False, "missing Model class"
    ModClass = getattr(mod, "Model")
    try:
        # ensure it's a class and has train_model
        if not isinstance(ModClass, type):
            return False, "Model is not a class"
        # duck-type: check method exists
        if not hasattr(ModClass, "train_model"):
            return False, "Model class has no train_model method"
        # check subclassing ModelScript if available
        try:
            from .base import ModelScript as _ModelScript  # type: ignore
            if not issubclass(ModClass, _ModelScript):
                return False, "Model class must subclass ModelScript"
        except Exception:
            # fallback: ignore if import fails
            pass
    except Exception:
        return False, "unable to validate Model class"

    return True, "ok"


class ModelScript(ABC):
    """Abstract base class that all model scripts must follow."""

    MODEL_NAME: str
    SUPPORTED_PROBLEM_TYPES: tuple[str, ...]  # subclasses must define this

    @abstractmethod
    def train_model(self, X_train, y_train, X_val=None, y_val=None, save_path=None, scale=True, **kwargs):
        """Train and return (model, metrics, metadata)."""
        pass

