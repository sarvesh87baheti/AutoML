from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .utils import _ensure_array, evaluate_model
from .base import ModelScript

MODEL_NAME = "linear"
SUPPORTED_PROBLEM_TYPES = ["regression"]


def _build_pipeline(scale: bool = True, **est_kwargs) -> Pipeline:
    if scale:
        return Pipeline([("scaler", StandardScaler()), ("est", LinearRegression(**est_kwargs))])
    return Pipeline([("est", LinearRegression(**est_kwargs))])


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    save_path: Optional[Path] = None,
    scale: bool = True,
    **kwargs,
) -> Tuple[Pipeline, Dict[str, Dict[str, float]], Dict[str, Any]]:
    X_train = _ensure_array(X_train)
    y_train = _ensure_array(y_train)
    pipe = _build_pipeline(scale=scale, **kwargs)
    pipe.fit(X_train, y_train)

    metrics = {"train": evaluate_model(pipe, X_train, y_train)}
    if X_val is not None and y_val is not None:
        metrics["val"] = evaluate_model(pipe, X_val, y_val)

    metadata = {"name": MODEL_NAME, "hyperparams": kwargs, "train_samples": int(len(X_train))}

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        import joblib

        joblib.dump(pipe, save_path)

    return pipe, metrics, metadata


class Model(ModelScript):
    MODEL_NAME = MODEL_NAME
    SUPPORTED_PROBLEM_TYPES = tuple(SUPPORTED_PROBLEM_TYPES)

    def train_model(self, X_train, y_train, X_val=None, y_val=None, save_path=None, scale=True, **kwargs):
        return train_model(X_train, y_train, X_val=X_val, y_val=y_val, save_path=save_path, scale=scale, **kwargs)
