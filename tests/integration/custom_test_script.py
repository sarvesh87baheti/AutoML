from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline

from main.model_scripts.base import ModelScript


MODEL_NAME = "custom_forest"
SUPPORTED_PROBLEM_TYPES = ("classification",)


def _as_array(value):
    return np.asarray(value)


def _classification_metrics(model: Pipeline, X, y) -> Dict[str, float]:
    predictions = model.predict(X)
    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "precision": float(precision_score(y, predictions, average="weighted", zero_division=0)),
        "recall": float(recall_score(y, predictions, average="weighted", zero_division=0)),
        "f1": float(f1_score(y, predictions, average="weighted", zero_division=0)),
    }


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    save_path: Optional[Path] = None,
    scale: bool = True,
    **kwargs,
) -> Tuple[Pipeline, Dict[str, Dict[str, float]], Dict[str, Any]]:
    X_train = _as_array(X_train)
    y_train = _as_array(y_train)

    forest_params = {
        "n_estimators": 16,
        "max_depth": 5,
        "random_state": 42,
        "n_jobs": 1,
    }
    forest_params.update(kwargs)

    pipe = Pipeline([("forest", RandomForestClassifier(**forest_params))])
    pipe.fit(X_train, y_train)

    metrics = {"train": _classification_metrics(pipe, X_train, y_train)}
    if X_val is not None and y_val is not None:
        metrics["val"] = _classification_metrics(pipe, _as_array(X_val), _as_array(y_val))

    metadata = {
        "name": MODEL_NAME,
        "hyperparams": forest_params,
        "train_samples": int(len(X_train)),
    }

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipe, save_path)

    return pipe, metrics, metadata


class Model(ModelScript):
    MODEL_NAME = MODEL_NAME
    SUPPORTED_PROBLEM_TYPES = SUPPORTED_PROBLEM_TYPES

    def train_model(self, X_train, y_train, X_val=None, y_val=None, save_path=None, scale=True, **kwargs):
        return train_model(
            X_train,
            y_train,
            X_val=X_val,
            y_val=y_val,
            save_path=save_path,
            scale=scale,
            **kwargs,
        )
