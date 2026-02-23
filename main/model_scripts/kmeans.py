from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from main.model_scripts.base import ModelScript
from main.model_scripts.utils import _ensure_array

MODEL_NAME = "kmeans"
SUPPORTED_PROBLEM_TYPES = ["kmeans_clustering"]


def _build_pipeline(n_clusters: int = 3, random_state: int = 42, **est_kwargs) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("est", KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10, **est_kwargs)),
        ]
    )


def _compute_metrics(X: np.ndarray, labels: np.ndarray) -> Dict[str, Optional[float]]:
    unique_labels = np.unique(labels)
    if len(unique_labels) <= 1:
        return {
            "silhouette": None,
            "calinski_harabasz": None,
            "davies_bouldin": None,
            "inertia": None,
        }

    return {
        "silhouette": float(silhouette_score(X, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
        "davies_bouldin": float(davies_bouldin_score(X, labels)),
        "inertia": None,
    }


def train_model(
    X_train: np.ndarray,
    y_train: Optional[np.ndarray] = None,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    save_path: Optional[Path] = None,
    n_clusters: int = 3,
    random_state: int = 42,
    **kwargs,
) -> Tuple[Pipeline, Dict[str, Dict[str, Optional[float]]], Dict[str, Any]]:
    X_train = _ensure_array(X_train)

    pipe = _build_pipeline(n_clusters=n_clusters, random_state=random_state, **kwargs)
    pipe.fit(X_train)

    train_labels = pipe.predict(X_train)
    metrics = {"train": _compute_metrics(X_train, train_labels)}

    est = pipe.named_steps["est"]
    metrics["train"]["inertia"] = float(est.inertia_)

    metadata = {
        "name": MODEL_NAME,
        "hyperparams": {"n_clusters": int(n_clusters), "random_state": int(random_state), **kwargs},
        "train_samples": int(len(X_train)),
    }

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        import joblib

        joblib.dump(pipe, save_path)

    return pipe, metrics, metadata


class Model(ModelScript):
    MODEL_NAME = MODEL_NAME
    SUPPORTED_PROBLEM_TYPES = tuple(SUPPORTED_PROBLEM_TYPES)

    def train_model(self, X_train, y_train=None, X_val=None, y_val=None, save_path=None, scale=True, **kwargs):
        return train_model(X_train, y_train=y_train, X_val=X_val, y_val=y_val, save_path=save_path, **kwargs)
