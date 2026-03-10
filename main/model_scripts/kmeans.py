from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from main.model_scripts.base import ModelScript
from main.model_scripts.utils import _ensure_array, evaluate_kmeans_metrics

MODEL_NAME = "kmeans"
SUPPORTED_PROBLEM_TYPES = ["kmeans_clustering"]

def _build_pipeline(
    scale: bool,
    n_clusters: int = 3,
    random_state: int = 42,
    **est_kwargs,
) -> Pipeline:

    steps = []

    if scale:
        steps.append(("scaler", StandardScaler()))

    steps.append(
        (
            "est",
            KMeans(
                n_clusters=n_clusters,
                random_state=random_state,
                n_init=10,
                **est_kwargs,
            ),
        )
    )

    return Pipeline(steps)

def train_model(
    X_train: np.ndarray,
    y_train: Optional[np.ndarray] = None,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    save_path: Optional[Path] = None,
    scale: bool = True,
    n_clusters: int = 3,
    random_state: int = 42,
    **kwargs,
) -> Tuple[Pipeline, Dict[str, Dict[str, Optional[float]]], Dict[str, Any]]:
    X_train = _ensure_array(X_train)

    pipe = _build_pipeline(
        scale=scale,
        n_clusters=n_clusters,
        random_state=random_state,
        **kwargs,
    )

    pipe.fit(X_train)

    train_labels = pipe.predict(X_train)
    metrics = {"train": evaluate_kmeans_metrics(X_train, train_labels)}

    est = pipe.named_steps["est"]
    metrics["train"]["inertia"] = float(est.inertia_)

    metadata = {
        "name": MODEL_NAME,
        "hyperparams": {
            "n_clusters": int(n_clusters),
            "random_state": int(random_state),
            "scale": scale,
            **kwargs,
        },
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

    def train_model(
        self,
        X_train,
        y_train=None,
        X_val=None,
        y_val=None,
        save_path=None,
        scale=True,
        **kwargs,
    ):
        return train_model(
            X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            save_path=save_path,
            scale=scale,
            **kwargs,
        )