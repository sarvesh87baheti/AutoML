from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

def _ensure_array(x):
    """Convert pandas objects to numpy arrays, otherwise return numpy array.
    """
    if isinstance(x, pd.DataFrame) or isinstance(x, pd.Series):
        return x.values
    return np.asarray(x)


def evaluate_model(model: Any, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Evaluate a fitted model and return common regression metrics.

    Returns a dict containing mse, rmse, mae and r2.
    """
    X = _ensure_array(X)
    y = _ensure_array(y)
    preds = model.predict(X)
    mse = mean_squared_error(y, preds)
    mae = mean_absolute_error(y, preds)
    r2 = r2_score(y, preds)
    rmse = np.sqrt(mse)
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}



def evaluate_classification_model(model: Any, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Evaluate classification model with common metrics."""
    X = _ensure_array(X)
    y = _ensure_array(y)
    preds = model.predict(X)
    metrics = {
        "accuracy": accuracy_score(y, preds),
        "precision": precision_score(y, preds, average="weighted", zero_division=0),
        "recall": recall_score(y, preds, average="weighted", zero_division=0),
        "f1": f1_score(y, preds, average="weighted", zero_division=0),
    }
    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(X)
            if probs.shape[1] == 2:
                metrics["roc_auc"] = roc_auc_score(y, probs[:, 1])
        except Exception:
            pass
    return metrics

def evaluate_kmeans_metrics(X: np.ndarray, labels: np.ndarray) -> Dict[str, Optional[float]]:
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
