from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


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
