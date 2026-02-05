from __future__ import annotations

from typing import Any, List, Dict, Union
import logging

import numpy as np

logger = logging.getLogger(__name__)


def _extract_estimator(model: Any) -> Any:
    try:
        from sklearn.pipeline import Pipeline
    except ImportError:
        Pipeline = None

    if Pipeline is not None and isinstance(model, Pipeline):
        return model.steps[-1][1]
    return model


def _normalize_importances(importances: np.ndarray) -> np.ndarray:
    importances = np.asarray(importances, dtype=float)
    if importances.ndim != 1:
        importances = importances.ravel()
    total = float(np.sum(importances))
    if total <= 0:
        if total < 0:
            logger.warning("Sum of importances is negative, which is unexpected. Returning zeros.")
        return np.zeros_like(importances, dtype=float)
    return (importances / total) * 100.0


def compute_feature_priorities(
    model: Any,
    X_val: Union[np.ndarray, Any],  # Also accepts scipy sparse matrices
    y_val: np.ndarray,
    feature_names: List[str],
    problem_type: str,
) -> List[Dict[str, float]]:
    estimator = _extract_estimator(model)
    importances = None

    if hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_)
        if coef.ndim > 1:
            coef = np.mean(np.abs(coef), axis=0)
        else:
            coef = np.abs(coef)
        importances = coef
    elif hasattr(estimator, "feature_importances_"):
        fi = np.asarray(estimator.feature_importances_)
        importances = np.abs(fi)
    else:
        try:
            from sklearn.inspection import permutation_importance
        except ImportError:
            permutation_importance = None

        if permutation_importance is not None:
            try:
                scoring = "r2" if problem_type == "regression" else "accuracy"
                result = permutation_importance(
                    model,
                    X_val,
                    y_val,
                    n_repeats=10,
                    random_state=42,
                    scoring=scoring,
                )
                # Permutation importance can produce negative values when permuting a feature
                # decreases model performance (i.e., the feature hurts the model).
                # For the purpose of normalized percentage display, we clip negative values to zero
                # so that only positive contributions are reflected in the importance distribution.
                signed_importances = np.asarray(result.importances_mean, dtype=float).ravel()
                importances = np.clip(signed_importances, a_min=0.0, a_max=None)
            except Exception as e:
                logger.warning(f"Permutation importance computation failed: {e}")
                return []

    if importances is None:
        logger.info("No feature importance method available for this model.")
        return []

    if len(feature_names) != len(importances):
        logger.warning(
            f"Feature names count ({len(feature_names)}) does not match importance count "
            f"({len(importances)}). Using generic feature names."
        )
        feature_names = [f"feature_{i}" for i in range(len(importances))]

    percentages = _normalize_importances(importances)
    priorities = [
        {
            "feature": feature_names[idx],
            "importance": float(importances[idx]),
            "percentage": float(percentages[idx]),
        }
        for idx in range(len(importances))
    ]
    priorities.sort(key=lambda item: item["percentage"], reverse=True)
    return priorities
