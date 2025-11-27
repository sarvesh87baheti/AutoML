def score_model(metrics, weights=None):
    # Regression scoring
    if "mse" in metrics.get("val", {}):
        if weights is None:
            weights = {"mse": 0.1, "rmse": 0.3, "mae": 0.2, "r2": 0.4}

        mse = metrics["val"].get("mse", 1e9)
        rmse = metrics["val"].get("rmse", 1e9)
        mae = metrics["val"].get("mae", 1e9)
        r2 = metrics["val"].get("r2", 0)

        score = (
            weights["mse"] * (1 / mse) +
            weights["rmse"] * (1 / rmse) +
            weights["mae"] * (1 / mae) +
            weights["r2"] * r2
        )
        return score

    # Classification scoring
    accuracy = metrics["val"].get("accuracy", 0)
    precision = metrics["val"].get("precision", 0)
    recall = metrics["val"].get("recall", 0)
    f1 = metrics["val"].get("f1", 0)

    # Weighted scoring for classification
    score = (0.5 * f1) + (0.3 * accuracy) + (0.1 * precision) + (0.1 * recall)
    return score


def compute_model_scores(results):
    model_scores = {}
    for model_name, info in results.items():
        score = score_model(info["metrics"])
        model_scores[model_name] = score

    best_model = max(model_scores, key=model_scores.get)
    # Extract best model’s weights from results JSON
    best_model_weights = results[best_model].get("weights", None)
    # model_Scores
    return best_model, best_model_weights
