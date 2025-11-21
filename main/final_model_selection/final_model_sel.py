def score_model(metrics, weights=None):
    if weights is None:
        weights = {"mse": 0.1, "rmse": 0.3, "mae": 0.2, "r2": 0.4}

    mse = metrics["val"]["mse"]
    rmse = metrics["val"]["rmse"]
    mae = metrics["val"]["mae"]
    r2 = metrics["val"]["r2"]

    score = (
        weights["mse"] * (1 / mse) +
        weights["rmse"] * (1 / rmse) +
        weights["mae"] * (1 / mae) +
        weights["r2"] * (r2)
    )
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
