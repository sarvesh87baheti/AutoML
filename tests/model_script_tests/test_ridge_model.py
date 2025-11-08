import numpy as np
from pathlib import Path

# Ensure project root is on path (if needed)
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from main.model_scripts.ridge import Model, MODEL_NAME


def test_ridge_model_training(tmp_path):
    """
    Test that the Ridge model trains properly and returns
    (pipeline, metrics, metadata) in the correct format.
    """

    # Simple regression dataset
    X_train = np.array([
        [1.0, 2.0],
        [2.0, 3.0],
        [3.0, 4.0],
        [4.0, 5.0]
    ])
    y_train = np.array([3.0, 5.0, 7.0, 9.0])

    model_script = Model()
    save_file = tmp_path / "ridge_model.joblib"

    # Train model
    pipeline, metrics, metadata = model_script.train_model(
        X_train=X_train,
        y_train=y_train,
        save_path=save_file,
        scale=True,
    )

    # ✅ Pipeline checks
    assert pipeline is not None
    assert hasattr(pipeline, "predict")
    preds = pipeline.predict(X_train)
    assert preds.shape == y_train.shape

    # ✅ Metrics checks
    assert "train" in metrics
    train_metrics = metrics["train"]
    for key in ("mse", "rmse", "mae", "r2"):
        assert key in train_metrics
        assert isinstance(train_metrics[key], float)

    # ✅ Metadata checks
    assert metadata["name"] == MODEL_NAME
    assert "hyperparams" in metadata
    assert "train_samples" in metadata
    assert metadata["train_samples"] == len(X_train)

    # ✅ Saved file exists
    assert save_file.exists()
