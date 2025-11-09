import numpy as np
from pathlib import Path
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from main.model_scripts.logistic import Model, MODEL_NAME


def test_logistic_model_training(tmp_path):
    """
    Test that the Logistic Regression model trains properly and returns
    (pipeline, metrics, metadata) in the correct format.
    """

    # Simple classification dataset (binary)
    X_train = np.array([
        [1.0, 2.0],
        [1.5, 1.8],
        [3.0, 3.5],
        [3.5, 4.0]
    ])
    y_train = np.array([0, 0, 1, 1])

    model_script = Model()
    save_file = tmp_path / "logistic_model.joblib"

    pipeline, metrics, metadata = model_script.train_model(
        X_train=X_train,
        y_train=y_train,
        save_path=save_file,
        scale=True
    )

    # ✅ Pipeline exists and can predict
    assert pipeline is not None
    assert hasattr(pipeline, "predict")

    preds = pipeline.predict(X_train)
    assert preds.shape == y_train.shape

    # ✅ Metrics structure
    assert "train" in metrics
    train_metrics = metrics["train"]
    for key in ("accuracy", "precision", "recall", "f1"):
        assert key in train_metrics
        assert isinstance(train_metrics[key], float)

    # ✅ Metadata checks
    assert metadata["name"] == MODEL_NAME
    assert "hyperparams" in metadata
    assert "train_samples" in metadata
    assert metadata["train_samples"] == len(X_train)

    # ✅ Saved model exists
    assert save_file.exists()
