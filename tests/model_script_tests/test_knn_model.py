import numpy as np
from pathlib import Path
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from main.model_scripts.knn import Model, MODEL_NAME


def test_knn_model_training(tmp_path):
    """
    Test that the KNN classifier trains and returns the expected outputs.
    """

    X_train = np.array([[1, 1], [1.2, 1.1], [3, 3], [3.1, 3.2]])
    y_train = np.array([0, 0, 1, 1])

    model_script = Model()
    save_file = tmp_path / "knn_model.joblib"

    pipeline, metrics, metadata = model_script.train_model(
        X_train=X_train,
        y_train=y_train,
        save_path=save_file,
        scale=True,
        n_neighbors=2
    )

    assert pipeline is not None
    assert hasattr(pipeline, "predict")
    preds = pipeline.predict(X_train)
    assert preds.shape == y_train.shape

    assert "train" in metrics
    for key in ("accuracy", "precision", "recall", "f1"):
        assert key in metrics["train"]
        assert isinstance(metrics["train"][key], float)

    assert metadata["name"] == MODEL_NAME
    assert "hyperparams" in metadata
    assert metadata["train_samples"] == len(X_train)
    assert save_file.exists()
