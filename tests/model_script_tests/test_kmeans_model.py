import numpy as np
from pathlib import Path
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from main.model_scripts.kmeans import Model, MODEL_NAME


def test_kmeans_model_training_and_saving(tmp_path):
    """
    Test that the KMeans model trains, returns expected outputs, and saves the pipeline.
    """
    # Two clearly separated clusters
    X_train = np.array(
        [
            [0.0, 0.0],
            [0.1, -0.1],
            [-0.2, 0.2],
            [5.0, 5.0],
            [5.1, 4.9],
            [4.9, 5.2],
        ],
        dtype=float,
    )

    model_script = Model()
    save_file = tmp_path / "kmeans_model.joblib"

    pipeline, metrics, metadata = model_script.train_model(
        X_train=X_train,
        y_train=None,
        save_path=save_file,
        scale=True,
        n_clusters=2,
        random_state=42,
    )

    assert pipeline is not None
    assert hasattr(pipeline, "predict")

    labels = pipeline.predict(X_train)
    assert labels.shape == (len(X_train),)
    assert labels.dtype.kind in ("i", "u")  # integer labels

    assert "train" in metrics
    assert isinstance(metrics["train"], dict)
    assert "inertia" in metrics["train"]
    assert isinstance(metrics["train"]["inertia"], float)

    # If silhouette is provided, it should be within valid range
    if "silhouette" in metrics["train"] and metrics["train"]["silhouette"] is not None:
        assert isinstance(metrics["train"]["silhouette"], float)
        assert -1.0 <= metrics["train"]["silhouette"] <= 1.0

    assert metadata["name"] == MODEL_NAME
    assert "hyperparams" in metadata
    assert metadata["hyperparams"]["n_clusters"] == 2
    assert metadata["hyperparams"]["random_state"] == 42
    assert metadata["hyperparams"]["scale"] is True
    assert metadata["train_samples"] == len(X_train)

    assert save_file.exists()


def test_kmeans_model_training_without_scaling(tmp_path):
    """
    Test that KMeans training works with scale=False and returns inertia metric.
    """
    X_train = np.array(
        [
            [10.0, 0.0],
            [10.2, -0.1],
            [9.8, 0.1],
            [0.0, 10.0],
            [-0.1, 10.2],
            [0.1, 9.8],
        ],
        dtype=float,
    )

    model_script = Model()
    save_file = tmp_path / "kmeans_model_no_scale.joblib"

    pipeline, metrics, metadata = model_script.train_model(
        X_train=X_train,
        y_train=None,
        save_path=save_file,
        scale=False,
        n_clusters=2,
        random_state=42,
    )

    assert pipeline is not None
    labels = pipeline.predict(X_train)
    assert labels.shape == (len(X_train),)

    assert "train" in metrics
    assert "inertia" in metrics["train"]
    assert isinstance(metrics["train"]["inertia"], float)

    assert metadata["name"] == MODEL_NAME
    assert metadata["hyperparams"]["scale"] is False
    assert metadata["hyperparams"]["n_clusters"] == 2
    assert metadata["train_samples"] == len(X_train)

    assert save_file.exists()