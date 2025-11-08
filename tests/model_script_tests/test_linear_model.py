import numpy as np
from pathlib import Path

# Import the script under test
from main.model_scripts.linear import Model, MODEL_NAME


def test_linear_model_training(tmp_path):
    """
    Test that the linear regression model trains properly and returns
    (pipeline, metrics, metadata) in correct format.
    """

    # Fake 2-feature regression dataset
    X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5]], dtype=float)
    y_train = np.array([3, 5, 7, 9], dtype=float)

    # Instantiate model script
    model_script = Model()

    # Temporary save path
    save_path = tmp_path / "linear_model.joblib"

    # Train the model
    pipeline, metrics, metadata = model_script.train_model(
        X_train=X_train,
        y_train=y_train,
        X_val=None,
        y_val=None,
        save_path=save_path,
        scale=True,
    )

    # ✅ --- Assertions ---

    # Pipeline should be a sklearn Pipeline
    assert pipeline is not None
    assert hasattr(pipeline, "predict")

    # Predictions should be numeric and correct shape
    preds = pipeline.predict(X_train)
    assert preds.shape == y_train.shape

    # Metrics should exist
    assert "train" in metrics
    train_metrics = metrics["train"]

    for key in ("mse", "rmse", "mae", "r2"):
        assert key in train_metrics
        assert isinstance(train_metrics[key], float)

    # Metadata should include expected fields
    assert metadata["name"] == MODEL_NAME
    assert "train_samples" in metadata
    assert metadata["train_samples"] == len(X_train)

    # Saved file should exist
    assert save_path.exists()
