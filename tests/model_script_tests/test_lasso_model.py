import numpy as np
from pathlib import Path

from main.model_scripts.lasso import Model, MODEL_NAME


def test_lasso_model_training(tmp_path):
    """
    Test that the Lasso model trains properly and returns
    (pipeline, metrics, metadata) in the correct format.
    """

    # Small regression dataset
    X_train = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])
    y_train = np.array([3.0, 5.0, 7.0, 9.0])

    # Instantiate Lasso model script
    model_script = Model()

    # Temporary save file
    save_file = tmp_path / "lasso_model.joblib"

    # Train the model
    pipeline, metrics, metadata = model_script.train_model(
        X_train=X_train,
        y_train=y_train,
        X_val=None,
        y_val=None,
        save_path=save_file,
        scale=True,
    )

    # ✅ Pipeline exists and has predict()
    assert pipeline is not None
    assert hasattr(pipeline, "predict")

    # ✅ Predictions are valid
    preds = pipeline.predict(X_train)
    assert preds.shape == y_train.shape

    # ✅ Metrics structure is correct
    assert "train" in metrics
    train_metrics = metrics["train"]
    for key in ("mse", "rmse", "mae", "r2"):
        assert key in train_metrics
        assert isinstance(train_metrics[key], float)

    # ✅ Metadata includes expected fields
    assert metadata["name"] == MODEL_NAME
    assert "hyperparams" in metadata
    assert "train_samples" in metadata
    assert metadata["train_samples"] == len(X_train)

    # ✅ Model saved correctly
    assert save_file.exists()
