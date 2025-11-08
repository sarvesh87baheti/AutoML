import numpy as np
from main.model_scripts.linear import Model


def test_linear_model_train_smoke():
    # tiny synthetic dataset
    rng = np.random.RandomState(0)
    X = rng.randn(20, 3)
    coef = np.array([1.2, -0.5, 0.3])
    y = X @ coef + 0.01 * rng.randn(20)

    m = Model()
    model, metrics, meta = m.train_model(X, y)

    # basic assertions
    assert hasattr(model, "predict")
    assert isinstance(metrics, dict)
    assert "train" in metrics
    tr = metrics["train"]
    assert all(k in tr for k in ("mse", "rmse", "mae", "r2"))
    assert isinstance(meta, dict)
    assert meta.get("name") == "linear"
    assert meta.get("train_samples") == 20
