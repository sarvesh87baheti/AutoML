import numpy as np
from main.model_training.regression import RegressionTrainer

def test_multioutput_regression(tmp_path):
    # Create fake model script
    model_scripts_dir = tmp_path / "model_scripts"
    model_scripts_dir.mkdir()

    fake_script = model_scripts_dir / "fake_multi.py"
    fake_script.write_text("""
from main.model_scripts.base import ModelScript
MODEL_NAME = "fake_multi"
SUPPORTED_PROBLEM_TYPES = ("regression",)

class Model(ModelScript):
    MODEL_NAME = MODEL_NAME
    SUPPORTED_PROBLEM_TYPES = SUPPORTED_PROBLEM_TYPES

    def train_model(self, X_train, y_train, X_val=None, y_val=None, save_path=None, scale=True, **kwargs):
        assert y_train.ndim == 2
        assert y_train.shape[1] == 2
        return "PIPE", {"mse": 0.1}, {"name": MODEL_NAME}
""")

    # Fake 2-target dataset
    X_train = np.random.randn(10, 4)
    y_train = np.random.randn(10, 2)      # <--- 2 targets
    X_val = np.random.randn(5, 4)
    y_val = np.random.randn(5, 2)

    trainer = RegressionTrainer(
        scripts_path=model_scripts_dir,
        output_path=tmp_path
    )

    results = trainer.train_all(X_train, y_train, X_val, y_val)

    assert "fake_multi" in results
    assert "mse" in results["fake_multi"]["metrics"]
