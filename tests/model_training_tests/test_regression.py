import numpy as np
from main.model_training.regression import RegressionTrainer

def test_regression_trainer_runs(tmp_path):
    # ----------------------------------------------------
    # 1. Create a fake model script dynamically
    # ----------------------------------------------------
    model_scripts_dir = tmp_path / "model_scripts"
    model_scripts_dir.mkdir()

    fake_script = model_scripts_dir / "fake_model.py"
    fake_script.write_text("""
from main.model_scripts.base import ModelScript

MODEL_NAME = "fake_regressor"
SUPPORTED_PROBLEM_TYPES = ("regression",)

class Model(ModelScript):
    MODEL_NAME = MODEL_NAME
    SUPPORTED_PROBLEM_TYPES = SUPPORTED_PROBLEM_TYPES

    def train_model(self, X_train, y_train, X_val=None, y_val=None, save_path=None, scale=True, **kwargs):
        return "PIPE", {"mse": 0.1}, {"name": "fake_regressor"}
""")


    # ----------------------------------------------------
    # 2. Setup fake numeric dataset
    # ----------------------------------------------------
    X_train = np.array([[1, 2], [3, 4]])
    y_train = np.array([1, 2])

    X_val = np.array([[5, 6]])
    y_val = np.array([3])

    # ----------------------------------------------------
    # 3. Run RegressionTrainer
    # ----------------------------------------------------
    trainer = RegressionTrainer(
        scripts_path=model_scripts_dir,
        output_path=tmp_path
    )

    results = trainer.train_all(X_train, y_train, X_val, y_val)

    # ----------------------------------------------------
    # 4. Assertions
    # ----------------------------------------------------
    assert "fake_regressor" in results
    assert results["fake_regressor"]["metadata"]["name"] == "fake_regressor"
    assert "mse" in results["fake_regressor"]["metrics"]
