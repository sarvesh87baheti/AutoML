from pathlib import Path
import numpy as np
from main.model_training.regression import RegressionTrainer


def test_regression_trainer_runs(tmp_path):
    project_root = Path(__file__).resolve().parents[2]
    model_scripts_dir = project_root / "main" / "model_scripts"

    # Create fake model script inside real package
    fake_script_path = model_scripts_dir / "fake_model.py"
    fake_script_path.write_text("""
from main.model_scripts.base import ModelScript

MODEL_NAME = "fake_regressor"
SUPPORTED_PROBLEM_TYPES = ("regression",)

class Model(ModelScript):
    MODEL_NAME = MODEL_NAME
    SUPPORTED_PROBLEM_TYPES = SUPPORTED_PROBLEM_TYPES

    def train_model(self, X_train, y_train, X_val=None, y_val=None, save_path=None, scale=True, **kwargs):
        return "PIPE", {"train": {"mse": 0.1}}, {"name": MODEL_NAME}
""")

    # Dataset
    X_train = np.array([[1, 2], [3, 4]])
    y_train = np.array([1, 2])
    X_val = np.array([[5, 6],[2, 4]])
    y_val = np.array([3, 2])

    trainer = RegressionTrainer(
        scripts_path=model_scripts_dir,
        output_path=tmp_path
    )

    results = trainer.train_all(X_train, y_train, X_val, y_val)

    # Assertions
    assert "fake_regressor" in results
    assert "train" in results["fake_regressor"]["metrics"]

    # ✅ Cleanup
    fake_script_path.unlink()
