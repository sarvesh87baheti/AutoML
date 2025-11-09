from pathlib import Path
import numpy as np
from main.model_training.regression import RegressionTrainer

def test_multioutput_regression(tmp_path):
    project_root = Path(__file__).resolve().parents[2]
    model_scripts_dir = project_root / "main" / "model_scripts"

    fake_script_path = model_scripts_dir / "fake_multi.py"
    fake_script_path.write_text("""
from main.model_scripts.base import ModelScript

MODEL_NAME = "fake_multi"
SUPPORTED_PROBLEM_TYPES = ("regression",)

class Model(ModelScript):
    MODEL_NAME = MODEL_NAME
    SUPPORTED_PROBLEM_TYPES = SUPPORTED_PROBLEM_TYPES

    def train_model(self, X_train, y_train, X_val=None, y_val=None, save_path=None, scale=True, **kwargs):
        assert y_train.ndim == 2
        assert y_train.shape[1] == 2
        return "PIPE", {"train": {"mse": 0.1}}, {"name": MODEL_NAME}
""")

    # Fake dataset
    X_train = np.random.randn(10, 4)
    y_train = np.random.randn(10, 2)
    X_val = np.random.randn(5, 4)
    y_val = np.random.randn(5, 2)

    trainer = RegressionTrainer(
        scripts_path=model_scripts_dir,
        output_path=tmp_path
    )

    results = trainer.train_all(X_train, y_train, X_val, y_val)

    assert "fake_multi" in results

    fake_script_path.unlink()
