from pathlib import Path

import numpy as np

from main.model_training.classification import ClassificationTrainer


def test_classification_trainer_skips_failing_optional_models(tmp_path):
    project_root = Path(__file__).resolve().parents[2]
    model_scripts_dir = project_root / "main" / "model_scripts"

    good_script_path = model_scripts_dir / "fake_optional_good.py"
    bad_script_path = model_scripts_dir / "fake_optional_bad.py"

    good_script = """
from main.model_scripts.base import ModelScript

MODEL_NAME = "fake_optional_good"
SUPPORTED_PROBLEM_TYPES = ("classification",)

class DummyModel:
    def predict(self, X):
        import numpy as np
        return np.zeros(len(X), dtype=int)

class Model(ModelScript):
    MODEL_NAME = MODEL_NAME
    SUPPORTED_PROBLEM_TYPES = SUPPORTED_PROBLEM_TYPES

    def train_model(self, X_train, y_train, X_val=None, y_val=None, save_path=None, scale=True, **kwargs):
        return DummyModel(), {"train": {"accuracy": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0}}, {"name": MODEL_NAME}
"""
    bad_script = """
from main.model_scripts.base import ModelScript

MODEL_NAME = "fake_optional_bad"
SUPPORTED_PROBLEM_TYPES = ("classification",)

class Model(ModelScript):
    MODEL_NAME = MODEL_NAME
    SUPPORTED_PROBLEM_TYPES = SUPPORTED_PROBLEM_TYPES

    def train_model(self, X_train, y_train, X_val=None, y_val=None, save_path=None, scale=True, **kwargs):
        raise RuntimeError("boom")
"""

    good_script_path.write_text(good_script)
    bad_script_path.write_text(bad_script)

    try:
        trainer = ClassificationTrainer(
            scripts_path=model_scripts_dir,
            output_path=tmp_path,
        )
        X_train = np.array([[0.0, 1.0], [1.0, 0.0], [0.2, 0.8], [0.8, 0.2]])
        y_train = np.array([0, 1, 0, 1])
        X_val = np.array([[0.1, 0.9], [0.9, 0.1]])
        y_val = np.array([0, 1])

        results = trainer.train_all(X_train, y_train, X_val, y_val)

        assert "fake_optional_good" in results
        assert "fake_optional_bad" not in results
    finally:
        if good_script_path.exists():
            good_script_path.unlink()
        if bad_script_path.exists():
            bad_script_path.unlink()
