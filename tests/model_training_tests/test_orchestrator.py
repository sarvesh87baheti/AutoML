import numpy as np
import json
from main.model_training.orchestrator import Orchestrator

def test_orchestrator_regression(tmp_path, monkeypatch):
    """
    Test that Orchestrator correctly:
    - loads processed dataset
    - detects regression problem type
    - calls RegressionTrainer.train_all
    - returns output from the trainer
    """

    # --------------------------------------------------------------------
    # 1. Create a fake processed dataset directory that orchestrator loads
    # --------------------------------------------------------------------
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    # fake numpy arrays for train and val data
    np.save(dataset_dir / "X_train.npy", np.array([[1, 2], [3, 4]]))
    np.save(dataset_dir / "y_train.npy", np.array([1, 2]))
    np.save(dataset_dir / "X_val.npy", np.array([[5, 6]]))
    np.save(dataset_dir / "y_val.npy", np.array([3]))

    # metadata.json: choose regression
    metadata = {"problem_type": "regression"}
    with open(dataset_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)

    # --------------------------------------------------------------------
    # 2. Mock RegressionTrainer so orchestrator doesn't actually train anything
    # --------------------------------------------------------------------
    class FakeRegressionTrainer:
        def __init__(self, scripts_path, output_path):
            self.called = False

        def train_all(self, X_train, y_train, X_val, y_val):
            self.called = True
            # dummy training result
            return {"fake_model": {"metrics": {"mse": 0.123}, "metadata": {"name": "fake_model"}}}

    # Monkeypatch RegressionTrainer inside orchestrator module
    monkeypatch.setattr(
        "main.model_training.orchestrator.RegressionTrainer",
        FakeRegressionTrainer
    )

    # --------------------------------------------------------------------
    # 3. Run orchestrator
    # --------------------------------------------------------------------
    orch = Orchestrator(
        dataset_path=dataset_dir,
        model_scripts_path=tmp_path / "model_scripts",  # doesn't matter due to mocking
        output_path=tmp_path
    )

    results = orch.run()

    # --------------------------------------------------------------------
    # 4. Assertions
    # --------------------------------------------------------------------
    assert "fake_model" in results
    assert results["fake_model"]["metrics"]["mse"] == 0.123
    assert results["fake_model"]["metadata"]["name"] == "fake_model"
