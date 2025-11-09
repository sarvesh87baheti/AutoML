from pathlib import Path
import json
import numpy as np
from typing import Dict, Any

from .regression import RegressionTrainer
# from .classification import ClassificationTrainer

def load_processed_dataset(path: Path):
    X_train = np.load(path / "X_train.npy")
    y_train = np.load(path / "y_train.npy")
    X_val = np.load(path / "X_val.npy")
    y_val = np.load(path / "y_val.npy")

    with open(path / "metadata.json", "r") as f:
        metadata = json.load(f)

    return X_train, y_train, X_val, y_val, metadata


class Orchestrator:
    def __init__(self, dataset_path: Path, model_scripts_path: Path, output_path: Path):
        self.dataset_path = dataset_path
        self.model_scripts_path = model_scripts_path
        self.output_path = output_path

    def run(self):
        X_train, y_train, X_val, y_val, metadata = load_processed_dataset(self.dataset_path)

        problem_type = metadata["problem_type"]

        if problem_type == "regression":
            trainer = RegressionTrainer(self.model_scripts_path, self.output_path)
        # elif problem_type == "classification":
        #     trainer = ClassificationTrainer(self.model_scripts_path, self.output_path)
        else:
            raise ValueError(f"Unsupported problem type: {problem_type}")

        results = trainer.train_all(X_train, y_train, X_val, y_val)
        return results
