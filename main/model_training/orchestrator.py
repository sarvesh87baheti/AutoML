from pathlib import Path
import json
import numpy as np
from typing import Dict, Any

from .regression import RegressionTrainer
from .classification import ClassificationTrainer
from .kmclustering import KMClusteringTrainer

def load_processed_dataset(path: Path):
    with open(path / "metadata.json", "r") as f:
        metadata = json.load(f)

    problem_type = metadata.get("problem_type")
    x_train_npz = path / "X_train.npz"

    if x_train_npz.exists():
        from scipy.sparse import load_npz

        X_train = load_npz(x_train_npz)
    else:
        X_train = np.load(path / "X_train.npy")

    if problem_type in {"regression", "classification"}:
        x_val_npz = path / "X_val.npz"

        if x_val_npz.exists():
            from scipy.sparse import load_npz

            X_val = load_npz(x_val_npz)
        else:
            X_val = np.load(path / "X_val.npy")

        y_train = np.load(path / "y_train.npy")
        y_val = np.load(path / "y_val.npy")
    elif problem_type == "kmeans_clustering":
        X_val = None
        y_train = None
        y_val = None
    else:
        raise ValueError(f"Unsupported problem type in metadata: {problem_type}")

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
        elif problem_type == "classification":
            trainer = ClassificationTrainer(self.model_scripts_path, self.output_path)
        elif problem_type == "kmeans_clustering":
            n_clusters = int(metadata.get("n_clusters", 3))
            trainer = KMClusteringTrainer(self.model_scripts_path, self.output_path)
            return trainer.train_all(X_train, y_train=None, X_val=None, y_val=None, n_clusters=n_clusters)
        else:
            raise ValueError(f"Unsupported problem type: {problem_type}")

        results = trainer.train_all(X_train, y_train, X_val, y_val)
        return results
