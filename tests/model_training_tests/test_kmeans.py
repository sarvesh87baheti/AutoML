from pathlib import Path
import numpy as np
from main.model_training.kmclustering import KMClusteringTrainer


def test_kmeans_trainer_runs(tmp_path):
    project_root = Path(__file__).resolve().parents[2]
    model_scripts_dir = project_root / "main" / "model_scripts"
    # Create fake clustering model script
    fake_script_path = model_scripts_dir / "fake_kmeans.py"

    fake_script_path.write_text("""
from main.model_scripts.base import ModelScript

MODEL_NAME = "fake_kmeans"
SUPPORTED_PROBLEM_TYPES = ("kmeans_clustering",)

class Model(ModelScript):
    MODEL_NAME = MODEL_NAME
    SUPPORTED_PROBLEM_TYPES = SUPPORTED_PROBLEM_TYPES

    def train_model(self, X_train, y_train=None, X_val=None, y_val=None,
                    save_path=None, scale=True, **kwargs):

        metrics = {
            "train": {
                "silhouette": 0.5,
                "calinski_harabasz": 100.0,
                "davies_bouldin": 0.8,
                "inertia": 10.0
            }
        }

        metadata = {"name": MODEL_NAME}

        return "PIPE", metrics, metadata
""")

    # --------------------------------------------------
    # Fake unsupervised dataset
    # --------------------------------------------------
    X_train = np.array([
        [1.0, 2.0],
        [1.1, 2.1],
        [8.0, 9.0],
        [8.2, 9.1],
    ])

    trainer = KMClusteringTrainer(
        scripts_path=model_scripts_dir,
        output_path=tmp_path
    )

    results = trainer.train_all(X_train)

    # --------------------------------------------------
    # Assertions
    # --------------------------------------------------
    assert "fake_kmeans" in results
    assert "train" in results["fake_kmeans"]["metrics"]
    assert "silhouette" in results["fake_kmeans"]["metrics"]["train"]

    # --------------------------------------------------
    # Cleanup
    # --------------------------------------------------
    fake_script_path.unlink()