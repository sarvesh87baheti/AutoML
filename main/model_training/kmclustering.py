import importlib
import pkgutil
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

from main.model_scripts.base import validate_module


class KMClusteringTrainer:
    def __init__(self, scripts_path: Path, output_path: Path):
        self.scripts_path = scripts_path
        self.output_path = output_path

    def _load_models(self):
        model_classes = []
        package_name = "main.model_scripts"

        for _, module_name, _ in pkgutil.iter_modules([str(self.scripts_path)]):
            full_module_name = f"{package_name}.{module_name}"
            module = importlib.import_module(full_module_name)

            ok, _ = validate_module(module)
            if not ok:
                continue

            supported = getattr(module, "SUPPORTED_PROBLEM_TYPES", [])
            if "kmeans_clustering" not in supported:
                continue

            ModelClass = getattr(module, "Model", None)
            if ModelClass is not None:
                model_classes.append(ModelClass)

        return model_classes

    def train_all(self, X_train, y_train=None, X_val=None, y_val=None, n_clusters: int = 3):
        models = self._load_models()
        results = {}

        for ModelClass in models:
            model_name = ModelClass.MODEL_NAME
            save_path = self.output_path / f"{model_name}.joblib"

            model = ModelClass()
            pipe, metrics, metadata = model.train_model(
                X_train=X_train,
                y_train=None,
                X_val=None,
                y_val=None,
                save_path=save_path,
                n_clusters=n_clusters,
            )

            try:
                labels = pipe.predict(X_train).tolist()
            except Exception:
                labels = None

            # Attempt to generate a 2D cluster visualization and save it to the results folder.
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                # Prepare dense numpy array
                if hasattr(X_train, "toarray"):
                    X_arr = X_train.toarray()
                else:
                    X_arr = np.asarray(X_train)

                # If the pipeline has a scaler, apply it before PCA so centers and points align
                scaled = X_arr
                if hasattr(pipe, "named_steps") and "scaler" in pipe.named_steps:
                    try:
                        scaled = pipe.named_steps["scaler"].transform(X_arr)
                    except Exception:
                        scaled = X_arr

                # Reduce to 2D (PCA if needed)
                if scaled.shape[1] > 2:
                    pca = PCA(n_components=2, random_state=0)
                    X2 = pca.fit_transform(scaled)
                    centers_2 = None
                    est = pipe.named_steps.get("est") if hasattr(pipe, "named_steps") else None
                    if est is not None and hasattr(est, "cluster_centers_"):
                        try:
                            centers = est.cluster_centers_
                            centers_2 = pca.transform(centers)
                        except Exception:
                            centers_2 = None
                else:
                    X2 = scaled[:, :2]
                    est = pipe.named_steps.get("est") if hasattr(pipe, "named_steps") else None
                    centers_2 = None
                    if est is not None and hasattr(est, "cluster_centers_"):
                        try:
                            centers_2 = est.cluster_centers_[:, :2]
                        except Exception:
                            centers_2 = None

                # Plot
                if X2 is not None and labels is not None:
                    fig, ax = plt.subplots(figsize=(6, 5))
                    scatter = ax.scatter(X2[:, 0], X2[:, 1], c=labels, cmap="tab10", s=20, alpha=0.8)
                    if centers_2 is not None:
                        ax.scatter(centers_2[:, 0], centers_2[:, 1], c="black", s=100, marker="x")
                    ax.set_title(f"{model_name} clusters")
                    ax.set_xlabel("Component 1")
                    ax.set_ylabel("Component 2")
                    ax.grid(False)

                    out_path = self.output_path / f"{model_name}_clusters.png"
                    fig.tight_layout()
                    fig.savefig(out_path, dpi=150)
                    plt.close(fig)
            except Exception:
                # Do not fail training if plotting cannot run
                pass

            results[model_name] = {
                "metrics": metrics,
                "metadata": metadata,
                "cluster_labels": labels,
            }

        return results
