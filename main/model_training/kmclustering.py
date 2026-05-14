import importlib
import importlib.util
import pkgutil
import sys
from pathlib import Path
from typing import List

import numpy as np
from sklearn.decomposition import PCA

from main.model_scripts.base import validate_module

# File stems that must never be loaded as model scripts.
_SKIP_STEMS = {"__init__", "base", "utils", "script_validator", "example_custom_model"}


class KMClusteringTrainer:
    def __init__(
        self,
        scripts_path: Path,
        output_path: Path,
        custom_script_paths: List[Path] | None = None,
    ):
        self.scripts_path = scripts_path
        self.output_path = output_path
        self.custom_script_paths: List[Path] = custom_script_paths or []

    # ── Model discovery ───────────────────────────────────────────────────────

    def _load_models(self) -> list:
        """
        Load all kmeans_clustering-compatible model classes.

        1. Scans built-in model_scripts/ via pkgutil.
        2. Appends user-supplied custom scripts loaded by file path.

        Custom Model classes are tagged with is_custom = True.
        """
        model_classes = []
        package_name = "main.model_scripts"

        # ── Built-in scripts ──────────────────────────────────────────────────
        for _, module_name, _ in pkgutil.iter_modules([str(self.scripts_path)]):
            if module_name in _SKIP_STEMS:
                continue

            full_module_name = f"{package_name}.{module_name}"
            try:
                module = importlib.import_module(full_module_name)
            except Exception as exc:
                print(f"⚠️ Skipping {module_name}: import failed ({exc})")
                continue

            ok, _ = validate_module(module)
            if not ok:
                continue

            supported = getattr(module, "SUPPORTED_PROBLEM_TYPES", [])
            if "kmeans_clustering" not in supported:
                continue

            ModelClass = getattr(module, "Model", None)
            if ModelClass is not None:
                model_classes.append(ModelClass)

        # ── Custom scripts ────────────────────────────────────────────────────
        for custom_path in self.custom_script_paths:
            custom_path = Path(custom_path)

            if not custom_path.exists():
                print(f"⚠️ Custom script not found, skipping: {custom_path}")
                continue

            unique_name = f"_custom_{custom_path.stem}_{id(custom_path)}"
            try:
                spec = importlib.util.spec_from_file_location(unique_name, custom_path)
                module = importlib.util.module_from_spec(spec)
                sys.modules[unique_name] = module
                spec.loader.exec_module(module)  # type: ignore[union-attr]
            except Exception as exc:
                print(f"⚠️ Custom script {custom_path.name}: import failed ({exc})")
                continue

            ok, reason = validate_module(module)
            if not ok:
                print(f"⚠️ Custom script {custom_path.name}: {reason}")
                continue

            supported = getattr(module, "SUPPORTED_PROBLEM_TYPES", [])
            if "kmeans_clustering" not in supported:
                print(
                    f"⚠️ Custom script {custom_path.name} does not support "
                    "'kmeans_clustering', skipping."
                )
                continue

            ModelClass = getattr(module, "Model", None)
            if ModelClass is not None:
                ModelClass.is_custom = True
                model_classes.append(ModelClass)
                print(f"✅ Loaded custom script: {custom_path.name}")

        return model_classes

    # ── Training ──────────────────────────────────────────────────────────────

    def train_all(
        self,
        X_train,
        y_train=None,
        X_val=None,
        y_val=None,
        n_clusters: int = 3,
    ):
        """
        Train all clustering model scripts.

        Returns:
            dict[str, dict]: model_name → {
                "metrics": {...},
                "metadata": {...},
                "cluster_labels": list | None,
                "is_custom": bool,
            }
        """
        models = self._load_models()
        results = {}

        for ModelClass in models:
            model_name = ModelClass.MODEL_NAME
            save_path = self.output_path / f"{model_name}.joblib"
            is_custom = bool(getattr(ModelClass, "is_custom", False))

            print(f"🚀 Training {model_name}{'  [custom]' if is_custom else ''}...")

            try:
                model = ModelClass()
                pipe, metrics, metadata = model.train_model(
                    X_train=X_train,
                    y_train=None,
                    X_val=None,
                    y_val=None,
                    save_path=save_path,
                    n_clusters=n_clusters,
                )
            except Exception as exc:
                print(f"⚠️ Skipping {model_name}: training failed ({exc})")
                continue

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
                "is_custom": is_custom,
            }

            print(f"✅ Completed {model_name}")

        return results