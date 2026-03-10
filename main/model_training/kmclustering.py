import importlib
import pkgutil
from pathlib import Path

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

            results[model_name] = {
                "metrics": metrics,
                "metadata": metadata,
                "cluster_labels": labels,
            }

        return results
