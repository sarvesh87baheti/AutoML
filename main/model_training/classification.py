import importlib.util
from pathlib import Path
from typing import Dict, Any

from main.model_scripts.base import validate_module


class ClassificationTrainer:
    def __init__(self, scripts_path: Path, output_path: Path):
        """
        Handles automatic discovery, validation, and training of classification model scripts.

        Args:
            scripts_path (Path): Directory containing classification model scripts.
            output_path (Path): Directory to save trained model pipelines.
        """
        self.scripts_path = scripts_path
        self.output_path = output_path

    def _load_models(self):
        """Dynamically discover and validate all compatible classification model scripts."""
        model_classes = []

        for f in self.scripts_path.glob("*.py"):
            if f.name in ("__init__.py", "base.py", "utils.py"):
                continue

            spec = importlib.util.spec_from_file_location(f.stem, f)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            ok, reason = validate_module(module)
            if not ok:
                print(f"⚠️ Skipping {f.name}: {reason}")
                continue

            if "classification" in module.SUPPORTED_PROBLEM_TYPES:
                model_classes.append(module.Model)

        return model_classes

    def train_all(self, X_train, y_train, X_val=None, y_val=None):
        """
        Trains all discovered classification models and saves their pipelines.

        Args:
            X_train, y_train: Training data and labels.
            X_val, y_val: Optional validation data.

        Returns:
            Dict[str, Dict[str, Any]]: model_name → {"metrics": {...}, "metadata": {...}}
        """
        models = self._load_models()
        results = {}

        for ModelClass in models:
            model_name = ModelClass.MODEL_NAME
            save_path = self.output_path / f"{model_name}.joblib"

            model_obj = ModelClass()
            print(f"🚀 Training {model_name}...")

            pipe, metrics, metadata = model_obj.train_model(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                save_path=save_path
            )

            results[model_name] = {
                "metrics": metrics,
                "metadata": metadata
            }

            print(f"✅ Completed {model_name}")

        return results
