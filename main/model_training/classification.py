import importlib.util
import sys
from pathlib import Path
from typing import Dict, Any, List

from main.model_scripts.base import validate_module

# File names that must never be loaded as model scripts regardless of where
# they appear (built-in directory or custom paths).
_SKIP_NAMES = {"__init__.py", "base.py", "utils.py", "script_validator.py", "example_custom_model.py"}


class ClassificationTrainer:
    def __init__(
        self,
        scripts_path: Path,
        output_path: Path,
        custom_script_paths: List[Path] | None = None,
    ):
        """
        Handles automatic discovery, validation, and training of classification
        model scripts, including any user-supplied custom scripts.

        Args:
            scripts_path (Path): Directory containing built-in classification
                model scripts.
            output_path (Path): Directory to save trained model pipelines.
            custom_script_paths (list[Path] | None): Optional list of paths to
                user-uploaded custom model scripts.  Each is validated before
                being added to the training run.
        """
        self.scripts_path = scripts_path
        self.output_path = output_path
        self.custom_script_paths: List[Path] = custom_script_paths or []

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _load_module_from_path(self, f: Path) -> object | None:
        """Load a single .py file as a module.  Returns None on failure."""
        try:
            spec = importlib.util.spec_from_file_location(f.stem, f)
            module = importlib.util.module_from_spec(spec)
            sys.modules[f.stem] = module
            spec.loader.exec_module(module)
            return module
        except Exception as exc:
            print(f"⚠️ Skipping {f.name}: import failed ({exc})")
            return None

    # ── Model discovery ───────────────────────────────────────────────────────

    def _load_models(self) -> list:
        """
        Discover and validate all compatible classification model scripts.

        1. Scans the built-in model_scripts directory.
        2. Appends any user-supplied custom scripts (already validated by
           script_validator, but we re-check here for defence-in-depth).

        Custom Model classes are tagged with is_custom = True so downstream
        result aggregation can distinguish them.
        """
        model_classes = []

        # ── Built-in scripts ──────────────────────────────────────────────────
        for f in self.scripts_path.glob("*.py"):
            if f.name in _SKIP_NAMES:
                continue

            module = self._load_module_from_path(f)
            if module is None:
                continue

            ok, reason = validate_module(module)
            if not ok:
                print(f"⚠️ Skipping {f.name}: {reason}")
                continue

            if "classification" in module.SUPPORTED_PROBLEM_TYPES:
                model_classes.append(module.Model)

        # ── Custom scripts ────────────────────────────────────────────────────
        for custom_path in self.custom_script_paths:
            custom_path = Path(custom_path)

            if not custom_path.exists():
                print(f"⚠️ Custom script not found, skipping: {custom_path}")
                continue

            module = self._load_module_from_path(custom_path)
            if module is None:
                continue

            ok, reason = validate_module(module)
            if not ok:
                print(f"⚠️ Skipping custom script {custom_path.name}: {reason}")
                continue

            supported = getattr(module, "SUPPORTED_PROBLEM_TYPES", ())
            if "classification" not in supported:
                print(
                    f"⚠️ Custom script {custom_path.name} does not support "
                    "'classification', skipping."
                )
                continue

            # Tag the class so train_all() can mark the result as custom.
            ModelClass = module.Model
            ModelClass.is_custom = True
            model_classes.append(ModelClass)
            print(f"✅ Loaded custom script: {custom_path.name}")

        return model_classes

    # ── Training ──────────────────────────────────────────────────────────────

    def train_all(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train all discovered classification models and save their pipelines.

        Args:
            X_train, y_train: Training data and labels.
            X_val, y_val: Optional validation data.

        Returns:
            dict[str, dict]: model_name → {
                "metrics": {...},
                "metadata": {...},
                "is_custom": bool,
            }
        """
        models = self._load_models()
        results = {}

        for ModelClass in models:
            model_name = ModelClass.MODEL_NAME
            save_path = self.output_path / f"{model_name}.joblib"
            is_custom = bool(getattr(ModelClass, "is_custom", False))

            model_obj = ModelClass()
            print(f"🚀 Training {model_name}{'  [custom]' if is_custom else ''}...")

            try:
                pipe, metrics, metadata = model_obj.train_model(
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    save_path=save_path,
                )
            except Exception as exc:
                print(f"⚠️ Skipping {model_name}: training failed ({exc})")
                continue

            results[model_name] = {
                "metrics": metrics,
                "metadata": metadata,
                "is_custom": is_custom,
            }

            print(f"✅ Completed {model_name}")

        return results