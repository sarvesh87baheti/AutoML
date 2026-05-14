import importlib
import importlib.util
import pkgutil
import sys
from pathlib import Path
from typing import List

from main.model_scripts.base import validate_module
import logging

# File stems that must never be loaded as model scripts.
_SKIP_STEMS = {"__init__", "base", "utils", "script_validator", "example_custom_model"}


class RegressionTrainer:
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
        Load all regression-compatible model classes.

        1. Scans built-in model_scripts/ via pkgutil (preserving the original
           package-import approach so relative imports inside scripts work).
        2. Appends any user-supplied custom scripts loaded via importlib.util
           (they are not part of the package and must be loaded by file path).

        Custom Model classes are tagged with is_custom = True.
        """
        model_classes = []
        package_name = "main.model_scripts"

        # ── Built-in scripts (pkgutil / package import) ───────────────────────
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
            if "regression" not in supported:
                continue

            ModelClass = getattr(module, "Model", None)
            if ModelClass is not None:
                model_classes.append(ModelClass)

        # ── Custom scripts (file-path import) ─────────────────────────────────
        for custom_path in self.custom_script_paths:
            custom_path = Path(custom_path)

            if not custom_path.exists():
                print(f"⚠️ Custom script not found, skipping: {custom_path}")
                continue

            # Use a collision-safe module name
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
            if "regression" not in supported:
                print(
                    f"⚠️ Custom script {custom_path.name} does not support "
                    "'regression', skipping."
                )
                continue

            ModelClass = getattr(module, "Model", None)
            if ModelClass is not None:
                ModelClass.is_custom = True
                model_classes.append(ModelClass)
                print(f"✅ Loaded custom script: {custom_path.name}")

        return model_classes

    # ── Training ──────────────────────────────────────────────────────────────

    def train_all(self, X_train, y_train, X_val, y_val):
        """
        Train all regression model scripts found in model_scripts/ plus any
        custom scripts.  Each model handles its own saving via save_path.

        Returns:
            dict[str, dict]: model_name → {
                "metrics": {...},
                "metadata": {...},
                "val_predictions": list | None,
                "val_actual": list | None,
                "is_custom": bool,
            }
        """
        models = self._load_models()
        results = {}

        def _extract_weights(pipe):
            """Extract linear coefficients or tree feature importances."""
            try:
                from sklearn.pipeline import Pipeline
            except Exception:
                Pipeline = None

            est = pipe
            try:
                if Pipeline is not None and isinstance(pipe, Pipeline):
                    est = pipe.steps[-1][1]
            except Exception:
                est = pipe

            weights = None
            try:
                if hasattr(est, "coef_"):
                    coef = getattr(est, "coef_")
                    intercept = getattr(est, "intercept_", None)
                    try:
                        coef = coef.tolist()
                    except Exception:
                        pass
                    try:
                        if intercept is not None:
                            intercept = float(intercept)
                    except Exception:
                        pass
                    weights = {"coef": coef, "intercept": intercept}
                elif hasattr(est, "feature_importances_"):
                    fi = getattr(est, "feature_importances_")
                    try:
                        fi = fi.tolist()
                    except Exception:
                        pass
                    weights = {"feature_importances": fi}
            except Exception:
                weights = None

            return weights

        for ModelClass in models:
            model_name = ModelClass.MODEL_NAME
            save_path = self.output_path / f"{model_name}.joblib"
            is_custom = bool(getattr(ModelClass, "is_custom", False))

            print(f"🚀 Training {model_name}{'  [custom]' if is_custom else ''}...")

            try:
                model = ModelClass()
                pipe, metrics, metadata = model.train_model(
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    save_path=save_path,
                )
            except Exception as exc:
                print(f"⚠️ Skipping {model_name}: training failed ({exc})")
                continue

            # Validation predictions for Actual vs Predicted visualisation
            try:
                val_preds = pipe.predict(X_val).tolist()
                val_actual = y_val.tolist()
            except Exception:
                val_preds = None
                val_actual = None

            results[model_name] = {
                "metrics": metrics,
                "metadata": metadata,
                "val_predictions": val_preds,
                "val_actual": val_actual,
                "is_custom": is_custom,
            }
            print(f"✅ Completed {model_name}")

            print(f"✅ Completed {model_name}")

        return results