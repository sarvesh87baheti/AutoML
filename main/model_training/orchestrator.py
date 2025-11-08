from __future__ import annotations

from pathlib import Path
import importlib
import pkgutil
from typing import Any, Dict, List, Optional, Tuple
from importlib import import_module

from main.model_scripts.base import validate_module

import joblib
import numpy as np


PACKAGE = "main.model_scripts"
DEFAULT_EXCLUDE = {"utils", "base", "__init__", "regression"}


def discover_model_modules(package: str = PACKAGE, exclude: Optional[set] = None) -> List[Any]:
    """Discover and import modules under the package.

    Returns a list of imported modules. Skips names in `exclude`.
    """
    exclude = exclude or DEFAULT_EXCLUDE
    pkg = importlib.import_module(package)
    modules: List[Any] = []
    for finder, name, ispkg in pkgutil.iter_modules(pkg.__path__):
        if name in exclude:
            continue
        full_name = f"{package}.{name}"
        try:
            mod = importlib.import_module(full_name)
            modules.append(mod)
        except Exception as exc:  # import errors shouldn't stop discovery
            print(f"Warning: failed to import {full_name}: {exc}")
    return modules


def run_models_on_data(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    save_dir: Optional[Path] = None,
    scale: bool = True,
    package: str = PACKAGE,
    exclude: Optional[set] = None,
    strict: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Discover model scripts and run their `train_model` entrypoint.

    Returns (models, metrics, metadata) where:
      - models[name] = fitted pipeline (or None on error)
      - metrics[name] = metrics dict or {'error': str()} on failure
      - metadata[name] = metadata dict returned by module (or empty dict)
    """
    modules = discover_model_modules(package=package, exclude=exclude)
    models: Dict[str, Any] = {}
    metrics: Dict[str, Dict[str, Any]] = {}
    metadata: Dict[str, Dict[str, Any]] = {}

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    for mod in modules:
        # validate module contract
        try:
            ok, reason = validate_module(mod)
            if not ok:
                msg = f"Module {getattr(mod, 'MODEL_NAME', mod.__name__)} failed validation: {reason}"
                if strict:
                    raise ValueError(msg)
                else:
                    print("Warning:", msg)
                    continue
        except Exception as exc:
            if strict:
                raise
            else:
                print("Warning: validation check raised:", exc)
                continue
        try:
            model_name = getattr(mod, "MODEL_NAME", mod.__name__.split(".")[-1])
            supported = getattr(mod, "SUPPORTED_PROBLEM_TYPES", ["regression"]) or ["regression"]
            if "regression" not in supported:
                # skip modules that do not support regression
                continue

            save_path = None
            if save_dir is not None:
                save_path = Path(save_dir) / f"{model_name}.joblib"

            # Prefer a class-based Model implementation
            if not hasattr(mod, "Model"):
                raise AttributeError("module has no Model class")

            ModClass = getattr(mod, "Model")
            inst = ModClass()
            result = inst.train_model(
                X_train, y_train, X_val=X_val, y_val=y_val, save_path=save_path, scale=scale
            )

            # support both (model, metrics) and (model, metrics, metadata)
            if isinstance(result, tuple) and len(result) == 3:
                model_obj, metric_obj, meta_obj = result
            elif isinstance(result, tuple) and len(result) == 2:
                model_obj, metric_obj = result
                meta_obj = {}
            else:
                # unexpected return value
                model_obj = None
                metric_obj = {"error": f"unexpected return from {mod.__name__}: {type(result)}"}
                meta_obj = {}

            models[model_name] = model_obj
            metrics[model_name] = metric_obj
            metadata[model_name] = meta_obj or {}

            # If module didn't save the model itself, save here as a fallback
            if save_path is not None and model_obj is not None and not save_path.exists():
                try:
                    joblib.dump(model_obj, save_path)
                except Exception:
                    # best-effort save; ignore failures
                    pass

        except Exception as exc:
            name = getattr(mod, "MODEL_NAME", mod.__name__.split(".")[-1])
            models[name] = None
            metrics[name] = {"error": str(exc)}
            metadata[name] = {}

    return models, metrics, metadata


def _example_run():
    """Run a tiny smoke test on discovered modules using synthetic data."""
    X = np.random.rand(50, 4)
    # create a simple linear target
    y = X @ np.array([1.0, -2.0, 0.5, 0.0]) + 0.1 * np.random.randn(50)

    models, metrics, metadata = run_models_on_data(X, y, save_dir=Path("models_test"))
    print("Completed run. Models:")
    for k, v in metrics.items():
        print(k, v)


if __name__ == "__main__":
    _example_run()
