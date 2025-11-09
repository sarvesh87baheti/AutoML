import importlib
import pkgutil
from pathlib import Path

from main.model_scripts.base import validate_module


class RegressionTrainer:
    def __init__(self, scripts_path: Path, output_path: Path):
        self.scripts_path = scripts_path
        self.output_path = output_path

    def _load_models(self):
        """
        Properly import model scripts as part of the package `main.model_scripts`
        so that relative imports inside those scripts work correctly.
        """
        model_classes = []
        package_name = "main.model_scripts"

        # Discover modules inside model_scripts/ using pkgutil
        for _, module_name, _ in pkgutil.iter_modules([str(self.scripts_path)]):

            # Import module as: main.model_scripts.module_name
            full_module_name = f"{package_name}.{module_name}"
            module = importlib.import_module(full_module_name)

            # Validate module implements required structure
            ok, _ = validate_module(module)
            if not ok:
                continue

            # Filter only regression models
            supported = getattr(module, "SUPPORTED_PROBLEM_TYPES", [])
            if "regression" not in supported:
                continue

            ModelClass = getattr(module, "Model", None)
            if ModelClass is not None:
                model_classes.append(ModelClass)

        return model_classes

    def train_all(self, X_train, y_train, X_val, y_val):
        """
        Train all regression model scripts found in model_scripts/.
        Each model script handles its own saving via save_path.
        """
        models = self._load_models()
        results = {}

        for ModelClass in models:
            model_name = ModelClass.MODEL_NAME
            save_path = self.output_path / f"{model_name}.joblib"

            model = ModelClass()
            pipe, metrics, metadata = model.train_model(
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

        return results
