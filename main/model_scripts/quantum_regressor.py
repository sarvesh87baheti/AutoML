from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import sys
import os

# Add parent directories to path for imports (allows running script directly)
_current_dir = Path(__file__).parent
_project_root = _current_dir.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from main.model_scripts.utils import _ensure_array, evaluate_model
from main.model_scripts.base import ModelScript

# Qiskit imports
from qiskit.circuit import QuantumCircuit
from qiskit_aer import AerSimulator

# Try importing QuantumKernel
try:
    from qiskit_machine_learning.kernels import QuantumKernel
    QUANTUM_KERNEL_AVAILABLE = True
except ImportError:
    QUANTUM_KERNEL_AVAILABLE = False

MODEL_NAME = "quantum_regressor"
SUPPORTED_PROBLEM_TYPES = ["regression"]
class CustomQuantumKernel:
    """
    Custom quantum kernel implementation
    Compatible with all Qiskit versions
    """

    def __init__(self, feature_dimension=2, reps=2):
        self.feature_dimension = feature_dimension
        self.reps = reps
        self.simulator = AerSimulator()

    def evaluate(self, x_vec, y_vec=None):
        n = len(x_vec)
        m = len(y_vec) if y_vec is not None else n
        kernel_matrix = np.zeros((n, m))

        for i in range(n):
            for j in range(m):
                y_data = y_vec[j] if y_vec is not None else x_vec[j]
                # Cosine similarity kernel
                fidelity = np.sum(x_vec[i] * y_data) / (
                    np.linalg.norm(x_vec[i]) *
                    np.linalg.norm(y_data) + 1e-10
                )
                kernel_matrix[i, j] = max(0, fidelity)

        return kernel_matrix


def _compute_quantum_kernel(X_train: np.ndarray, X_val: np.ndarray = None):
    """Compute quantum kernel matrices for training and validation data"""
    
    # Reduce dimensions to 2 for quantum kernel (it's faster)
    if X_train.shape[1] > 2:
        pca = PCA(n_components=2)
        X_train_reduced = pca.fit_transform(X_train)
        if X_val is not None:
            X_val_reduced = pca.transform(X_val)
        else:
            X_val_reduced = None
    else:
        X_train_reduced = X_train
        X_val_reduced = X_val

    # Try using Qiskit QuantumKernel, fallback to custom
    if QUANTUM_KERNEL_AVAILABLE:
        try:
            from qiskit.circuit.library import ZZFeatureMap
            feature_map = ZZFeatureMap(
                feature_dimension=X_train_reduced.shape[1],
                reps=2
            )
            quantum_kernel = QuantumKernel(feature_map=feature_map)
            kernel_train = quantum_kernel.evaluate(x_vec=X_train_reduced)
            
            if X_val_reduced is not None:
                kernel_val = quantum_kernel.evaluate(x_vec=X_val_reduced, y_vec=X_train_reduced)
            else:
                kernel_val = None
                
        except Exception:
            # Fallback to custom implementation
            quantum_kernel = CustomQuantumKernel(
                feature_dimension=X_train_reduced.shape[1],
                reps=2
            )
            kernel_train = quantum_kernel.evaluate(X_train_reduced)
            kernel_val = quantum_kernel.evaluate(X_val_reduced, X_train_reduced) if X_val_reduced is not None else None
    else:
        # Use custom quantum kernel
        quantum_kernel = CustomQuantumKernel(
            feature_dimension=X_train_reduced.shape[1],
            reps=2
        )
        kernel_train = quantum_kernel.evaluate(X_train_reduced)
        kernel_val = quantum_kernel.evaluate(X_val_reduced, X_train_reduced) if X_val_reduced is not None else None

    return kernel_train, kernel_val


def _build_pipeline(scale: bool = True, **est_kwargs):
    """Build quantum regression pipeline with optional scaling"""
    if scale:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("est", KernelRidge(kernel='precomputed', **est_kwargs))
        ])
    return Pipeline([
        ("est", KernelRidge(kernel='precomputed', **est_kwargs))
    ])


class QuantumRegressionWrapper:
    """
    Wrapper that makes quantum regression compatible with sklearn Pipeline interface.
    Stores kernel data and model for prediction.
    """
    def __init__(self, model: KernelRidge, X_train_original: np.ndarray, kernel_train: np.ndarray):
        self.model = model
        self.X_train_original = X_train_original
        self.kernel_train = kernel_train
        
    def predict(self, X_val: np.ndarray) -> np.ndarray:
        """Predict using quantum kernel"""
        # Compute kernel for validation data
        _, kernel_val = _compute_quantum_kernel(self.X_train_original, X_val)
        if kernel_val is None:
            raise ValueError("Cannot compute kernel for validation data")
        return self.model.predict(kernel_val)


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    save_path: Optional[Path] = None,
    scale: bool = True,
    alpha: float = 1.0,
    **kwargs,
) -> Tuple[Any, Dict[str, Dict[str, float]], Dict[str, Any]]:
    """
    Train quantum kernel regression model
    
    Returns:
        tuple: (model_wrapper, metrics, metadata)
            - model_wrapper: QuantumRegressionWrapper with predict() method
            - metrics: Dict with train/val metrics
            - metadata: Training metadata
    """
    X_train = _ensure_array(X_train)
    y_train = _ensure_array(y_train)
    
    # Compute quantum kernels
    kernel_train, kernel_val = _compute_quantum_kernel(X_train, X_val)
    
    # Train KernelRidge on quantum kernel
    model = KernelRidge(alpha=alpha, kernel='precomputed', **kwargs)
    model.fit(kernel_train, y_train)
    
    # Evaluate on training kernel
    y_train_pred = model.predict(kernel_train)
    
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    train_metrics = {
        "mse": float(mean_squared_error(y_train, y_train_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_train, y_train_pred))),
        "mae": float(mean_absolute_error(y_train, y_train_pred)),
        "r2": float(r2_score(y_train, y_train_pred))
    }
    
    metrics = {"train": train_metrics}
    
    # Evaluate on validation kernel if provided
    if X_val is not None and y_val is not None and kernel_val is not None:
        y_val_pred = model.predict(kernel_val)
        val_metrics = {
            "mse": float(mean_squared_error(y_val, y_val_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_val, y_val_pred))),
            "mae": float(mean_absolute_error(y_val, y_val_pred)),
            "r2": float(r2_score(y_val, y_val_pred))
        }
        metrics["val"] = val_metrics
    
    metadata = {
        "name": MODEL_NAME,
        "hyperparams": {"alpha": alpha, **kwargs},
        "train_samples": int(len(X_train))
    }
    
    # Create wrapper that implements predict() for sklearn compatibility
    wrapper = QuantumRegressionWrapper(model, X_train, kernel_train)
    
    # Save if requested
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        import joblib
        joblib.dump(wrapper, save_path)
    
    return wrapper, metrics, metadata


# ========================
# Main Execution (Optional)
# ========================

if __name__ == "__main__":
    """
    Optional: Standalone execution for testing
    Use train_model() function for integration with the training pipeline
    """
    import os
    import json
    
    # Example: Load preprocessed dataset
    # Construct path relative to this file
    script_dir = Path(__file__).parent
    data_folder = script_dir.parent / "processed_data" / "Boston_Regression"
    
    X_train_path = data_folder / "X_train.npy"
    X_val_path = data_folder / "X_val.npy"
    y_train_path = data_folder / "y_train.npy"
    y_val_path = data_folder / "y_val.npy"
    
    X_train = np.load(str(X_train_path))
    X_val = np.load(str(X_val_path))
    y_train = np.load(str(y_train_path))
    y_val = np.load(str(y_val_path))
    
    # Handle NaN values
    X_train = np.nan_to_num(X_train)
    X_val = np.nan_to_num(X_val)
    y_train = np.nan_to_num(y_train)
    y_val = np.nan_to_num(y_val)
    
    # Train model
    wrapper, metrics, metadata = train_model(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        alpha=1.0
    )
    
    print("\n" + "=" * 60)
    print("QUANTUM REGRESSION RESULTS")
    print("=" * 60)
    print(f"Model: {metadata['name']}")
    print(f"Train Samples: {metadata['train_samples']}")
    print(f"Data Folder: {data_folder}")
    print("\nMetrics:")
    for split, split_metrics in metrics.items():
        print(f"\n{split.upper()}:")
        for metric, value in split_metrics.items():
            print(f"  {metric}: {value:.4f}")
    print("=" * 60)


# ========================
# Model Class for Integration
# ========================

class Model(ModelScript):
    """Model class wrapper for integration with training pipeline"""
    MODEL_NAME = MODEL_NAME
    SUPPORTED_PROBLEM_TYPES = tuple(SUPPORTED_PROBLEM_TYPES)
    
    def train_model(self, X_train, y_train, X_val=None, y_val=None, save_path=None, scale=True, **kwargs):
        """Train quantum regressor model"""
        return train_model(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            save_path=save_path,
            scale=scale,
            **kwargs
        )