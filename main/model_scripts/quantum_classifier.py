from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler

from main.model_scripts.base import ModelScript
from main.model_scripts.utils import _ensure_array, evaluate_classification_model

MODEL_NAME = "quantum_classifier"
SUPPORTED_PROBLEM_TYPES = ["classification"]


def _load_quantum_dependencies():
    from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
    from qiskit_machine_learning.algorithms import QSVC, VQC
    from qiskit_machine_learning.kernels import FidelityQuantumKernel
    from qiskit_algorithms.optimizers import COBYLA

    return {
        "ZZFeatureMap": ZZFeatureMap,
        "RealAmplitudes": RealAmplitudes,
        "QSVC": QSVC,
        "VQC": VQC,
        "FidelityQuantumKernel": FidelityQuantumKernel,
        "COBYLA": COBYLA,
    }


class QuantumClassificationWrapper:
    """Small serializable wrapper so the quantum model behaves like the other models."""

    def __init__(self, estimator, scaler, feature_indices, selected_model: str):
        self.estimator = estimator
        self.scaler = scaler
        self.feature_indices = np.asarray(feature_indices, dtype=int)
        self.selected_model = selected_model

    def _transform(self, X):
        X = _ensure_array(X)
        X = self.scaler.transform(X)
        return X[:, self.feature_indices]

    def predict(self, X):
        return self.estimator.predict(self._transform(X))


def _prepare_features(X_train: np.ndarray, X_other: Optional[np.ndarray], max_qubits: int):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    n_features = X_train_scaled.shape[1]
    selected_count = min(max_qubits, n_features)
    feature_variances = np.var(X_train_scaled, axis=0)
    feature_indices = np.argsort(feature_variances)[::-1][:selected_count]

    X_train_selected = X_train_scaled[:, feature_indices]
    X_other_selected = None
    if X_other is not None:
        X_other = scaler.transform(X_other)
        X_other_selected = X_other[:, feature_indices]

    return X_train_selected, X_other_selected, scaler, feature_indices


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    save_path: Optional[Path] = None,
    scale: bool = True,
    max_qubits: int = 3,
    vqc_maxiter: int = 2,
    include_vqc: bool = False,
    **kwargs,
) -> Tuple[QuantumClassificationWrapper, Dict[str, Dict[str, float]], Dict[str, Any]]:
    if not scale:
        raise ValueError("quantum_classifier requires scaled features.")

    if max_qubits < 1:
        raise ValueError("max_qubits must be at least 1.")

    X_train = _ensure_array(X_train)
    y_train = _ensure_array(y_train)
    X_val = None if X_val is None else _ensure_array(X_val)
    y_val = None if y_val is None else _ensure_array(y_val)

    deps = _load_quantum_dependencies()
    X_train_prepared, X_val_prepared, scaler, feature_indices = _prepare_features(
        X_train=X_train,
        X_other=X_val,
        max_qubits=max_qubits,
    )

    feature_map = deps["ZZFeatureMap"](feature_dimension=X_train_prepared.shape[1], reps=1)
    candidates = []

    qsvc = deps["QSVC"](quantum_kernel=deps["FidelityQuantumKernel"](feature_map=feature_map))
    qsvc.fit(X_train_prepared, y_train)
    qsvc_wrapper = QuantumClassificationWrapper(qsvc, scaler, feature_indices, selected_model="QSVC")
    qsvc_metrics = {"train": evaluate_classification_model(qsvc_wrapper, X_train, y_train)}
    if X_val is not None and y_val is not None:
        qsvc_metrics["val"] = evaluate_classification_model(qsvc_wrapper, X_val, y_val)
    candidates.append(("QSVC", qsvc_wrapper, qsvc_metrics))

    if include_vqc:
        optimizer = deps["COBYLA"](maxiter=vqc_maxiter)
        ansatz = deps["RealAmplitudes"](num_qubits=X_train_prepared.shape[1], reps=1)
        try:
            vqc = deps["VQC"](
                feature_map=feature_map,
                ansatz=ansatz,
                optimizer=optimizer,
            )
            vqc.fit(X_train_prepared, y_train)
            vqc_wrapper = QuantumClassificationWrapper(vqc, scaler, feature_indices, selected_model="VQC")
            vqc_metrics = {"train": evaluate_classification_model(vqc_wrapper, X_train, y_train)}
            if X_val is not None and y_val is not None:
                vqc_metrics["val"] = evaluate_classification_model(vqc_wrapper, X_val, y_val)
            candidates.append(("VQC", vqc_wrapper, vqc_metrics))
        except Exception as exc:
            kwargs = {**kwargs, "vqc_status": f"skipped: {exc}"}

    score_split = "val" if X_val is not None and y_val is not None else "train"
    best_name, best_model, best_metrics = max(
        candidates,
        key=lambda item: item[2][score_split].get("f1", item[2][score_split].get("accuracy", 0.0)),
    )

    metadata = {
        "name": MODEL_NAME,
        "selected_quantum_model": best_name,
        "candidate_models": [name for name, _, _ in candidates],
        "hyperparams": {
            "max_qubits": int(max_qubits),
            "vqc_maxiter": int(vqc_maxiter),
            "include_vqc": bool(include_vqc),
            **kwargs,
        },
        "train_samples": int(len(X_train)),
        "selected_feature_indices": feature_indices.astype(int).tolist(),
        "selected_feature_count": int(len(feature_indices)),
    }

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        import joblib
        joblib.dump(best_model, save_path)

    return best_model, best_metrics, metadata


class Model(ModelScript):
    MODEL_NAME = MODEL_NAME
    SUPPORTED_PROBLEM_TYPES = tuple(SUPPORTED_PROBLEM_TYPES)

    def train_model(self, X_train, y_train, X_val=None, y_val=None, save_path=None, scale=True, **kwargs):
        return train_model(
            X_train,
            y_train,
            X_val=X_val,
            y_val=y_val,
            save_path=save_path,
            scale=scale,
            **kwargs,
        )
