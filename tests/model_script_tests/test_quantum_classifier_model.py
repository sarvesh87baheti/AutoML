import sys
import types

import numpy as np

from main.model_scripts.quantum_classifier import Model, MODEL_NAME


def test_quantum_classifier_model_training_with_stubbed_qiskit(tmp_path, monkeypatch):
    class FakeFeatureMap:
        def __init__(self, feature_dimension, reps=1):
            self.feature_dimension = feature_dimension
            self.reps = reps

    class FakeAnsatz:
        def __init__(self, num_qubits, reps=1):
            self.num_qubits = num_qubits
            self.reps = reps

    class FakeOptimizer:
        def __init__(self, maxiter=25):
            self.maxiter = maxiter

    class FakeKernel:
        def __init__(self, feature_map=None):
            self.feature_map = feature_map

    class FakeQSVC:
        def __init__(self, quantum_kernel=None):
            self.quantum_kernel = quantum_kernel

        def fit(self, X, y):
            self.majority_class = int(np.bincount(y).argmax())
            return self

        def predict(self, X):
            return np.full(len(X), self.majority_class, dtype=int)

    class FakeVQC(FakeQSVC):
        def __init__(self, feature_map=None, ansatz=None, optimizer=None):
            self.feature_map = feature_map
            self.ansatz = ansatz
            self.optimizer = optimizer

    monkeypatch.setitem(sys.modules, "qiskit", types.ModuleType("qiskit"))
    monkeypatch.setitem(sys.modules, "qiskit.circuit", types.ModuleType("qiskit.circuit"))
    monkeypatch.setitem(sys.modules, "qiskit.circuit.library", types.ModuleType("qiskit.circuit.library"))
    monkeypatch.setitem(sys.modules, "qiskit_machine_learning", types.ModuleType("qiskit_machine_learning"))
    monkeypatch.setitem(sys.modules, "qiskit_machine_learning.algorithms", types.ModuleType("qiskit_machine_learning.algorithms"))
    monkeypatch.setitem(sys.modules, "qiskit_machine_learning.kernels", types.ModuleType("qiskit_machine_learning.kernels"))
    monkeypatch.setitem(sys.modules, "qiskit_algorithms", types.ModuleType("qiskit_algorithms"))
    monkeypatch.setitem(sys.modules, "qiskit_algorithms.optimizers", types.ModuleType("qiskit_algorithms.optimizers"))

    sys.modules["qiskit.circuit.library"].ZZFeatureMap = FakeFeatureMap
    sys.modules["qiskit.circuit.library"].RealAmplitudes = FakeAnsatz
    sys.modules["qiskit_machine_learning.algorithms"].QSVC = FakeQSVC
    sys.modules["qiskit_machine_learning.algorithms"].VQC = FakeVQC
    sys.modules["qiskit_machine_learning.kernels"].FidelityQuantumKernel = FakeKernel
    sys.modules["qiskit_algorithms.optimizers"].COBYLA = FakeOptimizer

    X_train = np.array([
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.2, 0.1, 0.4, 0.3, 0.6],
        [0.9, 0.8, 0.7, 0.6, 0.5],
        [0.8, 0.9, 0.6, 0.7, 0.4],
    ])
    y_train = np.array([0, 0, 1, 1])
    X_val = np.array([
        [0.15, 0.25, 0.35, 0.45, 0.55],
        [0.85, 0.75, 0.65, 0.55, 0.45],
    ])
    y_val = np.array([0, 1])

    model_script = Model()
    save_file = tmp_path / "quantum_model.joblib"

    model, metrics, metadata = model_script.train_model(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        save_path=save_file,
        max_qubits=3,
        vqc_maxiter=2,
    )

    assert model is not None
    assert hasattr(model, "predict")
    preds = model.predict(X_val)
    assert preds.shape == y_val.shape

    assert "train" in metrics
    assert "val" in metrics
    assert metadata["name"] == MODEL_NAME
    assert metadata["selected_feature_count"] == 3
    assert metadata["selected_quantum_model"] in {"QSVC", "VQC"}
    assert save_file.exists()
