"""
tests/test_phase1_custom_scripts.py
------------------------------------
Unit and integration tests for the Phase 1 custom-scripts feature.

Run with:
    python -m pytest tests/test_phase1_custom_scripts.py -v

The tests work entirely with in-memory temporary files and do NOT require a
trained dataset — they stub out the parts of the pipeline that touch disk
(preprocessing, feature importance, etc.) so they run quickly in CI.
"""

from __future__ import annotations

import json
import sys
import textwrap
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers — build minimal valid / invalid script files in a tmp dir
# ---------------------------------------------------------------------------

VALID_CLASSIFICATION_SCRIPT = textwrap.dedent(
    """\
    import joblib
    import numpy as np
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.pipeline import Pipeline

    MODEL_NAME = "test_decision_tree"
    SUPPORTED_PROBLEM_TYPES = ("classification",)

    class ModelScript:
        pass

    class Model(ModelScript):
        MODEL_NAME = "test_decision_tree"
        SUPPORTED_PROBLEM_TYPES = ("classification",)

        def train_model(self, X_train, y_train, X_val=None, y_val=None,
                        save_path=None, scale=True, **kwargs):
            clf = DecisionTreeClassifier(max_depth=3, random_state=0)
            clf.fit(X_train, y_train)
            pipe = Pipeline([("clf", clf)])
            metrics = {"accuracy": float((clf.predict(X_val) == y_val).mean())}
            metadata = {"model": "test_decision_tree"}
            if save_path:
                joblib.dump(pipe, save_path)
            return pipe, metrics, metadata
    """
)

VALID_REGRESSION_SCRIPT = textwrap.dedent(
    """\
    import joblib
    import numpy as np
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline

    MODEL_NAME = "test_ridge"
    SUPPORTED_PROBLEM_TYPES = ("regression",)

    class ModelScript:
        pass

    class Model(ModelScript):
        MODEL_NAME = "test_ridge"
        SUPPORTED_PROBLEM_TYPES = ("regression",)

        def train_model(self, X_train, y_train, X_val=None, y_val=None,
                        save_path=None, scale=True, **kwargs):
            model = Ridge()
            model.fit(X_train, y_train)
            pipe = Pipeline([("ridge", model)])
            from sklearn.metrics import r2_score, mean_squared_error
            preds = model.predict(X_val)
            metrics = {"r2": float(r2_score(y_val, preds)),
                       "mse": float(mean_squared_error(y_val, preds))}
            metadata = {"model": "test_ridge"}
            if save_path:
                joblib.dump(pipe, save_path)
            return pipe, metrics, metadata
    """
)

VALID_CLUSTERING_SCRIPT = textwrap.dedent(
    """\
    import joblib
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.pipeline import Pipeline

    MODEL_NAME = "test_kmeans_custom"
    SUPPORTED_PROBLEM_TYPES = ("kmeans_clustering",)

    class ModelScript:
        pass

    class Model(ModelScript):
        MODEL_NAME = "test_kmeans_custom"
        SUPPORTED_PROBLEM_TYPES = ("kmeans_clustering",)

        def train_model(self, X_train, y_train=None, X_val=None, y_val=None,
                        save_path=None, scale=True, n_clusters=3, **kwargs):
            km = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto")
            km.fit(X_train)
            pipe = Pipeline([("km", km)])
            metrics = {"inertia": float(km.inertia_)}
            metadata = {"model": "test_kmeans_custom"}
            if save_path:
                joblib.dump(pipe, save_path)
            return pipe, metrics, metadata
    """
)

MISSING_MODEL_NAME_SCRIPT = textwrap.dedent(
    """\
    SUPPORTED_PROBLEM_TYPES = ("classification",)
    class Model:
        def train_model(self, *a, **kw): pass
    """
)

MISSING_MODEL_CLASS_SCRIPT = textwrap.dedent(
    """\
    MODEL_NAME = "no_class"
    SUPPORTED_PROBLEM_TYPES = ("classification",)
    """
)

WRONG_PROBLEM_TYPE_SCRIPT = textwrap.dedent(
    """\
    MODEL_NAME = "only_regression"
    SUPPORTED_PROBLEM_TYPES = ("regression",)
    class ModelScript: pass
    class Model(ModelScript):
        MODEL_NAME = "only_regression"
        SUPPORTED_PROBLEM_TYPES = ("regression",)
        def train_model(self, *a, **kw): pass
    """
)

SYNTAX_ERROR_SCRIPT = "def broken(: pass"

LARGE_SCRIPT = "# " + "x" * 120 + "\n"  # 123 bytes per line
LARGE_SCRIPT = LARGE_SCRIPT * 900  # ~110 KB — exceeds the 100 KB limit


def _write(tmp_path: Path, filename: str, content: str) -> Path:
    p = tmp_path / filename
    p.write_text(content, encoding="utf-8")
    return p


# ===========================================================================
# Tests for script_validator.validate_custom_script
# ===========================================================================

class TestValidateCustomScript:
    """Tests for main/model_scripts/script_validator.py"""

    def _validator(self):
        """Import the module under test (avoids top-level package requirement)."""
        spec_path = Path(__file__).parent.parent / "main" / "model_scripts" / "script_validator.py"
        import importlib.util
        spec = importlib.util.spec_from_file_location("script_validator", spec_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_valid_classification_script(self, tmp_path):
        p = _write(tmp_path, "good_clf.py", VALID_CLASSIFICATION_SCRIPT)
        v = self._validator()
        ok, reason = v.validate_custom_script(p, "classification")
        assert ok, f"Expected valid, got: {reason}"
        assert reason == "ok"

    def test_valid_regression_script(self, tmp_path):
        p = _write(tmp_path, "good_reg.py", VALID_REGRESSION_SCRIPT)
        v = self._validator()
        ok, reason = v.validate_custom_script(p, "regression")
        assert ok, reason

    def test_valid_clustering_script(self, tmp_path):
        p = _write(tmp_path, "good_km.py", VALID_CLUSTERING_SCRIPT)
        v = self._validator()
        ok, reason = v.validate_custom_script(p, "kmeans_clustering")
        assert ok, reason

    def test_file_not_found(self, tmp_path):
        v = self._validator()
        ok, reason = v.validate_custom_script(tmp_path / "ghost.py", "classification")
        assert not ok
        assert "not found" in reason

    def test_non_py_extension(self, tmp_path):
        p = _write(tmp_path, "model.txt", "nothing")
        v = self._validator()
        ok, reason = v.validate_custom_script(p, "classification")
        assert not ok
        assert "not a Python file" in reason

    def test_file_too_large(self, tmp_path):
        p = _write(tmp_path, "huge.py", LARGE_SCRIPT)
        v = self._validator()
        ok, reason = v.validate_custom_script(p, "classification")
        assert not ok
        assert "too large" in reason

    def test_syntax_error(self, tmp_path):
        p = _write(tmp_path, "broken.py", SYNTAX_ERROR_SCRIPT)
        v = self._validator()
        ok, reason = v.validate_custom_script(p, "classification")
        assert not ok
        assert "syntax error" in reason.lower() or "import error" in reason.lower()

    def test_missing_model_name(self, tmp_path):
        p = _write(tmp_path, "no_name.py", MISSING_MODEL_NAME_SCRIPT)
        v = self._validator()
        ok, reason = v.validate_custom_script(p, "classification")
        assert not ok
        assert "MODEL_NAME" in reason

    def test_missing_model_class(self, tmp_path):
        p = _write(tmp_path, "no_class.py", MISSING_MODEL_CLASS_SCRIPT)
        v = self._validator()
        ok, reason = v.validate_custom_script(p, "classification")
        assert not ok
        assert "Model" in reason

    def test_wrong_problem_type(self, tmp_path):
        p = _write(tmp_path, "wrong_type.py", WRONG_PROBLEM_TYPE_SCRIPT)
        v = self._validator()
        ok, reason = v.validate_custom_script(p, "classification")
        assert not ok
        assert "classification" in reason

    def test_reserved_name_rejected(self, tmp_path):
        p = _write(tmp_path, "base.py", VALID_CLASSIFICATION_SCRIPT)
        v = self._validator()
        ok, reason = v.validate_custom_script(p, "classification")
        assert not ok
        assert "reserved" in reason

    def test_idempotent_multiple_calls(self, tmp_path):
        """Calling the validator twice on the same file should not raise."""
        p = _write(tmp_path, "idempotent.py", VALID_CLASSIFICATION_SCRIPT)
        v = self._validator()
        ok1, _ = v.validate_custom_script(p, "classification")
        ok2, _ = v.validate_custom_script(p, "classification")
        assert ok1 and ok2


# ===========================================================================
# Tests for ClassificationTrainer with custom scripts
# ===========================================================================

class TestClassificationTrainerCustomScripts:

    def _make_trainer(self, scripts_path, output_path, custom_paths=None):
        spec_path = Path(__file__).parent.parent / "main" / "model_training" / "classification.py"
        import importlib.util as ilu
        # Provide a stub for the package-level import
        base_stub = types.ModuleType("main.model_scripts.base")
        base_stub.validate_module = _stub_validate_module
        sys.modules.setdefault("main", types.ModuleType("main"))
        sys.modules.setdefault("main.model_scripts", types.ModuleType("main.model_scripts"))
        sys.modules["main.model_scripts.base"] = base_stub

        spec = ilu.spec_from_file_location("classification_trainer", spec_path)
        mod = ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.ClassificationTrainer(scripts_path, output_path, custom_paths)

    def test_custom_script_loaded_and_tagged(self, tmp_path):
        p = _write(tmp_path, "my_clf.py", VALID_CLASSIFICATION_SCRIPT)
        trainer = self._make_trainer(
            scripts_path=tmp_path / "builtin",   # empty dir
            output_path=tmp_path / "out",
            custom_paths=[p],
        )
        (tmp_path / "builtin").mkdir(exist_ok=True)
        (tmp_path / "out").mkdir(exist_ok=True)

        models = trainer._load_models()
        assert len(models) == 1
        assert getattr(models[0], "is_custom", False) is True

    def test_invalid_custom_script_skipped(self, tmp_path):
        p = _write(tmp_path, "broken.py", MISSING_MODEL_NAME_SCRIPT)
        trainer = self._make_trainer(
            scripts_path=tmp_path / "builtin",
            output_path=tmp_path / "out",
            custom_paths=[p],
        )
        (tmp_path / "builtin").mkdir(exist_ok=True)
        (tmp_path / "out").mkdir(exist_ok=True)

        models = trainer._load_models()
        assert len(models) == 0

    def test_nonexistent_custom_path_skipped(self, tmp_path):
        trainer = self._make_trainer(
            scripts_path=tmp_path / "builtin",
            output_path=tmp_path / "out",
            custom_paths=[tmp_path / "ghost.py"],
        )
        (tmp_path / "builtin").mkdir(exist_ok=True)
        models = trainer._load_models()
        assert len(models) == 0

    def test_train_all_is_custom_flag_in_results(self, tmp_path):
        p = _write(tmp_path, "my_clf.py", VALID_CLASSIFICATION_SCRIPT)
        (tmp_path / "builtin").mkdir(exist_ok=True)
        (tmp_path / "out").mkdir(exist_ok=True)

        trainer = self._make_trainer(
            scripts_path=tmp_path / "builtin",
            output_path=tmp_path / "out",
            custom_paths=[p],
        )

        rng = np.random.default_rng(0)
        X = rng.random((60, 4))
        y = rng.integers(0, 2, 60)

        results = trainer.train_all(X[:50], y[:50], X[50:], y[50:])
        assert "test_decision_tree" in results
        assert results["test_decision_tree"]["is_custom"] is True


# ===========================================================================
# Tests for RegressionTrainer with custom scripts
# ===========================================================================

class TestRegressionTrainerCustomScripts:

    def _make_trainer(self, scripts_path, output_path, custom_paths=None):
        spec_path = Path(__file__).parent.parent / "main" / "model_training" / "regression.py"
        import importlib.util as ilu

        base_stub = types.ModuleType("main.model_scripts.base")
        base_stub.validate_module = _stub_validate_module
        sys.modules.setdefault("main", types.ModuleType("main"))
        sys.modules.setdefault("main.model_scripts", types.ModuleType("main.model_scripts"))
        sys.modules["main.model_scripts.base"] = base_stub

        spec = ilu.spec_from_file_location("regression_trainer", spec_path)
        mod = ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.RegressionTrainer(scripts_path, output_path, custom_paths)

    def test_custom_regression_script_loaded(self, tmp_path):
        p = _write(tmp_path, "my_ridge.py", VALID_REGRESSION_SCRIPT)
        (tmp_path / "builtin").mkdir(exist_ok=True)
        (tmp_path / "out").mkdir(exist_ok=True)

        trainer = self._make_trainer(
            scripts_path=tmp_path / "builtin",
            output_path=tmp_path / "out",
            custom_paths=[p],
        )
        models = trainer._load_models()
        assert any(getattr(m, "is_custom", False) for m in models)

    def test_train_all_is_custom_in_result(self, tmp_path):
        p = _write(tmp_path, "my_ridge.py", VALID_REGRESSION_SCRIPT)
        (tmp_path / "builtin").mkdir(exist_ok=True)
        (tmp_path / "out").mkdir(exist_ok=True)

        trainer = self._make_trainer(
            scripts_path=tmp_path / "builtin",
            output_path=tmp_path / "out",
            custom_paths=[p],
        )

        rng = np.random.default_rng(1)
        X = rng.random((60, 4))
        y = rng.random(60)

        results = trainer.train_all(X[:50], y[:50], X[50:], y[50:])
        assert "test_ridge" in results
        assert results["test_ridge"]["is_custom"] is True
        assert "val_predictions" in results["test_ridge"]


# ===========================================================================
# Tests for KMClusteringTrainer with custom scripts
# ===========================================================================

class TestKMClusteringTrainerCustomScripts:

    def _make_trainer(self, scripts_path, output_path, custom_paths=None):
        spec_path = Path(__file__).parent.parent / "main" / "model_training" / "kmclustering.py"
        import importlib.util as ilu

        base_stub = types.ModuleType("main.model_scripts.base")
        base_stub.validate_module = _stub_validate_module
        sys.modules.setdefault("main", types.ModuleType("main"))
        sys.modules.setdefault("main.model_scripts", types.ModuleType("main.model_scripts"))
        sys.modules["main.model_scripts.base"] = base_stub

        spec = ilu.spec_from_file_location("kmclustering_trainer", spec_path)
        mod = ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.KMClusteringTrainer(scripts_path, output_path, custom_paths)

    def test_custom_clustering_script_loaded(self, tmp_path):
        p = _write(tmp_path, "my_km.py", VALID_CLUSTERING_SCRIPT)
        (tmp_path / "builtin").mkdir(exist_ok=True)
        (tmp_path / "out").mkdir(exist_ok=True)

        trainer = self._make_trainer(
            scripts_path=tmp_path / "builtin",
            output_path=tmp_path / "out",
            custom_paths=[p],
        )
        models = trainer._load_models()
        assert any(getattr(m, "is_custom", False) for m in models)

    def test_train_all_is_custom_in_result(self, tmp_path):
        p = _write(tmp_path, "my_km.py", VALID_CLUSTERING_SCRIPT)
        (tmp_path / "builtin").mkdir(exist_ok=True)
        (tmp_path / "out").mkdir(exist_ok=True)

        trainer = self._make_trainer(
            scripts_path=tmp_path / "builtin",
            output_path=tmp_path / "out",
            custom_paths=[p],
        )

        rng = np.random.default_rng(2)
        X = rng.random((60, 4))

        results = trainer.train_all(X, n_clusters=3)
        assert "test_kmeans_custom" in results
        assert results["test_kmeans_custom"]["is_custom"] is True


# ===========================================================================
# Tests for validate_only() in runner.py
# ===========================================================================

class TestRunnerValidateOnly:

    def _runner_mod(self):
        spec_path = Path(__file__).parent.parent / "runner.py"
        import importlib.util as ilu

        # Build stubs for every project module that runner.py imports at the
        # top level, so loading runner.py doesn't require the full package tree.
        def _stub(name):
            m = types.ModuleType(name)
            sys.modules[name] = m
            return m

        # Ensure parent packages exist first
        for pkg in ("main", "main.preprocessing", "main.model_training",
                    "main.final_model_selection", "main.model_scripts"):
            sys.modules.setdefault(pkg, types.ModuleType(pkg))

        # datacleaning
        dc = _stub("main.preprocessing.datacleaning")
        dc.clean_dataframe = lambda df: df

        # preprocessor
        pp = _stub("main.preprocessing.preprocessor")
        pp.process_features = lambda *a, **kw: None

        # orchestrator
        orch = _stub("main.model_training.orchestrator")
        orch.Orchestrator = MagicMock()

        # image classification
        img = _stub("main.model_training.imageclassification_multi_train")
        img.run_training = MagicMock(return_value={})

        # feature importance
        fi = _stub("main.model_training.feature_importance")
        fi.compute_feature_priorities = MagicMock(return_value=[])

        # final model selection
        fms = _stub("main.final_model_selection.final_model_sel")
        fms.compute_model_scores = MagicMock(return_value=("best", {}))

        # script_validator — use the real one from disk
        sv_path = Path(__file__).parent.parent / "main" / "model_scripts" / "script_validator.py"
        sv_spec = ilu.spec_from_file_location("main.model_scripts.script_validator", sv_path)
        sv_mod = ilu.module_from_spec(sv_spec)
        sv_spec.loader.exec_module(sv_mod)
        sys.modules["main.model_scripts.script_validator"] = sv_mod

        spec = ilu.spec_from_file_location("runner_under_test", spec_path)
        mod = ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_validate_only_returns_correct_structure(self, tmp_path):
        p = _write(tmp_path, "dummy.py", "# placeholder")
        runner = self._runner_mod()
        result = runner.validate_only([str(p)], "classification")
        assert "results" in result
        assert len(result["results"]) == 1
        entry = result["results"][0]
        assert entry["filename"] == "dummy.py"
        assert "valid" in entry
        assert "reason" in entry

    def test_validate_only_empty_list(self, tmp_path):
        runner = self._runner_mod()
        result = runner.validate_only([], "classification")
        assert result == {"results": []}


# ===========================================================================
# Shared stub — replaces base.validate_module in test contexts where the
# full package tree is not available
# ===========================================================================

def _stub_validate_module(mod):
    """Minimal inline validate_module for tests that cannot import base.py."""
    if not hasattr(mod, "MODEL_NAME"):
        return False, "missing MODEL_NAME"
    if not hasattr(mod, "SUPPORTED_PROBLEM_TYPES"):
        return False, "missing SUPPORTED_PROBLEM_TYPES"
    if not hasattr(mod, "Model"):
        return False, "missing Model class"
    ModelClass = getattr(mod, "Model")
    if not isinstance(ModelClass, type):
        return False, "Model is not a class"
    if not hasattr(ModelClass, "train_model"):
        return False, "Model class has no train_model method"
    return True, "ok"