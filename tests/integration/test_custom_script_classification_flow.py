import importlib
import sys
from pathlib import Path

import pandas as pd

from main.model_scripts.script_validator import validate_custom_script
from main.model_training.orchestrator import Orchestrator
from main.preprocessing.datacleaning import clean_dataframe
from main.preprocessing.preprocessor import process_features


def _restore_real_model_script_base():
    sys.modules.pop("main.model_scripts.base", None)
    base_module = importlib.import_module("main.model_scripts.base")
    assert hasattr(base_module, "ModelScript")


def test_custom_classification_script_trains_and_is_marked_custom(tmp_path):
    project_root = Path(__file__).resolve().parents[2]
    raw_csv = project_root / "main" / "raw_data" / "Wine_dataset.csv"
    custom_script = Path(__file__).with_name("custom_test_script.py")

    assert raw_csv.exists(), f"Raw data not found at: {raw_csv}"
    assert custom_script.exists(), f"Custom script fixture not found at: {custom_script}"

    _restore_real_model_script_base()

    valid, reason = validate_custom_script(custom_script, "classification")
    assert valid, reason

    processed_dir = tmp_path / "processed_data" / "wine_custom_script"
    empty_model_scripts_dir = tmp_path / "empty_model_scripts"
    output_dir = tmp_path / "trained_output"
    empty_model_scripts_dir.mkdir()

    df = clean_dataframe(pd.read_csv(raw_csv))
    process_features(
        df,
        target_col="class",
        problem_type="classification",
        save_dir=str(processed_dir),
        apply_pca=False,
    )

    orchestrator = Orchestrator(
        dataset_path=processed_dir,
        model_scripts_path=empty_model_scripts_dir,
        output_path=output_dir,
        custom_script_paths=[custom_script],
    )

    results = orchestrator.run()

    assert set(results) == {"custom_forest"}

    custom_result = results["custom_forest"]
    assert custom_result["is_custom"] is True
    assert custom_result["metadata"]["name"] == "custom_forest"
    assert custom_result["metadata"]["train_samples"] > 0
    assert (output_dir / "custom_forest.joblib").exists()

    for split in ("train", "val"):
        assert split in custom_result["metrics"]
        for metric in ("accuracy", "precision", "recall", "f1"):
            value = custom_result["metrics"][split][metric]
            assert isinstance(value, float)
            assert 0.0 <= value <= 1.0
