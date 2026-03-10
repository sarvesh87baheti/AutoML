from pathlib import Path

import pandas as pd

from main.model_training.orchestrator import Orchestrator
from main.preprocessing.datacleaning import clean_dataframe
from main.preprocessing.preprocessor import process_features


def test_classification_flow_with_raw_data(tmp_path):
    """
    End-to-end integration test:
    - Load actual classification dataset CSV from main/raw_data/
    - Preprocess into processed_data/ folder
    - Run orchestrator to train classification models
    - Assert all models trained successfully
    """

    project_root = Path(__file__).resolve().parents[2]
    raw_csv = project_root / "main" / "raw_data" / "Wine_dataset.csv"   # Example classification dataset

    assert raw_csv.exists(), f"Raw data not found at: {raw_csv}"

    # Preprocess into tmp_path (mirrors runner.py)
    processed_dir = tmp_path / "processed_data" / "Wine_dataset_test"
    df = pd.read_csv(raw_csv)
    df = clean_dataframe(df)
    process_features(
        df,
        target_col="class",
        problem_type="classification",
        save_dir=str(processed_dir),
        apply_pca=False,
    )

    # Validate preprocessing output
    required_files = [
        "X_train.npy",
        "y_train.npy",
        "X_val.npy",
        "y_val.npy",
        "metadata.json",
    ]
    for f in required_files:
        assert (processed_dir / f).exists(), f"Missing file: {f}"

    # Run orchestrator (training)
    orchestrator = Orchestrator(
        dataset_path=processed_dir,
        model_scripts_path=project_root / "main" / "model_scripts",
        output_path=tmp_path / "trained_output",
    )

    results = orchestrator.run()

    # Check all expected classification models ran
    expected_models = {"logistic", "randomforest", "svm", "knn"}

    for model_name in expected_models:
        assert model_name in results, f"{model_name} not trained!"

        # Metrics structure
        assert "metrics" in results[model_name]
        assert "train" in results[model_name]["metrics"]

        train_metrics = results[model_name]["metrics"]["train"]
        for key in ("accuracy", "precision", "recall", "f1"):
            assert key in train_metrics, f"{key} missing in metrics for {model_name}"
            assert isinstance(train_metrics[key], float)

        # Metadata structure
        assert "metadata" in results[model_name]
        metadata = results[model_name]["metadata"]
        assert "name" in metadata
        assert metadata["name"] == model_name

    print("✅ Full classification AutoML flow using real raw_data PASSED.")
