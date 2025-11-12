from pathlib import Path
from main.fakes.fake_preprocess import fake_preprocess
from main.model_training.orchestrator import Orchestrator


def test_classification_flow_with_raw_data(tmp_path):
    """
    End-to-end integration test:
    - Load actual classification dataset CSV from main/raw_data/
    - Preprocess into processed_data/ folder
    - Run orchestrator to train classification models
    - Assert all models trained successfully
    """

    # ----------------------------------------------------------------------
    # 1. Locate the actual raw CSV inside your project
    # ----------------------------------------------------------------------
    project_root = Path(__file__).resolve().parents[2]
    raw_csv = project_root / "main" / "raw_data" / "Wine_dataset.csv"   # Example classification dataset

    assert raw_csv.exists(), f"Raw data not found at: {raw_csv}"

    # ----------------------------------------------------------------------
    # 2. Preprocess into processed_data/iris_test/
    # ----------------------------------------------------------------------
    processed_dir = tmp_path / "processed_data" / "Wine_dataset_test"
    fake_preprocess(str(raw_csv), str(processed_dir))

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

    # ----------------------------------------------------------------------
    # 3. Run orchestrator (training)
    # ----------------------------------------------------------------------
    orchestrator = Orchestrator(
        dataset_path=processed_dir,
        model_scripts_path=project_root / "main" / "model_scripts",
        output_path=tmp_path / "trained_output"
    )

    results = orchestrator.run()

    # ----------------------------------------------------------------------
    # 4. Check all expected classification models ran
    # ----------------------------------------------------------------------
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
