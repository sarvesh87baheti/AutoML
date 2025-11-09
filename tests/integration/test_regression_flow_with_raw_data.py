from pathlib import Path
from main.fakes.fake_preprocess import fake_preprocess
from main.model_training.orchestrator import Orchestrator


def test_regression_flow_with_raw_data(tmp_path):
    """
    End-to-end integration test:
    - Load actual advertising.csv from main/raw_data/
    - Preprocess into processed_data/ folder
    - Run orchestrator to train regression models
    - Assert all models trained successfully
    """

    # ----------------------------------------------------------------------
    # 1. Locate the actual raw CSV inside your project
    # ----------------------------------------------------------------------
    project_root = Path(__file__).resolve().parents[2]
    raw_csv = project_root / "main" / "raw_data" / "advertising.csv"

    assert raw_csv.exists(), f"Raw data not found at: {raw_csv}"

    # ----------------------------------------------------------------------
    # 2. Preprocess into processed_data/advertising_test/
    # ----------------------------------------------------------------------
    processed_dir = tmp_path / "processed_data" / "advertising_test"
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
    # 4. Check all expected regression models ran
    # ----------------------------------------------------------------------
    expected_models = {"linear", "ridge", "lasso", "elasticnet"}

    for model_name in expected_models:
        assert model_name in results, f"{model_name} not trained!"

        # Metrics structure
        assert "metrics" in results[model_name]
        assert "train" in results[model_name]["metrics"]
        assert "mse" in results[model_name]["metrics"]["train"]
        assert isinstance(results[model_name]["metrics"]["train"]["mse"], float)

        # Metadata
        assert "metadata" in results[model_name]
        assert "name" in results[model_name]["metadata"]

    print("✅ Full regression AutoML flow using real raw_data PASSED.")
