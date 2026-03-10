from pathlib import Path

import pandas as pd

from main.model_training.orchestrator import Orchestrator
from main.preprocessing.datacleaning import clean_dataframe
from main.preprocessing.preprocessor import process_features


def test_kmeans_flow_with_raw_data(tmp_path):
    project_root = Path(__file__).resolve().parents[2]
    raw_csv = project_root / "main" / "raw_data" / "Wine_dataset.csv"

    assert raw_csv.exists(), f"Raw data not found at: {raw_csv}"

    processed_dir = tmp_path / "processed_data" / "Wine_dataset_kmeans_test"
    df = pd.read_csv(raw_csv)
    df = clean_dataframe(df)
    process_features(
        df,
        target_col=None,
        problem_type="kmeans_clustering",
        save_dir=str(processed_dir),
        apply_pca=False,
        n_clusters=3,
    )

    orchestrator = Orchestrator(
        dataset_path=processed_dir,
        model_scripts_path=project_root / "main" / "model_scripts",
        output_path=tmp_path / "trained_output",
    )

    results = orchestrator.run()

    assert "kmeans" in results
    assert "metrics" in results["kmeans"]
    assert "train" in results["kmeans"]["metrics"]
    assert "silhouette" in results["kmeans"]["metrics"]["train"]
    assert "cluster_labels" in results["kmeans"]
