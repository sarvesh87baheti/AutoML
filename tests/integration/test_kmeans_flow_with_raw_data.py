from pathlib import Path

from runner import run_pipeline


def test_kmeans_flow_with_raw_data():
    project_root = Path(__file__).resolve().parents[2]
    raw_csv = project_root / "main" / "raw_data" / "Wine_dataset.csv"

    assert raw_csv.exists(), f"Raw data not found at: {raw_csv}"

    results = run_pipeline(
        file_path=str(raw_csv),
        problem_type="kmeans_clustering",
        target_col=None,
        n_clusters=3,
    )

    assert "kmeans" in results
    assert "metrics" in results["kmeans"]
    assert "train" in results["kmeans"]["metrics"]
    assert "silhouette" in results["kmeans"]["metrics"]["train"]
    assert "cluster_labels" in results["kmeans"]
