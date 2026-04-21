# runner.py
import argparse
import json
import io
import logging
from contextlib import redirect_stdout
import pandas as pd
from pathlib import Path
import zipfile
import sys, os
import joblib
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

# Make project importable regardless of run context
ROOT = Path(__file__).resolve().parent
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

# Project imports
from main.preprocessing.datacleaning import clean_dataframe
from main.preprocessing.preprocessor import process_features
from main.model_training.orchestrator import Orchestrator
from main.model_training.imageclassification_train import run_training as run_image_classification_training
from main.model_training.feature_importance import compute_feature_priorities
from main.final_model_selection.final_model_sel import compute_model_scores


def run_pipeline(file_path: str, problem_type: str, target_col: str = None, n_clusters: int = None):
    print("\n===============================")
    print("🚀 Starting AutoML Pipeline")
    print("===============================\n")

    dataset_path = Path(file_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"❌ Dataset not found: {file_path}")

    # -------------------------------------------------------
    # HANDLE IMAGE CLASSIFICATION SEPARATELY
    # -------------------------------------------------------
    if problem_type == "image_classification":
        print(f"📂 Loading image dataset: {dataset_path.name}")
        
        if not dataset_path.suffix.lower() == ".zip":
            raise ValueError("❌ Image classification requires a ZIP file with class folders containing images.")
        
        try:
            # Call the dedicated image classification pipeline
            run_image_classification_training(str(dataset_path))

            dataset_name = dataset_path.stem
            result_dir = ROOT / "main" / "model_results" / dataset_name
            summary_path = result_dir / "training_summary.json"
            metrics_path = result_dir / "metrics.json"
            confusion_matrix_path = result_dir / "confusion_matrix.json"

            summary = {
                "problem_type": "image_classification",
                "dataset_name": dataset_name,
                "best_model": "image_classification",
                "results": {
                    "image_classification": {
                        "metrics": {
                            "val": {
                                "accuracy": None,
                                "precision": None,
                                "recall": None,
                                "f1": None,
                            }
                        },
                        "confusion_matrix": None
                    }
                }
            }

            if metrics_path.exists():
                with open(metrics_path, "r") as f:
                    report = json.load(f)
                summary["results"]["image_classification"]["metrics"]["val"] = {
                    "accuracy": report.get("accuracy"),
                    "precision": report.get("macro avg", {}).get("precision") or report.get("weighted avg", {}).get("precision"),
                    "recall": report.get("macro avg", {}).get("recall") or report.get("weighted avg", {}).get("recall"),
                    "f1": report.get("macro avg", {}).get("f1-score") or report.get("weighted avg", {}).get("f1-score"),
                }

            if confusion_matrix_path.exists():
                with open(confusion_matrix_path, "r") as f:
                    summary["results"]["image_classification"]["confusion_matrix"] = json.load(f)

            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=4)

            return summary
        except Exception as e:
            raise ValueError(f"❌ Image classification training failed: {str(e)}")

    # -------------------------------------------------------
    # STANDARD PIPELINE FOR REGRESSION, CLASSIFICATION, KMEANS
    # -------------------------------------------------------
    dataset_name = dataset_path.stem
    project_root = ROOT / "main"

    # -------------------------------------------------------
    # 1) LOAD DATASET
    # -------------------------------------------------------
    print(f"📂 Loading dataset: {dataset_path.name}")

    if dataset_path.suffix.lower() == ".csv":
        df = pd.read_csv(dataset_path)
    elif dataset_path.suffix.lower() in [".xls", ".xlsx"]:
        df = pd.read_excel(dataset_path)
    elif dataset_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(dataset_path, 'r') as z:
            file_list = z.namelist()
            csv_files = [f for f in file_list if f.lower().endswith(".csv")]
            xlsx_files = [f for f in file_list if f.lower().endswith((".xls", ".xlsx"))]

            if csv_files:
                df = pd.read_csv(z.open(csv_files[0]))
            elif xlsx_files:
                df = pd.read_excel(z.open(xlsx_files[0]))
            else:
                raise ValueError("❌ ZIP contains no CSV/XLSX file")
    else:
        raise ValueError("❌ Unsupported format. Use .csv, .xlsx, or .zip")

    # -------------------------------------------------------
    # 2) CLEAN & VALIDATE TARGET
    # -------------------------------------------------------
    print("🧹 Cleaning dataset...")
    df = clean_dataframe(df)

    if problem_type in ["regression", "classification"]:
        if not target_col:
            raise ValueError(f"❌ Target column must be provided for {problem_type}.")
        if target_col not in df.columns:
            raise ValueError(f"❌ Target column '{target_col}' not found in dataset.")
    elif problem_type == "kmeans_clustering":
        target_col = None
        if n_clusters is None:
            raise ValueError("❌ Number of clusters (k) must be provided for kmeans_clustering.")
        if int(n_clusters) < 2:
            raise ValueError("❌ Number of clusters (k) must be >= 2.")
    else:
        raise ValueError(f"❌ Unsupported problem type: {problem_type}")

    # -------------------------------------------------------
    # 3) PREPROCESS & SAVE PROCESSED DATA
    # -------------------------------------------------------
    processed_dir = project_root / "processed_data" / dataset_name
    print("⚙️ Preprocessing features...")
    process_features(
        df,
        target_col=target_col,
        problem_type=problem_type,
        save_dir=str(processed_dir),
        apply_pca=False,
        n_clusters=n_clusters,
    )
    print(f"✅ Processed data saved at: {processed_dir}")

    meta_file = processed_dir / "metadata.json"
    if meta_file.exists():
        with open(meta_file, "r") as f:
            meta = json.load(f)
        problem_type = meta.get("problem_type", problem_type)
    else:
        meta = {}

    # -------------------------------------------------------
    # 4) TRAIN MODELS
    # -------------------------------------------------------
    print(f"🤖 Training {problem_type} models...")
    results_dir = project_root / "model_results" / dataset_name
    results_dir.mkdir(parents=True, exist_ok=True)

    orchestrator = Orchestrator(
        dataset_path=processed_dir,
        model_scripts_path=project_root / "model_scripts",
        output_path=results_dir
    )

    results = orchestrator.run()

    # -------------------------------------------------------
    # 5) BEST MODEL SELECTION
    # -------------------------------------------------------
    if problem_type in {"regression", "classification"}:
        best_model, scores = compute_model_scores(results)
        results["best_model"] = best_model
        results["model_scores"] = scores
    elif problem_type == "kmeans_clustering":
        best_model = next(iter(results.keys()), None)
        results["best_model"] = best_model
        results["model_scores"] = None
        
        # Extract clustering visualization data
        try:
            x_train_path = processed_dir / "X_train.npy"
            x_train_sparse_path = processed_dir / "X_train.npz"
            
            if x_train_sparse_path.exists():
                from scipy.sparse import load_npz
                X_train = load_npz(x_train_sparse_path).toarray()
            elif x_train_path.exists():
                X_train = np.load(x_train_path)
            else:
                X_train = None
            
            if X_train is not None and best_model and best_model in results:
                cluster_labels = results[best_model].get("cluster_labels", [])
                
                # Get original number of features
                n_features = X_train.shape[1]
                
                # If more than 3 features, reduce to 2D using PCA
                if n_features > 3:
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=2)
                    X_viz = pca.fit_transform(X_train)
                    explained_var = pca.explained_variance_ratio_.tolist()
                    viz_dims = 2
                else:
                    X_viz = X_train[:, :min(n_features, 3)]
                    explained_var = None
                    viz_dims = min(n_features, 3)
                
                # Get best model object for cluster centers
                model_path = results_dir / f"{best_model}.joblib"
                if model_path.exists():
                    pipe = joblib.load(model_path)
                    try:
                        # Get cluster centers from the kmeans model
                        kmeans = pipe.named_steps.get("kmeans", pipe)
                        if hasattr(kmeans, "cluster_centers_"):
                            centers = kmeans.cluster_centers_
                            if n_features > 3:
                                # Transform centers to 2D
                                centers_viz = pca.transform(centers)
                            else:
                                centers_viz = centers[:, :min(n_features, 3)]
                        else:
                            centers_viz = None
                    except:
                        centers_viz = None
                else:
                    centers_viz = None
                
                # Store clustering visualization data
                results["clustering_visualization"] = {
                    "points": X_viz.tolist() if X_viz is not None else [],
                    "labels": cluster_labels if cluster_labels else [],
                    "centers": centers_viz.tolist() if centers_viz is not None else [],
                    "dimensions": viz_dims,
                    "original_features": n_features,
                    "explained_variance": explained_var,
                }
        except Exception as e:
            logger.warning(f"Could not create clustering visualization data: {e}")

    # -------------------------------------------------------
    # 6) FEATURE IMPORTANCE EXTRACTION
    # -------------------------------------------------------
    if meta_file.exists():
        feature_names = meta.get("feature_names", [])

        # Verify PCA was not applied (to ensure feature names match model features)
        if meta.get("pca_applied", False):
            logger.warning(
                "PCA was applied during preprocessing. Feature importance may not be meaningful "
                "for PCA-transformed features. Skipping feature importance extraction."
            )
        else:
            model_path = results_dir / f"{best_model}.joblib"
            x_val_path = processed_dir / "X_val.npy"
            x_val_sparse_path = processed_dir / "X_val.npz"
            y_val_path = processed_dir / "y_val.npy"

            if problem_type in {"regression", "classification"} and model_path.exists() and y_val_path.exists():
                # Load validation features (sparse or dense)
                if x_val_sparse_path.exists():
                    from scipy.sparse import load_npz
                    X_val = load_npz(x_val_sparse_path)
                elif x_val_path.exists():
                    X_val = np.load(x_val_path)
                else:
                    logger.error(
                        f"Validation feature matrix not found in '{processed_dir}'. "
                        f"Expected 'X_val.npz' (sparse) or 'X_val.npy' (dense)."
                    )
                    X_val = None

                if X_val is not None:
                    y_val = np.load(y_val_path)
                    pipe = joblib.load(model_path)
                    priorities = compute_feature_priorities(
                        pipe,
                        X_val,
                        y_val,
                        feature_names,
                        problem_type,
                    )
                    results["feature_importance"] = priorities
            else:
                logger.warning(
                    "Model or validation data not found. Skipping feature importance extraction."
                )

    # Save summary JSON (after feature importance enrichment)
    summary_path = results_dir / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"📄 Summary saved: {summary_path}")
    print("\n🎉 AutoML Pipeline completed.\n")

    return results


# =======================================================
# CLI WRAPPER
# =======================================================
def main():
    parser = argparse.ArgumentParser(description="Run AutoML pipeline.")
    parser.add_argument("--file", required=True)
    parser.add_argument("--problem", required=True, choices=["regression", "classification", "kmeans_clustering", "image_classification"])
    parser.add_argument("--target", required=False)
    parser.add_argument("--k", required=False, type=int)
    parser.add_argument("--json", action="store_true")

    args = parser.parse_args()

    if args.json:
        buf = io.StringIO()
        with redirect_stdout(buf):
            result = run_pipeline(args.file, args.problem, args.target, args.k)
        print(json.dumps(result))
    else:
        result = run_pipeline(args.file, args.problem, args.target, args.k)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
