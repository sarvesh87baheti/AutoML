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
from main.model_training.imageclassification_multi_train import run_training as run_image_classification_training
from main.model_training.feature_importance import compute_feature_priorities
from main.final_model_selection.final_model_sel import compute_model_scores
from main.model_scripts.script_validator import validate_custom_script


def run_pipeline(
    file_path: str,
    problem_type: str,
    target_col: str = None,
    n_clusters: int = None,
    image_mode: str = "standard",
    custom_script_paths: list[str] = None,
):
    print("\n===============================")
    print("🚀 Starting AutoML Pipeline")
    print("===============================\n")

    dataset_path = Path(file_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"❌ Dataset not found: {file_path}")

    # Normalise custom scripts to a list of Path objects
    custom_paths: list[Path] = []
    if custom_script_paths:
        for p in custom_script_paths:
            cp = Path(p)
            if cp.exists():
                custom_paths.append(cp)
            else:
                logger.warning(f"Custom script path not found, ignoring: {p}")

    # -------------------------------------------------------
    # HANDLE IMAGE CLASSIFICATION SEPARATELY
    # -------------------------------------------------------
    if problem_type == "image_classification":
        print(f"📂 Loading image dataset: {dataset_path.name}")
        if not dataset_path.suffix.lower() == ".zip":
            raise ValueError("❌ Image classification requires a ZIP file with class folders containing images.")

        try:
            result = run_image_classification_training(str(dataset_path), image_mode=image_mode)
            return result
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
            # Filter out system files and directories
            file_list = [f for f in file_list if not f.startswith('__MACOSX') and not f.endswith('/')]
            
            csv_files = [f for f in file_list if f.lower().endswith(".csv")]
            xlsx_files = [f for f in file_list if f.lower().endswith((".xls", ".xlsx"))]
            if csv_files:
                # Use the first CSV file found
                with z.open(csv_files[0]) as f:
                    df = pd.read_csv(f)
            elif xlsx_files:
                # Use the first XLSX file found
                with z.open(xlsx_files[0]) as f:
                    df = pd.read_excel(f)
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
    # 3) VALIDATE CUSTOM SCRIPTS (before spending time on preprocessing)
    # -------------------------------------------------------
    valid_custom_paths: list[Path] = []
    if custom_paths:
        print(f"\n🔍 Validating {len(custom_paths)} custom script(s)...")
        for cp in custom_paths:
            ok, reason = validate_custom_script(cp, problem_type)
            if ok:
                print(f"  ✅ {cp.name} — valid")
                valid_custom_paths.append(cp)
            else:
                print(f"  ⚠️  {cp.name} — INVALID: {reason} (will be skipped)")

    # -------------------------------------------------------
    # 4) PREPROCESS & SAVE PROCESSED DATA
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
    # 5) TRAIN MODELS
    # -------------------------------------------------------
    print(f"🤖 Training {problem_type} models...")
    results_dir = project_root / "model_results" / dataset_name
    results_dir.mkdir(parents=True, exist_ok=True)

    orchestrator = Orchestrator(
        dataset_path=processed_dir,
        model_scripts_path=project_root / "model_scripts",
        output_path=results_dir,
        custom_script_paths=valid_custom_paths,
    )

    results = orchestrator.run()

    # -------------------------------------------------------
    # 6) BEST MODEL SELECTION
    # -------------------------------------------------------
    if problem_type in {"regression", "classification"}:
        best_model, scores = compute_model_scores(results)
        results["best_model"] = best_model
        results["model_scores"] = scores
    elif problem_type == "kmeans_clustering":
        best_model = next(iter(results.keys()), None)
        results["best_model"] = best_model
        results["model_scores"] = None

    # -------------------------------------------------------
    # 7) FEATURE IMPORTANCE EXTRACTION
    # -------------------------------------------------------
    if meta_file.exists():
        feature_names = meta.get("feature_names", [])

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


def validate_only(
    custom_script_paths: list[str],
    problem_type: str,
) -> dict:
    """
    Validate a list of custom script paths without running the training
    pipeline.  Used by the --validate-only CLI flag and the Next.js
    /api/validate-scripts endpoint.

    Returns a dict:
        {
            "results": [
                {"filename": "...", "valid": true/false, "reason": "..."},
                ...
            ]
        }
    """
    output = []
    for p in custom_script_paths:
        cp = Path(p)
        ok, reason = validate_custom_script(cp, problem_type)
        output.append({
            "filename": cp.name,
            "valid": ok,
            "reason": reason,
        })
    return {"results": output}


# =======================================================
# CLI WRAPPER
# =======================================================

def main():
    parser = argparse.ArgumentParser(description="Run AutoML pipeline.")
    parser.add_argument("--file", required=False)
    parser.add_argument(
        "--problem",
        required=False,
        choices=["regression", "classification", "kmeans_clustering", "image_classification"],
    )
    parser.add_argument("--target", required=False)
    parser.add_argument("--k", required=False, type=int)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--image-mode", choices=["light", "standard"], default="standard")
    parser.add_argument(
        "--custom-scripts",
        nargs="*",
        default=[],
        dest="custom_scripts",
        metavar="SCRIPT_PATH",
        help="Paths to one or more custom model .py scripts (max 3).",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        dest="validate_only",
        help=(
            "Validate --custom-scripts against --problem type and print JSON "
            "results without running the training pipeline."
        ),
    )

    args = parser.parse_args()

    # ── Validate-only mode ────────────────────────────────────────────────────
    if args.validate_only:
        if not args.problem:
            parser.error("--problem is required with --validate-only")
        result = validate_only(args.custom_scripts or [], args.problem)
        print(json.dumps(result, indent=2))
        return

    # ── Normal training mode ──────────────────────────────────────────────────
    if not args.file:
        parser.error("--file is required")
    if not args.problem:
        parser.error("--problem is required")

    # Enforce the 3-script limit at the CLI layer
    custom_scripts = args.custom_scripts or []
    if len(custom_scripts) > 3:
        print(
            f"⚠️  More than 3 custom scripts provided ({len(custom_scripts)}); "
            "only the first 3 will be used."
        )
        custom_scripts = custom_scripts[:3]

    if args.json:
        buf = io.StringIO()
        with redirect_stdout(buf):
            result = run_pipeline(
                args.file,
                args.problem,
                args.target,
                args.k,
                args.image_mode,
                custom_script_paths=custom_scripts,
            )
        print(json.dumps(result))
    else:
        result = run_pipeline(
            args.file,
            args.problem,
            args.target,
            args.k,
            args.image_mode,
            custom_script_paths=custom_scripts,
        )
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()