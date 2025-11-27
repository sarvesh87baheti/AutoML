# runner.py
import argparse
import json
import io
from contextlib import redirect_stdout
import pandas as pd
from pathlib import Path
import zipfile
import sys, os
import joblib
import numpy as np

# Make project importable regardless of run context
ROOT = Path(__file__).resolve().parent
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

# Project imports
from main.preprocessing.datacleaning import clean_dataframe
from main.preprocessing.preprocessor import process_features
from main.model_training.orchestrator import Orchestrator
from main.final_model_selection.final_model_sel import compute_model_scores


def run_pipeline(file_path: str, problem_type: str, target_col: str = None):
    print("\n===============================")
    print("🚀 Starting AutoML Pipeline")
    print("===============================\n")

    dataset_path = Path(file_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"❌ Dataset not found: {file_path}")

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
    else:
        target_col = None

    # -------------------------------------------------------
    # 3) PREPROCESS & SAVE PROCESSED DATA
    # -------------------------------------------------------
    processed_dir = project_root / "processed_data" / dataset_name
    print("⚙️ Preprocessing features...")
    process_features(df, target_col=target_col, save_dir=str(processed_dir))
    print(f"✅ Processed data saved at: {processed_dir}")

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
    best_model, scores = compute_model_scores(results)
    results["best_model"] = best_model
    results["model_scores"] = scores

    # Save summary JSON
    summary_path = results_dir / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"📄 Summary saved: {summary_path}")
    print("\n🎉 AutoML Pipeline completed.\n")

    # -------------------------------------------------------
    # 6) SAFE COEFFICIENT EXTRACTION (Regression only)
    # -------------------------------------------------------
    meta_file = processed_dir / "metadata.json"
    if meta_file.exists():
        with open(meta_file, "r") as f:
            meta = json.load(f)

        if meta.get("problem_type") == "regression":
            feature_names = meta.get("numeric_cols", [])

            model_path = results_dir / f"{best_model}.joblib"
            if model_path.exists():
                pipe = joblib.load(model_path)
                est = getattr(pipe, "named_steps", {}).get("est", pipe)

                coef = getattr(est, "coef_", None)
                intercept = getattr(est, "intercept_", None)

                if coef is not None:
                    coef = np.asarray(coef)
                    if coef.ndim > 1:
                        coef = coef[0]

                    if len(coef) != len(feature_names):
                        feature_names = [f"feature_{i}" for i in range(len(coef))]

                    coef_map = {feature_names[i]: float(coef[i]) for i in range(len(coef))}
                    coef_map["intercept"] = float(intercept) if intercept is not None else 0.0
                    results["coefficients"] = coef_map

    return results


# =======================================================
# CLI WRAPPER
# =======================================================
def main():
    parser = argparse.ArgumentParser(description="Run AutoML pipeline.")
    parser.add_argument("--file", required=True)
    parser.add_argument("--problem", required=True, choices=["regression", "classification", "clustering"])
    parser.add_argument("--target", required=False)
    parser.add_argument("--json", action="store_true")

    args = parser.parse_args()

    if args.json:
        buf = io.StringIO()
        with redirect_stdout(buf):
            result = run_pipeline(args.file, args.problem, args.target)
        print(json.dumps(result))
    else:
        result = run_pipeline(args.file, args.problem, args.target)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
