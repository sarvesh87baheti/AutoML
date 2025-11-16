# runner.py
import argparse
import json
import pandas as pd
from pathlib import Path
import zipfile  
import sys, os

# --- Universal import fix (works anywhere: CLI or UI) ---
ROOT = Path(__file__).resolve().parent
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

# --- Import project modules ---
from main.preprocessing.datacleaning import clean_dataframe
from main.preprocessing.preprocessor import process_features
from main.model_training.orchestrator import Orchestrator


def run_pipeline(file_path: str, problem_type: str, target_col: str = None):
    """
    Runs the full AutoML pipeline:
      1. Load and clean raw dataset
      2. Preprocess and save processed data
      3. Train ML models
      4. Save trained models and summary report

    Args:
        file_path (str): Path to the dataset file (CSV/XLSX)
        problem_type (str): Type of ML task (regression, classification, clustering)
        target_col (str, optional): Target column for supervised tasks

    Returns:
        dict: Model training results summary
    """
    print("\n===============================")
    print("🚀 Starting AutoML Pipeline")
    print("===============================")

    dataset_path = Path(file_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"❌ Dataset not found: {file_path}")

    dataset_name = dataset_path.stem
    project_root = ROOT / "main"

    # === 1. Load and clean data ===
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
                with z.open(csv_files[0]) as f:
                    df = pd.read_csv(f)
            elif xlsx_files:
                with z.open(xlsx_files[0]) as f:
                    df = pd.read_excel(f)
            else:
                raise ValueError("ZIP file does not contain a CSV or XLSX file.")
        
    
    else:
        raise ValueError("Unsupported file format. Use .csv or .xlsx")
    

    print("🧹 Cleaning dataset...")
    df = clean_dataframe(df)

    # --- Validate target column if supervised ---
    if problem_type in ["regression", "classification"]:
        if not target_col:
            raise ValueError(f"❌ Target column must be specified for {problem_type} tasks.")
        if target_col not in df.columns:
            raise ValueError(f"❌ Target column '{target_col}' not found in dataset.")
    else:
        target_col = None  # not used for unsupervised problems

    # === 2. Preprocess features ===
    processed_dir = project_root / "processed_data" / dataset_name
    print("⚙️  Preprocessing features and saving processed arrays...")
    process_result = process_features(
        cleaned_df=df,
        target_col=target_col,
        save_dir=str(processed_dir)
    )
    print(f"✅ Processed data saved to: {processed_dir}")

    # === 3. Train models ===
    print(f"🤖 Running {problem_type.capitalize()} model training...")
    results_dir = project_root / "model_results" / dataset_name
    results_dir.mkdir(parents=True, exist_ok=True)

    # Currently, orchestrator handles regression; extend later for others
    orchestrator = Orchestrator(
        dataset_path=processed_dir,
        model_scripts_path=project_root / "model_scripts",
        output_path=results_dir
    )

    if problem_type == "regression":
        results = orchestrator.run()
    else:
        print(f"⚠️ {problem_type.capitalize()} model training not implemented yet.")
        results = {"status": "skipped", "problem_type": problem_type}

    # === 4. Save summary ===
    summary_path = results_dir / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"📄 Training summary saved at: {summary_path}")

    print("\n✅ AutoML Pipeline completed successfully.")
    print(f"All trained models saved in: {results_dir}\n")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run AutoML pipeline on a dataset.")
    parser.add_argument("--file", required=True, help="Path to raw dataset (CSV/XLSX)")
    parser.add_argument("--problem", required=True, choices=["regression", "classification", "clustering"],
                        help="Problem type: regression, classification, or clustering")
    parser.add_argument("--target", required=False, help="Target column name for supervised tasks")

    args = parser.parse_args()
    run_pipeline(args.file, args.problem, args.target)


if __name__ == "__main__":
    main()
