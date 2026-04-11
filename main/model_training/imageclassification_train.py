import os
import numpy as np
import sys
from pathlib import Path

# Handle imports for both direct execution and module import
try:
    from preprocessing.imageclassification_preprocess import preprocess
    from model_scripts.imageclassification_model_builder import build_model
    from model_scripts.imageclassification_trainer import train_model
    from model_scripts.imageclassification_evaluator import evaluate
except ImportError:
    from main.preprocessing.imageclassification_preprocess import preprocess
    from main.model_scripts.imageclassification_model_builder import build_model
    from main.model_scripts.imageclassification_trainer import train_model
    from main.model_scripts.imageclassification_evaluator import evaluate

def run_training(zip_path):

    dataset_name = os.path.splitext(os.path.basename(zip_path))[0]
    
    # Determine base directory (main directory)
    main_dir = Path(__file__).resolve().parent.parent
    
    result_dir = main_dir / "model_results" / dataset_name
    result_dir.mkdir(parents=True, exist_ok=True)
    
    processed_data_dir = main_dir / "processed_data" / dataset_name

    # 🔥 PREPROCESS
    train_ds, val_ds, test_ds = preprocess(zip_path)

    # load classes
    class_names = np.load(processed_data_dir / "classes.npy")

    # load y_test (needed for metrics)
    y_test = np.load(processed_data_dir / "y_test.npy")

    # 🔥 MODEL
    model, base_model = build_model(len(class_names))

    # 🔥 TRAIN
    model = train_model(
        model,
        base_model,
        train_ds,
        val_ds,
        result_dir / "model.keras"
    )

    # 🔥 EVALUATE
    evaluate(
        model,
        test_ds,
        y_test,
        class_names,
        str(result_dir)
    )

    print(f"\nResults saved in: {result_dir}")

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python train.py <dataset.zip>")
        exit()

    run_training(sys.argv[1])
