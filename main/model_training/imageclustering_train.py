import os
import numpy as np

from preprocessing.imageclassification_preprocess import preprocess
from model_scripts.imageclustering_model_builder import build_model
from model_scripts.imageclustering_trainer import train_model
from model_scripts.imageclustering_evaluator import evaluate

def run_training(zip_path):

    dataset_name = os.path.splitext(os.path.basename(zip_path))[0]
    result_dir = f"./model_results/{dataset_name}"
    os.makedirs(result_dir, exist_ok=True)

    # 🔥 PREPROCESS
    train_ds, val_ds, test_ds = preprocess(zip_path)

    # load classes
    class_names = np.load(f"./processed_data/{dataset_name}/classes.npy")

    # load y_test (needed for metrics)
    y_test = np.load(f"./processed_data/{dataset_name}/y_test.npy")

    # 🔥 MODEL
    model, base_model = build_model(len(class_names))

    # 🔥 TRAIN
    model = train_model(
        model,
        base_model,
        train_ds,
        val_ds,
        os.path.join(result_dir, "model.keras")
    )

    # 🔥 EVALUATE
    evaluate(
        model,
        test_ds,
        y_test,
        class_names,
        result_dir
    )

    print(f"\nResults saved in: {result_dir}")

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python train.py <dataset.zip>")
        exit()

    run_training(sys.argv[1])