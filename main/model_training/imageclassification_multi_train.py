"""
Multi-Model Image Classification Training
Trains multiple transfer learning backbones and compares performance.
"""
import os
import json
import numpy as np
import sys
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

# Handle imports
try:
    from ..preprocessing.imageclassification_preprocess import preprocess
    from .imageclassification_model_registry import (
        build_model,
        compile_model,
        get_callbacks,
        train,
        evaluate,
        ALL_BACKBONES,
    )
except ImportError:
    from main.preprocessing.imageclassification_preprocess import preprocess
    from main.model_training.imageclassification_model_registry import (
        build_model,
        compile_model,
        get_callbacks,
        train,
        evaluate,
        ALL_BACKBONES,
    )

LIGHT_MODE_BACKBONES = ["mobilenetv2"]
STANDARD_MODE_BACKBONES = ["mobilenetv2", "mobilenetv3small", "efficientnetb0"]


def train_multiple_models(
    train_ds,
    val_ds,
    test_ds,
    dataset_name: str,
    num_classes: int,
    class_names: List[str],
    result_dir: Path,
    backbones: List[str],
    image_mode: str,
) -> Dict:
    """
    Train multiple backbone models and return comparison results.

    Returns:
        {
            "models": {
                "model_name": {
                    "accuracy": float,
                    "loss": float,
                    "report": dict,
                    "training_time": float
                }
            },
            "best_model": "model_name",
            "comparison_metrics": [...],
            "all_predictions": {...}
        }
    """
    import time

    results = {
        "models": {},
        "best_model": None,
        "class_names": list(class_names),
        "num_classes": num_classes,
    }

    best_accuracy = -1
    for backbone_name in backbones:
        print(f"\n{'='*60}")
        print(f"Training: {backbone_name.upper()}")
        print(f"{'='*60}")

        try:
            # Build model
            model, preprocess_fn = build_model(
                backbone_name=backbone_name,
                num_classes=num_classes,
                dropout_rate=0.3,
                freeze_backbone=True,
            )

            # Compile
            compile_model(model, num_classes=num_classes, learning_rate=1e-3)

            start_time = time.time()

            _, h1, h2 = train(
                train_ds=train_ds,
                val_ds=val_ds,
                dataset_name=dataset_name,
                backbone_name=backbone_name,
                num_classes=num_classes,
                epochs_frozen=8 if image_mode == "light" else 10,
                epochs_finetune=4 if image_mode == "light" else 5,
                fine_tune_at=None,
                save_final_model=False,
            )

            training_time = time.time() - start_time

            checkpoint_path = (
                result_dir.parent.parent
                / "saved_models"
                / dataset_name
                / backbone_name
                / "best_model.weights.h5"
            )
            if checkpoint_path.exists():
                model.load_weights(str(checkpoint_path))
            eval_metrics = evaluate(
                model,
                test_ds,
                class_names=list(class_names),
                save_dir=str(result_dir),
            )

            accuracy = eval_metrics.get("accuracy", 0)
            loss = eval_metrics.get("loss", 0)

            results["models"][backbone_name] = {
                "accuracy": float(accuracy),
                "loss": float(loss),
                "training_time": float(training_time),
                "report": eval_metrics.get("classification_report", {}),
                "model_size_mb": os.path.getsize(checkpoint_path) / (1024 * 1024),
            }

            print(f"\n✅ {backbone_name}: Accuracy={accuracy:.4f}, Loss={loss:.4f}")

            # Track best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                results["best_model"] = backbone_name

        except Exception as e:
            print(f"❌ Error training {backbone_name}: {str(e)}")
            results["models"][backbone_name] = {
                "error": str(e),
                "accuracy": 0,
                "loss": 999999.0,  # Use large finite value instead of inf for JSON compatibility
            }
        finally:
            try:
                from tensorflow.keras import backend as keras_backend

                keras_backend.clear_session()
            except Exception:
                pass

    return results


def cleanup_non_best_checkpoints(dataset_name: str, best_model: str | None):
    saved_models_dir = Path(__file__).resolve().parent.parent / "saved_models" / dataset_name
    if not saved_models_dir.exists():
        return

    for child in saved_models_dir.iterdir():
        if not child.is_dir():
            continue
        if best_model and child.name == best_model:
            continue
        shutil.rmtree(child, ignore_errors=True)


def create_comparison_table(results: Dict) -> List[Dict]:
    """Create a comparison table for UI display."""
    table = []
    for model_name, metrics in results["models"].items():
        if "error" not in metrics:
            table.append(
                {
                    "name": model_name,
                    "value": metrics["accuracy"],
                    "loss": metrics.get("loss", "N/A"),
                    "time": metrics.get("training_time", "N/A"),
                    "size_mb": metrics.get("model_size_mb", "N/A"),
                }
            )
    return sorted(table, key=lambda x: x["value"], reverse=True)


def run_training(zip_path: str, image_mode: str = "standard"):
    """Main entry point for multi-model training."""
    from tensorflow.keras.models import load_model

    dataset_name = os.path.splitext(os.path.basename(zip_path))[0]
    normalized_mode = (image_mode or "standard").strip().lower()
    if normalized_mode not in {"light", "standard"}:
        raise ValueError(f"Unsupported image mode: {image_mode}")
    selected_backbones = (
        LIGHT_MODE_BACKBONES if normalized_mode == "light" else STANDARD_MODE_BACKBONES
    )

    # Paths
    main_dir = Path(__file__).resolve().parent.parent
    result_dir = main_dir / "model_results" / dataset_name
    processed_data_dir = main_dir / "processed_data" / dataset_name
    saved_models_dir = main_dir / "saved_models" / dataset_name

    for generated_dir in (result_dir, processed_data_dir, saved_models_dir):
        if generated_dir.exists():
            shutil.rmtree(generated_dir, ignore_errors=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    # 🔥 PREPROCESS
    print("🔥 Preprocessing images...")
    train_ds, val_ds, test_ds, extracted_dir = preprocess(zip_path)

    try:
        # Load metadata
        class_names = np.load(processed_data_dir / "classes.npy")
        num_classes = len(class_names)

        # 🔥 TRAIN MULTIPLE MODELS
        print(f"🔥 Training {len(selected_backbones)} models in {normalized_mode} mode...")
        results = train_multiple_models(
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=test_ds,
            dataset_name=dataset_name,
            num_classes=num_classes,
            class_names=class_names,
            result_dir=result_dir,
            backbones=selected_backbones,
            image_mode=normalized_mode,
        )
        cleanup_non_best_checkpoints(dataset_name, results["best_model"])
    finally:
        shutil.rmtree(extracted_dir, ignore_errors=True)

    # 🔥 CREATE COMPARISON
    comparison_table = create_comparison_table(results)

    # 🔥 SAVE RESULTS
    summary = {
        "image_classification": {
            "metrics": {
                "val": {
                    "accuracy": results["models"].get(
                        results["best_model"], {}
                    ).get("accuracy", 0) if results["best_model"] else 0,
                    "loss": results["models"].get(
                        results["best_model"], {}
                    ).get("loss", 0) if results["best_model"] else 0,
                }
            },
            "models_comparison": comparison_table,
            "all_models": results["models"],
            "classification_report": results["models"].get(results["best_model"], {}).get("report", {}),
        },
        "best_model": results["best_model"] or "No successful models",
        "problem_type": "image_classification",
        "image_mode": normalized_mode,
        "dataset_name": dataset_name,
        "class_names": list(class_names),
        "num_classes": num_classes,
    }

    summary_path = result_dir / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    print(f"\n✅ Results saved in: {result_dir}")
    print(f"📄 Summary saved: {summary_path}")
    
    if results['best_model'] is not None:
        print(f"\n🏆 Best Model: {results['best_model'].upper()}")
        print(f"   Accuracy: {results['models'][results['best_model']]['accuracy']:.4f}")
    else:
        print("\n❌ No successful model training. All models failed.")

    return summary


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python imageclassification_multi_train.py <dataset.zip>")
        sys.exit(1)

    run_training(sys.argv[1])
