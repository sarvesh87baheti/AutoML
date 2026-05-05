# Suppress TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

"""
EasyFlow ML - Image Classification Model Registry
==================================================
Central integration file for all supported transfer learning backbones.

Supported model families:
  - EfficientNet  (B0–B3, V2S, V2M)         → efficientnet_models.py
  - MobileNet     (V2, V3Small, V3Large)     → mobilenet_models.py
  - ResNet        (50V2, 101V2)              → resnet_models.py
  - Inception     (V3, ResNetV2, Xception)   → inception_models.py
  - DenseNet      (121, 169, 201)            → densenet_models.py
  - VGG & NASNet  (VGG16, VGG19, NASNetMobile) → vgg_nasnet_models.py

Usage:
    from imageclassification_model_registry import build_model, train, evaluate, list_all_models

    model, preprocess_fn = build_model("efficientnetb0", num_classes=5)
    model, h1, h2 = train(train_ds, val_ds, dataset_name="flowers", backbone_name="efficientnetb0")
    evaluate(model, test_ds, class_names=["daisy", "rose", ...])
"""

import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)

# --- Import all model families ---
from ..model_scripts.efficientnet_models import (
    build_efficientnet_model,
    EFFICIENTNET_CONFIGS,
    list_efficientnets,
)
from ..model_scripts.mobilenet_models import (
    build_mobilenet_model,
    MOBILENET_CONFIGS,
    list_mobilenets,
)
from ..model_scripts.resnet_models import (
    build_resnet_model,
    RESNET_CONFIGS,
    list_resnets,
)
from ..model_scripts.inception_models import (
    build_inception_model,
    INCEPTION_CONFIGS,
    list_inceptions,
)
from ..model_scripts.densenet_models import (
    build_densenet_model,
    DENSENET_CONFIGS,
    list_densenets,
)
from ..model_scripts.vgg_nasnet_models import (
    build_vgg_nasnet_model,
    VGG_NASNET_CONFIGS,
    list_vgg_nasnet,
)


# ==============================
# PATHS
# ==============================
_MAIN_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = str(_MAIN_DIR / "processed_data")
MODELS_DIR = str(_MAIN_DIR / "saved_models")


# ==============================
# REGISTRY
# ==============================

# Maps every supported backbone name → its builder function
_REGISTRY = {}

for _name in EFFICIENTNET_CONFIGS:
    _REGISTRY[_name] = build_efficientnet_model

for _name in MOBILENET_CONFIGS:
    _REGISTRY[_name] = build_mobilenet_model

for _name in RESNET_CONFIGS:
    _REGISTRY[_name] = build_resnet_model

for _name in INCEPTION_CONFIGS:
    _REGISTRY[_name] = build_inception_model

for _name in DENSENET_CONFIGS:
    _REGISTRY[_name] = build_densenet_model

for _name in VGG_NASNET_CONFIGS:
    _REGISTRY[_name] = build_vgg_nasnet_model

ALL_BACKBONES = list(_REGISTRY.keys())

# Recommended input sizes per backbone (falls back to 224 if not listed)
_RECOMMENDED_SIZES = {}
for _configs in [
    EFFICIENTNET_CONFIGS,
    MOBILENET_CONFIGS,
    RESNET_CONFIGS,
    INCEPTION_CONFIGS,
    DENSENET_CONFIGS,
    VGG_NASNET_CONFIGS,
]:
    for _name, _cfg in _configs.items():
        _RECOMMENDED_SIZES[_name] = _cfg["recommended_size"]


# ==============================
# PUBLIC API
# ==============================

def list_all_models():
    """Prints all supported backbones grouped by family."""
    print("\n" + "=" * 60)
    print("  EasyFlow ML — Supported Backbones")
    print("=" * 60)
    list_efficientnets()
    print()
    list_mobilenets()
    print()
    list_resnets()
    print()
    list_inceptions()
    print()
    list_densenets()
    print()
    list_vgg_nasnet()
    print("=" * 60 + "\n")


def get_recommended_input_shape(backbone_name: str) -> tuple:
    """Returns the recommended (H, W, C) input shape for a backbone."""
    name = backbone_name.lower().strip()
    size = _RECOMMENDED_SIZES.get(name, 224)
    return (size, size, 3)


def build_model(
    backbone_name: str,
    num_classes: int,
    input_shape: tuple = None,
    dropout_rate: float = 0.3,
    freeze_backbone: bool = True,
):
    """
    Builds a transfer learning model for any supported backbone.

    Args:
        backbone_name:   Any name from ALL_BACKBONES.
        num_classes:     Number of output classes.
        input_shape:     (H, W, C). If None, uses recommended size for backbone.
        dropout_rate:    Dropout rate for classification head.
        freeze_backbone: Freeze backbone weights (True for Phase 1 training).

    Returns:
        (model, preprocess_fn)

    Example:
        model, preprocess_fn = build_model("efficientnetb0", num_classes=5)
    """
    name = backbone_name.lower().strip()

    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown backbone '{name}'.\n"
            f"Call list_all_models() to see all options."
        )

    if input_shape is None:
        input_shape = get_recommended_input_shape(name)
        print(f"Using recommended input shape for '{name}': {input_shape}")

    builder = _REGISTRY[name]
    return builder(
        name=name,
        num_classes=num_classes,
        input_shape=input_shape,
        dropout_rate=dropout_rate,
        freeze_backbone=freeze_backbone,
    )


def compile_model(model, num_classes: int, learning_rate: float = 1e-3):
    """
    Compiles a model with appropriate loss and metrics.

    Binary:      binary_crossentropy + accuracy + AUC
    Multiclass:  categorical_crossentropy + accuracy + Top-3 accuracy
    """
    if num_classes == 2:
        loss = "binary_crossentropy"
        metrics = [
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
        ]
    else:
        loss = "categorical_crossentropy"
        metrics = [
            "accuracy",
            tf.keras.metrics.TopKCategoricalAccuracy(k=min(3, num_classes), name="top3_acc"),
        ]

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=metrics,
    )
    return model


def get_callbacks(checkpoint_path: str, patience: int = 10):
    """Standard EarlyStopping + ReduceLROnPlateau + ModelCheckpoint callbacks."""
    return [
        EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=max(patience // 2, 3),
            min_lr=1e-7,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
    ]


def train(
    train_ds,
    val_ds,
    dataset_name: str,
    backbone_name: str = "efficientnetb0",
    num_classes: int = None,
    input_shape: tuple = None,
    epochs_frozen: int = 20,
    epochs_finetune: int = 20,
    fine_tune_at: int = None,
    dropout_rate: float = 0.3,
    lr_frozen: float = 1e-3,
    lr_finetune: float = 1e-5,
    save_final_model: bool = False,
):
    """
    Two-phase transfer learning trainer.

    Phase 1 — Frozen backbone: only the classification head is trained.
    Phase 2 — Fine-tune:       top 30% of backbone layers are unfrozen at a lower LR.

    Args:
        train_ds:        tf.data.Dataset returning (images, labels).
        val_ds:          tf.data.Dataset returning (images, labels).
        dataset_name:    Name used to organise saved checkpoints/models.
        backbone_name:   Backbone architecture (see ALL_BACKBONES).
        num_classes:     Auto-read from metadata.json if None.
        input_shape:     (H, W, C). Defaults to backbone's recommended size.
        epochs_frozen:   Epochs for Phase 1.
        epochs_finetune: Epochs for Phase 2.
        fine_tune_at:    Backbone layer index to start unfreezing from.
                         Defaults to 70% of total layers (top 30% unfrozen).
        dropout_rate:    Dropout rate for classification head.
        lr_frozen:       Learning rate for Phase 1.
        lr_finetune:     Learning rate for Phase 2.

    Returns:
        (model, history_phase1, history_phase2)

    Example:
        model, h1, h2 = train(
            train_ds, val_ds,
            dataset_name="flowers102",
            backbone_name="efficientnetb0",
            epochs_frozen=15,
            epochs_finetune=15,
        )
    """
    # --- Load num_classes from metadata if not given ---
    if num_classes is None:
        meta_path = os.path.join(PROCESSED_DIR, dataset_name, "metadata.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"metadata.json not found at '{meta_path}'. "
                "Run the preprocessor first, or pass num_classes explicitly."
            )
        with open(meta_path) as f:
            meta = json.load(f)
        num_classes = meta["num_classes"]
        print(f"Loaded metadata: {num_classes} classes — {meta['class_names']}")

    # --- Resolve input_shape ---
    if input_shape is None:
        input_shape = get_recommended_input_shape(backbone_name)
        print(f"Input shape: {input_shape}")

    # --- Output directory ---
    save_dir = os.path.join(MODELS_DIR, dataset_name, backbone_name)
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, "best_model.weights.h5")

    # ==========================
    # PHASE 1 — Frozen backbone
    # ==========================
    print(f"\n{'='*55}")
    print(f"  PHASE 1 — Head training only  ({epochs_frozen} epochs)")
    print(f"  Backbone: {backbone_name}  |  Classes: {num_classes}")
    print(f"{'='*55}")

    model, preprocess_fn = build_model(
        backbone_name=backbone_name,
        num_classes=num_classes,
        input_shape=input_shape,
        dropout_rate=dropout_rate,
        freeze_backbone=True,
    )
    model = compile_model(model, num_classes, learning_rate=lr_frozen)
    model.summary(line_length=80)

    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_frozen,
        callbacks=get_callbacks(ckpt_path, patience=8),
        verbose=1,
    )

    # ==========================
    # PHASE 2 — Fine-tuning
    # ==========================
    print(f"\n{'='*55}")
    print(f"  PHASE 2 — Fine-tuning  ({epochs_finetune} epochs)")
    print(f"{'='*55}")

    # Locate the nested application backbone instead of assuming a fixed layer index.
    backbone_layer = next((layer for layer in model.layers if hasattr(layer, "layers")), None)
    if backbone_layer is None:
        raise ValueError("Could not locate the transfer-learning backbone for fine-tuning.")

    total_layers = len(backbone_layer.layers)

    if fine_tune_at is None:
        fine_tune_at = int(total_layers * 0.7)

    backbone_layer.trainable = True
    for layer in backbone_layer.layers[:fine_tune_at]:
        layer.trainable = False

    unfrozen = total_layers - fine_tune_at
    print(f"Unfreezing top {unfrozen}/{total_layers} backbone layers "
          f"(from layer index {fine_tune_at}).")

    # Recompile at lower LR
    model = compile_model(model, num_classes, learning_rate=lr_finetune)

    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_finetune,
        callbacks=get_callbacks(ckpt_path, patience=10),
        verbose=1,
    )

    if os.path.exists(ckpt_path):
        model.load_weights(ckpt_path)

    if save_final_model:
        final_path = os.path.join(save_dir, "final_model.keras")
        model.save(final_path)
        print(f"\nFinal model saved: {final_path}")

    # --- Save training config ---
    config = {
        "backbone": backbone_name,
        "num_classes": num_classes,
        "input_shape": list(input_shape),
        "epochs_frozen": epochs_frozen,
        "epochs_finetune": epochs_finetune,
        "fine_tune_at": fine_tune_at,
        "lr_frozen": lr_frozen,
        "lr_finetune": lr_finetune,
        "dropout_rate": dropout_rate,
        "save_final_model": save_final_model,
    }
    with open(os.path.join(save_dir, "training_config.json"), "w") as f:
        json.dump(config, f, indent=4)

    return model, history1, history2


def evaluate(model, test_ds, class_names=None, save_dir: str = None):
    """
    Evaluates model on test_ds and prints a detailed report.

    Args:
        model:       Trained Keras model.
        test_ds:     tf.data.Dataset returning (images, labels).
        class_names: Optional list of class names for per-class report.

    Returns:
        dict of metric_name → value
    """
    print("\nEvaluating on test set...")
    results = model.evaluate(test_ds, verbose=1)
    metric_names = model.metrics_names

    print("\n--- Test Results ---")
    for name, val in zip(metric_names, results):
        print(f"  {name}: {val:.4f}")

    classification_report_data = None
    if class_names is not None:
        y_true, y_pred = [], []
        for x_batch, y_batch in test_ds:
            preds = model.predict(x_batch, verbose=0)
            if preds.shape[-1] == 1:
                y_pred.extend((preds.squeeze() > 0.5).astype(int).tolist())
                y_true.extend(y_batch.numpy().tolist())
            else:
                y_pred.extend(np.argmax(preds, axis=1).tolist())
                y_true.extend(np.argmax(y_batch.numpy(), axis=1).tolist())

        try:
            from sklearn.metrics import classification_report, confusion_matrix
            classification_report_data = classification_report(
                y_true,
                y_pred,
                target_names=class_names,
                output_dict=True,
            )
            print("\n--- Classification Report ---")
            print(classification_report(y_true, y_pred, target_names=class_names))
            print("--- Confusion Matrix ---")
            print(confusion_matrix(y_true, y_pred))
        except ImportError:
            print("(Install scikit-learn for per-class report: pip install scikit-learn)")

    result_dict = dict(zip(metric_names, results))
    if results:
        result_dict["loss"] = float(results[0])

    if classification_report_data is not None:
        result_dict["classification_report"] = classification_report_data
        result_dict["accuracy"] = float(classification_report_data.get("accuracy", result_dict.get("accuracy", 0.0)))

        if save_dir is not None:
            try:
                import matplotlib.pyplot as plt
                import seaborn as sns
                from sklearn.metrics import confusion_matrix

                cm = confusion_matrix(y_true, y_pred)
                plt.figure(figsize=(8, 8))
                sns.heatmap(
                    cm,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    xticklabels=class_names,
                    yticklabels=class_names,
                )
                plt.xlabel("Predicted")
                plt.ylabel("Actual")
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
                plt.close()
            except Exception as exc:
                print(f"Could not save confusion matrix image: {exc}")

    return result_dict


def load_model(dataset_name: str, backbone_name: str, best: bool = True):
    """
    Loads a previously saved model.

    Args:
        dataset_name:  Dataset name used during training.
        backbone_name: Backbone name used during training.
        best:          If True, loads best_model.weights.h5 into a rebuilt model;
                       else loads final_model.keras.

    Returns:
        Loaded Keras model.
    """
    save_dir = os.path.join(MODELS_DIR, dataset_name, backbone_name)
    if best:
        weights_path = os.path.join(save_dir, "best_model.weights.h5")
        config_path = os.path.join(save_dir, "training_config.json")
        if not os.path.exists(weights_path) or not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Best model artifacts not found in '{save_dir}'. "
                "Check dataset_name and backbone_name, or train first."
            )
        with open(config_path) as f:
            config = json.load(f)
        model, _ = build_model(
            backbone_name=config["backbone"],
            num_classes=config["num_classes"],
            input_shape=tuple(config["input_shape"]),
            dropout_rate=config.get("dropout_rate", 0.3),
            freeze_backbone=False,
        )
        model.load_weights(weights_path)
        print(f"Model weights loaded from: {weights_path}")
        return model

    model_path = os.path.join(save_dir, "final_model.keras")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No saved model at '{model_path}'. "
            "Check dataset_name and backbone_name, or train first."
        )
    model = tf.keras.models.load_model(model_path)
    print(f"Model loaded from: {model_path}")
    return model


def auto_select_backbone(num_samples: int, num_classes: int) -> str:
    """
    Heuristic backbone selector based on dataset size and class count.

    Args:
        num_samples: Total number of training images.
        num_classes: Number of output classes.

    Returns:
        Recommended backbone name.
    """
    print(f"\nAuto-selecting backbone for {num_samples} samples, {num_classes} classes...")

    if num_samples < 500:
        # Small dataset — use lightest model to avoid overfitting
        choice = "mobilenetv2"
        reason = "small dataset (<500 samples) — lightweight model reduces overfitting"
    elif num_samples < 2000:
        choice = "densenet121"
        reason = "small-medium dataset — DenseNet's dense connections help with limited data"
    elif num_samples < 10000:
        choice = "efficientnetb0"
        reason = "medium dataset — best accuracy/speed tradeoff"
    elif num_samples < 50000:
        choice = "efficientnetb2"
        reason = "large dataset — slightly bigger EfficientNet for improved accuracy"
    else:
        choice = "efficientnetb3"
        reason = "very large dataset — higher capacity backbone justified"

    print(f"  Recommended: {choice}  ({reason})")
    return choice


# ==============================
# QUICK DEMO
# ==============================

if __name__ == "__main__":
    list_all_models()

    print("\nAuto-selector examples:")
    auto_select_backbone(num_samples=300, num_classes=2)
    auto_select_backbone(num_samples=1500, num_classes=5)
    auto_select_backbone(num_samples=8000, num_classes=10)
    auto_select_backbone(num_samples=60000, num_classes=100)

    print("\nBuilding a quick test model (efficientnetb0, 5 classes)...")
    model, preprocess_fn = build_model("efficientnetb0", num_classes=5)
    model.summary(line_length=80)
