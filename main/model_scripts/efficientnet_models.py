from tensorflow.keras.applications import (
    EfficientNetB0,
    EfficientNetB1,
    EfficientNetB2,
    EfficientNetB3,
    EfficientNetV2S,
    EfficientNetV2M,
)
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess_input

from .imageclassification_backbone_utils import build_transfer_learning_model


EFFICIENTNET_CONFIGS = {
    "efficientnetb0": {
        "builder": EfficientNetB0,
        "recommended_size": 224,
    },
    "efficientnetb1": {
        "builder": EfficientNetB1,
        "recommended_size": 240,
    },
    "efficientnetb2": {
        "builder": EfficientNetB2,
        "recommended_size": 260,
    },
    "efficientnetb3": {
        "builder": EfficientNetB3,
        "recommended_size": 300,
    },
    "efficientnetv2s": {
        "builder": EfficientNetV2S,
        "recommended_size": 300,
    },
    "efficientnetv2m": {
        "builder": EfficientNetV2M,
        "recommended_size": 384,
    },
}


def list_efficientnets():
    print("EfficientNet:", ", ".join(EFFICIENTNET_CONFIGS.keys()))


def build_efficientnet_model(
    *,
    name,
    num_classes,
    input_shape,
    dropout_rate=0.3,
    freeze_backbone=True,
):
    config = EFFICIENTNET_CONFIGS[name]
    return build_transfer_learning_model(
        base_model_fn=config["builder"],
        preprocess_input=efficientnet_preprocess_input,
        input_shape=input_shape,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        freeze_backbone=freeze_backbone,
    )
