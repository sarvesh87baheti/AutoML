from tensorflow.keras.applications import ResNet50V2, ResNet101V2
from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnet_preprocess_input

from .imageclassification_backbone_utils import build_transfer_learning_model


RESNET_CONFIGS = {
    "resnet50v2": {
        "builder": ResNet50V2,
        "recommended_size": 224,
    },
    "resnet101v2": {
        "builder": ResNet101V2,
        "recommended_size": 224,
    },
}


def list_resnets():
    print("ResNet:", ", ".join(RESNET_CONFIGS.keys()))


def build_resnet_model(
    *,
    name,
    num_classes,
    input_shape,
    dropout_rate=0.3,
    freeze_backbone=True,
):
    config = RESNET_CONFIGS[name]
    return build_transfer_learning_model(
        base_model_fn=config["builder"],
        preprocess_input=resnet_preprocess_input,
        input_shape=input_shape,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        freeze_backbone=freeze_backbone,
    )
