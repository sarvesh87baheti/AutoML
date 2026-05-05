from tensorflow.keras.applications import MobileNetV2, MobileNetV3Large, MobileNetV3Small
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenetv2_preprocess_input
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input as mobilenetv3_preprocess_input

from .imageclassification_backbone_utils import build_transfer_learning_model


MOBILENET_CONFIGS = {
    "mobilenetv2": {
        "builder": MobileNetV2,
        "preprocess_input": mobilenetv2_preprocess_input,
        "recommended_size": 224,
    },
    "mobilenetv3small": {
        "builder": MobileNetV3Small,
        "preprocess_input": mobilenetv3_preprocess_input,
        "recommended_size": 224,
    },
    "mobilenetv3large": {
        "builder": MobileNetV3Large,
        "preprocess_input": mobilenetv3_preprocess_input,
        "recommended_size": 224,
    },
}


def list_mobilenets():
    print("MobileNet:", ", ".join(MOBILENET_CONFIGS.keys()))


def build_mobilenet_model(
    *,
    name,
    num_classes,
    input_shape,
    dropout_rate=0.3,
    freeze_backbone=True,
):
    config = MOBILENET_CONFIGS[name]
    return build_transfer_learning_model(
        base_model_fn=config["builder"],
        preprocess_input=config["preprocess_input"],
        input_shape=input_shape,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        freeze_backbone=freeze_backbone,
    )
