from tensorflow.keras.applications import DenseNet121, DenseNet169, DenseNet201
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess_input

from .imageclassification_backbone_utils import build_transfer_learning_model


DENSENET_CONFIGS = {
    "densenet121": {
        "builder": DenseNet121,
        "recommended_size": 224,
    },
    "densenet169": {
        "builder": DenseNet169,
        "recommended_size": 224,
    },
    "densenet201": {
        "builder": DenseNet201,
        "recommended_size": 224,
    },
}


def list_densenets():
    print("DenseNet:", ", ".join(DENSENET_CONFIGS.keys()))


def build_densenet_model(
    *,
    name,
    num_classes,
    input_shape,
    dropout_rate=0.3,
    freeze_backbone=True,
):
    config = DENSENET_CONFIGS[name]
    return build_transfer_learning_model(
        base_model_fn=config["builder"],
        preprocess_input=densenet_preprocess_input,
        input_shape=input_shape,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        freeze_backbone=freeze_backbone,
    )
