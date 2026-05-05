from tensorflow.keras.applications import InceptionResNetV2, InceptionV3, Xception
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input as inception_resnet_preprocess_input
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_v3_preprocess_input
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess_input

from .imageclassification_backbone_utils import build_transfer_learning_model


INCEPTION_CONFIGS = {
    "inceptionv3": {
        "builder": InceptionV3,
        "preprocess_input": inception_v3_preprocess_input,
        "recommended_size": 299,
    },
    "inceptionresnetv2": {
        "builder": InceptionResNetV2,
        "preprocess_input": inception_resnet_preprocess_input,
        "recommended_size": 299,
    },
    "xception": {
        "builder": Xception,
        "preprocess_input": xception_preprocess_input,
        "recommended_size": 299,
    },
}


def list_inceptions():
    print("Inception:", ", ".join(INCEPTION_CONFIGS.keys()))


def build_inception_model(
    *,
    name,
    num_classes,
    input_shape,
    dropout_rate=0.3,
    freeze_backbone=True,
):
    config = INCEPTION_CONFIGS[name]
    return build_transfer_learning_model(
        base_model_fn=config["builder"],
        preprocess_input=config["preprocess_input"],
        input_shape=input_shape,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        freeze_backbone=freeze_backbone,
    )
