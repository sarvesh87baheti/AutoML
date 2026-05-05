from tensorflow.keras.applications import NASNetMobile, VGG16, VGG19
from tensorflow.keras.applications.nasnet import preprocess_input as nasnet_preprocess_input
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg19_preprocess_input

from .imageclassification_backbone_utils import build_transfer_learning_model


VGG_NASNET_CONFIGS = {
    "vgg16": {
        "builder": VGG16,
        "preprocess_input": vgg16_preprocess_input,
        "recommended_size": 224,
    },
    "vgg19": {
        "builder": VGG19,
        "preprocess_input": vgg19_preprocess_input,
        "recommended_size": 224,
    },
    "nasnetmobile": {
        "builder": NASNetMobile,
        "preprocess_input": nasnet_preprocess_input,
        "recommended_size": 224,
    },
}


def list_vgg_nasnet():
    print("VGG / NASNet:", ", ".join(VGG_NASNET_CONFIGS.keys()))


def build_vgg_nasnet_model(
    *,
    name,
    num_classes,
    input_shape,
    dropout_rate=0.3,
    freeze_backbone=True,
):
    config = VGG_NASNET_CONFIGS[name]
    return build_transfer_learning_model(
        base_model_fn=config["builder"],
        preprocess_input=config["preprocess_input"],
        input_shape=input_shape,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        freeze_backbone=freeze_backbone,
    )
