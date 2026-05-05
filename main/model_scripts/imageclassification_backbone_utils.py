import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_transfer_learning_model(
    *,
    base_model_fn,
    preprocess_input,
    input_shape,
    num_classes,
    dropout_rate=0.3,
    freeze_backbone=True,
    dense_units=256,
):
    """
    Build a standard transfer-learning classifier around a Keras Applications backbone.

    Returns:
        (model, preprocess_input)
    """
    base_model = base_model_fn(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = not freeze_backbone

    inputs = keras.Input(shape=input_shape)
    x = layers.Lambda(preprocess_input, name="preprocess")(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)

    if dense_units:
        x = layers.Dense(dense_units, activation="relu")(x)
        x = layers.Dropout(dropout_rate)(x)

    if num_classes == 2:
        outputs = layers.Dense(1, activation="sigmoid")(x)
    else:
        outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name=f"{base_model.name}_classifier")
    return model, preprocess_input
