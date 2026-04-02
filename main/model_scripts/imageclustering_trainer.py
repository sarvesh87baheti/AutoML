from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import tensorflow as tf

def train_model(model, base_model, train_ds, val_ds, save_path):

    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True),
        ReduceLROnPlateau(patience=3),
        ModelCheckpoint(save_path, save_best_only=True)
    ]

    # Phase 1
    model.fit(train_ds, validation_data=val_ds, epochs=20, callbacks=callbacks)

    # Phase 2 (fine-tuning)
    base_model.trainable = True
    for layer in base_model.layers[:-40]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=model.loss,
        metrics=["accuracy"]
    )

    model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=callbacks)

    return model