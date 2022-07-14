import datetime
import os

import cv2
import tensorflow as tf
from tensorflow import keras

shape = cv2.imread(
    "data/training/forward/" + os.listdir("data/training/forward")[0]
).shape

train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=40,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    fill_mode="nearest",
)

train_dir = "data/training"
train_generator = train_datagen.flow_from_directory(
    train_dir, batch_size=30, class_mode="categorical", target_size=(shape[0], shape[1])
)

validation_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)

val_dir = "data/testing"
validation_generator = validation_datagen.flow_from_directory(
    val_dir, batch_size=30, class_mode="categorical", target_size=(shape[0], shape[1])
)


class EarlyStopping(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get("accuracy") > 0.95:
            print(
                "\n95% accuracy reached in epoch {}\nStopping training 👻👻👻👻👻👻\n".format(
                    epoch
                )
            )
            self.model.stop_training = True


model = tf.keras.models.Sequential(
    [
        keras.layers.Conv2D(
            24, (5, 5), activation="relu", strides=(2, 2), input_shape=shape
        ),
        keras.layers.Conv2D(36, (5, 5), activation="relu", strides=(2, 2)),
        keras.layers.Conv2D(48, (5, 5), activation="relu", strides=(2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation="relu"),
        keras.layers.Conv2D(64, (3, 3), activation="relu"),
        keras.layers.Dropout(0.5),
        keras.layers.Flatten(),
        keras.layers.Dense(100, activation="relu"),
        keras.layers.Dense(50, activation="relu"),
        keras.layers.Dense(10, activation="relu"),
        keras.layers.Dense(4, activation="softmax"),
    ]
)

model.compile(
    loss=keras.losses.categorical_crossentropy,
    optimizer=keras.optimizers.Adam(),
    metrics=["accuracy"],
)

model_name = "Autonomus-model-{}".format(datetime.datetime.now().strftime("%m%d-%H%M"))
log_dir = "logs/" + model_name
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)


history = model.fit(
    train_generator,
    steps_per_epoch=1,
    epochs=200,
    validation_data=validation_generator,
    validation_steps=1,
    verbose=1,
    shuffle=1,
    callbacks=[
        EarlyStopping(),
        tf.keras.callbacks.EarlyStopping(monitor="loss", patience=10),
        tensorboard_callback,
    ],
)

keras_file = f"models/h5/{model_name}.h5"
tflite_file = f"models/tflite/{model_name}.tflite"

keras.models.save_model(model, keras_file, save_format="h5")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open(tflite_file, "wb") as f:
    f.write(tflite_model)
