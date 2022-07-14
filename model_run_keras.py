import cv2
import numpy as np
from tensorflow import keras

import camera
import control
import profiling

action_map = {
    "w": control.forward,
    "a": control.left,
    "d": control.right,
    "o": control.stop,
}


def take_action(x):
    action_map[x]()


@profiling.profile
def main():
    actions = ["w", "a", "d", "o"]

    loaded_model = keras.models.load_model("models/h5/Autonomus-model-0508-1145.h5")
    height = loaded_model.input_shape[1]
    width = loaded_model.input_shape[2]

    try:
        while True:
            frame = camera.capture()
            frame = camera.flip(frame)
            frame = camera.resize(frame, width=width, height=height)
            frame_array = np.array(np.expand_dims(frame, axis=0), dtype=np.float32)

            prediction = loaded_model.predict(frame_array)
            take_action(actions[prediction.argmax()])

    except KeyboardInterrupt:
        control.stop()
        camera.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    control.main()
    main()
