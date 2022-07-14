# for windows users:
# pip install --extra-index-url https://google-coral.github.io/py-repo tflite_runtime
import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter

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

    interpreter = Interpreter(
        model_path="models/tflite/Autonomus-model-0508-1145.tflite"
    )
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    height = input_details[0]["shape"][1]
    width = input_details[0]["shape"][2]

    try:
        while True:
            frame = camera.capture()
            frame = camera.resize(frame, width=width, height=height)
            frame_array = np.array(np.expand_dims(frame, axis=0), dtype=np.float32)

            interpreter.set_tensor(input_details[0]["index"], frame_array)
            interpreter.invoke()

            prediction = interpreter.get_tensor(output_details[0]["index"])
            take_action(actions[prediction.argmax()])

    except KeyboardInterrupt:
        control.stop()
        camera.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    control.main()
    main()
