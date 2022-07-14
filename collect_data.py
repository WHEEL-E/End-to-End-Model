import datetime
import os
import time

import cv2
import keyboard_control_gui
import pygame

import camera
import control


def take_snapshot(frame, path):
    """
    Take a snapshot of the current frame and save it to the given path
    """
    file_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S%f.jpg")
    if not os.path.exists(path):
        os.makedirs(path)
    cv2.imwrite(os.path.join(path, file_name), frame)


def main(delay: float = 0.0, preview: bool = False):
    """
    Control the wheelchair using the keyboard,
    and collect data for training the model.
    """
    pygame.init()
    pygame.display.set_mode((100, 100))
    stop_flag = True

    try:
        while True:
            frame = camera.capture()
            if preview:
                camera.preview(frame)

            if keyboard_control_gui.getKey("UP") or keyboard_control_gui.getKey("w"):
                control.forward()
                take_snapshot(frame, "data/raw/forward")
                stop_flag = False
            elif keyboard_control_gui.getKey("LEFT") or keyboard_control_gui.getKey("a"):
                control.left()
                take_snapshot(frame, "data/raw/left")
                stop_flag = False
            elif keyboard_control_gui.getKey("DOWN") or keyboard_control_gui.getKey("s"):
                control.backward()
                stop_flag = False
            elif keyboard_control_gui.getKey("RIGHT") or keyboard_control_gui.getKey("d"):
                control.right()
                take_snapshot(frame, "data/raw/right")
                stop_flag = False
            elif keyboard_control_gui.getKey("SPACE") or keyboard_control_gui.getKey("o"):
                control.stop()
                take_snapshot(frame, "data/raw/stop")
            else:
                if not stop_flag:
                    control.stop()
                    stop_flag = True
            time.sleep(delay)

    except KeyboardInterrupt:
        control.stop()
        camera.cap.release()
        cv2.destroyAllWindows()
        pygame.quit()


if __name__ == "__main__":
    control.main()
    main(delay=0.1, preview=True)
