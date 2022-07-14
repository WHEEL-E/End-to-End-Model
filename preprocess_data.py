import os
import random
from pathlib import Path
from shutil import copyfile

import cv2
import numpy as np


def random_flip(img, action):
    if np.random.rand() < 0.5:
        img = cv2.flip(img, 1)
        action = -action
    return img, action


def random_shadow(img):
    bright_factor = 0.3
    x = random.randint(0, img.shape[1])
    y = random.randint(0, img.shape[0])

    width = random.randint(img.shape[1], img.shape[1])
    if x + width > img.shape[1]:
        x = img.shape[1] - x

    height = random.randint(img.shape[0], img.shape[0])
    if y + height > img.shape[0]:
        y = img.shape[0] - y

    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img[y : y + height, x : x + width, 2] = (
        img[y : y + height, x : x + width, 2] * bright_factor
    )
    return cv2.cvtColor(img, cv2.COLOR_HSV2RGB)


def random_brightness(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + (np.random.rand() - 0.5)
    img[:, :, 2] = img[:, :, 2] * ratio
    return cv2.cvtColor(img, cv2.COLOR_HSV2RGB)


def split_data(source, training, testing, split_size):
    files = []

    for f in os.listdir(source):
        f_path = source + f
        if os.path.getsize(f_path):
            files.append(f)

    split_point = int(len(files) * split_size)
    random.shuffle(files)
    train_set = files[:split_point]
    val_set = files[split_point:]

    for f in train_set:
        copyfile(os.path.join(source, f), os.path.join(training, f))

    for f in val_set:
        copyfile(os.path.join(source, f), os.path.join(testing, f))


def main():
    train_forward_dir = "data/training/forward"
    Path(train_forward_dir).mkdir(parents=True, exist_ok=True)
    train_left_dir = "data/training/left"
    Path(train_left_dir).mkdir(parents=True, exist_ok=True)
    train_right_dir = "data/training/right"
    Path(train_right_dir).mkdir(parents=True, exist_ok=True)
    train_stop_dir = "data/training/stop"
    Path(train_stop_dir).mkdir(parents=True, exist_ok=True)

    val_forward_dir = "data/testing/forward"
    Path(val_forward_dir).mkdir(parents=True, exist_ok=True)
    val_left_dir = "data/testing/left"
    Path(val_left_dir).mkdir(parents=True, exist_ok=True)
    val_right_dir = "data/testing/right"
    Path(val_right_dir).mkdir(parents=True, exist_ok=True)
    val_stop_dir = "data/testing/stop"
    Path(val_stop_dir).mkdir(parents=True, exist_ok=True)

    split_data("data/forward/", train_forward_dir, val_forward_dir, 0.8)
    split_data("data/left/", train_left_dir, val_left_dir, 0.8)
    split_data("data/right/", train_right_dir, val_right_dir, 0.8)
    split_data("data/stop/", train_stop_dir, val_stop_dir, 0.8)


main()
