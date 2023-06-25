"""
Integration testing
"""
import sys
import pytest
import argparse

import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

sys.path.append("..")
from couro.processing import process_image, predict, visualize_keypoints, calculate_2D_joint_angles, add_angles_to_keypoints
from app import load_model

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, default="../data/_aJOs5B9T-Q/00146.png")
    parser.add_argument("--model_path", type=str, default="../models/lite-model_movenet_singlepose_thunder_3.tflite")
    args = parser.parse_args()

    dim_size = 256
    model = load_model(args.model_path)
    file = "../data/_aJOs5B9T-Q/00146.png"
    image = Image.open(args.image_path)
    x = process_image(image)
    image = tf.image.resize_with_pad(image, dim_size, dim_size)
    tmp = tf.keras.utils.array_to_img(image)

    keypoints = predict(model, x)
    left_angles, right_angles = calculate_2D_joint_angles(keypoints)
    output_overlay = visualize_keypoints(image, keypoints, left_angles, right_angles)
