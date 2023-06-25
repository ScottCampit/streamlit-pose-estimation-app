"""
Unit testing
"""

import pytest
import sys

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

sys.path.append("/Users/scottcampit/Projects/pose-estimation/")
from couro.processing import predict, process_image, calculate_2D_joint_angles, _get_best_keypoints, add_angles_to_keypoints, draw_keypoints, draw_edges

def test_num_mov_frames(test_mov):
    """Tests that the number of frames in the video is correct."""
    assert len(test_mov) == 20

def test_num_mp4_frames(test_mp4):
    """Tests that the number of frames in the video is correct."""
    assert len(test_mp4) == 20

def test_process_image_shape(test_png):
    """Tests that the processed image has the correct shape."""
    x = process_image(test_png)
    assert x.shape == (1, 256, 256, 3)

def test_process_image_type(test_png):
    """Tests that the processed image has a dtype of float32."""
    x = process_image(test_png)
    assert x.dtype == "float32"

def test_original_image_shape(test_png):
    """Tests that the original image has the correct shape."""
    image = tf.image.resize_with_pad(test_png, 256, 256)
    image = tf.cast(image, dtype=tf.uint8)
    assert image.shape == (256, 256, 3)

def test_original_image_type(test_png):
    """Tests that the original image has a dtype of uint8."""
    image = tf.image.resize_with_pad(test_png, 256, 256)
    image = tf.cast(image, dtype=tf.uint8)
    assert image.dtype == "uint8"

def test_image_expand_dim(test_png):
    """Tests that the image has the correct shape after expanding the dimensions."""
    image = tf.image.resize_with_pad(test_png, 256, 256)
    image = tf.cast(image, dtype=tf.uint8)
    image = tf.expand_dims(image, axis=0)
    assert image.shape == (1, 256, 256, 3)

def test_inference_output(test_model, test_png):
    """Tests that the inference output has the correct shape."""
    x = process_image(test_png)
    keypoints = predict(test_model, x)
    assert keypoints.shape == (1, 1, 17, 3)

def test_inference_output_type(test_model, test_png):
    """Tests that the inference output has a dtype of float32."""
    x = process_image(test_png)
    keypoints = predict(test_model, x)
    assert keypoints.dtype == "float32"

def test_getting_best_keypoints(test_model, test_png):
    """Tests that the best keypoints are returned from the inference output. Note that we are doing 2D pose estimation, and thus we are only returning the x,y coordinates."""
    image = tf.image.resize_with_pad(test_png, 256, 256)
    height, width, _ = image.numpy().shape
    x = process_image(test_png)
    keypoints_with_scores = predict(test_model, x)
    final_keypoints = _get_best_keypoints(keypoints_with_scores, height, width)
    assert final_keypoints[0].shape == (17, 2)

def test_getting_best_keypoints_type(test_model, test_png):
    """Tests that the best keypoints are returned from the inference output. Note that we are doing 2D pose estimation, and thus we are only returning the x,y coordinates."""
    image = tf.image.resize_with_pad(test_png, 256, 256)
    height, width, _ = image.numpy().shape
    x = process_image(test_png)
    keypoints_with_scores = predict(test_model, x)
    final_keypoints = _get_best_keypoints(keypoints_with_scores, height, width)
    assert final_keypoints[0].dtype == "float32"

def test_return_2D_join_coord(test_model, test_png):
    """Tests that the 2D joint coordinates are returned."""
    image = tf.image.resize_with_pad(test_png, 256, 256)
    height, width, _ = image.numpy().shape
    x = process_image(test_png)
    keypoints_with_scores = predict(test_model, x)
    final_keypoints = _get_best_keypoints(keypoints_with_scores, height, width)
    left_angles, right_angles = calculate_2D_joint_angles(frame=np.squeeze(image.numpy()), keypoints=keypoints_with_scores)
    assert len(left_angles) == 3

def test_add_angles_to_keypoints_size(test_model, test_png):
    """Tests that the 2D joint coordinates are returned."""
    image = tf.image.resize_with_pad(test_png, 256, 256)
    image = image.numpy()
    height, width, _ = image.shape
    x = process_image(test_png)
    keypoints_with_scores = predict(test_model, x)
    final_keypoints = _get_best_keypoints(keypoints_with_scores, height, width)
    left_angles, right_angles = calculate_2D_joint_angles(frame=np.squeeze(image), keypoints=keypoints_with_scores)
    frame = add_angles_to_keypoints(image, final_keypoints, left_angles, right_angles)
    assert frame.shape == (256, 256, 3)

@pytest.mark.skip(reason="This test is for visual inspection only.")
def test_plt_show_image(test_png):
    image = tf.image.resize_with_pad(test_png, 256, 256)
    image = tf.cast(image, dtype=tf.uint8)
    plt.imshow(image)
    plt.show()

@pytest.mark.skip(reason="This test is for visual inspection only.")
def test_draw_keypoints_onto_image(test_model, test_png):
    image = tf.image.resize_with_pad(test_png, 256, 256)
    image = tf.cast(image, dtype=tf.uint8)
    image = image.numpy()
    height, width, _ = image.shape
    x = process_image(test_png)
    keypoints_with_scores = predict(test_model, x)
    final_keypoints = _get_best_keypoints(keypoints_with_scores, height, width)
    frame = draw_keypoints(image, final_keypoints)
    plt.imshow(frame)
    plt.show()

def test_draw_edges_onto_image(test_model, test_png):
    image = tf.image.resize_with_pad(test_png, 256, 256)
    image = tf.cast(image, dtype=tf.uint8)
    image = image.numpy()
    height, width, _ = image.shape
    x = process_image(test_png)
    keypoints_with_scores = predict(test_model, x)
    final_keypoints = _get_best_keypoints(keypoints_with_scores, height, width)
    frame = draw_edges(image, final_keypoints)
    plt.imshow(frame)
    plt.show()

@pytest.mark.skip(reason="This test is for visual inspection only.")
def test_plt_show_annot_frame(test_model, test_png):
    image = tf.image.resize_with_pad(test_png, 256, 256)
    image = tf.cast(image, dtype=tf.uint8)
    image = image.numpy()
    height, width, _ = image.shape
    x = process_image(test_png)
    keypoints_with_scores = predict(test_model, x)
    final_keypoints = _get_best_keypoints(keypoints_with_scores, height, width)
    left_angles, right_angles = calculate_2D_joint_angles(frame=np.squeeze(image), keypoints=keypoints_with_scores)
    frame = add_angles_to_keypoints(image, final_keypoints, left_angles, right_angles)
    plt.imshow(frame)
    plt.show()