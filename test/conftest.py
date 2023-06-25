"""
conftest.py
"""

import pytest
import tensorflow as tf
import cv2
from PIL import Image

from couro.processing import process_image, predict

def get_frames(_video):
  """"""
  cap = cv2.VideoCapture(_video)

  total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  step_size = total_frames // 20
  frame_counts = 0

  frames = list()
  for i in range(0, total_frames, step_size):
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ret, frame = cap.read()
    if not ret:
      break
    frames.append(frame)
    frame_counts += 1
    if frame_counts > 20-1:
       break
  return frames

@pytest.fixture
def test_mp4():
    return get_frames('/Users/scottcampit/Projects/pose-estimator/data/_aJOs5B9T-Q.mp4')

@pytest.fixture
def test_mov():
    return get_frames('/Users/scottcampit/Projects/pose-estimator/data/Video Apr 02, 3 46 51 PM.mov')

@pytest.fixture
def test_jpg():
    pass

@pytest.fixture
def test_png():
    return Image.open("/Users/scottcampit/Projects/pose-estimator/data/_aJOs5B9T-Q/00146.png")

@pytest.fixture
def test_model():
  model_path="/Users/scottcampit/Projects/pose-estimator/models/lite-model_movenet_singlepose_thunder_3.tflite"
  interpreter = tf.lite.Interpreter(model_path=model_path)
  interpreter.allocate_tensors()
  return interpreter

@pytest.fixture
def test_model_prediction(test_png, test_model):
  x = process_image(test_png)
  keypoints = predict(test_model, x)
  return keypoints
   


