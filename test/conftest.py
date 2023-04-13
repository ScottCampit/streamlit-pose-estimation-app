"""
conftest.py
"""

import pytest
import cv2
from PIL import Image

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
    return get_frames('../data/_aJOs5B9T-Q.mp4')

@pytest.fixture
def test_mov():
    return get_frames('../data/Video Apr 02, 3 46 51 PM.mov')

@pytest.fixture
def test_jpg():
    pass

@pytest.fixture
def test_png():
    return Image.open("../data/_aJOs5B9T-Q/00146.png")

