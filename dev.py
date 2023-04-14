"""
predict.py

Predicts the pose of a single image using the MoveNet model.
"""

import os
from datetime import datetime

import cv2
#import supervisely_lib as sly
import numpy as np

import tensorflow as tf
import onnx
import onnxruntime as ort

from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.patches as patches



def load_local_test_image():
  """Load a local test image"""
  image_path = "/mnt/c/Users/owner/Software/couro-models/data/A2D/RunClips/_aJOs5B9T-Q/00141.png"
  image = tf.io.read_file(image_path)
  image = tf.image.decode_jpeg(image, channels=3)
  return image

def resize_image(image, dim_size=256):
  """Resize the image to 256x256."""
  x = tf.expand_dims(image, axis=0)
  x = tf.image.resize_with_pad(x, dim_size, dim_size)
  x = tf.cast(x, dtype=tf.float32)
  return x

def load_remote_test_image():
  pass

def load_movenet_model(model_path:str="/mnt/c/Users/owner/Software/couro-models/base/lite-model_movenet_singlepose_thunder_3.tflite"):
  """Loads the model from the model directory."""
  interpreter = tf.lite.Interpreter(model_path=model_path)
  interpreter.allocate_tensors()
  return interpreter

def load_blazepose_model(model_path:str="/mnt/c/Users/owner/Software/couro-models/base/blazepose_heatmap_v1.1.onnx"):
  """Loads the model from the model directory."""
  model = onnx.load(model_path)
  onnx.checker.check_model(model)
  return model

def blazepose_inference(x, model_path:str="/mnt/c/Users/owner/Software/couro-models/base/blazepose_heatmap_v1.1.onnx"):
  """"""
  session = ort.InferenceSession(model_path)
  output = session.run(None, {'input': x.numpy()})
  return output

def movenet_predict(interpreter, x):
  """Runs the model on the input image and returns the keypoints."""
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()
  interpreter.set_tensor(input_details[0]['index'], x.numpy())
  interpreter.invoke()
  keypoints = interpreter.get_tensor(output_details[0]['index'])
  return keypoints

def calculate_angle(proximal:float, medial:float, distal:float):
  """Calculate the angle given 3 keypoints (proxima, medial, and distal)."""
  radians = np.arctan2(distal[1]-medial[1], distal[0]-medial[0]) - np.arctan2(proximal[1]-medial[1], proximal[0]-medial[0])
  angle = np.abs((radians * 180.0) / np.pi)
  
  if angle > 180.0:
      angle = 360 - angle
      
  return angle

def return_2D_joint_coord(keypoints:np.ndarray):
  """Returns the joint angles using key points from MoveNet model."""

  keypoints = np.squeeze(keypoints)

  left_elbow   = [keypoints[KEYPOINT_DICT['left_elbow'], :][0],      
                    keypoints[KEYPOINT_DICT['left_elbow'], :][1]]
  left_shoulder = [keypoints[KEYPOINT_DICT['left_shoulder'], :][0],
                    keypoints[KEYPOINT_DICT['left_shoulder'], :][1]]
  left_hip      = [keypoints[KEYPOINT_DICT['left_hip'], :][0],
                    keypoints[KEYPOINT_DICT['left_hip'], :][1]]
  left_knee     = [keypoints[KEYPOINT_DICT['left_knee'], :][0],
                    keypoints[KEYPOINT_DICT['left_knee'], :][1]]
  left_ankle    = [keypoints[KEYPOINT_DICT['left_ankle'], :][0],
                    keypoints[KEYPOINT_DICT['left_ankle'], :][1]]

  right_elbow   = [keypoints[KEYPOINT_DICT['right_elbow'], :][0],
                    keypoints[KEYPOINT_DICT['right_elbow'], :][1]]
  right_shoulder = [keypoints[KEYPOINT_DICT['right_shoulder'], :][0],
                    keypoints[KEYPOINT_DICT['right_shoulder'], :][1]]
  right_hip      = [keypoints[KEYPOINT_DICT['right_hip'], :][0],
                    keypoints[KEYPOINT_DICT['right_hip'], :][1]]
  right_knee     = [keypoints[KEYPOINT_DICT['right_knee'], :][0],
                    keypoints[KEYPOINT_DICT['right_knee'], :][1]]
  right_ankle    = [keypoints[KEYPOINT_DICT['right_ankle'], :][0],
                    keypoints[KEYPOINT_DICT['right_ankle'], :][1]]

  left = (left_shoulder, left_elbow, left_hip, left_knee, left_ankle)
  right = (right_shoulder, right_elbow, right_hip, right_knee, right_ankle)
  return left, right

def calculate_2D_joint_angles(keypoints:np.ndarray):
  """Calculates the 2D joint angles of the pose landmarks."""
  left, right = return_2D_joint_coord(keypoints)
  left_shoulder, left_elbow, left_hip, left_knee, left_ankle = left
  right_shoulder, right_elbow, right_hip, right_knee, right_ankle = right

  right_angle_shoulder = calculate_angle(right_elbow, right_shoulder, right_hip)
  right_angle_knee = calculate_angle(right_hip, right_knee, right_ankle)
  right_angle_hip = calculate_angle(right_shoulder, right_hip, right_knee)

  left_angle_shoulder = calculate_angle(left_elbow, left_shoulder, left_hip)
  left_angle_knee = calculate_angle(left_hip, left_knee, left_ankle)
  left_angle_hip = calculate_angle(left_shoulder, left_hip, left_knee)

  left_angle = (left_angle_shoulder, left_angle_knee, left_angle_hip)
  right_angle = (right_angle_shoulder, right_angle_knee, right_angle_hip)
  return left_angle, right_angle

def _keypoints_and_edges_for_display(keypoints_with_scores,
                                     height,
                                     width,
                                     keypoint_threshold=0.11):
  """Returns high confidence keypoints and edges for visualization.

  Args:
    keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
      the keypoint coordinates and scores returned from the MoveNet model.
    height: height of the image in pixels.
    width: width of the image in pixels.
    keypoint_threshold: minimum confidence score for a keypoint to be
      visualized.

  Returns:
    A (keypoints_xy, edges_xy, edge_colors) containing:
      * the coordinates of all keypoints of all detected entities;
      * the coordinates of all skeleton edges of all detected entities;
      * the colors in which the edges should be plotted.
  """
  keypoints_all = []
  keypoint_edges_all = []
  edge_colors = []
  num_instances, _, _, _ = keypoints_with_scores.shape
  for idx in range(num_instances):
    kpts_x = keypoints_with_scores[0, idx, :, 1]
    kpts_y = keypoints_with_scores[0, idx, :, 0]
    kpts_scores = keypoints_with_scores[0, idx, :, 2]
    kpts_absolute_xy = np.stack(
        [width * np.array(kpts_x), height * np.array(kpts_y)], axis=-1)
    kpts_above_thresh_absolute = kpts_absolute_xy[
        kpts_scores > keypoint_threshold, :]
    keypoints_all.append(kpts_above_thresh_absolute)

    for edge_pair, color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
      if (kpts_scores[edge_pair[0]] > keypoint_threshold and
          kpts_scores[edge_pair[1]] > keypoint_threshold):
        x_start = kpts_absolute_xy[edge_pair[0], 0]
        y_start = kpts_absolute_xy[edge_pair[0], 1]
        x_end = kpts_absolute_xy[edge_pair[1], 0]
        y_end = kpts_absolute_xy[edge_pair[1], 1]
        line_seg = np.array([[x_start, y_start], [x_end, y_end]])
        keypoint_edges_all.append(line_seg)
        edge_colors.append(color)
  if keypoints_all:
    keypoints_xy = np.concatenate(keypoints_all, axis=0)
  else:
    keypoints_xy = np.zeros((0, 17, 2))

  if keypoint_edges_all:
    edges_xy = np.stack(keypoint_edges_all, axis=0)
  else:
    edges_xy = np.zeros((0, 2, 2))
  return keypoints_xy, edges_xy, edge_colors

def draw_prediction_on_image(
    image, keypoints_with_scores, crop_region=None, close_figure=False,
    output_image_height=None):
  """Draws the keypoint predictions on image.

  Args:
    image: A numpy array with shape [height, width, channel] representing the
      pixel values of the input image.
    keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
      the keypoint coordinates and scores returned from the MoveNet model.
    crop_region: A dictionary that defines the coordinates of the bounding box
      of the crop region in normalized coordinates (see the init_crop_region
      function below for more detail). If provided, this function will also
      draw the bounding box on the image.
    output_image_height: An integer indicating the height of the output image.
      Note that the image aspect ratio will be the same as the input image.

  Returns:
    A numpy array with shape [out_height, out_width, channel] representing the
    image overlaid with keypoint predictions.
  """
  height, width, channel = image.shape
  aspect_ratio = float(width) / height
  fig, ax = plt.subplots(figsize=(12 * aspect_ratio, 12))
  # To remove the huge white borders
  fig.tight_layout(pad=0)
  ax.margins(0)
  ax.set_yticklabels([])
  ax.set_xticklabels([])
  plt.axis('off')

  im = ax.imshow(image)
  line_segments = LineCollection([], linewidths=(4), linestyle='solid')
  ax.add_collection(line_segments)
  # Turn off tick labels
  scat = ax.scatter([], [], s=60, color='#FF1493', zorder=3)

  (keypoint_locs, keypoint_edges,
   edge_colors) = _keypoints_and_edges_for_display(
       keypoints_with_scores, height, width)

  line_segments.set_segments(keypoint_edges)
  line_segments.set_color(edge_colors)
  if keypoint_edges.shape[0]:
    line_segments.set_segments(keypoint_edges)
    line_segments.set_color(edge_colors)
  if keypoint_locs.shape[0]:
    scat.set_offsets(keypoint_locs)

  if crop_region is not None:
    xmin = max(crop_region['x_min'] * width, 0.0)
    ymin = max(crop_region['y_min'] * height, 0.0)
    rec_width = min(crop_region['x_max'], 0.99) * width - xmin
    rec_height = min(crop_region['y_max'], 0.99) * height - ymin
    rect = patches.Rectangle(
        (xmin, ymin),rec_width,rec_height,
        linewidth=1,edgecolor='b',facecolor='none')
    ax.add_patch(rect)

  fig.canvas.draw()
  image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
  image_from_plot = image_from_plot.reshape(
      fig.canvas.get_width_height()[::-1] + (3,))
  plt.close(fig)

  if output_image_height is not None:
    output_image_width = int(output_image_height / height * width)
    image_from_plot = cv2.resize(
        image_from_plot, dsize=(output_image_width, output_image_height),
         interpolation=cv2.INTER_CUBIC)
  return image_from_plot

def visualize_keypoints(image, keypoints, dim_size=256):
  """Visualizes the keypoints on the image."""
  display_image = tf.expand_dims(image, axis=0)
  display_image = tf.cast(tf.image.resize_with_pad(
      display_image, dim_size, dim_size), dtype=tf.int32)
  output_overlay = draw_prediction_on_image(np.squeeze(display_image.numpy(), axis=0), keypoints)
  plt.figure(figsize=(5, 5))
  plt.imshow(output_overlay)
  plt.savefig(os.path.join('./test/images/', datetime.now().strftime('%Y%m-%d%H-%M%S')))
  _ = plt.axis('off')

def movenet_video_inference(mp4_path:str, num_of_frames_you_want:int=20):
  """Inference on a video file."""
  cap = cv2.VideoCapture(mp4_path)
  model = load_movenet_model()
  dim_size = 256

  total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  step_size = total_frames // num_of_frames_you_want
  frame_count = 0

  while True:
    ret, frame = cap.read()
    if ret:
      if frame_count % step_size == 0:
        x = resize_image(frame, dim_size=dim_size)
        image = tf.image.resize_with_pad(frame, dim_size, dim_size)

        keypoints = movenet_predict(model, x)
        visualize_keypoints(image, keypoints)
      frame_count += 1
    else:
      break

    # Display the frame
    cv2.imshow('Video', frame)
    cv2.waitKey(1)
  
  cap.release()
  cv2.destroyAllWindows()

def visualize_movenet_models(model_list:list, input_sizes:list):
  """"""
  base = "/mnt/c/Users/owner/Software/couro-models/base/"
  for idx, model_path in enumerate(model_list):
    dim_size = input_sizes[idx]
    image = load_local_test_image()
    x = resize_image(image, dim_size=dim_size)
    image = tf.image.resize_with_pad(image, dim_size, dim_size)
    
    model = load_movenet_model(os.path.join(base, model_path))
    keypoints = movenet_predict(model, x)
    visualize_keypoints(image, keypoints)

if __name__ == "__main__":

  # Joint analysis on a static image with Thunder MoveNet model
  #model = load_movenet_model()
  #image = load_local_test_image()
  #x = resize_image(image)
  #image = tf.image.resize_with_pad(image, 256, 256)

  #keypoints = movenet_predict(model, x)
  #left_angle, right_angle = calculate_2D_joint_angles(keypoints)
  #print(left_angle, right_angle)

  #print(keypoints[KEYPOINT_DICT['left_elbow']])
  # Static images on different MoveNet models
  #models = ["lite-model_movenet_singlepose_lightning_3.tflite", "lite-model_movenet_singlepose_thunder_3.tflite"]
  #input_sizes = [192, 256]
  #visualize_movenet_models(models, input_sizes)

  # BlazePose model
  #model = load_blazepose_model()

  # Inference on a video file
  mp4_path = "/mnt/c/Users/owner/Software/couro-models/data/A2D/Run/_aJOs5B9T-Q.mp4"
  movenet_video_inference(mp4_path)