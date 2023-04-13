"""
app.py

Create MVP development application using Streamlit.
"""

import sys
import os
import io
import tempfile

import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.patches as patches

import streamlit as st

@st.cache_resource()
def load_model():
    """Loads the model from the model directory."""
    model_path = "./lite-model_movenet_singlepose_thunder_3.tflite"
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def predict(interpreter, x):
    """Runs the model on the input image and returns the keypoints."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], x.numpy())
    interpreter.invoke()
    keypoints = interpreter.get_tensor(output_details[0]['index'])
    return keypoints

def process_image(_image):
    """Process image for prediction."""
    x = tf.expand_dims(_image, axis=0)
    x = tf.image.resize_with_pad(x, 256, 256)
    x = tf.cast(x, dtype=tf.float32)
    return x

def get_angles(_keypoints):
  """"""
  # Store joint angles
  left_angles, right_angles = calculate_2D_joint_angles(_keypoints)
  left_joint_str = ""
  right_joint_str = ""

  for i in range(len(left_joints)):
    left_joint_str += f"{left_joints[i]}: {round(left_angles[i], 2)}" + '\n'
    right_joint_str += f"{right_joints[i]}: {round(right_angles[i], 2)}" + '\n'
  st.session_state['all_left_joint_annot'].append(left_joint_str)
  st.session_state['all_right_joint_annot'].append(right_joint_str)

@st.cache_data(experimental_allow_widgets=True)
def get_frames(_video):
  """"""
  cap = cv2.VideoCapture(_video)

  total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  step_size = total_frames // 20

  frames = list()
  for i in range(0, total_frames, step_size):
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ret, frame = cap.read()
    if not ret:
      break
    frames.append(frame)
  return frames

@st.cache_data(experimental_allow_widgets=True)
def infer_skeleton_from_frames(frames: list):
  """"""
  frame_count = 0
  for frame in frames:
    x = process_image(frame)
    image = tf.image.resize_with_pad(frame, dim_size, dim_size)
    keypoints = predict(model, x)

    annotated_img = visualize_keypoints(image, keypoints)
    get_angles(keypoints)
    st.session_state['frame_count'].append(frame_count)
    st.session_state['annotated_img'].append(annotated_img)
    frame_count += 1
  
# Dictionary that maps from joint names to keypoint indices.
KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

# Maps bones to a matplotlib color name.
KEYPOINT_EDGE_INDS_TO_COLOR = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}

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
        (xmin,ymin),rec_width,rec_height,
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

def visualize_keypoints(image, keypoints):
    """Visualizes the keypoints on the image."""
    display_image = tf.expand_dims(image, axis=0)
    display_image = tf.cast(tf.image.resize_with_pad(
        display_image, 256, 256), dtype=tf.int32)
    output_overlay = draw_prediction_on_image(np.squeeze(display_image.numpy(), axis=0), keypoints)
    return output_overlay

def calculate_angle(proximal:float, medial:float, distal:float):
  """Calculate the angle given 3 keypoints (proxima, medial, and distal)."""
  radians = np.arctan2(distal[1]-medial[1], distal[0]-medial[0]) - np.arctan2(proximal[1]-medial[1], proximal[0]-medial[0])
  angle = np.abs((radians * 180.0) / np.pi)
  
  if angle > 180.0:
      angle = 360 - angle
      
  return angle

def movenet_video_inference(mp4_path:str, num_of_frames_you_want:int=20):
  """Inference on a video file."""
  cap = cv2.VideoCapture(mp4_path)

  total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  step_size = total_frames // num_of_frames_you_want
  frame_count = 0

  while True:
    ret, frame = cap.read()
    if ret:
      if frame_count % step_size == 0:
        x = process_image(frame)
        image = tf.image.resize_with_pad(frame, dim_size, dim_size)

        keypoints = predict(model, x)
        visualize_keypoints(image, keypoints)
      frame_count += 1
    else:
      break

    # Display the frame
    cv2.imshow('Video', frame)
    cv2.waitKey(1)
  
  cap.release()
  cv2.destroyAllWindows()

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

if __name__ == "__main__":
    st.set_page_config(layout="wide", 
                       page_title="Couro Pose Estimator")
    
    hide_default_format = """
          <style>
          #MainMenu {visibility: hidden; }
          footer {visibility: hidden;}
          </style>
          """
    st.markdown(hide_default_format, unsafe_allow_html=True)
    st.title("Couro Pose Estimator")
    st.write("This is a demo of the Couro Pose Estimation model. It is a 2D pose estimation model that can be used to detect the pose of a person in an image. The model is trained on the COCO dataset and can detect 17 keypoints.")

    left_joints = ['Left Shoulder', 'Left Hip', 'Left Knee']
    right_joints = ['Right Shoulder', 'Right Hip', 'Right Knee']

    dim_size = 256
    model = load_model()

    if 'annotated_img' not in st.session_state:
      st.session_state['annotated_img'] = list()
    
    if 'all_left_joint_annot' not in st.session_state:
      st.session_state['all_left_joint_annot'] = list()

    if 'all_right_joint_annot' not in st.session_state:
      st.session_state['all_right_joint_annot'] = list()

    if 'frame_count' not in st.session_state:
      st.session_state['frame_count'] = list()

    uploaded_files = st.file_uploader(
          label="Choose an image or video to annotate", 
          type=["jpg", "png", "mp4", "mov"], 
          accept_multiple_files=True)
    
    col1, col2 = st.columns(2)
    if uploaded_files is not None:
      try:
        if uploaded_files[0].name.split('.')[-1] == ('png' or 'jpg' or 'jpeg'):
          with col1:
            for file in uploaded_files:
              image = Image.open(file)
              x = process_image(image)
              image = tf.image.resize_with_pad(image, dim_size, dim_size)
              tmp = tf.keras.utils.array_to_img(image)
              st.image(tmp, use_column_width=True)

              if st.button("Annotate"):
                with col2:
                  keypoints = predict(model, x)
                  output_overlay = visualize_keypoints(image, keypoints)
                  left_angles, right_angles = calculate_2D_joint_angles(keypoints)

                  st.image(output_overlay)
                  left_joint_str = ""
                  right_joint_str = ""
                  for i in range(len(left_joints)):
                    left_joint_str += f"{left_joints[i]}: {round(left_angles[i], 2)}" + '\n'
                    right_joint_str += f"{right_joints[i]}: {round(right_angles[i], 2)}" + '\n'
                
                  st.subheader("Predicted joint angles:")
                  col1x, col2x, _ = st.columns(3)

                  with col1x:
                    st.text_area("Left joint angles: ", 
                              value=left_joint_str, 
                              height=100, max_chars=None, key=None)
                    
                  with col2x:
                    st.text_area("Right joint angles: ", 
                              value=right_joint_str, 
                              height=100, max_chars=None, key=None)

        elif uploaded_files[0].name.split('.')[-1] == ('mp4' or 'mov'):
          with col1:
            st.subheader("Uploaded video:")
            for file in uploaded_files:
              temp = tempfile.NamedTemporaryFile(delete=False)
              temp.write(file.read())
              if file.name.split('.')[-1] == 'mp4':
                st.video(temp.name)
              else:
                pass
              
          if st.button("Annotate"):
            with col2:
              st.subheader("Model predictions: ")
              col1x, col2x, col3x = st.columns(3, gap='large')
              get_frames(temp.name)
  
              for idx, img in enumerate(st.session_state['annotated_img']):  
                with st.container(): 
                  with col1x:
                    st.image(img, 
                            caption=st.session_state['frame_count'][idx], 
                            width=200, 
                            use_column_width=False)             

                  with col2x:
                    st.text_area(f"Left joint angles for image {st.session_state['frame_count'][idx]}: ", 
                              value=st.session_state['all_left_joint_annot'][idx], 
                              height=200, max_chars=None, key=None)
                    
                  with col3x:
                    st.text_area(f"Right joint angles for image {st.session_state['frame_count'][idx]}: ", 
                              value=st.session_state['all_right_joint_annot'][idx], 
                              height=200, max_chars=None, key=None)

        else:
          pass
      except:
        pass