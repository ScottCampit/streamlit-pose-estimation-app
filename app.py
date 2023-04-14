"""
app.py

Create MVP development application using Streamlit.
"""

import sys
import os
import io
import subprocess
import tempfile

import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.patches as patches

import streamlit as st

from couro import process_image, predict, visualize_keypoints, calculate_2D_joint_angles

@st.cache_resource()
def load_model():
    """Loads the model from the model directory."""
    model_path = "./lite-model_movenet_singlepose_thunder_3.tflite"
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

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
              if uploaded_files[0].name.split('.')[-1] == 'mov':
                output_file = "temp.mp4"
                command = ["ffmpeg", "-i", file, output_file]
                subprocess.run(command, check=True)
              temp.write(output_file.read())
              st.video(temp.name)

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