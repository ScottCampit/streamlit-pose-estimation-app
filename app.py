"""
app.py

Create MVP development application using Streamlit.

Note that this is NOT a production ready application.
"""

import sys
import os
import io

import tensorflow as tf
import numpy as np
import streamlit as st
from PIL import Image
import cv2

sys.path.append('/mnt/c/Users/owner/Software/couro-models/expts/movenet/')
import predict

@st.cache(allow_output_mutation=True)
def load_model():
    return predict.load_model()

if __name__ == "__main__":

    st.title("Couro Pose Estimation Demo")
    st.write("This is a demo of the Couro Pose Estimation model. Currently, the foundation model in-use is the MoveNet model (from Tensorflow team @ Google).")



    uploaded_files = st.file_uploader(
        label="Choose an image", 
        type=["jpg", "png"], 
        accept_multiple_files=True)
    if uploaded_files is None:
        image_path = "/mnt/c/Users/owner/Software/couro-models/data/A2D/RunClips/_aJOs5B9T-Q/00141.png"
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        x = tf.expand_dims(image, axis=0)
        x = tf.image.resize_with_pad(x, 256, 256)
        x = tf.cast(x, dtype=tf.float32)

    elif uploaded_files is not None:
        for uploaded_file in uploaded_files:
            image = uploaded_file.getvalue()
            st.image(image)
            tmp = Image.open(io.BytesIO(image))
            image_arr = np.array(tmp)
            x = tf.expand_dims(image_arr, axis=0)
            x = tf.image.resize_with_pad(x, 256, 256)
            x = tf.cast(x, dtype=tf.float32)

    if st.button("Annotate"):
        model = load_model()
        keypoints = predict.predict(model, x)
        output_overlay = predict.visualize_keypoints(image_arr, keypoints)
        
        st.image(output_overlay)