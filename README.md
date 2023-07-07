# streamlit-pose-estimation-app

## Problem statement
Athletes engage in complex movement patterns. By exercising with optimal movement patterns, athletes can realize significant gains for competition, health, or other personal reasons related to physical performance. However, it is challenging to quantify what constitutes optimal movement patterns. 

## Description
I wrote a proof-of-concept application using Streamlit for the frontend and Tensorflow for the backend to create a pose estimation application with a pre-trained MoveNet model. This POC does the following:
1. Takes a single image, or frames from a video (.mp4) file.
2. It runs pose estimation on the frame.
3. Finally, it calculates the joint angles using the keypoints and edges determined from pose estimation.

While not incredibly complicated, this serves as the basis for more complicated analyses including biomechanical determination for injury prevention, measuring fatigue during exercises, and other applications that can potentially be used for improving athletic performance.

## Usage
### Development
To develop on this application, run the following to set up a Python environment for this project and run a development application using Streamlit.
```Bash
python3 -m venv venv
source venv/bin/activate

pip3 install -r requirements.txt

# run application
streamlit run app.py
```

### Production
To visit the production variant, visit couro-ai.streamlit.app/.

## Conclusions and Learnings
This release was incredibly informative for developing ML applications quickly using Streamlit for the frontend with pretrained CV models for the backend. This work can be expanded by incorporating model fine-tuning with new annotated data for different use-cases.
