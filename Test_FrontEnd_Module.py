import streamlit as st
import cv2
import numpy as np
from assets.FrontEnd_Module import Pipeline

st.title("Live Visual Odometry (VO) with Webcam")

stframe = st.empty()
traj_frame = st.empty()
text_placeholder = st.empty()

VO = Pipeline()

camera = cv2.VideoCapture(0)
FeatureList = {"FD": [], "FM": [], "T": []}
for function in dir(VO):
    if callable(getattr(VO, function)):
        if "FD_" in function:
            FeatureList["FD"].append(function)
        elif "FM_" in function:
            FeatureList["FM"].append(function)
        elif function.startswith("T_"):
            FeatureList["T"].append(function)

with st.sidebar:
    st.header("Pipeline Controls")
    run = st.toggle("Run VO", value=True)
    featureDetector = st.selectbox("Feature Detector", FeatureList["FD"])
    featureMatcher = st.selectbox("Feature Matcher", FeatureList["FM"])
    transformation1 = st.selectbox("Transformation 1", FeatureList["T"])
    transformation2 = st.selectbox("Transformation 2", FeatureList["T"])

if not camera.isOpened():
    st.error("Cannot open webcam")
else:
    while run:
        ret, img_frame = camera.read()
        if not ret:
            st.warning("Camera disconnected")
            break

        result_frame, E, F, traj_map, detector_used = VO.VisualOdometry(
            img_frame,
            FeatureDetector=featureDetector,
            FeatureMatcher=featureMatcher,
            Transformation1=transformation1,
            Transformation2=transformation2
        )

        # Display frames
        stframe.image(result_frame, channels="BGR", caption="Frame with Keypoints & Matches")
        traj_frame.image(traj_map, channels="BGR", caption="2D Camera Trajectory")

        # Display matrices
        with text_placeholder.container():
            st.subheader("Feature Detector Used")
            st.code(detector_used)
            st.subheader("Essential Matrix")
            st.write(E if E is not None else "Not computed")
            st.subheader("Fundamental Matrix")
            st.write(F if F is not None else "Not computed")
    else:
        camera.release()
