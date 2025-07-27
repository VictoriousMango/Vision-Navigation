import streamlit as st
import cv2 
import os
from assets.FrontEnd_Module import Pipeline
from assets.LoopDetection import BOVW
import plotly.graph_objects as go

# CSS ----------------------
st.markdown("""
<style>
            .st-emotion-cache-1w723zb {
    width: 100%;
    padding: 6rem 1rem 10rem;
    max-width: 100%;
}
            </style>""", unsafe_allow_html=True)
# --------------------------

# Session Management -------
if "ImageControls" not in st.session_state:
    st.session_state["ImageControls"] = {
        "Sequence": "00",                  # Sequence Selected
        "Image": None                      # Image Selected
    }
if "ImagePath" not in st.session_state:
    st.session_state["ImagePath"] = {
        "FolderPath" : None,              # Sequence Folder Path
        "Image" : None                    # Selected Image Path
    }
# --------------------------

# Constants ----------------
BASE_DIR = r"D:/coding/Temp_Download/data_odometry_color/dataset/sequences"
SEQUENCE = [f"{index:02d}" for index in range(11)]
VO = Pipeline()
bovw = BOVW()
# --------------------------

# Functions/Classes --------
# --------------------------

# layout of Web Page -------
st.title("Single Image Processing")
with st.sidebar:
    ImageControls = st.empty()
ImageSelected = st.empty()
FeatureDetector = st.empty()
# --------------------------

# Logic of Page ------------
with ImageControls.container():
    # Control -> Sequence Control
    st.session_state["ImageControls"]["Sequence"]=st.selectbox('Selected Sequence:', SEQUENCE)
    # ---------------------------

    st.session_state["ImagePath"]["FolderPath"] = os.path.join(BASE_DIR, st.session_state["ImageControls"]["Sequence"], "image_2")
    images = [img for img in os.listdir(st.session_state["ImagePath"]["FolderPath"]) if img.endswith('.png')]

    # Control -> Image from folder
    st.session_state["ImageControls"]["Image"]=st.selectbox('Select Image', images)
    # ---------------------------

    st.session_state["ImagePath"]["Image"] = os.path.join(os.path.abspath(st.session_state["ImagePath"]["FolderPath"]), st.session_state["ImageControls"]["Image"])

    col1, col2 = st.columns(2)
    col1.metric("Selected Sequence", st.session_state["ImageControls"]["Sequence"])
    col1.metric("Number of Images", len(images))
    col2.metric("Selected Image", st.session_state["ImageControls"]["Image"][-8:-4])

with ImageSelected.container():
    img = cv2.imread(st.session_state["ImagePath"]["Image"])
    img2 = images[images.index(st.session_state["ImageControls"]["Image"]) + 1]
    img2 = cv2.imread(os.path.join(st.session_state["ImagePath"]["FolderPath"], img2))
    col1, col2 = st.columns(2)
    col1.image(img, channels="BGR", caption="Selected Image")
    col2.image(img2, channels="BGR", caption="Next Image")
with FeatureDetector.container():
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Affine SIFT", "Affine ORB", "SIFT", "ORB", "BRISK"])
    with tab1:
        col1, col2 = st.columns(2)
        kp, des, Descriptor = VO.FD_AffineSIFT(img)
        kp2, des2, Descriptor2 = VO.FD_AffineSIFT(img2)
        col1.image(Descriptor, channels="BGR")
        col2.image(Descriptor, channels="BGR")
        col11, col12, col13, col14 = st.columns(4)
        col11.metric("Keypoints", len(kp))
        col12.metric("Descriptors", len(des))
        col13.metric("Keypoints", len(kp2))
        col14.metric("Descriptors", len(des2))
        VO.FM_BF_NORM_L2()
        matches, pts1, pts2 = VO.BruteForce(kp, des, kp2, des2, max_matches=32)
        matchedIMG = cv2.drawMatches(img, kp, img2, kp2, matches, None, flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
        st.image(matchedIMG, caption="Matches", channels="BGR")
        col1, col2, col3 = st.columns(3)
        col1.metric("Matches", len(matches))
        col2.metric("Points 1", len(pts1))
        col3.metric("Points 2", len(pts2))
        st.write()
    with tab2:
        col1, col2 = st.columns(2)
        kp, des, Descriptor = VO.FD_AffineORB(img)
        kp2, des2, Descriptor2 = VO.FD_AffineORB(img2)
        col1.image(Descriptor, channels="BGR")
        col2.image(Descriptor, channels="BGR")
        col11, col12, col13, col14 = st.columns(4)
        col11.metric("Keypoints", len(kp))
        col12.metric("Descriptors", len(des))
        col13.metric("Keypoints", len(kp2))
        col14.metric("Descriptors", len(des2))
        VO.FM_BF_NORM_Hamming()
        matches, pts1, pts2 = VO.BruteForce(kp, des, kp2, des2, max_matches=32)
        matchedIMG = cv2.drawMatches(img, kp, img2, kp2, matches, None, flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
        st.image(matchedIMG, caption="Matches", channels="BGR")
        col1, col2, col3 = st.columns(3)
        col1.metric("Matches", len(matches))
        col2.metric("Points 1", len(pts1))
        col3.metric("Points 2", len(pts2))
    with tab3:
        col1, col2 = st.columns(2)
        kp, des, Descriptor = VO.FD_SIFT(img)
        hist1 = bovw.Histogram(des)
        kp2, des2, Descriptor2 = VO.FD_SIFT(img2)
        hist2 = bovw.Histogram(des2)
        fig1 = go.Figure(data=[go.Bar(x=list(range(len(hist1))), y=hist1)])
        fig1.update_layout(
            title="BoVW Histogram",
            xaxis_title="Visual Word Index",
            yaxis_title="Frequency",
            bargap=0.1
        )
        fig2 = go.Figure(data=[go.Bar(x=list(range(len(hist2))), y=hist2)])
        fig2.update_layout(
            title="BoVW Histogram",
            xaxis_title="Visual Word Index",
            yaxis_title="Frequency",
            bargap=0.1
        )

        col1.plotly_chart(fig1, use_container_width=True)
        col2.plotly_chart(fig2, use_container_width=True)

        col1.image(Descriptor, channels="BGR")
        col2.image(Descriptor, channels="BGR")
        col11, col12, col13, col14 = st.columns(4)
        col11.metric("Keypoints", len(kp))
        col12.metric("Descriptors", len(des))
        col13.metric("Keypoints", len(kp2))
        col14.metric("Descriptors", len(des2))
        VO.FM_BF_NORM_L2()
        matches, pts1, pts2 = VO.BruteForce(kp, des, kp2, des2, max_matches=32)
        matchedIMG = cv2.drawMatches(img, kp, img2, kp2, matches, None, flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
        st.image(matchedIMG, caption="Matches", channels="BGR")
        col1, col2, col3 = st.columns(3)
        col1.metric("Matches", len(matches))
        col2.metric("Points 1", len(pts1))
        col3.metric("Points 2", len(pts2))
    with tab4:
        col1, col2 = st.columns(2)
        kp, des, Descriptor = VO.FD_ORB(img)
        kp2, des2, Descriptor2 = VO.FD_ORB(img2)
        col1.image(Descriptor, channels="BGR")
        col2.image(Descriptor, channels="BGR")
        col11, col12, col13, col14 = st.columns(4)
        col11.metric("Keypoints", len(kp))
        col12.metric("Descriptors", len(des))
        col13.metric("Keypoints", len(kp2))
        col14.metric("Descriptors", len(des2))
        VO.FM_BF_NORM_Hamming()
        matches, pts1, pts2 = VO.BruteForce(kp, des, kp2, des2, max_matches=32)
        matchedIMG = cv2.drawMatches(img, kp, img2, kp2, matches, None, flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
        st.image(matchedIMG, caption="Matches", channels="BGR")
        col1, col2, col3 = st.columns(3)
        col1.metric("Matches", len(matches))
        col2.metric("Points 1", len(pts1))
        col3.metric("Points 2", len(pts2))
    with tab5:
        col1, col2 = st.columns(2)
        kp, des, Descriptor = VO.FD_BRISK(img)
        kp2, des2, Descriptor2 = VO.FD_BRISK(img2)
        col1.image(Descriptor, channels="BGR")
        col2.image(Descriptor, channels="BGR")
        col11, col12, col13, col14 = st.columns(4)
        col11.metric("Keypoints", len(kp))
        col12.metric("Descriptors", len(des))
        col13.metric("Keypoints", len(kp2))
        col14.metric("Descriptors", len(des2))
        VO.FM_BF_NORM_Hamming()
        matches, pts1, pts2 = VO.BruteForce(kp, des, kp2, des2, max_matches=32)
        matchedIMG = cv2.drawMatches(img, kp, img2, kp2, matches, None, flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
        st.image(matchedIMG, caption="Matches", channels="BGR")
        col1, col2, col3 = st.columns(3)
        col1.metric("Matches", len(matches))
        col2.metric("Points 1", len(pts1))
        col3.metric("Points 2", len(pts2))
# --------------------------