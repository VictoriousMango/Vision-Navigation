import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
import tempfile
import zipfile
from PIL import Image
import matplotlib.pyplot as plt
from assets.FrontEnd_Module import Pipeline, KITTIDataset
import time

st.title("Visual Odometry (VO) with Live Webcam or Dataset")

# ---------
# Session Management
# ---------
if "trajectory" not in st.session_state:
    st.session_state.trajectory = []

# Placeholders
stframe = st.empty()
col1, col2 = st.columns(2)
traj_frame = col1.empty()
text_placeholder = col2.empty()

# Initialize VO Pipeline
VO = Pipeline()
kitti = KITTIDataset()

# Get feature detection, matching and transformation methods dynamically
FeatureList = {"FD": [], "FM": [], "T": []}
for function in dir(VO):
    if callable(getattr(VO, function)):
        if "FD_" in function:
            FeatureList["FD"].append(function)
        elif "FM_" in function:
            FeatureList["FM"].append(function)
        elif function.startswith("T_"):
            FeatureList["T"].append(function)
if "trajectory_GT" not in st.session_state:
    st.session_state.trajectory_GT = []
# -----------------------
# Sidebar controls
# -----------------------
with st.sidebar:
    st.header("Pipeline Controls")
    input_mode = st.radio("Input Source", ["Live Camera", "Image Dataset"])
    run = st.toggle("Run VO", value=False)
    featureDetector = st.selectbox("Feature Detector", FeatureList["FD"])
    featureMatcher = st.selectbox("Feature Matcher", FeatureList["FM"])
    # transformation1 = st.selectbox("Transformation 1", FeatureList["T"])
    # transformation2 = st.selectbox("Transformation 2", FeatureList["T"])

    if input_mode == "Image Dataset":
        image_dir_path = st.text_input("Enter path to image dataset directory")
        # uploaded_zip = st.file_uploader("Upload Image Dataset (.zip)", type="zip")
        calib_M = st.file_uploader("Upload Calibration (.txt)", type="txt")
        pose_GT = st.file_uploader("Upload Ground Truth Poses (.txt)", type="txt")
        processing_speed = st.slider("Processing Speed (seconds between frames)", 0.1, 2.0, 0.5)

# -------------------
# Dataset Processing
# -------------------
@st.cache_data
def load_images_from_directory(folder_path):
    """Load and cache images from a local directory"""
    images = []
    filenames = []

    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    for root, dirs, files in os.walk(folder_path):
        for file in sorted(files):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                img_path = os.path.join(root, file)
                images.append(img_path)
                filenames.append(file)
    return images, filenames

# Load images if dataset mode is selected
image_data = None
if input_mode == "Image Dataset" and image_dir_path is not None:
    try:
        images, filenames = load_images_from_directory(image_dir_path)
        if filenames:
            image_data = (images, filenames)
            st.success(f"Loaded {len(filenames)} images from dataset")
            # Show preview of first image
            preview_img = cv2.cvtColor(cv2.imread(images[0]), cv2.COLOR_BGR2RGB)
            st.image(preview_img, caption=f"Preview: {filenames[0]}", width=400)
        else:
            st.error("No valid images directory")
    except Exception as e:
        st.error(f"Error processing directory: {str(e)}")
    if calib_M is not None:
        try:
            K, P = kitti.load_calib(calib_M)
            VO.set_k_intrinsic(K)
            st.success("Calibration file loaded successfully")
            col1, col2 = st.columns(2)
            col2.subheader("Intrinsic Parameters (K)")
            col2.dataframe(VO.get_k_intrinsic())
            col2.subheader("Projection Matrix (P)")
            col2.dataframe(P)
            
        except Exception as e:
            st.error(f"Error reading calibration file: {str(e)}")
    if pose_GT is not None:
        try:
            poses = kitti.load_poses(pose_GT)
            for pose in poses:
                # Translation is stored in the last column (index 3) of the transformation matrix
                x, y, z = pose[0, 3], pose[1, 3], pose[2, 3]
                st.session_state.trajectory_GT.append([x, y, z])
            trajectory_GT = np.array(st.session_state.trajectory_GT)
            st.success("Pose file loaded successfully")
            fig = plt.figure(figsize=(12, 3))
            ax1 = fig.add_subplot(131)
            ax1.plot(trajectory_GT[:, 0], trajectory_GT[:, 2], 'b-', linewidth=2, alpha=0.7)
            ax1.scatter(trajectory_GT[0, 0], trajectory_GT[0, 2], color='green', s=100, label='Start', zorder=5)
            ax1.scatter(trajectory_GT[-1, 0], trajectory_GT[-1, 2], color='red', s=100, label='End', zorder=5)
            ax1.set_xlabel('X (m)')
            ax1.set_ylabel('Z (m)')
            ax1.set_title('Bird\'s Eye View (X-Z plane)')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            ax1.axis('equal')
            col1.pyplot(fig, use_container_width=False)
        except Exception as e:
            st.error(f"Error reading pose file: {str(e)}")

# ---------------------
# Run VO - Main Logic
# ---------------------
 
if input_mode == "Live Camera":
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        st.error("Cannot open webcam")
    else:
        while run:
            ret, img_frame = camera.read()
            if not ret:
                st.warning("Camera disconnected")
                break

            result_frame, E, F, traj_path, detector_used = VO.VisualOdometry(
                img_frame,
                FeatureDetector=featureDetector,
                FeatureMatcher=featureMatcher,
                # Transformation1=transformation1,
                # Transformation2=transformation2
            )
            
            if traj_path: st.session_state.trajectory = traj_path

            # Show visual output
            stframe.image(result_frame, channels="BGR", caption="Frame with Keypoints & Matches")

            if st.session_state.trajectory:
                traj_np = np.array(st.session_state.trajectory)
                fig, ax = plt.subplots()
                ax.plot(traj_np[:, 0], traj_np[:, 1], marker='o', linewidth=1, color='blue')
                ax.set_title("2D Camera Trajectory")
                ax.set_xlabel("X")
                ax.set_ylabel("Z")
                ax.grid(True)
                traj_frame.pyplot(fig)

            # Info
            with text_placeholder.container():
                st.subheader("Feature Detector Used")
                st.code(detector_used)
                st.subheader("Essential Matrix")
                st.write(E if E is not None else "Not computed")
                st.subheader("Fundamental Matrix")
                st.write(F if F is not None else "Not computed")

        camera.release()
elif input_mode == "Image Dataset" and image_data is not None:
    images, filenames = image_data
    
    if not images:
        st.error("No images available for processing")
        st.stop()
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Create stop button for dataset processing
    stop_button_placeholder = st.empty()
    
    for i, (img_frame, filename) in enumerate(zip(images, filenames)):
        # Check if user wants to stop
        with stop_button_placeholder:
            if st.button("Stop Processing", key=f"stop_{i}"):
                break
        
        # Check if toggle is still on
        if not run:
            break
        
        try:
            result_frame, E, F, traj_path, detector_used = VO.VisualOdometry(
                cv2.imread(img_frame),
                FeatureDetector=featureDetector,
                FeatureMatcher=featureMatcher,
                # Transformation1=transformation1,
                # Transformation2=transformation2
            )
            
            if traj_path: 
                st.session_state.trajectory = traj_path

            # Update progress
            progress = (i + 1) / len(images)
            progress_bar.progress(progress)
            status_text.text(f"Processing: {filename} ({i+1}/{len(images)})")

            # Show current frame
            stframe.image(result_frame, channels="BGR", caption=f"Processing {filename}")

            # Plot trajectory
            if st.session_state.trajectory and len(st.session_state.trajectory) > 1:
                traj_np = np.array(st.session_state.trajectory)
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.plot(traj_np[:, 0], traj_np[:, 1], marker='o', linewidth=1, markersize=3, color='green')
                ax.set_title("2D Camera Trajectory")
                ax.set_xlabel("X")
                ax.set_ylabel("Z")
                ax.grid(True)
                ax.axis('equal')
                traj_frame.pyplot(fig)
                plt.close(fig)

            # Display information
            with text_placeholder.container():
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.subheader("Current Image")
                    st.write(f"**File:** {filename}")
                    st.write(f"**Progress:** {i+1}/{len(images)}")
                with col2:
                    st.subheader("Feature Detector")
                    st.code(detector_used)
                with col3:
                    st.subheader("Trajectory Points")
                    st.write(f"Total: {len(st.session_state.trajectory) if st.session_state.trajectory else 0}")
                
                col21, col22 = st.columns(2)
                if E is not None:
                    col21.subheader("Essential Matrix")
                    col21.write(E)
                if F is not None:
                    col22.subheader("Fundamental Matrix")
                    col22.write(F) 
        except Exception as e:
            st.error(f"Error processing {filename}: {str(e)}")
            continue
        
        # Delay between frames
        time.sleep(processing_speed)
    
    # Clean up progress indicators
    progress_bar.empty()
    status_text.empty()
    stop_button_placeholder.empty()
    
    if st.session_state.trajectory:
        st.success(f"Dataset processing completed! Final trajectory has {len(st.session_state.trajectory)} points.")

elif input_mode == "Image Dataset" and image_dir_path is None:
    st.warning("Please upload a zip file containing images to process.")

elif input_mode == "Image Dataset" and image_data is None:
    st.warning("Please wait for the images to load, or upload a valid zip file.")

if not run:
    # Show instructions when not running
    if input_mode == "Image Dataset":
        st.info("Upload a zip file containing images and toggle 'Run VO' to start processing.")
    else:
        st.info("Toggle 'Run VO' to start live camera processing.")
with st.expander("Predicted, Ground Truth Dataset Poses", expanded=True):
            # st.dataframe(pd.DataFrame([traj_np[:, 0], traj_np[:, 1], st.session_state.trajectory_GT[:, 0], st.session_state.trajectory_GT[:, 2]]))
            # st.write(type(st.session_state.trajectory), type(st.session_state.trajectory_GT))
            traj_np = pd.DataFrame(np.array(st.session_state.trajectory))
            traj_gt = pd.DataFrame(np.array(st.session_state.trajectory_GT))
            summary = pd.concat([traj_np, traj_gt], axis=1, keys=['Predicted XZ', 'Ground Truth XZ'])
            # summary["MSE_X"] = np.sqrt(((summary['Predicted XZ']['X'] - summary['Ground Truth XZ']['X_GT']) ** 2).mean())
            # summary["MSE_Z"] = np.sqrt(((summary['Predicted XZ']['Z'] - summary['Ground Truth XZ']['Z_GT']) ** 2).mean())
            st.dataframe(summary)