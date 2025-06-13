import streamlit as st
import cv2
import numpy as np
import os
import tempfile
import zipfile
from PIL import Image
import matplotlib.pyplot as plt
from assets.FrontEnd_Module import Pipeline
import time

st.title("Visual Odometry (VO) with Live Webcam or Dataset")

# ---------
# Session Management
# ---------
if "trajectory" not in st.session_state:
    st.session_state.trajectory = []

# Placeholders
stframe = st.empty()
traj_frame = st.empty()
text_placeholder = st.empty()

# Initialize VO Pipeline
VO = Pipeline()

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

# Sidebar controls
with st.sidebar:
    st.header("Pipeline Controls")
    input_mode = st.radio("Input Source", ["Live Camera", "Image Dataset"])
    run = st.toggle("Run VO", value=False)
    featureDetector = st.selectbox("Feature Detector", FeatureList["FD"])
    featureMatcher = st.selectbox("Feature Matcher", FeatureList["FM"])
    transformation1 = st.selectbox("Transformation 1", FeatureList["T"])
    transformation2 = st.selectbox("Transformation 2", FeatureList["T"])

    if input_mode == "Image Dataset":
        uploaded_zip = st.file_uploader("Upload Image Dataset (.zip)", type="zip")
        if input_mode == "Image Dataset":
            processing_speed = st.slider("Processing Speed (seconds between frames)", 0.1, 2.0, 0.5)

# -------------------
# Dataset Processing
# -------------------
@st.cache_data
def load_images_from_zip(uploaded_file):
    """Load and cache images from uploaded zip file"""
    images = []
    filenames = []
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save uploaded file
        zip_path = os.path.join(temp_dir, "uploaded.zip")
        with open(zip_path, "wb") as f:
            f.write(uploaded_file.read())
        
        # Extract zip
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # Find all image files
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
        image_files = []
        
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_files.append(os.path.join(root, file))
        
        # Sort files naturally
        image_files.sort()
        
        # Load images into memory
        for img_path in image_files:
            try:
                img = cv2.imread(img_path)
                if img is not None:
                    images.append(img)
                    filenames.append(os.path.basename(img_path))
                else:
                    st.warning(f"Could not load image: {os.path.basename(img_path)}")
            except Exception as e:
                st.error(f"Error loading {os.path.basename(img_path)}: {str(e)}")
    
    return images, filenames

# Load images if dataset mode is selected
image_data = None
if input_mode == "Image Dataset" and uploaded_zip is not None:
    try:
        images, filenames = load_images_from_zip(uploaded_zip)
        if images:
            image_data = (images, filenames)
            st.success(f"Loaded {len(images)} images from dataset")
            # Show preview of first image
            preview_img = cv2.cvtColor(images[0], cv2.COLOR_BGR2RGB)
            st.image(preview_img, caption=f"Preview: {filenames[0]}", width=400)
        else:
            st.error("No valid images found in the uploaded zip file")
    except Exception as e:
        st.error(f"Error processing zip file: {str(e)}")

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
                Transformation1=transformation1,
                Transformation2=transformation2
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
                img_frame,
                FeatureDetector=featureDetector,
                FeatureMatcher=featureMatcher,
                Transformation1=transformation1,
                Transformation2=transformation2
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
                
                if E is not None:
                    with st.expander("Essential Matrix"):
                        st.write(E)
                
                if F is not None:
                    with st.expander("Fundamental Matrix"):
                        st.write(F)

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

elif input_mode == "Image Dataset" and uploaded_zip is None:
    st.warning("Please upload a zip file containing images to process.")

elif input_mode == "Image Dataset" and image_data is None:
    st.warning("Please wait for the images to load, or upload a valid zip file.")

if not run:
    # Show instructions when not running
    if input_mode == "Image Dataset":
        st.info("Upload a zip file containing images and toggle 'Run VO' to start processing.")
    else:
        st.info("Toggle 'Run VO' to start live camera processing.")
