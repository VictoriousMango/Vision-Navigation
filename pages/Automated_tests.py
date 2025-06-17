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

st.title("Visual Odometry (VO) with KITTI Dataset")

# ---------
# Session Management
# ---------
if "trajectory" not in st.session_state:
    st.session_state.trajectory = []
if "trajectory_GT" not in st.session_state:
    st.session_state.trajectory_GT = []
if "poses_dataframe" not in st.session_state:
    st.session_state.poses_dataframe = pd.DataFrame()

# Placeholders
stframe = st.empty()
col1, col2 = st.columns(2)
traj_frame = col1.empty()
text_placeholder = col2.empty()

# Add placeholder for the dataframe table
table_placeholder = st.empty()

# Initialize VO Pipeline
VO = Pipeline()
kitti = KITTIDataset()

# Define feature detector and matcher pairs to test
test_pairs = [
    {"featureDetector": ["FD_AffineSIFT", "FD_SIFT"], "featureMatcher": "FM_BF_NORM_L2"},
    {"featureDetector": ["FD_ORB", "FD_BRISK", "FD_AffineORB"], "featureMatcher": "FM_BF_NORM_Hamming"} #
]

# Base directory for KITTI dataset
BASE_DIR = r"D:/coding/Temp_Download/data_odometry_color/dataset/sequences"
POSE_DIR = r"D:/coding/Temp_Download/data_odometry_poses/dataset/poses"

# -----------------------
# Sidebar controls
# -----------------------
with st.sidebar:
    st.header("Pipeline Controls")
    run = st.toggle("Run VO", value=False)
    sequence = st.selectbox("Select Sequence", [f"{i:02d}" for i in range(11)])
    processing_speed = st.slider("Processing Speed (seconds between frames)", 0.1, 2.0, 0.5)
    
    # Add option to control table display
    show_table = st.checkbox("Show Poses Table", value=True)
    # table_max_rows = st.slider("Max rows to display in table", 10, 100, 20)

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

# Function to create the poses dataframe (same format as CSV)
def create_poses_dataframe(trajectory, trajectory_GT):
    """Create the poses dataframe in the same format as will be saved to CSV"""
    if not trajectory or not trajectory_GT:
        return pd.DataFrame()
    
    # Ensure we have the same length
    min_len = min(len(trajectory), len(trajectory_GT))
    if min_len == 0:
        return pd.DataFrame()
    
    # Create dataframe matching CSV format
    traj_data = np.array(trajectory[:min_len])
    gt_data = np.array(trajectory_GT[:min_len])
    
    data = {
        "Frame": list(range(min_len)),
        "Predicted_X": traj_data[:, 0] if traj_data.shape[1] > 0 else [0] * min_len,
        "Predicted_Z": traj_data[:, 1] if traj_data.shape[1] > 1 else [0] * min_len,
        "Ground_Truth_X": gt_data[:, 0],
        "Ground_Truth_Z": gt_data[:, 2],
    }
    
    # Calculate errors
    data["Error_X"] = [abs(p - g) for p, g in zip(data["Predicted_X"], data["Ground_Truth_X"])]
    data["Error_Z"] = [abs(p - g) for p, g in zip(data["Predicted_Z"], data["Ground_Truth_Z"])]
    
    # Calculate MSE (same as original function)
    df = pd.DataFrame(data)
    # mse_x = np.sqrt(((df["Predicted_X"] - df["Ground_Truth_X"]) ** 2).mean()) if len(df) > 0 else 0
    # mse_z = np.sqrt(((df["Predicted_Z"] - df["Ground_Truth_Z"]) ** 2).mean()) if len(df) > 0 else 0
    # df["MSE_X"] = mse_x
    # df["MSE_Z"] = mse_z
    
    return df

# Function to save dataframe as CSV
def save_dataframe_to_csv(df, sequence, feature_detector, output_dir="results"):
    """Save the dataframe to CSV"""
    if df.empty:
        return
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    csv_filename = f"{output_dir}/{sequence}_image_2_{feature_detector}.csv"
    df.to_csv(csv_filename, index=False)
    return csv_filename

# Function to check if we should save CSV (every 10th percentile)
def should_save_csv(current_frame, total_frames):
    """Check if current frame is at a 10th percentile milestone"""
    if total_frames <= 10:
        return current_frame == total_frames - 1  # Save at the end for small datasets
    
    percentile_step = total_frames // 10  # 10th percentile step
    return (current_frame + 1) % percentile_step == 0 or current_frame == total_frames - 1

# Function to create and display the poses dataframe
def display_poses_table(poses_df, current_frame):
    """Create and display the poses comparison table"""
    if poses_df.empty:
        return
    
    with table_placeholder.container():
        st.subheader(f"Predicted vs Ground Truth Poses (Frame {current_frame})")
        st.write(f"Total frames processed: {len(poses_df)}")
        
        # Display only the last N rows for better performance, but show we have the full dataset
        st.write(f"Showing last {len(poses_df)} of {len(trajectory_GT)} total frames")
        st.dataframe(poses_df, use_container_width=True)
        
        # Show summary statistics
        if len(poses_df) > 1:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Frames Processed", len(poses_df))
            with col2:
                st.metric("Mean Error X", f"{poses_df['Error_X'].mean():.3f}")
            with col3:
                st.metric("Mean Error Z", f"{poses_df['Error_Z'].mean():.3f}")
            with col4:
                st.metric("Current Frame", current_frame)

# ---------------------
# Run VO - Main Logic
# ---------------------
if run:
    # Process selected sequence
    sequence_dir = os.path.join(BASE_DIR, sequence, "image_2")
    calib_file = os.path.join(BASE_DIR, sequence, "calib2.txt")
    pose_file = os.path.join(POSE_DIR, f"{sequence}.txt")

    # Load images
    try:
        images, filenames = load_images_from_directory(sequence_dir)
        if not images:
            st.error(f"No valid images found in {sequence_dir}")
            st.stop()
        st.success(f"Loaded {len(filenames)} images from sequence {sequence}")
        preview_img = cv2.cvtColor(cv2.imread(images[0]), cv2.COLOR_BGR2RGB)
        st.image(preview_img, caption=f"Preview: {filenames[0]}", width=400)
    except Exception as e:
        st.error(f"Error loading images: {str(e)}")
        st.stop()

    # Load calibration
    try:
        with open(calib_file, 'r') as f:
            K, P = kitti.load_calib(f)
            VO.set_k_intrinsic(K)
        st.success("Calibration file loaded successfully")
        col1, col2 = st.columns(2)
        col2.subheader("Intrinsic Parameters (K)")
        col2.dataframe(VO.get_k_intrinsic())
        col2.subheader("Projection Matrix (P)")
        col2.dataframe(P)
    except Exception as e:
        st.error(f"Error reading calibration file: {str(e)}")
        st.stop()

    # Load ground truth poses
    try:
        with open(pose_file, 'r') as f:
            poses = kitti.load_poses(f)
        st.session_state.trajectory_GT = []
        for pose in poses:
            x, y, z = pose[0, 3], pose[1, 3], pose[2, 3]
            st.session_state.trajectory_GT.append([x, y, z])
        trajectory_GT = np.array(st.session_state.trajectory_GT)
        st.success("Ground truth poses loaded successfully")
        fig = plt.figure(figsize=(12, 3))
        ax1 = fig.add_subplot(131)
        ax1.plot(trajectory_GT[:, 0], trajectory_GT[:, 2], 'b-', linewidth=2, alpha=0.7)
        ax1.scatter(trajectory_GT[0, 0], trajectory_GT[0, 2], color='green', s=100, label='Start', zorder=5)
        ax1.scatter(trajectory_GT[-1, 0], trajectory_GT[-1, 2], color='red', s=100, label='End', zorder=5)
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Z (m)')
        ax1.set_title('Ground Truth Trajectory (X-Z plane)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.axis('equal')
        col1.pyplot(fig, use_container_width=False)
    except Exception as e:
        st.error(f"Error reading pose file: {str(e)}")
        st.stop()

    # Process each feature detector and matcher pair
    mse_results = []
    for pair in test_pairs:
        feature_matcher = pair["featureMatcher"]
        for feature_detector in pair["featureDetector"]:
            st.session_state.trajectory = []  # Reset trajectory for each detector
            st.session_state.poses_dataframe = pd.DataFrame()  # Reset dataframe
            VO.prev_keypoints = None  # Reset keypoints
            VO.prev_descriptors = None  # Reset descriptors
            progress_bar = st.progress(0)
            status_text = st.empty()
            stop_button_placeholder = st.empty()
            csv_save_info = st.empty()

            total_images = len(images)

            for i, (img_frame, filename) in enumerate(zip(images, filenames)):
                with stop_button_placeholder:
                    if st.button("Stop Processing", key=f"stop_{sequence}_{feature_detector}_{i}"):
                        break

                try:
                    result_frame, E, F, traj_path, detector_used = VO.VisualOdometry(
                        cv2.imread(img_frame),
                        FeatureDetector=feature_detector,
                        FeatureMatcher=feature_matcher
                    )
                    
                    if traj_path:
                        st.session_state.trajectory = traj_path

                    # Update the poses dataframe
                    st.session_state.poses_dataframe = create_poses_dataframe(
                        st.session_state.trajectory, 
                        st.session_state.trajectory_GT
                    )

                    # Check if we should save CSV at this 10th percentile milestone
                    if should_save_csv(i, total_images):
                        csv_filename = save_dataframe_to_csv(
                            st.session_state.poses_dataframe, 
                            sequence, 
                            feature_detector
                        )
                        percentile = ((i + 1) / total_images) * 100
                        csv_save_info.success(f"CSV saved at {percentile:.0f}% completion: {csv_filename}")

                    progress = (i + 1) / len(images)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing: {filename} ({i+1}/{len(images)}) with {feature_detector}")

                    stframe.image(result_frame, channels="BGR", caption=f"Processing {filename} with {feature_detector}")

                    if st.session_state.trajectory and len(st.session_state.trajectory) > 1:
                        traj_np = np.array(st.session_state.trajectory)
                        fig, ax = plt.subplots(figsize=(8, 6))
                        ax.plot(traj_np[:, 0], traj_np[:, 1], marker='o', linewidth=1, markersize=3, color='green')
                        ax.plot(trajectory_GT[:, 0], trajectory_GT[:, 2], 'b-', linewidth=2, alpha=0.7, label='Ground Truth')
                        ax.set_title(f"2D Camera Trajectory ({feature_detector})")
                        ax.set_xlabel("X")
                        ax.set_ylabel("Z")
                        ax.grid(True)
                        ax.axis('equal')
                        ax.legend()
                        traj_frame.pyplot(fig)
                        plt.close(fig)

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

                    # Display the poses table after each frame if enabled
                    if show_table and not st.session_state.poses_dataframe.empty:
                        display_poses_table(
                            st.session_state.poses_dataframe, 
                            i
                        )

                except Exception as e:
                    st.error(f"Error processing {filename} with {feature_detector}: {str(e)}")
                    continue

                # time.sleep(processing_speed)
            
            # Final save of complete dataframe
            if not st.session_state.poses_dataframe.empty:
                final_csv = save_dataframe_to_csv(
                    st.session_state.poses_dataframe, 
                    sequence, 
                    feature_detector
                )
                
                # Calculate final MSE for results summary
                final_df = st.session_state.poses_dataframe
                mse_x = final_df["MSE_X"].iloc[0] if len(final_df) > 0 else 0
                mse_z = final_df["MSE_Z"].iloc[0] if len(final_df) > 0 else 0
                
                mse_results.append({
                    "Sequence": sequence,
                    "Feature_Detector": feature_detector,
                    "MSE_X": mse_x,
                    "MSE_Z": mse_z
                })
                st.success(f"Completed processing sequence {sequence} with {feature_detector}. Final CSV saved: {final_csv}")

            progress_bar.empty()
            status_text.empty()
            stop_button_placeholder.empty()
            csv_save_info.empty()
            # Clear the trajectory for the next feature detector
            st.session_state.trajectory = []
            st.session_state.poses_dataframe = pd.DataFrame()
            

    # Display MSE results
    if mse_results:
        mse_df = pd.DataFrame(mse_results)
        st.subheader("Mean Squared Error Results")
        st.dataframe(mse_df)

    if st.session_state.trajectory:
        st.success(f"Sequence {sequence} processing completed! Final trajectory has {len(st.session_state.trajectory)} points.")

else:
    st.info("Select a sequence and toggle 'Run VO' to start processing.")
    # Clear the table when not running
    table_placeholder.empty()

# Final display of complete poses dataframe (replaces the old expandable section)
with st.expander("Complete Poses DataFrame (CSV Format)", expanded=False):
    if not st.session_state.poses_dataframe.empty:
        st.write("This is the complete dataframe that matches the saved CSV format:")
        st.dataframe(st.session_state.poses_dataframe, use_container_width=True)
        
        # Show download button for the dataframe
        csv_data = st.session_state.poses_dataframe.to_csv(index=False)
        st.download_button(
            label="Download Complete DataFrame as CSV",
            data=csv_data,
            file_name=f"complete_poses_{sequence}.csv",
            mime="text/csv"
        )
    else:
        st.info("No pose data available. Run the visual odometry pipeline to generate data.")