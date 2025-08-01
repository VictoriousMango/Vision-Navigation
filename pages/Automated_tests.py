import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
from PIL import Image
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from assets.FrontEnd_Module import Pipeline, KITTIDataset
from assets.LoopDetection import BOVW
import random
import logging
from datetime import datetime

# Configure logging
def setup_logging():
    """Setup logging configuration"""
    # Create logs directory if it doesn't exist
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Create log filename with timestamp
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_dir, f"visual_odometry.log")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            # logging.StreamHandler()  # Also log to console
        ]
    )
    
    return logging.getLogger(__name__)

# Initialize logger
logger = setup_logging()

# Helper functions for logging with Streamlit display
def log_error(message, show_in_ui=False):
    """Log error message and optionally show in Streamlit UI"""
    logger.error(message)
    if show_in_ui:
        st.error(message)

def log_info(message, show_in_ui=False):
    """Log info message and optionally show in Streamlit UI"""
    logger.info(message)
    if show_in_ui:
        st.info(message)

def log_success(message, show_in_ui=False):
    """Log success message and optionally show in Streamlit UI"""
    logger.info(f"SUCCESS: {message}")
    if show_in_ui:
        st.success(message)

def log_warning(message, show_in_ui=False):
    """Log warning message and optionally show in Streamlit UI"""
    logger.warning(message)
    if show_in_ui:
        st.warning(message)

st.title("Visual Odometry (VO) with KITTI Dataset")

# Log application start
logger.info("Visual Odometry application started")

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
traj_frame = st.empty()
text_placeholder = st.empty()
loopClosure = st.empty()
with st.sidebar:
    panel = st.empty()
    sequence_information = st.empty()

# Add placeholder for the dataframe table
table_placeholder = st.empty()

# Initialize VO Pipeline
VO = Pipeline()
bovw = BOVW()
kitti = KITTIDataset()
logger.info("VO Pipeline and KITTI Dataset initialized")

# Define feature detector and matcher pairs to test
test_pairs = [
    {"featureDetector": ["FD_SIFT"], "featureMatcher": "FM_BF_NORM_L2"}, # "FD_AffineSIFT", 
    # {"featureDetector": ["FD_ORB", "FD_BRISK", "FD_AffineORB"], "featureMatcher": "FM_BF_NORM_Hamming"}
]

# Base directory for KITTI dataset
BASE_DIR = r"D:/coding/Temp_Download/data_odometry_color/dataset/sequences"
POSE_DIR = r"D:/coding/Temp_Download/data_odometry_poses/dataset/poses"

logger.info(f"Base directory set to: {BASE_DIR}")
logger.info(f"Pose directory set to: {POSE_DIR}")

# -----------------------
# Sidebar controls
# -----------------------
with panel.container():
    st.header("Pipeline Controls")
    panel1, panel2 = st.columns(2)
    run_all = panel1.toggle("Run All Sequences (00-10)", value=False)
    run_single = panel2.toggle("Run Single Sequence", value=False, disabled=run_all)
    sequence = panel1.selectbox("Select Sequence", [f"{i:02d}" for i in range(11)], disabled=run_all)
    
    batch_size = st.slider("Batch Size", min_value=100, max_value=5000, value=500, step=100, disabled=run_all)
    # batch_size = st.number_input("Batch Size", min_value=10, max_value=10000, value=500, step=100)    
    # Add option to control table display
    show_table = st.checkbox("Show Poses Table", value=True)

logger.info(f"Sidebar controls set - Run All: {run_all}, Run Single: {run_single}, Sequence: {sequence}, Batch Size: {batch_size}")

def show_seq_info(calib_file, pose_file):
    with open(calib_file, 'r') as f:
                K, P = kitti.load_calib(f)
                VO.set_k_intrinsic(K)
            
    success_msg = "Calibration file loaded successfully"
    log_success(success_msg)
    with sequence_information.container():
        col1 = st.empty()
        st.subheader("Intrinsic Parameters (K)")
        st.dataframe(VO.get_k_intrinsic())
        st.subheader("Projection Matrix (P)")
        st.dataframe(P)

    with open(pose_file, 'r') as f:
        poses = kitti.load_poses(f, batch_size=batch_size)
    st.session_state.trajectory_GT = []
    for pose in poses:
        x, y, z = pose[0, 3], pose[1, 3], pose[2, 3]
        st.session_state.trajectory_GT.append([x, y, z])
    panel2.metric(label="Number of points", value=len(st.session_state.trajectory_GT))
    trajectory_GT = np.array(st.session_state.trajectory_GT)

    success_msg = "Ground truth poses loaded successfully"
    log_success(success_msg)

    # Plotly 3D Trajectory Plot (Beautiful)
    fig = go.Figure()

    # Main trajectory line
    fig.add_trace(go.Scatter3d(
        x=trajectory_GT[:, 0],
        y=trajectory_GT[:, 2],  # Z as horizontal axis (floor)
        z=trajectory_GT[:, 1],  # Y as height
        mode='lines',
        line=dict(color='royalblue', width=6),
        name='Trajectory'
    ))

    # Start point
    fig.add_trace(go.Scatter3d(
        x=[trajectory_GT[0, 0]],
        y=[trajectory_GT[0, 2]],
        z=[trajectory_GT[0, 1]],
        mode='markers+text',
        marker=dict(size=8, color='green', symbol='circle'),
        name='Start',
        text=['Start'],
        textposition='top center'
    ))

    # End point
    fig.add_trace(go.Scatter3d(
        x=[trajectory_GT[-1, 0]],
        y=[trajectory_GT[-1, 2]],
        z=[trajectory_GT[-1, 1]],
        mode='markers+text',
        marker=dict(size=8, color='red', symbol='circle'),
        name='End',
        text=['End'],
        textposition='top center'
    ))

    fig.update_layout(
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Z (m)',  # Z as floor axis
            zaxis_title='Y (m)',  # Y as height
            xaxis=dict(showgrid=True, gridcolor='white', zeroline=False),
            yaxis=dict(showgrid=True, gridcolor='white', zeroline=False),
            zaxis=dict(showgrid=True, gridcolor='white', zeroline=False),
            aspectmode='data',
            camera=dict(
                eye=dict(x=1, y=-5, z=2)
            ),
            bgcolor='rgba(0, 0, 0, 0)'
        ),
        title=dict(
            text='Ground Truth Trajectory (3D)',
            font=dict(size=22, color='white')
        ),
        legend=dict(
            x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.7)', bordercolor='black'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    col1.plotly_chart(fig, use_container_width=True)
    return trajectory_GT
# -------------------
# Dataset Processing
# -------------------
@st.cache_data
def load_images_from_directory(folder_path, batch_size):
    """Load and cache images from a local directory"""
    logger.info(f"Loading images from directory: {folder_path} with batch size: {batch_size}")
    images = []
    filenames = []
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    files = os.listdir(folder_path)
    root = os.path.abspath(folder_path)
    batch_size = min(batch_size, len(files))
    for file in sorted(files)[:+batch_size]:
        if any(file.lower().endswith(ext) for ext in image_extensions):
            img_path = os.path.join(root, file)
            images.append(img_path)
            filenames.append(file)
    
    logger.info(f"Loaded {len(images)} images from {folder_path}")
    return images, filenames, batch_size

# Function to create the poses dataframe
def create_poses_dataframe(trajectory, trajectory_GT):
    """Create the poses dataframe in the same format as will be saved to CSV"""
    if not trajectory or not trajectory_GT:
        logger.warning("Empty trajectory or ground truth data provided to create_poses_dataframe")
        return pd.DataFrame()
    
    min_len = min(len(trajectory), len(trajectory_GT))
    if min_len == 0:
        logger.warning("Minimum length is 0 for trajectory data")
        return pd.DataFrame()
    
    traj_data = np.array(trajectory[:min_len])
    gt_data = np.array(trajectory_GT[:min_len])
    
    data = {
        "Frame": list(range(min_len)),
        "Predicted_X": traj_data[:, 0] if traj_data.shape[1] > 0 else [0] * min_len,
        "Predicted_Y": traj_data[:, 1] if traj_data.shape[1] > 0 else [0] * min_len,
        "Predicted_Z": traj_data[:, 2] if traj_data.shape[1] > 1 else [0] * min_len,
        "Ground_Truth_X": gt_data[:, 0],
        "Ground_Truth_Y": gt_data[:, 1],
        "Ground_Truth_Z": gt_data[:, 2],
    }
    
    data["Error_X"] = [np.sqrt((p - g)**2) for p, g in zip(data["Predicted_X"], data["Ground_Truth_X"])]
    data["Error_Y"] = [np.sqrt((p - g)**2) for p, g in zip(data["Predicted_Y"], data["Ground_Truth_Y"])]
    data["Error_Z"] = [np.sqrt((p - g)**2) for p, g in zip(data["Predicted_Z"], data["Ground_Truth_Z"])]
    
    df = pd.DataFrame(data)
    return df

# Function to save dataframe as CSV
def save_dataframe_to_csv(df, sequence, feature_detector, output_dir="results"):
    """Save the dataframe to CSV"""
    if df.empty:
        logger.warning("Attempted to save empty dataframe to CSV")
        return
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")
    
    csv_filename = f"{output_dir}/{sequence}_image_2_{feature_detector}.csv"
    df.to_csv(csv_filename, index=False)
    return csv_filename

# Function to check if we should save CSV
def should_save_csv(current_frame, total_frames):
    """Check if current frame is at a 10th percentile milestone"""
    if total_frames <= 10:
        return current_frame == total_frames - 1
    
    percentile_step = total_frames // 10
    return (current_frame + 1) % percentile_step == 0 or current_frame == total_frames - 1

# Function to display the poses table
def display_poses_table(poses_df, current_frame, totalFrames):
    """Create and display the poses comparison table"""
    if poses_df.empty:
        return
    
    with table_placeholder.container():
        st.subheader(f"Predicted vs Ground Truth Poses (Frame {current_frame})")
        st.write(f"Total frames processed: {len(poses_df)}")
        st.write(f"Showing last {len(poses_df)} of {totalFrames} total frames")
        st.dataframe(poses_df, use_container_width=True)
        
        if len(poses_df) > 1:
            col1, col2, col3, col4, col5 = st.columns(5)
            mean_x = poses_df["Error_X"].mean()
            mean_y = poses_df["Error_Y"].mean()
            mean_z = poses_df["Error_Z"].mean()
            mean_Error = f"{(mean_x + mean_y + mean_z)/3:.3f}"
            with col1:
                st.metric("Frames Processed", len(poses_df))
            with col2:
                st.metric("Mean Error X", f"{poses_df['Error_X'].mean():.3f}")
            with col3:
                st.metric("Mean Error Y", f"{poses_df['Error_Y'].mean():.3f}")
            with col4:
                st.metric("Mean Error Z", f"{poses_df['Error_Z'].mean():.3f}")
            with col5:
                st.metric("Mean Error", mean_Error)

# Function to process a single sequence
def process_sequence(sequence, show_table, batch_size, trajectory_GT):
    start_time = datetime.now()
    logger.info(f"Starting processing for sequence {sequence}")
    sequence_dir = os.path.join(BASE_DIR, sequence, "image_2")
    

    logger.info(f"Sequence directory: {sequence_dir}")
    # Load images
    try:
        images, filenames, batch_size = load_images_from_directory(sequence_dir, batch_size=batch_size)
        if not images:
            error_msg = f"No valid images found in {sequence_dir}"
            log_error(error_msg)
            return None
        
        success_msg = f"Loaded {len(filenames)} images from sequence {sequence}"
        log_success(success_msg)
        
        preview_img = cv2.cvtColor(cv2.imread(images[0]), cv2.COLOR_BGR2RGB)
        st.image(preview_img, caption=f"Preview: {filenames[0]}", width=400)
        
    except Exception as e:
        error_msg = f"Error loading images: {str(e)}"
        log_error(error_msg)
        return None

    # Process each feature detector and matcher pair
    mse_results = []
    numberOfDetectors = sum([len(pair["featureDetector"]) for pair in test_pairs])
    current = 0
    for pair in test_pairs:
        feature_matcher = pair["featureMatcher"]
        for feature_detector in pair["featureDetector"]:
            current += 1
            logger.info(f"Processing with feature detector: {feature_detector}, matcher: {feature_matcher}")
            
            st.session_state.trajectory = []
            st.session_state.poses_dataframe = pd.DataFrame()
            VO.prev_keypoints = None
            VO.prev_descriptors = None
            progress_bar = st.progress(0)
            status_text = st.empty()
            stop_button_placeholder = st.empty()
            csv_save_info = st.empty()
            traj_frame.empty()
            table_placeholder.empty()
            total_images = len(images)
            VO.reset()
            bovw.reset()
            
            for i, (img_frame, filename) in enumerate(zip(images, filenames)):
                # -AddedNew
                st.session_state.poses_dataframe = pd.DataFrame()
                st.session_state.trajectory = []
                with stop_button_placeholder:
                    if st.button("Stop Processing", key=f"stop_{sequence}_{feature_detector}_{i}"):
                        logger.info(f"Processing stopped by user at frame {i}")
                        break

                try:
                    logger.debug(f"Processing frame {i+1}/{total_images}: {filename}")
                    
                    result_frame, E, F, traj_path, detector_used, map_points, desc, kp = VO.VisualOdometry(
                        cv2.imread(img_frame),
                        FeatureDetector=feature_detector,
                        FeatureMatcher=feature_matcher
                    )
                    hist = bovw.Histogram(desc)
                    bovw.historyOfBOVW(visual_word=hist, desc=desc, kp=kp)
                    if traj_path:
                        st.session_state.trajectory = traj_path

                    st.session_state.poses_dataframe = create_poses_dataframe(
                        st.session_state.trajectory, 
                        st.session_state.trajectory_GT
                    )

                    if should_save_csv(i, total_images):
                        csv_filename = save_dataframe_to_csv(
                            st.session_state.poses_dataframe, 
                            sequence, 
                            feature_detector
                        )
                        percentile = ((i + 1) / total_images) * 100
                        success_msg = f"CSV saved at {percentile:.0f}% completion: {csv_filename}"
                        log_success(success_msg, show_in_ui=False)  # Don't show in UI to avoid clutter
                        csv_save_info.success(success_msg)

                    progress = (i + 1) / len(images)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing: {filename} ({i+1}/{len(images)}) with {feature_detector}")

                    stframe.image(result_frame, channels="BGR", caption=f"Processing {filename} with {feature_detector}")

                    if st.session_state.trajectory and len(st.session_state.trajectory) > 1:
                        traj_np = np.array(st.session_state.trajectory)
                        map_points_np = np.array(map_points)
                        # st.info(map_points)
                        # Create 3D scatter plot for traj_np (points)
                        scatter_traj = go.Scatter3d(
                            x=traj_np[:, 0],
                            y=traj_np[:, 2],
                            z=traj_np[:, 1],
                            mode='lines+markers',
                            marker=dict(size=2, color='red'),
                            line=dict(width=2, color='red'),
                            name='Estimated Trajectory'
                        )
                        # Create 3D line plot for trajectory_GT (ground truth)
                        scatter_gt = go.Scatter3d(
                            x=trajectory_GT[:len(traj_path), 0],
                            y=trajectory_GT[:len(traj_path), 2],
                            z=trajectory_GT[:len(traj_path), 1],
                            mode='lines+markers',
                            line=dict(width=4, color='green'),
                            marker=dict(size=4, color='green'),
                            opacity=0.7,
                            name='Ground Truth'
                        )
                        # If map points are available, add them to the plot
                        scatter_map_points = go.Scatter3d(
                                x=map_points_np[:, 0][:, 0],
                                y=map_points_np[:, 2][:, 2],
                                z=map_points_np[:, 1][:, 1],
                                mode='markers',
                                marker=dict(size=2, color='white', symbol='x'),
                                name='Map Points'
                            )
                            # # Add map points to the figure
                        # Create figure
                        fig = go.Figure(data=[scatter_traj, scatter_gt]) # , scatter_map_points
                        fig.update_layout(
                            scene=dict(
                                xaxis_title='X (m)',
                                yaxis_title='Z (m)',  # Z as floor axis
                                zaxis_title='Y (m)',  # Y as height
                                xaxis=dict(showgrid=True, gridcolor='white', zeroline=False),
                                yaxis=dict(showgrid=True, gridcolor='white', zeroline=False),
                                zaxis=dict(showgrid=True, gridcolor='white', zeroline=False),
                                aspectmode='data',
                                camera=dict(
                                    eye=dict(x=1, y=-5, z=10)
                                ),
                                bgcolor='rgba(0, 0, 0, 0)'
                            ),
                            title=dict(
                                text='Ground Truth Trajectory (3D)',
                                font=dict(size=22, color='white')
                            ),
                            legend=dict(
                                x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.7)', bordercolor='black'
                            ),
                            margin=dict(l=0, r=0, b=0, t=40)
                        )
                        traj_frame1, traj_frame2 = traj_frame.columns(2)
                        traj_frame1.plotly_chart(fig, use_container_width=True)
                        traj_frame21, traj_frame22 = traj_frame2.columns(2)
                        traj_frame21.metric(str(map_points is None), "No Map Points" if map_points is None else f"{len(map_points)} Map Points")
                    histogram = go.Figure(data=[go.Bar(x=list(range(len(hist))), y=hist)])
                    histogram.update_layout(
                        title="BoVW Histogram",
                        xaxis_title="Visual Word Index",
                        yaxis_title="Frequency",
                        bargap=0.1
                    )
                    traj_frame2.plotly_chart(histogram, use_container_width=True)
                        # temp_x=map_points_np[:, 0][:, 0],
                        # temp_y=map_points_np[:, 2][:, 2],
                        # temp_z=map_points_np[:, 1][:, 1],
                        # traj_frame2.write([temp_x, temp_y, temp_z])
                        # traj_frame2.write([temp_x[1], temp_y[1], temp_z[1]])
                        # traj_frame2.write([temp_x[2], temp_y[2], temp_z[2]])
                        # traj_frame2.plotly_chart(fig2, use_container_width=True)
                        # fig, ax = plt.subplots(figsize=(8, 6))
                        # ax.plot(traj_np[:, 0], traj_np[:, 1], marker='o', linewidth=1, markersize=3, color='green')
                        # ax.plot(trajectory_GT[:, 0], trajectory_GT[:, 2], 'b-', linewidth=2, alpha=0.7, label='Ground Truth')
                        # ax.set_title(f"2D Camera Trajectory ({feature_detector})")
                        # ax.set_xlabel("X")
                        # ax.set_ylabel("Z")
                        # ax.grid(True)
                        # ax.axis('equal')
                        # ax.legend()
                        # traj_frame.pyplot(fig)
                        # plt.close(fig)

                    with text_placeholder.container():
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.subheader("Current Image")
                            st.write(f"**File:** {filename}")
                            st.write(f"**Progress:** {i+1}/{len(images)}")
                            # st.progress(i+1/len(images))
                        with col2:
                            st.subheader("Feature Detector")
                            st.code(detector_used)
                        with col3:
                            st.subheader("Trajectory Points")
                            st.write(f"Total: {len(st.session_state.trajectory) if st.session_state.trajectory else 0}")
                        col11, col12 = st.columns(2)
                        col11.metric("Sequence", sequence)
                        col12.metric("Feature Detector", feature_detector)
                        st.progress(current/numberOfDetectors)

                        
                        col21, col22 = st.columns(2)
                        if E is not None:
                            col21.subheader("Essential Matrix")
                            col21.write(E)
                        if F is not None:
                            col22.subheader("Fundamental Matrix")
                            col22.write(F)

                    if show_table and not st.session_state.poses_dataframe.empty:
                        display_poses_table(
                            st.session_state.poses_dataframe, 
                            i, trajectory_GT.shape[0]
                        )

                except Exception as e:
                    error_msg = f"Error processing {filename} with {feature_detector}: {str(e)}"
                    log_error(error_msg)
                    continue
                loop_closure = bovw.LoopChecks() 
                with loopClosure.container():
                    st.dataframe(loop_closure)               

            
            if not st.session_state.poses_dataframe.empty:
                final_csv = save_dataframe_to_csv(
                    st.session_state.poses_dataframe, 
                    sequence, 
                    feature_detector
                )
                
                mse_results.append({
                    "Sequence": sequence,
                    "Feature_Detector": feature_detector,
                    "Time Taken": (datetime.now() - start_time).total_seconds(),
                    "MSE_X": st.session_state.poses_dataframe["Error_X"].mean() if not st.session_state.poses_dataframe.empty else 0,
                    "MSE_Y": st.session_state.poses_dataframe["Error_Y"].mean() if not st.session_state.poses_dataframe.empty else 0,
                    "MSE_Z": st.session_state.poses_dataframe["Error_Z"].mean() if not st.session_state.poses_dataframe.empty else 0,
                })
                
                success_msg = f"Completed processing sequence {sequence} with {feature_detector}. Final CSV saved: {final_csv}"
                log_success(success_msg)

            progress_bar.empty()
            status_text.empty()
            stop_button_placeholder.empty()
            csv_save_info.empty()
            st.session_state.trajectory = []
            st.session_state.poses_dataframe = pd.DataFrame()
    
    logger.info(f"Completed processing sequence {sequence}")
    return mse_results

# ---------------------
# Run VO - Main Logic
# ---------------------



if run_all:
    logger.info("Starting processing for all sequences (00-10)")
    all_mse_results = []
    for seq in [f"{i:02d}" for i in range(11)]:
        info_msg = f"Starting processing for sequence {seq}"
        log_info(info_msg)
        trajectory_GT = show_seq_info(calib_file = os.path.join(BASE_DIR, seq, "calib2.txt"), pose_file = os.path.join(POSE_DIR, f"{seq}.txt"))
        mse_results = process_sequence(seq, show_table, batch_size=batch_size, trajectory_GT=trajectory_GT)
        if mse_results:
            all_mse_results.extend(mse_results)
    
    if all_mse_results:
        mse_df = pd.DataFrame(all_mse_results)
        st.subheader("Mean Squared Error Results (All Sequences)")
        st.dataframe(mse_df)
        mse_df.to_csv("results/all_sequences_mse_results.csv", index=False)
    
    success_msg = "Completed processing all sequences (00-10)!"
    log_success(success_msg)

elif run_single:
    logger.info(f"Starting single sequence processing for sequence {sequence}")
    trajectory_GT = show_seq_info(calib_file = os.path.join(BASE_DIR, sequence, "calib2.txt"), pose_file = os.path.join(POSE_DIR, f"{sequence}.txt"))
    mse_results = process_sequence(sequence, show_table, batch_size=batch_size, trajectory_GT=trajectory_GT)
    
    if mse_results:
        mse_df = pd.DataFrame(mse_results)
        st.subheader("Mean Squared Error Results")
        st.dataframe(mse_df)
        mse_df.to_csv(f"results/sequence_{sequence}_mse_results.csv", index=False)
    
    if st.session_state.trajectory:
        success_msg = f"Sequence {sequence} processing completed! Final trajectory has {len(st.session_state.trajectory)} points."
        log_success(success_msg)

else:
    info_msg = "Toggle 'Run All Sequences' or 'Run Single Sequence' to start processing."
    trajectory_GT = show_seq_info(calib_file = os.path.join(BASE_DIR, sequence, "calib2.txt"), pose_file = os.path.join(POSE_DIR, f"{sequence}.txt"))
    log_info(info_msg)
    table_placeholder.empty()

# Final display of complete poses dataframe
with st.expander("Complete Poses DataFrame (CSV Format)", expanded=False):
    if not st.session_state.poses_dataframe.empty:
        st.write("This is the complete dataframe that matches the saved CSV format:")
        st.dataframe(st.session_state.poses_dataframe, use_container_width=True)
        
        csv_data = st.session_state.poses_dataframe.to_csv(index=False)
        st.download_button(
            label="Download Complete DataFrame as CSV",
            data=csv_data,
            file_name=f"complete_poses_{sequence}.csv",
            mime="text/csv"
        )
    else:
        info_msg = "No pose data available. Run the visual odometry pipeline to generate data."
        log_info(info_msg, show_in_ui=False)  # Only log, don't show in UI here
        st.info(info_msg)

logger.info("Visual Odometry application session ended")