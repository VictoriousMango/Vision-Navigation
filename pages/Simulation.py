import streamlit as st
import glob
import plotly.graph_objects as go
from pathlib import Path
import numpy as np
from assets.FrontEnd_Module import Pipeline, KITTIDataset
from assets.Mapping import Animated3DTrajectory
import os
import pandas as pd
import cv2
import ast
import json

st.title("Simulations of VSLAM Sequence")

# Session Management
if 'trajectory' not in st.session_state:
    st.session_state.trajectory = {
        "Grount Truth": [],
        "Estimated": [],
        "Loop Closure": [],
        "Keyframes Path": []
    }

# placeholders
image_sequence, trajectory_placeholder, Results = st.tabs(["Image Sequence", "Trajectory", "Results"])
map_placeholder = st.empty()
with st.sidebar:
    control_panel = st.empty()
Loop_closure = st.empty()
metrics = st.empty()
Animation_trajectory = st.empty()

# Functions --------------------------------------------------
def save_visual_word_as_string(visual_word):
    """Convert visual word list to string for CSV storage"""
    return json.dumps(visual_word.tolist() if isinstance(visual_word, np.ndarray) else visual_word)

def load_visual_word_from_string(visual_word_str):
    """Convert string back to list for plotting"""
    try:
        return json.loads(visual_word_str)
    except:
        # Fallback to ast.literal_eval if json fails
        try:
            return ast.literal_eval(visual_word_str)
        except:
            st.error(f"Could not parse visual word data: {visual_word_str}")
            return []
def get_available_results():
    """Get list of sequences that have stored results"""
    results_dir = "./results/simulation/"
    if not os.path.exists(results_dir):
        return []
    
    # Find all trajectory files
    trajectory_files = glob.glob(f"{results_dir}*_trajectory_estimated.csv")
    sequences = []
    
    for file in trajectory_files:
        # Extract sequence name from filename
        filename = os.path.basename(file)
        sequence_name = filename.replace("_trajectory_estimated.csv", "")
        sequences.append(sequence_name)
    
    return sorted(sequences)

def load_sequence_data(sequence_name):
    """Load all data for a specific sequence"""
    results_dir = "./results/simulation/"
    
    # Load trajectory data
    trajectory_file = f"{results_dir}{sequence_name}_trajectory_estimated.csv"
    if not os.path.exists(trajectory_file):
        return None, None
    
    df = pd.read_csv(trajectory_file)
    
    # Load keyframes if they exist
    keyframes_dir = f"./results/keyframes/{sequence_name}/"
    keyframes = []
    if os.path.exists(keyframes_dir):
        # Get all image files in the keyframes directory
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        for ext in image_extensions:
            keyframes.extend(glob.glob(f"{keyframes_dir}{ext}"))
        keyframes = sorted(keyframes)
    
    return df, keyframes
# ------------------------------------------------------------

# Import Classes
VO = Pipeline()
kitti = KITTIDataset()
animate_chart = Animated3DTrajectory()
with control_panel.container():
    if st.toggle("Reset Chart Var's"):
        animate_chart.reset()

with image_sequence.container():
    st.subheader("Image Sequence KITTI Dataset")
    sequence=st.slider(label='Select Sequence', min_value=0, max_value=10, value=0, step=1, format='%02d')
    col1, col2 = st.columns(2)
    with col1:
        folder_path = st.text_input(
            "Images Directory Path",
            help="Path to the directory containing KITTI sequence images", 
            value=f"D:/coding/Temp_Download/data_odometry_color/dataset/sequences/{sequence:02d}/image_2"
        )
        root = os.path.abspath(folder_path)
        
        calib_file = st.text_input(
            "Calibration File Path",
            help="Path to the calib.txt file", 
            value=f"D:/coding/Temp_Download/data_odometry_color/dataset/sequences/{sequence:02d}/calib2.txt"
        )
        with open(calib_file, 'r') as f:
            K, P = kitti.load_calib(f)
            VO.set_k_intrinsic(K)
        st.subheader("Camera Intrinsic Matrix K")
        st.dataframe(K)
    
    with col2:
        poses_file = st.text_input(
            "Ground Truth Poses File (Optional)",
            help="Path to the poses file for evaluation", 
            value=f"D:/coding/Temp_Download/data_odometry_poses/dataset/poses/{sequence:02d}.txt"
        )
        
        sequence_length = st.number_input(
            "Sequence Length (Optional)",
            min_value=1, max_value=10000, value=100,
            help="Limit the number of frames to process"
        )
        files = os.listdir(folder_path)
        files.sort()
        files = files[:min(len(files), sequence_length)]
        st.session_state.trajectory["Keyframes Path"] = [os.path.join(root, file) for file in files]

        with open(poses_file, 'r') as f:
            poses = kitti.load_poses(f, batch_size=sequence_length)
        st.subheader("Camera Projection Matrix P")
        st.dataframe(P)
        st.session_state.trajectory["Grount Truth"] = []
        for pose in poses:
            x, y, z = pose[0, 3], pose[1, 3], pose[2, 3]
            st.session_state.trajectory["Grount Truth"].append([x, y, z])
    with st.status("Visualizing pose"):
        st.write(len(animate_chart.lst_to_df(st.session_state.trajectory["Grount Truth"])))
        fig_gt = animate_chart.plotly_chart(animate_chart.lst_to_df(st.session_state.trajectory["Grount Truth"]))
        st.plotly_chart(fig_gt, use_container_width=True)

with trajectory_placeholder.container():
    # PlaceHolders
    with st.sidebar:
        processing = st.empty()
    toggle_button = st.empty()
    frame_viewer = st.empty()
    frame_plots = st.empty()
    if toggle_button.toggle("Start Processing"):
        with processing.status("Processing"):
            metrics = st.empty()
            st.session_state.trajectory["Estimated"] = []
            # ploting    
            hist_list = []        
            for index, frame in enumerate(st.session_state.trajectory["Keyframes Path"]):
                img = cv2.imread(frame)
                frame_viewer.image(img, channels="BGR", use_container_width=True)
                # Apply VSLAM Pipeline
                result_frame, E, F, st.session_state.trajectory["Estimated"], detector_used, map_points, desc, kp = VO.VisualOdometry(
                            img,
                            FeatureDetector="FD_SIFT",
                            FeatureMatcher="FM_BF_NORM_L2"
                        )
                trajectory, VBoW = frame_plots.columns(2)
                hist_list.append(VO.visual_word)
                hist = hist_list[-1]
                
                # Create histogram plot
                histogram = go.Figure(data=[go.Bar(x=list(range(len(hist))), y=hist)])
                histogram.update_layout(
                    title="BoVW Histogram",
                    xaxis_title="Visual Word Index",
                    yaxis_title="Frequency",
                    bargap=0.1
                )
                VBoW.plotly_chart(histogram, use_container_width=True, key=f"histogram_{index}")
                
                fig = animate_chart.static_chart(animate_chart.lst_to_df(st.session_state.trajectory["Estimated"]))
                trajectory.plotly_chart(fig, use_container_width=True, key=f"trajectory_{index}")
                with metrics.container():
                    st.subheader("Metrics")
                    st.write(f"Frame ID: {index}")
                    st.dataframe(VO.loop_detector.LoopChecks())
            
            # Save results to CSV
            df = animate_chart.lst_to_df(st.session_state.trajectory["Estimated"])
            st.write(f"{len(df)} : {len(VO.loop_detector.bovw_history)}")
            df['frame_index'] = df.index
            st.write(f"{len(df)} : {len(VO.loop_detector.bovw_history)}")
            
            # Convert visual words to strings for CSV storage
            visual_words_str = [save_visual_word_as_string(hist) for hist in hist_list[:len(df)]]
            df['Visual Word'] = visual_words_str
            
            # Ensure results directory exists
            os.makedirs("./results/simulation", exist_ok=True)
            df.to_csv(f"./results/simulation/{sequence}_trajectory_estimated.csv", index=False)
    else:
        processing.empty()

with Results.container():
    st.subheader("Stored Results Viewer")
    
    # Get available results
    available_sequences = get_available_results()
    
    if not available_sequences:
        st.warning("No stored results found. Please run some simulations first.")
        st.info("Results will be stored in: ./results/simulation/")
    else:
        # Sequence selector
        col1, col2 = st.columns([3, 1])
        
        with col1:
            selected_sequence = st.selectbox(
                "Select sequence to view results:",
                available_sequences,
                help="Choose from previously processed sequences"
            )
        
        with col2:
            if st.button("🔄 Refresh Results List"):
                st.rerun()
        
        if selected_sequence:
            # Load data for selected sequence
            df, keyframes = load_sequence_data(selected_sequence)
            
            if df is not None:
                st.success(f"Loaded results for sequence: **{selected_sequence}**")
                
                # Display basic info
                info_col1, info_col2, info_col3 = st.columns(3)
                with info_col1:
                    st.metric("Total Frames", len(df))
                with info_col2:
                    st.metric("Keyframes Available", len(keyframes))
                with info_col3:
                    if 'Visual Word' in df.columns:
                        st.metric("Histogram Data", "✅ Available")
                    else:
                        st.metric("Histogram Data", "❌ Not Available")
                
                # Main display area
                frame_viewer = st.empty()
                frame_plots, histogramplots = st.columns(2)
                frame_slider = st.empty()
                
                # Frame slider
                max_index = len(df) - 1
                if max_index >= 0:
                    index = frame_slider.slider(
                        "Select Frame", 
                        min_value=0, 
                        max_value=max_index, 
                        value=0, 
                        step=1,
                        key=f"frame_slider_{selected_sequence}"
                    )
                    
                    # Display frame image
                    if keyframes and index < len(keyframes):
                        try:
                            frame_path = keyframes[index]
                            img = cv2.imread(frame_path)
                            if img is not None:
                                frame_viewer.image(
                                    img, 
                                    caption=f"Frame {index}: {os.path.basename(frame_path)}", 
                                    channels="BGR", 
                                    use_container_width=True
                                )
                            else:
                                frame_viewer.error(f"Could not load image: {frame_path}")
                        except Exception as e:
                            frame_viewer.error(f"Error loading frame: {e}")
                    elif keyframes:
                        frame_viewer.warning(f"Frame {index} not available in keyframes")
                    else:
                        frame_viewer.info("No keyframe images available for this sequence")
                    
                    # Plot trajectory
                    try:
                        fig = animate_chart.plotly_chart(df)
                        frame_plots.plotly_chart(
                            fig, 
                            use_container_width=True, 
                            key=f"trajectory_{selected_sequence}_{index}"
                        )
                    except Exception as e:
                        frame_plots.error(f"Error plotting trajectory: {e}")
                    
                    # Plot histogram
                    if index < len(df) and 'Visual Word' in df.columns:
                        try:
                            hist_str = df["Visual Word"].iloc[index]
                            hist = load_visual_word_from_string(hist_str)
                            
                            if hist and len(hist) > 0:
                                histogram = go.Figure(data=[go.Bar(
                                    x=list(range(len(hist))), 
                                    y=hist,
                                    marker_color='lightblue'
                                )])
                                histogram.update_layout(
                                    title=f"BoVW Histogram - {selected_sequence} Frame {index}",
                                    xaxis_title="Visual Word Index",
                                    yaxis_title="Frequency",
                                    bargap=0.1,
                                    height=400
                                )
                                histogramplots.plotly_chart(
                                    histogram, 
                                    use_container_width=True, 
                                    key=f"histogram_{selected_sequence}_{index}"
                                )
                            else:
                                histogramplots.error("Could not load histogram data for this frame")
                        except Exception as e:
                            histogramplots.error(f"Error loading histogram: {e}")
                    else:
                        if 'Visual Word' not in df.columns:
                            histogramplots.warning("No histogram data column found in results")
                        else:
                            histogramplots.warning("No histogram data available for this frame")
                
                # Additional controls
                st.divider()
                
                # Export options
                # export_col1, export_col2, export_col3 = st.columns(3)
                
                # with export_col1:
                #     if st.button("📊 Export Data as CSV", key=f"export_{selected_sequence}"):
                #         csv = df.to_csv(index=False)
                #         st.download_button(
                #             label="Download CSV",
                #             data=csv,
                #             file_name=f"{selected_sequence}_results.csv",
                #             mime="text/csv"
                #         )
                
                # with export_col2:
                #     if st.button("📈 Show Full Trajectory", key=f"full_traj_{selected_sequence}"):
                #         st.plotly_chart(animate_chart.plotly_chart(df), use_container_width=True)
                
                # with export_col3:
                #     if st.button("🗑️ Delete Results", key=f"delete_{selected_sequence}"):
                #         if st.button("⚠️ Confirm Delete", key=f"confirm_delete_{selected_sequence}"):
                #             try:
                #                 os.remove(f"./results/simulation/{selected_sequence}_trajectory_estimated.csv")
                #                 st.success(f"Deleted results for {selected_sequence}")
                #                 st.rerun()
                #             except Exception as e:
                #                 st.error(f"Error deleting results: {e}")
                
            else:
                st.error(f"Could not load data for sequence: {selected_sequence}")
                st.info("The trajectory file might be corrupted or missing.")

# Optional: Add a summary view of all results
if st.checkbox("Show Results Summary"):
    st.subheader("All Stored Results Summary")
    
    summary_data = []
    for seq in available_sequences:
        try:
            df, keyframes = load_sequence_data(seq)
            if df is not None:
                summary_data.append({
                    "Sequence": seq,
                    "Frames": len(df),
                    "Keyframes": len(keyframes) if keyframes else 0,
                    "Has Histograms": "Visual Word" in df.columns,
                    "File Size": f"{os.path.getsize(f'./results/simulation/{seq}_trajectory_estimated.csv') / 1024:.1f} KB"
                })
        except Exception:
            continue
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)
    else:
        st.info("No valid results found for summary.")