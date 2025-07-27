import streamlit as st
import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import time
from typing import List, Dict, Optional
import tempfile
import zipfile
from PIL import Image
import io

# Import your VSLAM pipeline
try:
    from assets.Pipeline import VSLAMPipeline
    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False
    st.error("❌ Pipeline.py not found. Please ensure the VSLAM pipeline is in the same directory.")

# Page configuration
st.set_page_config(
    page_title="Visual SLAM Pipeline",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .status-success {
        color: #28a745;
        font-weight: bold;
    }
    .status-warning {
        color: #ffc107;
        font-weight: bold;
    }
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
if 'processing_results' not in st.session_state:
    st.session_state.processing_results = []
if 'is_processing' not in st.session_state:
    st.session_state.is_processing = False
if 'dataset_loaded' not in st.session_state:
    st.session_state.dataset_loaded = False

def initialize_pipeline():
    """Initialize the VSLAM pipeline with selected parameters"""
    if not PIPELINE_AVAILABLE:
        return None
    
    try:
        pipeline = VSLAMPipeline(
            feature_detector=st.session_state.feature_detector,
            feature_matcher=st.session_state.feature_matcher,
            enable_loop_detection=st.session_state.enable_loop_detection,
            enable_optimization=st.session_state.enable_optimization,
            optimization_interval=st.session_state.optimization_interval
        )
        return pipeline
    except Exception as e:
        st.error(f"Failed to initialize pipeline: {e}")
        return None

def load_kitti_dataset(images_path, calib_file, poses_file=None, sequence_length=None):
    """Load KITTI dataset into the pipeline"""
    if st.session_state.pipeline is None:
        return []
    
    try:
        image_files = st.session_state.pipeline.load_kitti_dataset(
            images_path=images_path,
            calib_file=calib_file,
            poses_file=poses_file,
            sequence_length=sequence_length
        )
        return image_files
    except Exception as e:
        st.error(f"Failed to load KITTI dataset: {e}")
        return []

def process_uploaded_images(uploaded_files, camera_matrix):
    """Process uploaded images"""
    image_files = []
    temp_dir = tempfile.mkdtemp()
    
    for i, uploaded_file in enumerate(uploaded_files):
        # Save uploaded file to temporary directory
        file_path = os.path.join(temp_dir, f"frame_{i:06d}.jpg")
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        image_files.append(file_path)
    
    # Set camera calibration
    if st.session_state.pipeline:
        st.session_state.pipeline.set_camera_calibration(camera_matrix)
    
    return image_files

def create_3d_trajectory_plot(pipeline):
    """Create interactive 3D trajectory plot using Plotly"""
    if not pipeline.trajectory_plot_data['estimated_x']:
        return None
    
    fig = go.Figure()
    
    # Estimated trajectory
    fig.add_trace(go.Scatter3d(
        x=pipeline.trajectory_plot_data['estimated_x'],
        y=pipeline.trajectory_plot_data['estimated_y'],
        z=pipeline.trajectory_plot_data['estimated_z'],
        mode='lines+markers',
        name='Estimated Trajectory',
        line=dict(color='blue', width=4),
        marker=dict(size=3)
    ))
    
    # Ground truth trajectory (if available)
    if pipeline.trajectory_plot_data['gt_x']:
        fig.add_trace(go.Scatter3d(
            x=pipeline.trajectory_plot_data['gt_x'],
            y=pipeline.trajectory_plot_data['gt_y'],
            z=pipeline.trajectory_plot_data['gt_z'],
            mode='lines+markers',
            name='Ground Truth',
            line=dict(color='red', width=4, dash='dash'),
            marker=dict(size=3)
        ))
    
    # Add loop closures
    if pipeline.loop_closures:
        loop_x, loop_y, loop_z = [], [], []
        for loop in pipeline.loop_closures:
            frame_id = loop.get('frame_id', 0)
            if frame_id < len(pipeline.trajectory_plot_data['estimated_x']):
                loop_x.append(pipeline.trajectory_plot_data['estimated_x'][frame_id])
                loop_y.append(pipeline.trajectory_plot_data['estimated_y'][frame_id])
                loop_z.append(pipeline.trajectory_plot_data['estimated_z'][frame_id])
        
        if loop_x:
            fig.add_trace(go.Scatter3d(
                x=loop_x, y=loop_y, z=loop_z,
                mode='markers',
                name='Loop Closures',
                marker=dict(color='green', size=8, symbol='diamond')
            ))
    
    fig.update_layout(
        title="3D SLAM Trajectory",
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)',
            aspectmode='cube'
        ),
        width=800,
        height=600
    )
    
    return fig

def create_error_plot(pipeline):
    """Create trajectory error plot"""
    if (not pipeline.trajectory_plot_data['estimated_x'] or 
        not pipeline.trajectory_plot_data['gt_x'] or
        len(pipeline.trajectory_plot_data['estimated_x']) != len(pipeline.trajectory_plot_data['gt_x'])):
        return None
    
    errors = []
    for i in range(len(pipeline.trajectory_plot_data['estimated_x'])):
        est_pos = np.array([
            pipeline.trajectory_plot_data['estimated_x'][i],
            pipeline.trajectory_plot_data['estimated_y'][i],
            pipeline.trajectory_plot_data['estimated_z'][i]
        ])
        gt_pos = np.array([
            pipeline.trajectory_plot_data['gt_x'][i],
            pipeline.trajectory_plot_data['gt_y'][i],
            pipeline.trajectory_plot_data['gt_z'][i]
        ])
        error = np.linalg.norm(est_pos - gt_pos)
        errors.append(error)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(errors))),
        y=errors,
        mode='lines',
        name='Position Error',
        line=dict(color='red', width=2)
    ))
    
    fig.update_layout(
        title="Trajectory Error Over Time",
        xaxis_title="Frame",
        yaxis_title="Position Error (m)",
        width=800,
        height=400
    )
    
    return fig

def create_performance_plots(pipeline):
    """Create performance analysis plots"""
    if not pipeline.processing_times:
        return None
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Processing Time per Frame', 'Processing Time Distribution', 
                       'Optimization Times', 'Feature Statistics'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Processing time per frame
    fig.add_trace(
        go.Scatter(x=list(range(len(pipeline.processing_times))), 
                  y=pipeline.processing_times,
                  mode='lines', name='Processing Time',
                  line=dict(color='blue')),
        row=1, col=1
    )
    
    # Processing time histogram
    fig.add_trace(
        go.Histogram(x=pipeline.processing_times, name='Time Distribution',
                    marker_color='lightblue'),
        row=1, col=2
    )
    
    # Optimization times
    if pipeline.optimization_times:
        fig.add_trace(
            go.Scatter(x=list(range(len(pipeline.optimization_times))), 
                      y=pipeline.optimization_times,
                      mode='lines+markers', name='Optimization Time',
                      line=dict(color='green')),
            row=2, col=1
        )
    
    # Feature statistics (if available in results)
    if hasattr(pipeline, 'feature_counts') and pipeline.feature_counts:
        fig.add_trace(
            go.Scatter(x=list(range(len(pipeline.feature_counts))), 
                      y=pipeline.feature_counts,
                      mode='lines', name='Features Detected',
                      line=dict(color='orange')),
            row=2, col=2
        )
    
    fig.update_layout(height=800, showlegend=True, title_text="Performance Analysis")
    return fig

# Main application
def main():
    st.markdown("<h1 class='main-header'>🤖 Visual SLAM Pipeline</h1>", unsafe_allow_html=True)
    
    if not PIPELINE_AVAILABLE:
        st.error("❌ VSLAM Pipeline not available. Please check the Pipeline.py file.")
        return
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Pipeline parameters
    st.sidebar.subheader("Pipeline Parameters")
    feature_detector = st.sidebar.selectbox(
        "Feature Detector",
        ['FD_SIFT', 'FD_ORB', 'FD_BRISK'],
        key='feature_detector'
    )
    
    feature_matcher = st.sidebar.selectbox(
        "Feature Matcher",
        ['FM_BF_NORM_L2', 'FM_BF_NORM_Hamming'],
        key='feature_matcher'
    )
    
    enable_loop_detection = st.sidebar.checkbox(
        "Enable Loop Detection", 
        value=True,
        key='enable_loop_detection'
    )
    
    enable_optimization = st.sidebar.checkbox(
        "Enable Bundle Adjustment", 
        value=True,
        key='enable_optimization'
    )
    
    optimization_interval = st.sidebar.slider(
        "Optimization Interval (frames)",
        min_value=1, max_value=20, value=5,
        key='optimization_interval'
    )
    
    # Initialize pipeline button
    if st.sidebar.button("🔄 Initialize Pipeline"):
        with st.spinner("Initializing VSLAM Pipeline..."):
            st.session_state.pipeline = initialize_pipeline()
            if st.session_state.pipeline:
                st.success("✅ Pipeline initialized successfully!")
            else:
                st.error("❌ Failed to initialize pipeline")
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📁 Dataset", "🚀 Processing", "📊 Results", "⚙️ Analysis"])
    
    # Dataset tab
    with tab1:
        st.header("Dataset Configuration")
        
        dataset_type = st.radio(
            "Select Dataset Type",
            ["KITTI Dataset", "Upload Images", "Webcam (Live)"]
        )
        
        if dataset_type == "KITTI Dataset":
            st.subheader("KITTI Dataset Configuration")
            sequence=st.slider(label='Select Sequence', min_value=0, max_value=10, value=0, step=1, format='%02d')
            col1, col2 = st.columns(2)
            
            with col1:
                images_path = st.text_input(
                    "Images Directory Path",
                    help="Path to the directory containing KITTI sequence images", value=f"D:/coding/Temp_Download/data_odometry_color/dataset/sequences/{sequence:02d}/image_2"
                )
                
                calib_file = st.text_input(
                    "Calibration File Path",
                    help="Path to the calib.txt file", value=f"D:/coding/Temp_Download/data_odometry_color/dataset/sequences/{sequence:02d}/calib2.txt"
                )
            
            with col2:
                poses_file = st.text_input(
                    "Ground Truth Poses File (Optional)",
                    help="Path to the poses file for evaluation", value=f"D:/coding/Temp_Download/data_odometry_poses/dataset/poses/{sequence:02d}.txt"
                )
                
                sequence_length = st.number_input(
                    "Sequence Length (Optional)",
                    min_value=1, max_value=10000, value=100,
                    help="Limit the number of frames to process"
                )
            
            if st.button("📂 Load KITTI Dataset"):
                if st.session_state.pipeline and images_path and calib_file:
                    with st.spinner("Loading KITTI dataset..."):
                        image_files = load_kitti_dataset(
                            images_path, calib_file, 
                            poses_file if poses_file else None,
                            sequence_length
                        )
                        
                        if image_files:
                            st.session_state.image_files = image_files
                            st.session_state.dataset_loaded = True
                            st.success(f"✅ Dataset loaded successfully! {len(image_files)} images ready for processing.")
                        else:
                            st.error("❌ Failed to load dataset")
                else:
                    st.warning("⚠️ Please initialize pipeline and provide required paths")
        
        elif dataset_type == "Upload Images":
            st.subheader("Upload Image Sequence")
            
            # Camera calibration parameters
            st.write("**Camera Calibration Parameters**")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                fx = st.number_input("fx", value=718.856, format="%.3f")
            with col2:
                fy = st.number_input("fy", value=718.856, format="%.3f")
            with col3:
                cx = st.number_input("cx", value=607.1928, format="%.3f")
            with col4:
                cy = st.number_input("cy", value=185.2157, format="%.3f")
            
            camera_matrix = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ])
            
            # File upload
            uploaded_files = st.file_uploader(
                "Choose image files",
                type=['png', 'jpg', 'jpeg'],
                accept_multiple_files=True,
                help="Upload a sequence of images for SLAM processing"
            )
            
            if uploaded_files and st.button("📤 Process Uploaded Images"):
                if st.session_state.pipeline:
                    with st.spinner("Processing uploaded images..."):
                        image_files = process_uploaded_images(uploaded_files, camera_matrix)
                        
                        if image_files:
                            st.session_state.image_files = image_files
                            st.session_state.dataset_loaded = True
                            st.success(f"✅ {len(image_files)} images uploaded and ready for processing!")
                        else:
                            st.error("❌ Failed to process uploaded images")
                else:
                    st.warning("⚠️ Please initialize pipeline first")
        
        elif dataset_type == "Webcam (Live)":
            st.subheader("Live Webcam Processing")
            st.info("🚧 Live webcam processing will be implemented in a future version.")
    
    # Processing tab
    with tab2:
        st.header("VSLAM Processing")
        
        if not st.session_state.dataset_loaded:
            st.warning("⚠️ Please load a dataset first in the Dataset tab")
            return
        
        if st.session_state.pipeline is None:
            st.warning("⚠️ Please initialize the pipeline first")
            return
        
        # Processing controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🚀 Start Processing", disabled=st.session_state.is_processing):
                st.session_state.is_processing = True
                
        with col2:
            if st.button("⏹️ Stop Processing"):
                st.session_state.is_processing = False
                
        with col3:
            if st.button("🔄 Reset Pipeline"):
                if st.session_state.pipeline:
                    st.session_state.pipeline.reset()
                    st.session_state.processing_results = []
                    st.success("✅ Pipeline reset completed")
        
        # Processing progress and real-time updates
        if st.session_state.is_processing and hasattr(st.session_state, 'image_files'):
            st.subheader("Processing Progress")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Real-time metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                frames_metric = st.empty()
            with col2:
                fps_metric = st.empty()
            with col3:
                features_metric = st.empty()
            with col4:
                loops_metric = st.empty()
            
            # Live trajectory plot
            trajectory_plot = st.empty()
            
            # Current frame display
            current_frame_display = st.empty()
            
            # Process frames
            total_frames = len(st.session_state.image_files)
            results = []
            
            start_time = time.time()
            
            for i, image_path in enumerate(st.session_state.image_files):
                if not st.session_state.is_processing:
                    break
                
                try:
                    # Load and process frame
                    frame = cv2.imread(image_path)
                    if frame is None:
                        continue
                    
                    result = st.session_state.pipeline.process_frame(frame)
                    results.append(result)
                    
                    # Update progress
                    progress = (i + 1) / total_frames
                    progress_bar.progress(progress)
                    
                    # Update metrics
                    elapsed_time = time.time() - start_time
                    fps = (i + 1) / elapsed_time if elapsed_time > 0 else 0
                    
                    frames_metric.metric("Frames Processed", f"{i+1}/{total_frames}")
                    fps_metric.metric("Processing FPS", f"{fps:.2f}")
                    features_metric.metric("Features", result.get('num_features', 0))
                    loops_metric.metric("Loop Closures", len(st.session_state.pipeline.loop_closures))
                    
                    # Update status
                    status_text.text(f"Processing frame {i+1}/{total_frames} - {progress*100:.1f}% complete")
                    
                    # Update trajectory plot every 10 frames
                    if (i + 1) % 10 == 0:
                        fig = create_3d_trajectory_plot(st.session_state.pipeline)
                        if fig:
                            trajectory_plot.plotly_chart(fig, use_container_width=True)
                    
                    # Show current processed frame
                    if result.get('processed_frame') is not None:
                        processed_frame = cv2.cvtColor(result['processed_frame'], cv2.COLOR_BGR2RGB)
                        current_frame_display.image(processed_frame, caption=f"Frame {i+1}", use_container_width=True)
                    
                    # Small delay to prevent overwhelming the interface
                    time.sleep(0.01)
                    
                except Exception as e:
                    st.error(f"Error processing frame {i}: {e}")
                    continue
            
            # Processing completed
            st.session_state.is_processing = False
            st.session_state.processing_results = results
            
            if results:
                st.success(f"✅ Processing completed! {len(results)} frames processed successfully.")
                
                # Final statistics
                stats = st.session_state.pipeline.get_statistics()
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Frames", stats['frames_processed'])
                with col2:
                    st.metric("Avg Processing Time", f"{stats['avg_processing_time']:.3f}s")
                with col3:
                    st.metric("Loop Closures", stats['loop_closures_detected'])
                with col4:
                    st.metric("Trajectory Points", stats['trajectory_length'])
            else:
                st.error("❌ Processing failed - no results generated")
    
    # Results tab
    with tab3:
        st.header("Processing Results")
        
        if not st.session_state.processing_results:
            st.info("ℹ️ No processing results available. Please run VSLAM processing first.")
            return
        
        # Results summary
        st.subheader("📈 Results Summary")
        
        stats = st.session_state.pipeline.get_statistics()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Frames Processed</h3>
                <h2>{stats['frames_processed']}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Processing Time</h3>
                <h2>{stats['avg_processing_time']:.3f}s</h2>
                <small>Average per frame</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Loop Closures</h3>
                <h2>{stats['loop_closures_detected']}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Trajectory Length</h3>
                <h2>{stats['trajectory_length']}</h2>
                <small>Pose estimates</small>
            </div>
            """, unsafe_allow_html=True)
        
        # Trajectory visualization
        st.subheader("🗺️ Trajectory Visualization")
        
        fig = create_3d_trajectory_plot(st.session_state.pipeline)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No trajectory data available for visualization")
        
        # Error analysis (if ground truth available)
        error_metrics = st.session_state.pipeline.compute_trajectory_error()
        if 'error' not in error_metrics:
            st.subheader("📏 Trajectory Error Analysis")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "ATE RMSE", 
                    f"{error_metrics['ate_rmse']:.4f}m",
                    help="Root Mean Square Error of Absolute Trajectory Error"
                )
            
            with col2:
                st.metric(
                    "ATE Mean", 
                    f"{error_metrics['ate_mean']:.4f}m",
                    help="Mean Absolute Trajectory Error"
                )
            
            with col3:
                st.metric(
                    "RPE RMSE", 
                    f"{error_metrics['rpe_rmse']:.4f}m",
                    help="Root Mean Square Error of Relative Pose Error"
                )
            
            # Error plot
            error_fig = create_error_plot(st.session_state.pipeline)
            if error_fig:
                st.plotly_chart(error_fig, use_container_width=True)
        
        # Export results
        st.subheader("💾 Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📄 Download Trajectory"):
                # Create trajectory file
                trajectory_data = []
                for pose in st.session_state.pipeline.estimated_trajectory:
                    pose_3x4 = pose[:3, :].flatten()
                    trajectory_data.append(' '.join(map(str, pose_3x4)))
                
                trajectory_text = '\n'.join(trajectory_data)
                
                st.download_button(
                    label="Download trajectory.txt",
                    data=trajectory_text,
                    file_name="trajectory.txt",
                    mime="text/plain"
                )
        
        with col2:
            if st.button("📊 Download Statistics"):
                # Create statistics JSON
                import json
                stats_json = json.dumps(stats, indent=2)
                
                st.download_button(
                    label="Download statistics.json",
                    data=stats_json,
                    file_name="vslam_statistics.json",
                    mime="application/json"
                )
    
    # Analysis tab
    with tab4:
        st.header("Performance Analysis")
        
        if not st.session_state.processing_results:
            st.info("ℹ️ No processing results available for analysis.")
            return
        
        # Performance plots
        perf_fig = create_performance_plots(st.session_state.pipeline)
        if perf_fig:
            st.plotly_chart(perf_fig, use_container_width=True)
        
        # Detailed statistics table
        st.subheader("📋 Detailed Statistics")
        
        stats = st.session_state.pipeline.get_statistics()
        
        # Convert stats to DataFrame for display
        stats_df = pd.DataFrame([
            {"Metric": "Frames Processed", "Value": stats['frames_processed']},
            {"Metric": "Total Processing Time", "Value": f"{stats['total_processing_time']:.3f}s"},
            {"Metric": "Average Processing Time", "Value": f"{stats['avg_processing_time']:.3f}s"},
            {"Metric": "Max Processing Time", "Value": f"{stats['max_processing_time']:.3f}s"},
            {"Metric": "Min Processing Time", "Value": f"{stats['min_processing_time']:.3f}s"},
            {"Metric": "Loop Closures Detected", "Value": stats['loop_closures_detected']},
            {"Metric": "Optimization Calls", "Value": stats['optimization_calls']},
            {"Metric": "Average Optimization Time", "Value": f"{stats['avg_optimization_time']:.3f}s"},
            {"Metric": "Feature Detector", "Value": stats['feature_detector']},
            {"Metric": "Feature Matcher", "Value": stats['feature_matcher']},
        ])
        
        st.dataframe(stats_df, use_container_width=True)
        
        # Loop closure analysis
        if st.session_state.pipeline.loop_closures:
            st.subheader("🔄 Loop Closure Analysis")
            
            loop_data = []
            for i, loop in enumerate(st.session_state.pipeline.loop_closures):
                loop_data.append({
                    "Loop ID": i + 1,
                    "Frame ID": loop.get('frame_id', 'N/A'),
                    "Matched Frame": loop.get('matched_frame', 'N/A'),
                    "Similarity Score": loop.get('similarity', 'N/A')
                })
            
            if loop_data:
                loop_df = pd.DataFrame(loop_data)
                st.dataframe(loop_df, use_container_width=True)
        
        # System information
        st.subheader("💻 System Information")
        
        system_info = {
            "OpenCV Version": cv2.__version__,
            "NumPy Version": np.__version__,
            "Matplotlib Version": plt.matplotlib.__version__,
        }
        
        for key, value in system_info.items():
            st.text(f"{key}: {value}")

if __name__ == "__main__":
    main()