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
import time

# Page configuration
st.set_page_config(
    page_title="VSLAM Simulator",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful styling
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #2E86AB;
        --secondary-color: #A23B72;
        --accent-color: #F18F01;
        --success-color: #C73E1D;
        --background-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Hide default Streamlit styling */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom header */
    .main-header {
        background: var(--background-gradient);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .main-title {
        font-size: 3.5rem;
        font-weight: 800;
        color: white;
        text-align: center;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-subtitle {
        font-size: 1.5rem;
        color: rgba(255,255,255,0.9);
        text-align: center;
        margin-top: 0.5rem;
        font-weight: 300;
    }
    
    /* Control panel styling */
    .control-panel {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid rgba(255,255,255,0.2);
        margin-bottom: 1rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Status cards */
    .status-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid var(--primary-color);
        margin: 1rem 0;
    }
    
    .status-card h3 {
        color: var(--primary-color);
        margin-top: 0;
    }
    
    /* Metrics styling */
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        margin: 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.8;
        margin: 0;
    }
    
    /* Progress styling */
    .stProgress .st-bo {
        background: linear-gradient(90deg, var(--primary-color), var(--accent-color));
        height: 8px;
        border-radius: 4px;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    /* Toggle styling */
    .stCheckbox > label {
        font-weight: 600;
        color: var(--primary-color);
    }
    
    /* Alert styling */
    .stAlert {
        border-radius: 10px;
        border: none;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    /* DataFrame styling */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    /* Animation for loading states */
    .loading-animation {
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    /* Custom spacing */
    .spacer {
        height: 2rem;
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: #333;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Configure logging
def setup_logging():
    """Setup logging configuration"""
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_filename = os.path.join(log_dir, f"visual_odometry.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
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
        st.error(f"❌ {message}")

def log_info(message, show_in_ui=False):
    """Log info message and optionally show in Streamlit UI"""
    logger.info(message)
    if show_in_ui:
        st.info(f"ℹ️ {message}")

def log_success(message, show_in_ui=False):
    """Log success message and optionally show in Streamlit UI"""
    logger.info(f"SUCCESS: {message}")
    if show_in_ui:
        st.success(f"✅ {message}")

def log_warning(message, show_in_ui=False):
    """Log warning message and optionally show in Streamlit UI"""
    logger.warning(message)
    if show_in_ui:
        st.warning(f"⚠️ {message}")

# Beautiful header
st.markdown("""
<div class="main-header">
    <h1 class="main-title">🎯 VSLAM Simulator</h1>
    <p class="main-subtitle">Visual Simultaneous Localization and Mapping with KITTI Dataset</p>
</div>
""", unsafe_allow_html=True)

# Log application start
logger.info("VSLAM Simulator application started")

# Session Management
if "trajectory" not in st.session_state:
    st.session_state.trajectory = []
if "trajectory_GT" not in st.session_state:
    st.session_state.trajectory_GT = []
if "poses_dataframe" not in st.session_state:
    st.session_state.poses_dataframe = pd.DataFrame()
if "processing_started" not in st.session_state:
    st.session_state.processing_started = False

# Create main containers
header_container = st.container()
control_container = st.container()
visualization_container = st.container()
metrics_container = st.container()
results_container = st.container()

# Initialize VO Pipeline
try:
    VO = Pipeline()
    bovw = BOVW()
    kitti = KITTIDataset()
    logger.info("VO Pipeline and KITTI Dataset initialized")
    
    with header_container:
        st.markdown('<div class="info-box">🚀 <strong>System Status:</strong> All modules loaded successfully! Ready for VSLAM simulation.</div>', unsafe_allow_html=True)
        
except Exception as e:
    with header_container:
        st.error(f"❌ Failed to initialize system: {str(e)}")
    st.stop()

# Define feature detector and matcher pairs to test
test_pairs = [
    {"featureDetector": ["FD_SIFT"], "featureMatcher": "FM_BF_NORM_L2"},
]

# Base directory for KITTI dataset
BASE_DIR = r"D:/coding/Temp_Download/data_odometry_color/dataset/sequences"
POSE_DIR = r"D:/coding/Temp_Download/data_odometry_poses/dataset/poses"

logger.info(f"Base directory set to: {BASE_DIR}")
logger.info(f"Pose directory set to: {POSE_DIR}")

# Enhanced Sidebar Controls
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem; background: rgba(255,255,255,0.1); border-radius: 10px; margin-bottom: 1rem;">
        <h2 style="color: white; margin: 0;">🎛️ Control Panel</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Processing mode selection
    st.markdown("### 🎯 Processing Mode")
    run_all = st.toggle("🔄 Run All Sequences (00-10)", value=False, help="Process all KITTI sequences automatically")
    run_single = st.toggle("🎯 Run Single Sequence", value=False, disabled=run_all, help="Process a specific sequence")
    
    # Sequence selection
    st.markdown("### 📁 Sequence Selection")
    sequence = st.selectbox(
        "Select KITTI Sequence", 
        [f"{i:02d}" for i in range(11)], 
        disabled=run_all,
        help="Choose which KITTI sequence to process"
    )
    
    # Processing parameters
    st.markdown("### ⚙️ Parameters")
    batch_size = st.slider(
        "Batch Size", 
        min_value=100, 
        max_value=5000, 
        value=500, 
        step=100, 
        disabled=run_all,
        help="Number of frames to process"
    )
    
    # Display options
    st.markdown("### 📊 Display Options")
    show_table = st.checkbox("📋 Show Poses Table", value=True, help="Display detailed pose comparison table")
    show_3d_trajectory = st.checkbox("🌐 Show 3D Trajectory", value=True, help="Display 3D trajectory visualization")
    show_histogram = st.checkbox("📈 Show BoVW Histogram", value=True, help="Display Bag of Visual Words histogram")
    
    # System info
    st.markdown("### 💻 System Info")
    st.info(f"🎯 Selected Sequence: **{sequence}**")
    st.info(f"📦 Batch Size: **{batch_size}**")
    st.info(f"🔧 Feature Detector: **SIFT**")

logger.info(f"Sidebar controls set - Run All: {run_all}, Run Single: {run_single}, Sequence: {sequence}, Batch Size: {batch_size}")

# Enhanced sequence info display
def show_seq_info(calib_file, pose_file):
    """Display sequence information with beautiful styling"""
    try:
        with open(calib_file, 'r') as f:
            K, P = kitti.load_calib(f)
            VO.set_k_intrinsic(K)
            
        log_success("Calibration file loaded successfully")
        
        # Create tabs for different information
        tab1, tab2, tab3 = st.tabs(["📷 Camera Calibration", "🎯 Ground Truth", "📊 Sequence Stats"])
        
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 🔍 Intrinsic Parameters (K)")
                st.dataframe(VO.get_k_intrinsic(), use_container_width=True)
            with col2:
                st.markdown("#### 🎥 Projection Matrix (P)")
                st.dataframe(P, use_container_width=True)

        with open(pose_file, 'r') as f:
            poses = kitti.load_poses(f, batch_size=batch_size)
            
        st.session_state.trajectory_GT = []
        for pose in poses:
            x, y, z = pose[0, 3], pose[1, 3], pose[2, 3]
            st.session_state.trajectory_GT.append([x, y, z])
            
        trajectory_GT = np.array(st.session_state.trajectory_GT)
        log_success("Ground truth poses loaded successfully")

        with tab2:
            # Enhanced 3D visualization
            fig = go.Figure()

            # Main trajectory line with gradient effect
            fig.add_trace(go.Scatter3d(
                x=trajectory_GT[:, 0],
                y=trajectory_GT[:, 2],
                z=trajectory_GT[:, 1],
                mode='lines+markers',
                line=dict(
                    color=np.arange(len(trajectory_GT)),
                    colorscale='Viridis',
                    width=6,
                    colorbar=dict(title="Time Progress")
                ),
                marker=dict(size=3, color='rgba(50,50,255,0.8)'),
                name='Ground Truth Trajectory',
                hovertemplate='<b>Position</b><br>X: %{x:.2f}m<br>Y: %{z:.2f}m<br>Z: %{y:.2f}m<extra></extra>'
            ))

            # Enhanced start point
            fig.add_trace(go.Scatter3d(
                x=[trajectory_GT[0, 0]],
                y=[trajectory_GT[0, 2]],
                z=[trajectory_GT[0, 1]],
                mode='markers+text',
                marker=dict(size=12, color='green', symbol='diamond'),
                name='🚀 Start Point',
                text=['START'],
                textposition='top center',
                textfont=dict(size=14, color='green')
            ))

            # Enhanced end point
            fig.add_trace(go.Scatter3d(
                x=[trajectory_GT[-1, 0]],
                y=[trajectory_GT[-1, 2]],
                z=[trajectory_GT[-1, 1]],
                mode='markers+text',
                marker=dict(size=12, color='red', symbol='diamond'),
                name='🏁 End Point',
                text=['END'],
                textposition='top center',
                textfont=dict(size=14, color='red')
            ))

            # Beautiful layout
            fig.update_layout(
                scene=dict(
                    xaxis_title='🡢 X Position (meters)',
                    yaxis_title='🡢 Z Position (meters)',
                    zaxis_title='🡡 Y Position (meters)',
                    xaxis=dict(
                        showgrid=True, 
                        gridcolor='rgba(255,255,255,0.3)', 
                        zeroline=False,
                        backgroundcolor='rgba(240,240,240,0.1)'
                    ),
                    yaxis=dict(
                        showgrid=True, 
                        gridcolor='rgba(255,255,255,0.3)', 
                        zeroline=False,
                        backgroundcolor='rgba(240,240,240,0.1)'
                    ),
                    zaxis=dict(
                        showgrid=True, 
                        gridcolor='rgba(255,255,255,0.3)', 
                        zeroline=False,
                        backgroundcolor='rgba(240,240,240,0.1)'
                    ),
                    aspectmode='data',
                    camera=dict(eye=dict(x=1.5, y=-2, z=1.5)),
                    bgcolor='rgba(0,0,0,0.05)'
                ),
                title=dict(
                    text=f'🌍 Ground Truth Trajectory - Sequence {sequence}',
                    font=dict(size=20, color='#2E86AB'),
                    x=0.5
                ),
                legend=dict(
                    x=0.02, y=0.98, 
                    bgcolor='rgba(255,255,255,0.9)', 
                    bordercolor='#2E86AB',
                    borderwidth=1
                ),
                margin=dict(l=0, r=0, b=0, t=60),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )

            st.plotly_chart(fig, use_container_width=True)
            
        with tab3:
            # Statistics
            col1, col2, col3, col4 = st.columns(4)
            total_distance = np.sum(np.linalg.norm(np.diff(trajectory_GT, axis=0), axis=1))
            
            with col1:
                st.markdown('<div class="metric-container"><p class="metric-value">{}</p><p class="metric-label">Total Frames</p></div>'.format(len(trajectory_GT)), unsafe_allow_html=True)
            with col2:
                st.markdown('<div class="metric-container"><p class="metric-value">{:.1f}m</p><p class="metric-label">Total Distance</p></div>'.format(total_distance), unsafe_allow_html=True)
            with col3:
                max_speed = np.max(np.linalg.norm(np.diff(trajectory_GT, axis=0), axis=1))
                st.markdown('<div class="metric-container"><p class="metric-value">{:.2f}m</p><p class="metric-label">Max Step Size</p></div>'.format(max_speed), unsafe_allow_html=True)
            with col4:
                st.markdown('<div class="metric-container"><p class="metric-value">{}</p><p class="metric-label">Sequence ID</p></div>'.format(sequence), unsafe_allow_html=True)
            
        return trajectory_GT
        
    except Exception as e:
        log_error(f"Error loading sequence info: {str(e)}", show_in_ui=True)
        return None

# Enhanced data loading
@st.cache_data
def load_images_from_directory(folder_path, batch_size):
    """Load and cache images from a local directory with progress indication"""
    logger.info(f"Loading images from directory: {folder_path} with batch size: {batch_size}")
    
    with st.spinner("🔍 Scanning image directory..."):
        images = []
        filenames = []
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
        files = os.listdir(folder_path)
        root = os.path.abspath(folder_path)
        batch_size = min(batch_size, len(files))
        
        progress_bar = st.progress(0)
        for i, file in enumerate(sorted(files)[:batch_size]):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                img_path = os.path.join(root, file)
                images.append(img_path)
                filenames.append(file)
            progress_bar.progress((i + 1) / batch_size)
        
        progress_bar.empty()
    
    logger.info(f"Loaded {len(images)} images from {folder_path}")
    return images, filenames, batch_size

# Enhanced poses dataframe creation
def create_poses_dataframe(trajectory, trajectory_GT):
    """Create the poses dataframe with enhanced error metrics"""
    if not trajectory or not trajectory_GT:
        logger.warning("Empty trajectory or ground truth data provided")
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
    
    # Enhanced error calculations
    data["Error_X"] = [abs(p - g) for p, g in zip(data["Predicted_X"], data["Ground_Truth_X"])]
    data["Error_Y"] = [abs(p - g) for p, g in zip(data["Predicted_Y"], data["Ground_Truth_Y"])]
    data["Error_Z"] = [abs(p - g) for p, g in zip(data["Predicted_Z"], data["Ground_Truth_Z"])]
    data["Total_Error"] = [np.sqrt(ex**2 + ey**2 + ez**2) for ex, ey, ez in zip(data["Error_X"], data["Error_Y"], data["Error_Z"])]
    
    df = pd.DataFrame(data)
    return df

# Enhanced CSV saving
def save_dataframe_to_csv(df, sequence, feature_detector, output_dir="results"):
    """Save the dataframe to CSV with timestamp"""
    if df.empty:
        logger.warning("Attempted to save empty dataframe to CSV")
        return
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"{output_dir}/{sequence}_image_2_{feature_detector}_{timestamp}.csv"
    df.to_csv(csv_filename, index=False)
    return csv_filename

# Enhanced CSV milestone check
def should_save_csv(current_frame, total_frames):
    """Check if current frame is at a milestone for CSV saving"""
    if total_frames <= 10:
        return current_frame == total_frames - 1
    
    percentile_step = total_frames // 10
    return (current_frame + 1) % percentile_step == 0 or current_frame == total_frames - 1

# Enhanced poses table display
def display_poses_table(poses_df, current_frame, totalFrames):
    """Create and display enhanced poses comparison table"""
    if poses_df.empty:
        return
    
    with st.expander("📊 Detailed Poses Analysis", expanded=False):
        st.markdown(f"### 📈 Frame {current_frame + 1} Analysis")
        
        # Summary metrics
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            st.metric("📊 Frames", len(poses_df), f"of {totalFrames}")
        with col2:
            st.metric("❌ Error X", f"{poses_df['Error_X'].mean():.3f}m")
        with col3:
            st.metric("❌ Error Y", f"{poses_df['Error_Y'].mean():.3f}m")
        with col4:
            st.metric("❌ Error Z", f"{poses_df['Error_Z'].mean():.3f}m")
        with col5:
            st.metric("🎯 Total Error", f"{poses_df['Total_Error'].mean():.3f}m")
        with col6:
            completion = (len(poses_df) / totalFrames) * 100
            st.metric("✅ Progress", f"{completion:.1f}%")
        
        # Data table
        st.dataframe(poses_df.tail(10), use_container_width=True)

# Enhanced sequence processing function
def process_sequence(sequence, show_table, batch_size, trajectory_GT):
    """Process a single sequence with enhanced UI and error handling"""
    start_time = datetime.now()
    logger.info(f"Starting processing for sequence {sequence}")
    
    sequence_dir = os.path.join(BASE_DIR, sequence, "image_2")
    logger.info(f"Sequence directory: {sequence_dir}")
    
    # Create processing status container
    status_container = st.container()
    
    try:
        with status_container:
            st.markdown("### 🔄 Loading Images...")
            images, filenames, batch_size = load_images_from_directory(sequence_dir, batch_size=batch_size)
            
            if not images:
                log_error(f"No valid images found in {sequence_dir}", show_in_ui=True)
                return None
            
            log_success(f"Loaded {len(filenames)} images from sequence {sequence}", show_in_ui=True)
            
            # Image preview
            col1, col2 = st.columns([2, 1])
            with col1:
                preview_img = cv2.cvtColor(cv2.imread(images[0]), cv2.COLOR_BGR2RGB)
                st.image(preview_img, caption=f"📷 Preview: {filenames[0]}", use_column_width=True)
            with col2:
                st.markdown("#### 📋 Dataset Info")
                st.info(f"🎯 **Sequence:** {sequence}")
                st.info(f"📦 **Total Images:** {len(images)}")
                st.info(f"🔧 **Feature Detector:** SIFT")
                st.info(f"📊 **Batch Size:** {batch_size}")
                
    except Exception as e:
        log_error(f"Error loading images: {str(e)}", show_in_ui=True)
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
            
            # Reset session state
            st.session_state.trajectory = []
            st.session_state.poses_dataframe = pd.DataFrame()
            VO.prev_keypoints = None
            VO.prev_descriptors = None
            VO.reset()
            bovw.reset()
            
            # Create processing UI containers
            processing_container = st.container()
            visualization_container = st.container()
            
            with processing_container:
                st.markdown("### 🚀 Processing Frames...")
                
                # Progress indicators
                overall_progress = st.progress(0)
                frame_progress = st.progress(0)
                status_text = st.empty()
                
                # Control buttons
                col1, col2, col3 = st.columns([1, 1, 2])
                with col1:
                    stop_button = st.button("⏹️ Stop", key=f"stop_{sequence}_{feature_detector}")
                with col2:
                    pause_button = st.button("⏸️ Pause", key=f"pause_{sequence}_{feature_detector}")
                
                csv_info_container = st.empty()
            
            total_images = len(images)
            
            for i, (img_frame, filename) in enumerate(zip(images, filenames)):
                # Check for stop signal
                if stop_button:
                    log_info(f"Processing stopped by user at frame {i}", show_in_ui=True)
                    break
                
                try:
                    logger.debug(f"Processing frame {i+1}/{total_images}: {filename}")
                    
                    # Process frame
                    result_frame, E, F, traj_path, detector_used, map_points, desc, kp = VO.VisualOdometry(
                        cv2.imread(img_frame),
                        FeatureDetector=feature_detector,
                        FeatureMatcher=feature_matcher
                    )
                    
                    # BoVW processing
                    hist = bovw.Histogram(desc)
                    bovw.historyOfBOVW(visual_word=hist, desc=desc, kp=kp)
                    
                    if traj_path:
                        st.session_state.trajectory = traj_path

                    # Update poses dataframe
                    st.session_state.poses_dataframe = create_poses_dataframe(
                        st.session_state.trajectory, 
                        st.session_state.trajectory_GT
                    )

                    # Save CSV at milestones
                    if should_save_csv(i, total_images):
                        csv_filename = save_dataframe_to_csv(
                            st.session_state.poses_dataframe, 
                            sequence, 
                            feature_detector
                        )
                        percentile = ((i + 1) / total_images) * 100
                        success_msg = f"💾 CSV saved at {percentile:.0f}% completion"
                        log_success(success_msg)
                        csv_info_container.success(success_msg)

                    # Update progress indicators
                    frame_progress_val = (i + 1) / total_images
                    overall_progress_val = (current - 1 + frame_progress_val) / numberOfDetectors
                    
                    frame_progress.progress(frame_progress_val)
                    overall_progress.progress(overall_progress_val)
                    status_text.markdown(f"🎯 **Processing:** `{filename}` | **Frame:** {i+1}/{total_images} | **Detector:** {feature_detector}")

                    # Display current processing frame
                    with visualization_container:
                        # Create tabs for different visualizations
                        tab1, tab2, tab3, tab4 = st.tabs(["🖼️ Current Frame", "🌍 3D Trajectory", "📊 BoVW Analysis", "🔍 Loop Closure"])
                        
                        with tab1:
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.image(result_frame, channels="BGR", caption=f"🎯 Frame {i+1}: {filename}", use_column_width=True)
                            
                            with col2:
                                st.markdown("#### 📋 Frame Info")
                                st.markdown(f"**📁 File:** `{filename}`")
                                st.markdown(f"**🎯 Progress:** {i+1}/{total_images}")
                                st.markdown(f"**🔧 Driver:** `{detector_used}`")
                                st.markdown(f"**📊 Trajectory Points:** {len(st.session_state.trajectory) if st.session_state.trajectory else 0}")
                                
                                # Matrix information
                                if E is not None:
                                    with st.expander("🎯 Essential Matrix", expanded=False):
                                        st.dataframe(E, use_container_width=True)
                                if F is not None:
                                    with st.expander("🔍 Fundamental Matrix", expanded=False):
                                        st.dataframe(F, use_container_width=True)
                        
                        with tab2:
                            if show_3d_trajectory and st.session_state.trajectory and len(st.session_state.trajectory) > 1:
                                traj_np = np.array(st.session_state.trajectory)
                                
                                # Enhanced 3D trajectory visualization
                                fig = go.Figure()
                                
                                # Estimated trajectory
                                fig.add_trace(go.Scatter3d(
                                    x=traj_np[:, 0],
                                    y=traj_np[:, 2],
                                    z=traj_np[:, 1],
                                    mode='lines+markers',
                                    marker=dict(
                                        size=4, 
                                        color=np.arange(len(traj_np)),
                                        colorscale='Reds',
                                        showscale=True,
                                        colorbar=dict(title="Frame", x=1.1)
                                    ),
                                    line=dict(width=4, color='red'),
                                    name='🎯 Estimated Trajectory',
                                    hovertemplate='<b>Estimated</b><br>Frame: %{marker.color}<br>X: %{x:.2f}m<br>Y: %{z:.2f}m<br>Z: %{y:.2f}m<extra></extra>'
                                ))
                                
                                # Ground truth trajectory (up to current frame)
                                current_gt = trajectory_GT[:len(traj_np)]
                                fig.add_trace(go.Scatter3d(
                                    x=current_gt[:, 0],
                                    y=current_gt[:, 2],
                                    z=current_gt[:, 1],
                                    mode='lines+markers',
                                    line=dict(width=3, color='green'),
                                    marker=dict(size=3, color='green'),
                                    opacity=0.7,
                                    name='✅ Ground Truth',
                                    hovertemplate='<b>Ground Truth</b><br>X: %{x:.2f}m<br>Y: %{z:.2f}m<br>Z: %{y:.2f}m<extra></extra>'
                                ))
                                
                                # Current position marker
                                fig.add_trace(go.Scatter3d(
                                    x=[traj_np[-1, 0]],
                                    y=[traj_np[-1, 2]],
                                    z=[traj_np[-1, 1]],
                                    mode='markers',
                                    marker=dict(size=12, color='orange', symbol='diamond'),
                                    name='📍 Current Position'
                                ))
                                
                                # Map points visualization (if available)
                                if map_points is not None and len(map_points) > 0:
                                    map_points_np = np.array(map_points)
                                    if map_points_np.shape[1] >= 3:
                                        fig.add_trace(go.Scatter3d(
                                            x=map_points_np[:, 0],
                                            y=map_points_np[:, 2],
                                            z=map_points_np[:, 1],
                                            mode='markers',
                                            marker=dict(size=1, color='lightblue', symbol='cross'),
                                            name='🗺️ Map Points',
                                            opacity=0.6
                                        ))
                                
                                # Enhanced layout
                                fig.update_layout(
                                    scene=dict(
                                        xaxis_title='🡢 X (meters)',
                                        yaxis_title='🡢 Z (meters)',
                                        zaxis_title='🡡 Y (meters)',
                                        xaxis=dict(showgrid=True, gridcolor='rgba(200,200,200,0.3)'),
                                        yaxis=dict(showgrid=True, gridcolor='rgba(200,200,200,0.3)'),
                                        zaxis=dict(showgrid=True, gridcolor='rgba(200,200,200,0.3)'),
                                        aspectmode='data',
                                        camera=dict(eye=dict(x=1.2, y=-2, z=1.5)),
                                        bgcolor='rgba(240,245,255,0.1)'
                                    ),
                                    title=dict(
                                        text=f'🌍 Real-time VSLAM Trajectory - Frame {i+1}',
                                        font=dict(size=18, color='#2E86AB'),
                                        x=0.5
                                    ),
                                    legend=dict(
                                        x=0.02, y=0.98,
                                        bgcolor='rgba(255,255,255,0.9)',
                                        bordercolor='#2E86AB'
                                    ),
                                    margin=dict(l=0, r=0, b=0, t=50),
                                    height=600
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Trajectory statistics
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("📊 Trajectory Points", len(traj_np))
                                with col2:
                                    if len(traj_np) > 1:
                                        distance = np.sum(np.linalg.norm(np.diff(traj_np, axis=0), axis=1))
                                        st.metric("📏 Distance Traveled", f"{distance:.2f}m")
                                with col3:
                                    if map_points is not None:
                                        st.metric("🗺️ Map Points", f"{len(map_points)}" if map_points else "0")
                                with col4:
                                    if not st.session_state.poses_dataframe.empty:
                                        avg_error = st.session_state.poses_dataframe["Total_Error"].mean()
                                        st.metric("❌ Avg Error", f"{avg_error:.3f}m")
                            else:
                                st.info("🔄 3D trajectory will appear after processing multiple frames...")
                        
                        with tab3:
                            if show_histogram and hist is not None:
                                # Enhanced BoVW histogram
                                histogram_fig = go.Figure()
                                histogram_fig.add_trace(go.Bar(
                                    x=list(range(len(hist))),
                                    y=hist,
                                    marker=dict(
                                        color=hist,
                                        colorscale='Viridis',
                                        showscale=True,
                                        colorbar=dict(title="Frequency")
                                    ),
                                    hovertemplate='<b>Visual Word %{x}</b><br>Frequency: %{y}<extra></extra>'
                                ))
                                
                                histogram_fig.update_layout(
                                    title=dict(
                                        text=f"📊 Bag of Visual Words - Frame {i+1}",
                                        font=dict(size=16, color='#2E86AB'),
                                        x=0.5
                                    ),
                                    xaxis_title="Visual Word Index",
                                    yaxis_title="Frequency",
                                    bargap=0.1,
                                    height=400,
                                    margin=dict(l=0, r=0, b=0, t=50)
                                )
                                
                                st.plotly_chart(histogram_fig, use_container_width=True)
                                
                                # BoVW statistics
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("📊 Visual Words", len(hist))
                                with col2:
                                    st.metric("🎯 Active Words", np.count_nonzero(hist))
                                with col3:
                                    st.metric("📈 Max Frequency", np.max(hist) if len(hist) > 0 else 0)
                                with col4:
                                    st.metric("📊 Total Features", np.sum(hist) if len(hist) > 0 else 0)
                            else:
                                st.info("📊 BoVW histogram will appear after feature extraction...")
                        
                        with tab4:
                            # Loop closure detection
                            loop_closure = bovw.LoopChecks()
                            if loop_closure is not None and len(loop_closure) > 0:
                                st.markdown("#### 🔄 Loop Closure Detection")
                                
                                # Enhanced loop closure table
                                loop_df = pd.DataFrame(loop_closure)
                                if not loop_df.empty:
                                    st.dataframe(loop_df, use_container_width=True)
                                    
                                    # Loop closure statistics
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("🔄 Total Checks", len(loop_df))
                                    with col2:
                                        if 'similarity' in loop_df.columns:
                                            max_sim = loop_df['similarity'].max() if not loop_df.empty else 0
                                            st.metric("🎯 Max Similarity", f"{max_sim:.3f}")
                                    with col3:
                                        if 'loop_detected' in loop_df.columns:
                                            loops = loop_df['loop_detected'].sum() if not loop_df.empty else 0
                                            st.metric("✅ Loops Detected", loops)
                                else:
                                    st.info("🔄 No loop closures detected yet...")
                            else:
                                st.info("🔄 Loop closure detection will start after processing multiple frames...")

                    # Display poses table if enabled
                    if show_table and not st.session_state.poses_dataframe.empty:
                        display_poses_table(
                            st.session_state.poses_dataframe, 
                            i, 
                            len(trajectory_GT)
                        )

                except Exception as e:
                    error_msg = f"Error processing {filename} with {feature_detector}: {str(e)}"
                    log_error(error_msg, show_in_ui=True)
                    continue
            
            # Final processing steps
            if not st.session_state.poses_dataframe.empty:
                final_csv = save_dataframe_to_csv(
                    st.session_state.poses_dataframe, 
                    sequence, 
                    feature_detector
                )
                
                # Calculate final metrics
                processing_time = (datetime.now() - start_time).total_seconds()
                mse_results.append({
                    "Sequence": sequence,
                    "Feature_Detector": feature_detector,
                    "Processing_Time": processing_time,
                    "Frames_Processed": len(st.session_state.poses_dataframe),
                    "MSE_X": st.session_state.poses_dataframe["Error_X"].mean(),
                    "MSE_Y": st.session_state.poses_dataframe["Error_Y"].mean(),
                    "MSE_Z": st.session_state.poses_dataframe["Error_Z"].mean(),
                    "Total_MSE": st.session_state.poses_dataframe["Total_Error"].mean(),
                    "Final_CSV": final_csv
                })
                
                success_msg = f"✅ Completed sequence {sequence} with {feature_detector}! Processing time: {processing_time:.2f}s"
                log_success(success_msg, show_in_ui=True)

            # Clean up UI elements
            overall_progress.empty()
            frame_progress.empty()
            status_text.empty()
            csv_info_container.empty()
    
    logger.info(f"Completed processing sequence {sequence}")
    return mse_results

# ---------------------
# MAIN APPLICATION LOGIC
# ---------------------

# Processing section
with control_container:
    st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)
    
    if run_all:
        st.markdown("### 🚀 Processing All Sequences (00-10)")
        logger.info("Starting processing for all sequences (00-10)")
        
        all_mse_results = []
        sequence_progress = st.progress(0)
        
        for seq_idx, seq in enumerate([f"{i:02d}" for i in range(11)]):
            st.markdown(f"#### 🎯 Processing Sequence {seq}")
            
            try:
                trajectory_GT = show_seq_info(
                    calib_file=os.path.join(BASE_DIR, seq, "calib2.txt"), 
                    pose_file=os.path.join(POSE_DIR, f"{seq}.txt")
                )
                
                if trajectory_GT is not None:
                    mse_results = process_sequence(seq, show_table, batch_size=batch_size, trajectory_GT=trajectory_GT)
                    if mse_results:
                        all_mse_results.extend(mse_results)
                
                sequence_progress.progress((seq_idx + 1) / 11)
                
            except Exception as e:
                log_error(f"Error processing sequence {seq}: {str(e)}", show_in_ui=True)
                continue
        
        # Display final results
        if all_mse_results:
            with results_container:
                st.markdown("### 📊 Final Results - All Sequences")
                
                mse_df = pd.DataFrame(all_mse_results)
                
                # Summary statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("✅ Sequences Processed", mse_df['Sequence'].nunique())
                with col2:
                    avg_time = mse_df['Processing_Time'].mean()
                    st.metric("⏱️ Avg Processing Time", f"{avg_time:.1f}s")
                with col3:
                    total_frames = mse_df['Frames_Processed'].sum()
                    st.metric("🎯 Total Frames", total_frames)
                with col4:
                    avg_error = mse_df['Total_MSE'].mean()
                    st.metric("❌ Overall Avg Error", f"{avg_error:.3f}m")
                
                # Detailed results table
                st.dataframe(mse_df, use_container_width=True)
                
                # Save results
                results_csv = "results/all_sequences_mse_results.csv"
                mse_df.to_csv(results_csv, index=False)
                
                # Download button
                csv_data = mse_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Complete Results",
                    data=csv_data,
                    file_name=f"vslam_results_all_sequences_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        sequence_progress.empty()
        success_msg = "🎉 Completed processing all sequences (00-10)!"
        log_success(success_msg, show_in_ui=True)

    elif run_single:
        st.markdown(f"### 🎯 Processing Single Sequence: {sequence}")
        logger.info(f"Starting single sequence processing for sequence {sequence}")
        
        try:
            trajectory_GT = show_seq_info(
                calib_file=os.path.join(BASE_DIR, sequence, "calib2.txt"), 
                pose_file=os.path.join(POSE_DIR, f"{sequence}.txt")
            )
            
            if trajectory_GT is not None:
                mse_results = process_sequence(sequence, show_table, batch_size=batch_size, trajectory_GT=trajectory_GT)
                
                if mse_results:
                    with results_container:
                        st.markdown("### 📊 Processing Results")
                        
                        mse_df = pd.DataFrame(mse_results)
                        
                        # Results summary
                        col1, col2, col3, col4 = st.columns(4)
                        result = mse_results[0]  # Single sequence result
                        
                        with col1:
                            st.metric("⏱️ Processing Time", f"{result['Processing_Time']:.2f}s")
                        with col2:
                            st.metric("🎯 Frames Processed", result['Frames_Processed'])
                        with col3:
                            st.metric("❌ Average Error", f"{result['Total_MSE']:.3f}m")
                        with col4:
                            fps = result['Frames_Processed'] / result['Processing_Time']
                            st.metric("📊 Processing Rate", f"{fps:.1f} FPS")
                        
                        # Detailed results
                        st.dataframe(mse_df, use_container_width=True)
                        
                        # Save and download options
                        results_csv = f"results/sequence_{sequence}_mse_results.csv"
                        mse_df.to_csv(results_csv, index=False)
                        
                        csv_data = mse_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results",
                            data=csv_data,
                            file_name=f"vslam_results_seq{sequence}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                
                if st.session_state.trajectory:
                    success_msg = f"🎉 Sequence {sequence} completed! Generated {len(st.session_state.trajectory)} trajectory points."
                    log_success(success_msg, show_in_ui=True)
            
        except Exception as e:
            log_error(f"Error processing sequence {sequence}: {str(e)}", show_in_ui=True)

    else:
        # Default state - show sequence info
        st.markdown("### 🎮 Ready to Start VSLAM Simulation")
        st.markdown('<div class="info-box">👆 <strong>Getting Started:</strong> Toggle "Run All Sequences" or "Run Single Sequence" in the sidebar to begin VSLAM processing.</div>', unsafe_allow_html=True)
        
        try:
            trajectory_GT = show_seq_info(
                calib_file=os.path.join(BASE_DIR, sequence, "calib2.txt"), 
                pose_file=os.path.join(POSE_DIR, f"{sequence}.txt")
            )
        except Exception as e:
            log_error(f"Error loading sequence info: {str(e)}", show_in_ui=True)

# Final results section
with st.expander("📁 Complete Trajectory Data Export", expanded=False):
    if not st.session_state.poses_dataframe.empty:
        st.markdown("### 💾 Export Complete Dataset")
        st.markdown("This section contains the complete trajectory data in CSV format, ready for analysis or further processing.")
        
        # Data preview
        st.dataframe(st.session_state.poses_dataframe.tail(10), use_container_width=True)
        
        # Export options
        col1, col2 = st.columns(2)
        with col1:
            csv_data = st.session_state.poses_dataframe.to_csv(index=False)
            st.download_button(
                label="📥 Download Complete CSV",
                data=csv_data,
                file_name=f"complete_trajectory_seq{sequence}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            json_data = st.session_state.poses_dataframe.to_json(orient='records', indent=2)
            st.download_button(
                label="📥 Download as JSON",
                data=json_data,
                file_name=f"complete_trajectory_seq{sequence}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    else:
        st.markdown('<div class="info-box">ℹ️ <strong>No Data Available:</strong> Run the VSLAM pipeline to generate trajectory data for export.</div>', unsafe_allow_html=True)

# Footer with system info
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**🎯 VSLAM Simulator v2.0**")
with col2:
    st.markdown(f"**📅 Session:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
with col3:
    st.markdown("**🔧 Powered by:** OpenCV + KITTI Dataset")

logger.info("VSLAM Simulator application session ended")