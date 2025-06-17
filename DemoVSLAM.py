import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np

# Configure page
st.set_page_config(
    page_title="VSLAM Concepts Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for dark theme
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0c0c0c 0%, #1a1a2e 50%, #16213e 100%);
    }
    
    .pipeline-container {
        background: rgba(30, 30, 60, 0.3);
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        border: 1px solid rgba(100, 100, 255, 0.2);
        backdrop-filter: blur(10px);
    }
    
    .step-card {
        background: linear-gradient(145deg, #1e1e3f, #2a2a5a);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(100, 150, 255, 0.3);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .algorithm-card {
        background: rgba(40, 40, 80, 0.6);
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #4a9eff;
        transition: all 0.3s ease;
    }
    
    .algorithm-card:hover {
        background: rgba(50, 50, 100, 0.8);
        border-left: 4px solid #66b3ff;
    }
    
    .preferred-algo {
        background: linear-gradient(135deg, #4a9eff, #0066cc);
        color: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        text-align: center;
        box-shadow: 0 8px 32px rgba(74, 158, 255, 0.3);
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from { box-shadow: 0 8px 32px rgba(74, 158, 255, 0.3); }
        to { box-shadow: 0 8px 32px rgba(74, 158, 255, 0.6); }
    }
    
    .metric-card {
        background: rgba(20, 20, 40, 0.8);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border: 1px solid rgba(100, 100, 255, 0.2);
    }
    
    h1, h2, h3 {
        color: #ffffff;
        text-shadow: 0 0 10px rgba(74, 158, 255, 0.3);
    }
    
    .stMarkdown {
        color: #e0e0ff;
    }
    
    .math-formula {
        background: rgba(0, 0, 0, 0.3);
        border-radius: 5px;
        padding: 0.5rem;
        border-left: 3px solid #4a9eff;
        font-family: 'Courier New', monospace;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div style="text-align: center; padding: 2rem 0;">
    <h1 style="font-size: 3rem; margin-bottom: 0;">🎯 VSLAM Concepts Dashboard</h1>
    <p style="font-size: 1.2rem; color: #a0a0ff; margin-top: 0;">
        Visual Simultaneous Localization and Mapping - Comprehensive Algorithm Guide
    </p>
</div>
""", unsafe_allow_html=True)

# VSLAM Pipeline Overview
st.markdown("""
<div class="pipeline-container">
    <h2 style="text-align: center; color: #4a9eff;">📊 VSLAM Pipeline Overview</h2>
    <p style="text-align: center; font-size: 1.1rem; color: #b0b0ff;">
        A comprehensive step-by-step approach to Visual SLAM implementation
    </p>
</div>
""", unsafe_allow_html=True)

# Create enhanced pipeline flow diagram
fig = go.Figure()

steps = [
    ("📷", "Image\nAcquisition"), ("🎯", "Feature\nDetection"), ("🔗", "Feature\nMatching"), 
    ("📐", "Motion\nEstimation"), ("🗺️", "Map\nInitialization"), ("👁️", "Tracking"), 
    ("🏗️", "Local\nMapping"), ("🔄", "Loop\nDetection"), ("⚖️", "Bundle\nAdjustment")
]

# Create two rows for better layout
x_positions_top = [0, 1, 2, 3, 4]
y_positions_top = [1, 1, 1, 1, 1]
x_positions_bottom = [0, 1, 2, 3]
y_positions_bottom = [0, 0, 0, 0]

# Colors for different stages
colors_top = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57']
colors_bottom = ['#ff9ff3', '#54a0ff', '#5f27cd', '#00d2d3']

# Top row nodes
for i, (icon, step) in enumerate(steps[:5]):
    fig.add_trace(go.Scatter(
        x=[x_positions_top[i]], y=[y_positions_top[i]],
        mode='markers',
        marker=dict(size=80, color=colors_top[i], line=dict(width=4, color='white')),
        showlegend=False,
        hovertemplate=f"<b>{step.replace('<br>', ' ')}</b><br>Click to learn more<extra></extra>"
    ))
    
    # Add text with dark background
    fig.add_annotation(
        x=x_positions_top[i], y=y_positions_top[i],
        text=f"{icon}<br>{step}",
        showarrow=False,
        font=dict(size=11, color='white', family="Arial Black"),
        bgcolor="rgba(0, 0, 0, 0.8)",
        bordercolor="rgba(255, 255, 255, 0.3)",
        borderwidth=1,
        borderpad=8
    )

# Bottom row nodes
for i, (icon, step) in enumerate(steps[5:]):
    fig.add_trace(go.Scatter(
        x=[x_positions_bottom[i]], y=[y_positions_bottom[i]],
        mode='markers',
        marker=dict(size=80, color=colors_bottom[i], line=dict(width=4, color='white')),
        showlegend=False,
        hovertemplate=f"<b>{step.replace('<br>', ' ')}</b><br>Click to learn more<extra></extra>"
    ))
    
    # Add text with dark background
    fig.add_annotation(
        x=x_positions_bottom[i], y=y_positions_bottom[i],
        text=f"{icon}<br>{step}",
        showarrow=False,
        font=dict(size=11, color='white', family="Arial Black"),
        bgcolor="rgba(0, 0, 0, 0.8)",
        bordercolor="rgba(255, 255, 255, 0.3)",
        borderwidth=1,
        borderpad=8
    )

# Add curved arrows for top row
for i in range(4):
    fig.add_annotation(
        x=i+0.35, y=1,
        ax=i+0.65, ay=1,
        xref='x', yref='y',
        axref='x', ayref='y',
        arrowhead=3,
        arrowsize=1.5,
        arrowwidth=3,
        arrowcolor='#ffffff',
        opacity=0.8
    )

# Add curved arrows for bottom row
for i in range(3):
    fig.add_annotation(
        x=i+0.35, y=0,
        ax=i+0.65, ay=0,
        xref='x', yref='y',
        axref='x', ayref='y',
        arrowhead=3,
        arrowsize=1.5,
        arrowwidth=3,
        arrowcolor='#ffffff',
        opacity=0.8
    )

# Connect top row to bottom row
fig.add_annotation(
    x=4, y=0.85,
    ax=0, ay=0.15,
    xref='x', yref='y',
    axref='x', ayref='y',
    arrowhead=3,
    arrowsize=1.5,
    arrowwidth=3,
    arrowcolor='#ffd700',
    opacity=0.9
)

# Add background shapes for visual appeal
fig.add_shape(
    type="rect",
    x0=-0.5, y0=0.7, x1=4.5, y1=1.3,
    fillcolor="rgba(74, 158, 255, 0.1)",
    line=dict(color="rgba(74, 158, 255, 0.3)", width=2),
    layer="below"
)

fig.add_shape(
    type="rect",
    x0=-0.5, y0=-0.3, x1=3.5, y1=0.3,
    fillcolor="rgba(255, 107, 107, 0.1)",
    line=dict(color="rgba(255, 107, 107, 0.3)", width=2),
    layer="below"
)

# Add stage labels
fig.add_annotation(
    x=2, y=1.45, text="<b>Frontend Processing</b>",
    showarrow=False, font=dict(size=14, color='#4a9eff'),
    bgcolor="rgba(74, 158, 255, 0.2)", bordercolor="#4a9eff"
)

fig.add_annotation(
    x=1.5, y=-0.45, text="<b>Backend Optimization</b>",
    showarrow=False, font=dict(size=14, color='#ff6b6b'),
    bgcolor="rgba(255, 107, 107, 0.2)", bordercolor="#ff6b6b"
)

fig.update_layout(
    title=dict(
        text="<b>🎯 VSLAM Processing Pipeline</b>",
        x=0.5,
        font=dict(size=20, color='white')
    ),
    showlegend=False,
    xaxis=dict(
        showgrid=False, 
        zeroline=False, 
        showticklabels=False,
        range=[-0.7, 4.7]
    ),
    yaxis=dict(
        showgrid=False, 
        zeroline=False, 
        showticklabels=False,
        range=[-0.7, 1.7]
    ),
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    height=400,
    margin=dict(l=20, r=20, t=60, b=20)
)

st.plotly_chart(fig, use_container_width=True)

# Define VSLAM steps and algorithms
vslam_steps = {
    "1. Image Acquisition": {
        "description": "Capturing visual data from camera sensors",
        "algorithms": {
            "Monocular Camera": {
                "working": "Uses single camera to capture sequential images",
                "description": "Most common setup, provides scale-ambiguous reconstruction",
                "math": "I(x,y,t) → Image at pixel (x,y) at time t"
            },
            "Stereo Camera": {
                "working": "Uses two cameras to capture simultaneous images with depth information",
                "description": "Provides direct depth estimation through disparity calculation",
                "math": "Depth = (f × B) / d, where f=focal length, B=baseline, d=disparity"
            },
            "RGB-D Camera": {
                "working": "Combines RGB image with depth sensor measurements",
                "description": "Direct 3D point cloud generation with color information",
                "math": "P(x,y,z) = (u-cx)/fx × d, (v-cy)/fy × d, d"
            }
        }
    },
    "2. Feature Detection": {
        "description": "Identifying distinctive points in images for tracking",
        "algorithms": {
            "SIFT (Scale-Invariant Feature Transform)": {
                "working": "Detects keypoints using Difference of Gaussians (DoG) at multiple scales",
                "description": "Robust to scale, rotation, and illumination changes but computationally expensive",
                "math": "DoG(x,y,σ) = G(x,y,kσ) - G(x,y,σ), where G is Gaussian kernel"
            },
            "SURF (Speeded Up Robust Features)": {
                "working": "Uses Hessian matrix and integral images for fast detection",
                "description": "Faster than SIFT while maintaining good performance",
                "math": "H(x,σ) = [Lxx(x,σ) Lxy(x,σ); Lxy(x,σ) Lyy(x,σ)]"
            },
            "ORB (Oriented FAST and Rotated BRIEF)": {
                "working": "Combines FAST keypoint detector with BRIEF descriptor",
                "description": "Very fast, free alternative to SIFT/SURF, used in real-time applications",
                "math": "Corner Response: R = Ixx·Iyy - (Ixy)²"
            },
            "FAST (Features from Accelerated Segment Test)": {
                "working": "Compares pixel intensities in a circle around candidate point",
                "description": "Extremely fast corner detection, foundation for ORB",
                "math": "Corner if |I(p) - I(pi)| > t for n contiguous pixels in circle"
            }
        }
    },
    "3. Feature Matching": {
        "description": "Establishing correspondences between features across images",
        "algorithms": {
            "Brute Force Matcher": {
                "working": "Computes distance between all descriptor pairs",
                "description": "Simple but computationally expensive O(n²) approach",
                "math": "Distance = ||d1 - d2||, typically L2 or Hamming distance"
            },
            "FLANN (Fast Library for Approximate Nearest Neighbors)": {
                "working": "Uses KD-trees or LSH for approximate nearest neighbor search",
                "description": "Much faster than brute force for large feature sets",
                "math": "Approximate NN search with error bound ε"
            },
            "Ratio Test": {
                "working": "Compares distance to closest vs second closest match",
                "description": "Filters out ambiguous matches, proposed by Lowe",
                "math": "Accept match if d1/d2 < threshold (typically 0.7-0.8)"
            }
        }
    },
    "4. Motion Estimation": {
        "description": "Estimating camera motion between consecutive frames",
        "algorithms": {
            "5-Point Algorithm": {
                "working": "Estimates essential matrix from 5 point correspondences",
                "description": "Minimal solver for calibrated cameras, used with RANSAC",
                "math": "x2ᵀEx1 = 0, where E is essential matrix"
            },
            "8-Point Algorithm": {
                "working": "Estimates fundamental matrix from 8+ point correspondences",
                "description": "Works with uncalibrated cameras, more stable with noise",
                "math": "x2ᵀFx1 = 0, where F is fundamental matrix"
            },
            "PnP (Perspective-n-Point)": {
                "working": "Estimates camera pose from n 3D-2D point correspondences",
                "description": "Used when 3D map points are available",
                "math": "Minimize ||x - π(K[R|t]X)||² over R,t"
            }
        }
    },
    "5. Map Initialization": {
        "description": "Creating initial 3D map from first few frames",
        "algorithms": {
            "Two-View Reconstruction": {
                "working": "Triangulates 3D points from two calibrated views",
                "description": "Standard approach for stereo or sequential monocular initialization",
                "math": "X = triangulate(x1, x2, P1, P2) using DLT or optimal methods"
            },
            "Homography vs Fundamental": {
                "working": "Chooses between planar (H) and general (F) scene reconstruction",
                "description": "Automatic selection based on scene geometry",
                "math": "Score_H vs Score_F based on symmetric transfer error"
            },
            "Bundle Adjustment": {
                "working": "Jointly optimizes camera poses and 3D points",
                "description": "Refines initial reconstruction for accuracy",
                "math": "min Σ ||x_ij - π(K_i[R_i|t_i]X_j)||²"
            }
        }
    },
    "6. Tracking": {
        "description": "Continuously estimating camera pose for new frames",
        "algorithms": {
            "KLT Tracker": {
                "working": "Tracks features using optical flow assumptions",
                "description": "Fast tracking assuming small motion and brightness constancy",
                "math": "I(x,y,t) = I(x+dx,y+dy,t+dt) → solve for (dx,dy)"
            },
            "Feature-based Tracking": {
                "working": "Matches detected features with existing map points",
                "description": "More robust to large motions and appearance changes",
                "math": "Find matches M = {(f_i, m_j)} then solve PnP"
            },
            "Direct Methods": {
                "working": "Minimizes photometric error directly on pixel intensities",
                "description": "Uses all pixel information, works in low-texture environments",
                "math": "min Σ (I1(x) - I2(w(x,ξ)))² over pose ξ"
            }
        }
    },
    "7. Local Mapping": {
        "description": "Creating and refining local 3D map around current position",
        "algorithms": {
            "Triangulation": {
                "working": "Creates new map points from feature matches across multiple views",
                "description": "Expands the map by adding well-observed 3D points",
                "math": "X = arg min Σ ||x_i - π(P_i X)||² subject to positive depth"
            },
            "Map Point Culling": {
                "working": "Removes unreliable map points based on observation statistics",
                "description": "Maintains map quality by filtering out bad points",
                "math": "Remove if observation_ratio < threshold or reprojection_error > max_error"
            },
            "Local Bundle Adjustment": {
                "working": "Optimizes poses and points in local sliding window",
                "description": "Maintains local accuracy while keeping computation bounded",
                "math": "min Σ_local ||x_ij - π(K_i[R_i|t_i]X_j)||²"
            }
        }
    },
    "8. Loop Detection": {
        "description": "Recognizing previously visited locations to correct drift",
        "algorithms": {
            "Bag of Words (BoW)": {
                "working": "Represents images as histograms of visual words from vocabulary",
                "description": "Fast place recognition using learned visual vocabulary",
                "math": "Similarity = Σ min(w1_i, w2_i) for word histograms w1, w2"
            },
            "DBoW2/DBoW3": {
                "working": "Hierarchical vocabulary tree for efficient image retrieval",
                "description": "State-of-the-art BoW implementation with scoring and temporal consistency",
                "math": "Score based on TF-IDF weighting and temporal constraints"
            },
            "NetVLAD": {
                "working": "Deep learning approach for place recognition",
                "description": "CNN-based global descriptor for robust place recognition",
                "math": "VLAD pooling: V(k) = Σ α_k(x_i)(x_i - c_k)"
            }
        }
    },
    "9. Bundle Adjustment": {
        "description": "Global optimization of all cameras and map points",
        "algorithms": {
            "Levenberg-Marquardt": {
                "working": "Iterative optimization combining Gauss-Newton and gradient descent",
                "description": "Standard solver for non-linear least squares in bundle adjustment",
                "math": "(J^T J + λI)δ = -J^T r, where λ controls damping"
            },
            "Sparse Bundle Adjustment": {
                "working": "Exploits sparsity structure of bundle adjustment problem",
                "description": "Efficient solving for large-scale problems using sparse matrices",
                "math": "Schur complement: S = J_c^T J_c - J_c^T J_p (J_p^T J_p)^{-1} J_p^T J_c"
            },
            "Pose Graph Optimization": {
                "working": "Optimizes only camera poses using relative motion constraints",
                "description": "Faster alternative focusing on trajectory rather than structure",
                "math": "min Σ ||log(T_ij^{-1} T_i T_j^{-1})||²_Σ over poses T_i"
            }
        }
    }
}

# Create columns for each step
for step_name, step_data in vslam_steps.items():
    st.markdown(f"""
    <div class="step-card">
        <h3 style="color: #4a9eff; margin-bottom: 0.5rem;">{step_name}</h3>
        <p style="color: #b0b0ff; font-style: italic; margin-bottom: 1rem;">{step_data['description']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create expandable sections for algorithms
    cols = st.columns(min(len(step_data['algorithms']), 3))
    
    for idx, (algo_name, algo_data) in enumerate(step_data['algorithms'].items()):
        with cols[idx % 3]:
            with st.expander(f"📋 {algo_name}", expanded=False):
                st.markdown(f"**🔧 How it works:**")
                st.write(algo_data['working'])
                
                st.markdown(f"**📝 Description:**")
                st.write(algo_data['description'])
                
                st.markdown(f"**🧮 Mathematical Background:**")
                st.code(algo_data['math'], language='text')
                
                st.markdown("---")

# Preferred Algorithms Section
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div class="pipeline-container">
    <h2 style="text-align: center; color: #4a9eff;">⭐ Recommended Algorithm Pipeline</h2>
    <p style="text-align: center; font-size: 1.1rem; color: #b0b0ff;">
        Industry-standard algorithms for robust VSLAM implementation
    </p>
</div>
""", unsafe_allow_html=True)

# Preferred algorithms
preferred_algos = [
    ("Image Acquisition", "Stereo Camera", "Provides direct depth estimation"),
    ("Feature Detection", "ORB Features", "Fast, rotation-invariant, patent-free"),
    ("Feature Matching", "FLANN + Ratio Test", "Fast approximate matching with reliability filtering"),
    ("Motion Estimation", "5-Point Algorithm + RANSAC", "Robust essential matrix estimation"),
    ("Map Initialization", "Two-View + Bundle Adjustment", "Accurate initial reconstruction"),
    ("Tracking", "Feature-based + PnP", "Robust to appearance changes"),
    ("Local Mapping", "Triangulation + Local BA", "Expanding map with local optimization"),
    ("Loop Detection", "DBoW3", "Fast and reliable place recognition"),
    ("Global Optimization", "Sparse Bundle Adjustment", "Efficient large-scale optimization")
]

# Create preferred algorithms grid
cols = st.columns(3)
for idx, (step, algo, reason) in enumerate(preferred_algos):
    with cols[idx % 3]:
        st.markdown(f"""
        <div class="preferred-algo">
            <h4 style="margin-bottom: 0.5rem;">{step}</h4>
            <h3 style="margin: 0.5rem 0; color: #ffffff;">{algo}</h3>
            <p style="margin-top: 0.5rem; font-size: 0.9rem; opacity: 0.9;">{reason}</p>
        </div>
        """, unsafe_allow_html=True)

# Performance Metrics
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div class="pipeline-container">
    <h2 style="text-align: center; color: #4a9eff;">📊 VSLAM Performance Metrics</h2>
</div>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="metric-card">
        <h3 style="color: #4a9eff;">Accuracy</h3>
        <h2 style="color: #66ff66;">±2cm</h2>
        <p>Typical positioning accuracy</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <h3 style="color: #4a9eff;">Frame Rate</h3>
        <h2 style="color: #66ff66;">30+ FPS</h2>
        <p>Real-time processing capability</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <h3 style="color: #4a9eff;">Map Points</h3>
        <h2 style="color: #66ff66;">10K+</h2>
        <p>Typical dense map size</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="metric-card">
        <h3 style="color: #4a9eff;">Loop Closure</h3>
        <h2 style="color: #66ff66;">&lt;1s</h2>
        <p>Detection latency</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; padding: 2rem 0; margin-top: 3rem; border-top: 1px solid rgba(100, 100, 255, 0.2);">
    <p style="color: #8080ff;">
        🎯 VSLAM Concepts Dashboard | Comprehensive Algorithm Reference
    </p>
</div>
""", unsafe_allow_html=True)