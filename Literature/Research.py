import streamlit as st
import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

# Configure page
st.set_page_config(
    page_title="VSLAM Algorithm Selection Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #667eea;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #667eea;
    }
    
    .algorithm-card {
        background: linear-gradient(135deg, #2d3748 0%, #4a5568 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    .pros-cons-container {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 1rem;
        margin-top: 1rem;
    }
    
    .pros-section {
        background: rgba(72, 187, 120, 0.1);
        padding: 1rem;
        border-radius: 8px;
        border-left: 3px solid #48bb78;
    }
    
    .cons-section {
        background: rgba(245, 101, 101, 0.1);
        padding: 1rem;
        border-radius: 8px;
        border-left: 3px solid #f56565;
    }
    
    .flowchart-container {
        background: linear-gradient(135deg, #1a202c 0%, #2d3748 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .step-indicator {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin: 2rem 0;
    }
    
    .step-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 2rem;
        border-radius: 10px;
        font-weight: bold;
        text-align: center;
        min-width: 200px;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .arrow {
        font-size: 2rem;
        color: #667eea;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Sample data (you can replace this with your JSON file loading)
sample_data = {
    "Affine-ORB": {
        "1-s2.0-S0030402620312572-main.pdf": {
            "pros": [
                "Keyframe loss rate significantly reduced from 0.5 % to 0.2 %",
                "Root mean square error (RMSE) of the running trajectory drastically reduced",
                "Key frame extraction speed is faster",
                "Keyframe loss rate is lower at the same moving speed",
                "Positioning accuracy is higher",
                "Takes advantage of more keyframes and lower keyframe losing rate under constant moving speed",
                "Robust feature extraction and accurate positioning",
                "Number of feature points and keyframe acquiring is increased",
                "Reliability of the front-end visual odometry of the SLAM system is enhanced",
                "Accuracy of positioning and mapping is improved by applying the mathematical method of affine transformation to ORB feature extraction"
            ],
            "cons": []
        }
    },
    "ORB": {
        "1-s2.0-S0030402620312572-main.pdf": {
            "pros": [
                "Can be used to detect local key points in an image",
                "Has better performance and lower computational load",
                "Images extracted have local feature invariance",
                "Combines FAST key point detection and BRIEF feature descriptor for improvement",
                "Scale and rotation descriptions are added for FAST algorithm shortcomings",
                "Principal directions of feature points are calculated",
                "Rotation invariance is added to the BRIEF method",
                "Greedy search solves the problem of substantial correlation between feature descriptors"
            ],
            "cons": []
        },
        "1-s2.0-S0952197622001853-main.pdf": {
            "pros": [
                "Calculation speed is faster than the FAST algorithm",
                "Less affected by noise",
                "Has good rotation invariance",
                "Has good scale invariance",
                "Can be applied to real-time SLAM systems",
                "Less computationally intensive compared to SIFT",
                "Considered better suited for applications of autonomous driving vehicles due to limited computing power"
            ],
            "cons": []
        },
        "1-s2.0-S0957417422010156-main.pdf": {
            "pros": [
                "Very fast"
            ],
            "cons": [
                "Less effective in terms of scale"
            ]
        }
    },
    "BRISK": {
        "1-s2.0-S0030402620312572-main.pdf": {
            "pros": [
                "Binary-based feature descriptor with constant constancy and direction of rotation",
                "Anti-noise capability has become stronger (improved from BRIEF)"
            ],
            "cons": []
        },
        "1-s2.0-S0957417422010156-main.pdf": {
            "pros": [
                "Robust to orientation, noise, and scale with low calculation complexity"
            ],
            "cons": []
        }
    },
    "SIFT": {
        "sample_paper.pdf": {
            "pros": [
                "Excellent scale invariance",
                "High accuracy feature detection",
                "Robust to illumination changes"
            ],
            "cons": [
                "Computationally expensive",
                "Slower than ORB algorithms"
            ]
        }
    },
    "SURF": {
        "sample_paper.pdf": {
            "pros": [
                "Faster than SIFT",
                "Good scale and rotation invariance",
                "Robust feature detection"
            ],
            "cons": [
                "Still computationally intensive",
                "Patent restrictions"
            ]
        }
    }
}

# Load your JSON data here
@st.cache_data
def load_algorithm_data():
    # Replace this with your actual JSON file loading
    with open('Literature/research.json', 'r') as f:
        return json.load(f)
    return sample_data

def create_flowchart(selected_detector, selected_matching, selected_trajectory, selected_evaluation):
    """Create dynamic flowchart based on selections"""
    fig = go.Figure()
    
    # Define positions
    positions = {
        'detection': {'x': 1, 'y': 4},
        'matching': {'x': 3, 'y': 4},
        'trajectory': {'x': 5, 'y': 4},
        'evaluation': {'x': 7, 'y': 4}
    }
    
    # Define colors
    colors = {
        'detection': '#667eea',
        'matching': '#764ba2',
        'trajectory': '#f093fb',
        'evaluation': '#f5576c'
    }
    
    # Add boxes
    boxes = [
        ('Feature Detection', selected_detector or 'Not Selected', 'detection'),
        ('Feature Matching', selected_matching or 'Not Selected', 'matching'),
        ('Trajectory Building', selected_trajectory or 'Not Selected', 'trajectory'),
        ('Result Evaluation', selected_evaluation or 'Not Selected', 'evaluation')
    ]
    
    for title, subtitle, key in boxes:
        pos = positions[key]
        color = colors[key]
        
        # Add box
        fig.add_shape(
            type="rect",
            x0=pos['x']-0.4, y0=pos['y']-0.3,
            x1=pos['x']+0.4, y1=pos['y']+0.3,
            line=dict(color=color, width=3),
            fillcolor=color,
            opacity=0.3
        )
        
        # Add title
        fig.add_annotation(
            x=pos['x'], y=pos['y']+0.1,
            text=f"<b>{title}</b>",
            showarrow=False,
            font=dict(size=14, color="white"),
            bgcolor="rgba(0,0,0,0)"
        )
        
        # Add subtitle
        fig.add_annotation(
            x=pos['x'], y=pos['y']-0.1,
            text=subtitle,
            showarrow=False,
            font=dict(size=11, color="#a0aec0"),
            bgcolor="rgba(0,0,0,0)"
        )
    
    # Add arrows
    for i in range(len(positions)-1):
        start_x = list(positions.values())[i]['x'] + 0.4
        end_x = list(positions.values())[i+1]['x'] - 0.4
        y = 4
        
        fig.add_annotation(
            x=end_x, y=y,
            ax=start_x, ay=y,
            xref="x", yref="y",
            axref="x", ayref="y",
            arrowhead=2,
            arrowsize=2,
            arrowwidth=3,
            arrowcolor="#667eea"
        )
    
    fig.update_layout(
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 8]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[3, 5]),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=300,
        margin=dict(l=0, r=0, t=0, b=0)
    )
    
    return fig

def display_algorithm_details(algorithm_name, algorithm_data, show_papers=True):
    """Display detailed information about an algorithm"""
    st.markdown(f"""
    <div class="algorithm-card">
        <h3 style="color: #667eea; margin-bottom: 1rem;">{algorithm_name}</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Aggregate pros and cons across all papers
    all_pros = []
    all_cons = []
    papers = []
    
    for paper, details in algorithm_data.items():
        papers.append(paper)
        all_pros.extend(details.get('pros', []))
        all_cons.extend(details.get('cons', []))
    
    # Remove duplicates while preserving order
    all_pros = list(dict.fromkeys(all_pros))
    all_cons = list(dict.fromkeys(all_cons))
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ✅ Advantages")
        if all_pros:
            for pro in all_pros:
                st.markdown(f"• {pro}")
        else:
            st.markdown("*No advantages listed*")
    
    with col2:
        st.markdown("### ❌ Disadvantages")
        if all_cons:
            for con in all_cons:
                st.markdown(f"• {con}")
        else:
            st.markdown("*No disadvantages listed*")
    
    if show_papers:
        st.markdown(f"### 📄 Papers referencing {algorithm_name} ({len(papers)})")
        for paper in papers:
            st.markdown(f"• `{paper}`")

def main():
    # Header
    st.markdown('<div class="main-header">🎯 VSLAM Algorithm Selection Dashboard</div>', unsafe_allow_html=True)
    
    # Load data
    algorithm_data = load_algorithm_data()
    
    # Sidebar for selections
    st.sidebar.markdown("## 🔧 Algorithm Selection")
    
    # Get available algorithms
    available_algorithms = list(algorithm_data.keys())
    
    # Selection widgets
    selected_detector = st.sidebar.selectbox(
        "🔍 Select Feature Detector",
        [""] + available_algorithms,
        help="Choose the feature detection algorithm"
    )
    
    selected_matching = st.sidebar.selectbox(
        "🔗 Select Feature Matching",
        [""] + available_algorithms,
        help="Choose the feature matching algorithm"
    )
    
    selected_trajectory = st.sidebar.selectbox(
        "📈 Select Trajectory Building",
        [""] + available_algorithms,
        help="Choose the trajectory building method"
    )
    
    selected_evaluation = st.sidebar.selectbox(
        "📊 Select Result Evaluation",
        [""] + available_algorithms,
        help="Choose the evaluation method"
    )
    
    # Main content area
    st.markdown("## 🔄 VSLAM Pipeline Flowchart")
    
    # Create and display flowchart
    flowchart = create_flowchart(selected_detector, selected_matching, selected_trajectory, selected_evaluation)
    st.plotly_chart(flowchart, use_container_width=True)
    
    # Display selected algorithms details
    st.markdown("## 📋 Selected Algorithms Details")
    
    selected_algorithms = {
        "Feature Detector": selected_detector,
        "Feature Matching": selected_matching,
        "Trajectory Building": selected_trajectory,
        "Result Evaluation": selected_evaluation
    }
    
    for stage, algorithm in selected_algorithms.items():
        if algorithm:
            st.markdown(f"### {stage}: {algorithm}")
            display_algorithm_details(algorithm, algorithm_data[algorithm], show_papers=False)
            st.markdown("---")
    
    # Algorithm comparison section
    st.markdown("## 📊 Algorithm Comparison")
    
    # Create comparison chart
    comparison_data = []
    for alg_name, alg_data in algorithm_data.items():
        total_pros = sum(len(details.get('pros', [])) for details in alg_data.values())
        total_cons = sum(len(details.get('cons', [])) for details in alg_data.values())
        total_papers = len(alg_data)
        
        comparison_data.append({
            'Algorithm': alg_name,
            'Advantages': total_pros,
            'Disadvantages': total_cons,
            'Papers': total_papers,
            'Advantage Ratio': total_pros / (total_pros + total_cons) if (total_pros + total_cons) > 0 else 0
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Advantages vs Disadvantages chart
        fig_comparison = px.bar(
            df_comparison, 
            x='Algorithm', 
            y=['Advantages', 'Disadvantages'],
            title="Pros vs Cons by Algorithm",
            color_discrete_map={'Advantages': '#48bb78', 'Disadvantages': '#f56565'},
            template='plotly_dark'
        )
        fig_comparison.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_comparison, use_container_width=True)
    
    with col2:
        # Papers count chart
        fig_papers = px.pie(
            df_comparison,
            values='Papers',
            names='Algorithm',
            title="Research Coverage by Algorithm",
            template='plotly_dark'
        )
        fig_papers.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_papers, use_container_width=True)
    
    # Algorithm database
    st.markdown("## 🗃️ Algorithm Database")
    
    # Search and filter
    search_term = st.text_input("🔍 Search algorithms", placeholder="Enter algorithm name or keyword...")
    
    # Filter algorithms based on search
    filtered_algorithms = algorithm_data
    if search_term:
        filtered_algorithms = {
            name: data for name, data in algorithm_data.items()
            if search_term.lower() in name.lower()
        }
    
    # Display filtered algorithms
    for alg_name, alg_data in filtered_algorithms.items():
        with st.expander(f"📚 {alg_name} - Detailed Analysis"):
            display_algorithm_details(alg_name, alg_data, show_papers=True)
    
    # Footer
    st.markdown("---")
    st.markdown("*💡 Select algorithms from the sidebar to build your VSLAM pipeline and see the dynamic flowchart update in real-time!*")

if __name__ == "__main__":
    main()