import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import random
import time

st.title("3D Real-time Testing Grounds")

# session variables ---------
if "x" not in st.session_state:
    st.session_state.x = []
if "y" not in st.session_state:
    st.session_state.y = []
if "z" not in st.session_state:
    st.session_state.z = []
if "frames" not in st.session_state:
    st.session_state.frames = [go.Frame(data=[go.Scatter3d(x=st.session_state.x, y=st.session_state.y, z=st.session_state.z)])]
if "N" not in st.session_state:
    st.session_state.N = 1000
if "is_adding" not in st.session_state:
    st.session_state.is_adding = False
if "plot_type" not in st.session_state:
    st.session_state.plot_type = "spiral"
# ---------------------------

# Create a placeholder for the chart
chart_placeholder = st.empty()

# Function to create/update the 3D figure
def create_3d_figure(x_data, y_data, z_data):
    fig = go.Figure(
        data=[go.Scatter3d(
            x=x_data, 
            y=y_data, 
            z=z_data,
            mode='lines+markers',
            name='3D Data',
            marker=dict(
                size=3,
                color=z_data if z_data else [0],
                colorscale='Viridis',
                showscale=True
            ),
            line=dict(
                color='blue',
                width=2
            )
        )],
        layout=go.Layout(
            scene=dict(
                xaxis=dict(title='X axis'),
                yaxis=dict(title='Y axis'),
                zaxis=dict(title='Z axis'),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            title=dict(text=f"Real-time 3D Plot - {len(x_data)} points"),
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[None, {"frame": {"duration": 50, "redraw": True}, "fromcurrent": True}]
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]
                    )
                ]
            )]
        )
    )
    return fig

# Function to generate different 3D patterns
def generate_3d_point(i, plot_type):
    if plot_type == "spiral":
        x = np.cos(i/10) * (i/100)
        y = np.sin(i/10) * (i/100)
        z = i/50
    elif plot_type == "helix":
        x = np.cos(i/20)
        y = np.sin(i/20)
        z = i/100
    elif plot_type == "wave":
        x = i/10
        y = np.sin(i/10)
        z = np.cos(i/10)
    elif plot_type == "tornado":
        radius = 1 + i/500
        x = radius * np.cos(i/5)
        y = radius * np.sin(i/5)
        z = i/20
    elif plot_type == "random_walk":
        if i == 0:
            x, y, z = 0, 0, 0
        else:
            x = st.session_state.x[-1] + np.random.normal(0, 0.1)
            y = st.session_state.y[-1] + np.random.normal(0, 0.1)
            z = st.session_state.z[-1] + np.random.normal(0, 0.1)
    else:  # default spiral
        x = np.cos(i/10) * (i/100)
        y = np.sin(i/10) * (i/100)
        z = i/50
    
    return x, y, z

# Sidebar controls
with st.sidebar:
    st.subheader("3D Plot Settings")
    
    # Plot type selection
    plot_type = st.selectbox(
        "Select 3D Pattern:",
        ["spiral", "helix", "wave", "tornado", "random_walk"],
        index=0
    )
    st.session_state.plot_type = plot_type
    
    # Number of points
    num_points = st.slider("Number of points to add:", 100, 2000, 1000)
    
    # Update frequency
    update_freq = st.slider("Update every N points:", 1, 50, 10)
    
    # Animation speed
    sleep_time = st.slider("Animation speed (seconds):", 0.01, 0.5, 0.05)
    
    st.divider()
    
    if st.button("Start Adding 3D Points", type="primary"):
        st.session_state.is_adding = True
        
        # Real-time loop with chart updates
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(num_points):
            # Generate new 3D data point
            x, y, z = generate_3d_point(i, st.session_state.plot_type)
            
            st.session_state.x.append(x)
            st.session_state.y.append(y)
            st.session_state.z.append(z)
            
            # Update the chart every N iterations
            if i % update_freq == 0 or i == num_points - 1:
                fig = create_3d_figure(st.session_state.x, st.session_state.y, st.session_state.z)
                chart_placeholder.plotly_chart(fig, use_container_width=True, key=f"chart_update_{i}")
            
            # Update progress
            progress_bar.progress((i + 1) / num_points)
            status_text.text(f'Adding 3D point {i + 1}/{num_points}')
            
            time.sleep(sleep_time)
        
        st.session_state.is_adding = False
        progress_bar.empty()
        status_text.empty()
        st.success(f"Finished adding {num_points} 3D points!")
    
    if st.button("Reset All Data"):
        st.session_state.x = []
        st.session_state.y = []
        st.session_state.z = []
        st.session_state.frames = [go.Frame(data=[go.Scatter3d(x=st.session_state.x, y=st.session_state.y, z=st.session_state.z)])]
        st.session_state.is_adding = False
        st.success("Data reset!")

# Display current chart (if not currently adding points)
if not st.session_state.is_adding:
    fig = create_3d_figure(st.session_state.x, st.session_state.y, st.session_state.z)
    chart_placeholder.plotly_chart(fig, use_container_width=True, key="static_chart")

# Display data statistics and sample
st.subheader("3D Data Information")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Points", len(st.session_state.x))

with col2:
    if st.session_state.z:
        st.metric("Z Range", f"{min(st.session_state.z):.2f} - {max(st.session_state.z):.2f}")

with col3:
    st.metric("Current Pattern", st.session_state.plot_type.replace('_', ' ').title())

# Display sample data
if st.session_state.x and st.session_state.y and st.session_state.z:
    st.subheader("Sample Data (Last 20 Points)")
    df = pd.DataFrame({
        "X": st.session_state.x[-20:],
        "Y": st.session_state.y[-20:], 
        "Z": st.session_state.z[-20:]
    })
    st.dataframe(df, use_container_width=True)
    
    # Download option
    if st.button("Download All Data as CSV"):
        full_df = pd.DataFrame({
            "X": st.session_state.x,
            "Y": st.session_state.y,
            "Z": st.session_state.z
        })
        csv = full_df.to_csv(index=False)
        st.download_button(
            label="Click to Download",
            data=csv,
            file_name=f"3d_data_{st.session_state.plot_type}_{len(st.session_state.x)}_points.csv",
            mime="text/csv"
        )
else:
    st.info("No data points yet. Select a pattern and click 'Start Adding 3D Points' to begin!")