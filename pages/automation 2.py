import pandas as pd
import plotly.express as px
import numpy as np
import streamlit as st

# Create sample 3D trajectory data
np.random.seed(42)
time_steps = np.linspace(0, 10, 100)
x = np.cumsum(np.random.randn(100) * 0.1) + time_steps * 0.5
y = np.cumsum(np.random.randn(100) * 0.1) + np.sin(time_steps)
z = np.cumsum(np.random.randn(100) * 0.1) + np.cos(time_steps)
xyz_list = list(zip(x, y, z))

class Animated3DTrajectory:
    def __init__(self):
        self.df_animated = None

    def build_dataframe(self, xyz_list):
        """
        xyz_list: list of [x, y, z] triples
        """
        df = pd.DataFrame(xyz_list, columns=['x', 'y', 'z'])
        df_animated = []
        for i in range(1, len(df) + 1):
            temp_df = df.iloc[:i].copy()
            temp_df['frame'] = i
            df_animated.append(temp_df)
        self.df_animated = pd.concat(df_animated, ignore_index=True)

    def get_figure(self, title="Animated 3D Trajectory with Trail"):
        if self.df_animated is None or self.df_animated.empty:
            return px.line_3d(title=title)
        df = self.df_animated
        fig = px.line_3d(
            df,
            x='x',
            y='y',
            z='z',
            animation_frame='frame',
            title=title,
            range_x=[df['x'].min()-1, df['x'].max()+1],
            range_y=[df['y'].min()-1, df['y'].max()+1],
            range_z=[df['z'].min()-1, df['z'].max()+1]
        )
        fig.update_traces(
            mode='lines+markers',
            line=dict(width=2),
            marker=dict(size=4)
        )
        return fig

# Show the plot
animator = Animated3DTrajectory()
animator.build_dataframe(xyz_list)
fig = animator.get_figure()
st.plotly_chart(fig, use_container_width=True)