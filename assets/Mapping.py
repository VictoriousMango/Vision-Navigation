import numpy as np
import pandas as pd
import plotly.express as px
import cv2
import plotly.graph_objects as go

class LocalMapper:
    def __init__(self):
        self.camera_matrix = None
        self.map_points = []
        # Added New => for Mapping
        self.keyframes = []
        self.last_keyframe_pose = np.eye(4)
    def setCameraMatrix(self, camera_matrix):
        """Set the camera intrinsic matrix"""
        self.camera_matrix = camera_matrix

    def triangulate_points(self, pose1, pose2, points1, points2):
        """Triangulate 3D points from matched features"""
        # Create projection matrices
        P1 = self.camera_matrix @ pose1[:3, :]
        P2 = self.camera_matrix @ pose2[:3, :]
        
        # Triangulate points
        points_4d = cv2.triangulatePoints(P1, P2, points1.T, points2.T)
        points_3d = points_4d[:3] / points_4d[3]
        
        return points_3d.T

class MapPoint:
    def __init__(self, position, descriptor):
        self.position = position
        self.descriptor = descriptor
        self.observations = []  # List of keyframes that observe this point
        self.is_outlier = False

class KeyFrame:
    def __init__(self, pose, features, descriptors, frame_id):
        self.pose = pose
        self.features = features
        self.descriptors = descriptors
        self.frame_id = frame_id
        self.map_points = []

class Animated3DTrajectory:
    def __init__(self):
        pass
    def lst_to_df(self, trajectory):
        """Convert trajectory list to DataFrame"""
        df = pd.DataFrame(trajectory, columns=['x', 'y', 'z'])
        df['frame_index'] = df.index
        return df
    def plotly_chart(self, df):
        # Plotly 3D Trajectory Plot (Beautiful)
        fig = px.scatter_3d(df, 
                                x='x', 
                                y='z',  
                                z='y',
                                animation_frame='frame_index',  # Note: correct column name
                                title='Animated Ground Truth Trajectory (3D)',
                                color_discrete_sequence=['red'])
        fig.add_trace(go.Scatter3d(
            x=df['x'],
            y=df['z'],  # Z as horizontal axis (floor)
            z=df['y'],  # Y as height
            mode='markers',
            marker=dict(size=1, color='green'),
            name='Keyframes'
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
                font=dict(size=11, color='white')
            ),
            legend=dict(
                x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.7)', bordercolor='black'
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )

        return fig
    
    def static_chart(self, df):
        """Static 3D Trajectory Plot"""
        fig = px.line_3d(df, 
                                x='x', 
                                y='z',  
                                z='y',
                                title='Trajectory (3D)',
                                color_discrete_sequence=['red'])
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
                font=dict(size=11, color='white')
            ),
            legend=dict(
                x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.7)', bordercolor='black'
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )
        return fig