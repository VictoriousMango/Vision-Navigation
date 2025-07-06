import numpy as np
import cv2
import g2o

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

class LocalBundleAdjustment:
    def __init__(self):
        self.keyframe_interval = 10000 # Temporal Threshold
        self.optimizer = g2o.SparseOptimizer()
        # Create a linear solver
        solver = g2o.BlockSolverSE3(g2o.LinearSolverDenseSE3())
        # Create the optimization algorithm with the solver
        algorithm = g2o.OptimizationAlgorithmLevenberg(solver)
        self.optimizer.set_algorithm(algorithm)
        
    def optimize_local_map(self, keyframes, map_points):
        """Optimize poses and 3D points using bundle adjustment"""
        # Add camera poses as vertices
        for i, kf in enumerate(keyframes):
            pose_vertex = g2o.VertexSE3Expmap()
            pose_vertex.set_id(i)
            # Extract rotation (3x3) and translation (3x1) from 4x4 pose matrix
            R = kf.pose[:3, :3]
            t = kf.pose[:3, 3].reshape(3, 1)
            pose_vertex.set_estimate(g2o.SE3Quat(R, t))
            self.optimizer.add_vertex(pose_vertex)
        
        # Add 3D points as vertices and create projection edges
        point_id = len(keyframes)
        for mp in map_points:
            point_vertex = g2o.VertexPointXYZ()
            point_vertex.set_id(point_id)
            point_vertex.set_estimate(mp)
            self.optimizer.add_vertex(point_vertex)
            point_id += 1
class PoseGraphOptimizer:
    def __init__(self):
        self.pose_graph = g2o.SparseOptimizer()
        self.pose_graph.set_algorithm(g2o.OptimizationAlgorithmLevenberg())
        
    def add_pose_vertex(self, pose_id, pose):
        """Add a pose vertex to the graph"""
        vertex = g2o.VertexSE3Expmap()
        vertex.set_id(pose_id)
        vertex.set_estimate(g2o.SE3Quat(pose))
        self.pose_graph.add_vertex(vertex)
        
    def add_odometry_edge(self, id1, id2, measurement, information):
        """Add odometry constraint between consecutive poses"""
        edge = g2o.EdgeSE3Expmap()
        edge.set_vertex(0, self.pose_graph.vertex(id1))
        edge.set_vertex(1, self.pose_graph.vertex(id2))
        edge.set_measurement(g2o.SE3Quat(measurement))
        edge.set_information(information)
        self.pose_graph.add_edge(edge)
        
    def add_loop_closure_edge(self, id1, id2, measurement, information):
        """Add loop closure constraint"""
        edge = g2o.EdgeSE3Expmap()
        edge.set_vertex(0, self.pose_graph.vertex(id1))
        edge.set_vertex(1, self.pose_graph.vertex(id2))
        edge.set_measurement(g2o.SE3Quat(measurement))
        edge.set_information(information)
        self.pose_graph.add_edge(edge)
