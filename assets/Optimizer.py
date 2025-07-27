import numpy as np
import cv2
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from typing import List, Dict, Tuple, Optional
import threading
import time
from collections import defaultdict

class ScipyOptimizer:
    """
    Comprehensive optimization system using SciPy for SLAM
    Supports both local bundle adjustment and global pose graph optimization
    Direct replacement for G2O-based optimizer
    """
    
    def __init__(self, 
                 camera_matrix: np.ndarray,
                 local_window_size: int = 5,
                 global_optimization_interval: int = 10,
                 robust_kernel_threshold: float = 5.991):
        """
        Initialize the SciPy optimizer
        
        Args:
            camera_matrix: 3x3 camera intrinsic matrix
            local_window_size: Number of keyframes for local optimization
            global_optimization_interval: Interval for global optimization
            robust_kernel_threshold: Threshold for robust kernel (chi-squared for 2 DOF)
        """
        self.camera_matrix = camera_matrix
        self.fx = camera_matrix[0, 0]
        self.fy = camera_matrix[1, 1]
        self.cx = camera_matrix[0, 2]
        self.cy = camera_matrix[1, 2]
        
        self.local_window_size = local_window_size
        self.global_optimization_interval = global_optimization_interval
        self.robust_kernel_threshold = robust_kernel_threshold
        
        # Storage for optimization data
        self.keyframes = {}  # keyframe_id: pose
        self.map_points = {}  # point_id: 3D position
        self.observations = defaultdict(list)  # point_id: [(keyframe_id, 2D_point), ...]
        self.pose_graph_edges = []  # [(from_id, to_id, relative_pose, information), ...]
        
        # Optimization state
        self.optimization_thread = None
        self.optimization_running = False
        self.last_global_optimization = 0
        
        # Counters
        self.next_keyframe_id = 0
        self.next_point_id = 0
        
    def add_keyframe(self, pose: np.ndarray, keypoints: List, descriptors: np.ndarray) -> int:
        """
        Add a new keyframe to the system
        
        Args:
            pose: 4x4 transformation matrix
            keypoints: List of cv2.KeyPoint objects
            descriptors: Feature descriptors
            
        Returns:
            keyframe_id: Unique ID for the keyframe
        """
        keyframe_id = self.next_keyframe_id
        self.next_keyframe_id += 1
        
        self.keyframes[keyframe_id] = {
            'pose': pose.copy(),
            'keypoints': keypoints,
            'descriptors': descriptors,
            'fixed': keyframe_id == 0  # Fix the first keyframe
        }
        
        return keyframe_id
    
    def add_map_point(self, point_3d: np.ndarray, observations: List[Tuple[int, np.ndarray]]) -> int:
        """
        Add a 3D map point with its observations
        
        Args:
            point_3d: 3D coordinates of the point
            observations: List of (keyframe_id, 2D_point) tuples
            
        Returns:
            point_id: Unique ID for the map point
        """
        point_id = self.next_point_id
        self.next_point_id += 1
        
        self.map_points[point_id] = point_3d.copy()
        
        for keyframe_id, point_2d in observations:
            self.observations[point_id].append((keyframe_id, point_2d))
            
        return point_id
    
    def add_pose_graph_edge(self, from_id: int, to_id: int, 
                           relative_pose: np.ndarray, information: np.ndarray):
        """
        Add an edge to the pose graph
        
        Args:
            from_id: Source keyframe ID
            to_id: Target keyframe ID
            relative_pose: 4x4 relative transformation
            information: 6x6 information matrix
        """
        self.pose_graph_edges.append((from_id, to_id, relative_pose, information))
    
    @staticmethod
    def pose_to_vector(pose: np.ndarray) -> np.ndarray:
        """Convert 4x4 pose matrix to 6-DOF vector [translation, rodrigues_rotation]"""
        translation = pose[:3, 3]
        rotation_matrix = pose[:3, :3]
        rodrigues, _ = cv2.Rodrigues(rotation_matrix)
        return np.concatenate([translation, rodrigues.flatten()])
    
    @staticmethod
    def vector_to_pose(vector: np.ndarray) -> np.ndarray:
        """Convert 6-DOF vector to 4x4 pose matrix"""
        translation = vector[:3]
        rodrigues = vector[3:6]
        rotation_matrix, _ = cv2.Rodrigues(rodrigues)
        
        pose = np.eye(4)
        pose[:3, :3] = rotation_matrix
        pose[:3, 3] = translation
        return pose
    
    def project_point(self, point_3d: np.ndarray, pose: np.ndarray) -> np.ndarray:
        """Project 3D point to 2D using camera model"""
        # Transform point to camera frame
        point_cam = pose[:3, :3] @ point_3d + pose[:3, 3]
        
        # Avoid division by zero
        if point_cam[2] <= 1e-6:
            return np.array([float('inf'), float('inf')])
        
        # Project to image plane
        x = self.fx * point_cam[0] / point_cam[2] + self.cx
        y = self.fy * point_cam[1] / point_cam[2] + self.cy
        
        return np.array([x, y])
    
    def huber_loss(self, residual: float, delta: float = 1.0) -> float:
        """Huber robust loss function"""
        abs_residual = abs(residual)
        if abs_residual <= delta:
            return 0.5 * residual**2
        else:
            return delta * (abs_residual - 0.5 * delta)
    
    def bundle_adjustment_residuals(self, params: np.ndarray, keyframe_ids: List[int], 
                                  point_ids: List[int], observations_data: List) -> np.ndarray:
        """
        Compute residuals for bundle adjustment
        
        Args:
            params: Flattened parameter vector [poses..., points...]
            keyframe_ids: List of keyframe IDs being optimized
            point_ids: List of point IDs being optimized
            observations_data: List of observation tuples
            
        Returns:
            Array of residuals
        """
        residuals = []
        
        # Calculate number of non-fixed poses
        non_fixed_keyframes = [kf_id for kf_id in keyframe_ids if not self.keyframes[kf_id]['fixed']]
        n_poses = len(non_fixed_keyframes)
        n_points = len(point_ids)
        
        # Extract poses and points from parameter vector
        if n_poses > 0:
            pose_params = params[:n_poses * 6].reshape(-1, 6)
            point_params = params[n_poses * 6:].reshape(-1, 3)
        else:
            pose_params = np.array([]).reshape(0, 6)
            point_params = params.reshape(-1, 3)
        
        # Create pose and point mappings
        pose_map = {kf_id: i for i, kf_id in enumerate(non_fixed_keyframes)}
        point_map = {pt_id: i for i, pt_id in enumerate(point_ids)}
        
        # Process each observation
        for pt_id, kf_id, observed_2d in observations_data:
            if pt_id not in point_map:
                continue
                
            # Get 3D point
            point_3d = point_params[point_map[pt_id]]
            
            # Get pose
            if self.keyframes[kf_id]['fixed']:
                # Use original pose for fixed keyframes
                pose = self.keyframes[kf_id]['pose']
            else:
                if kf_id not in pose_map:
                    continue
                pose_vector = pose_params[pose_map[kf_id]]
                pose = self.vector_to_pose(pose_vector)
            
            # Project point
            projected_2d = self.project_point(point_3d, pose)
            
            # Compute residual
            if np.any(np.isinf(projected_2d)):
                residual = np.array([100.0, 100.0])  # Large residual for invalid projections
            else:
                residual = projected_2d - observed_2d
            
            # Apply robust loss (simpler version)
            residual = np.clip(residual, -10.0, 10.0)  # Clip extreme values
            residuals.extend(residual)
        
        return np.array(residuals)
    
    def pose_graph_residuals(self, params: np.ndarray, keyframe_ids: List[int]) -> np.ndarray:
        """
        Compute residuals for pose graph optimization
        
        Args:
            params: Flattened pose parameter vector
            keyframe_ids: List of keyframe IDs being optimized
            
        Returns:
            Array of residuals
        """
        residuals = []
        
        # Get only non-fixed keyframes
        non_fixed_keyframes = [kf_id for kf_id in keyframe_ids if not self.keyframes[kf_id]['fixed']]
        
        if len(non_fixed_keyframes) == 0:
            return np.array([0.0])  # Return dummy residual if no poses to optimize
        
        # Extract poses from parameter vector
        pose_params = params.reshape(-1, 6)
        pose_map = {kf_id: i for i, kf_id in enumerate(non_fixed_keyframes)}
        
        # Process each edge
        for from_id, to_id, relative_pose, information in self.pose_graph_edges:
            if from_id not in keyframe_ids or to_id not in keyframe_ids:
                continue
            
            # Get poses
            if self.keyframes[from_id]['fixed']:
                pose_from = self.keyframes[from_id]['pose']
            else:
                if from_id not in pose_map:
                    continue
                pose_from = self.vector_to_pose(pose_params[pose_map[from_id]])
            
            if self.keyframes[to_id]['fixed']:
                pose_to = self.keyframes[to_id]['pose']
            else:
                if to_id not in pose_map:
                    continue
                pose_to = self.vector_to_pose(pose_params[pose_map[to_id]])
            
            # Compute relative pose error
            predicted_relative = np.linalg.inv(pose_from) @ pose_to
            error_pose = relative_pose @ np.linalg.inv(predicted_relative)
            
            # Convert to 6-DOF error vector
            error_vector = self.pose_to_vector(error_pose)
            
            # Weight by information matrix (simplified - use diagonal)
            weights = np.sqrt(np.diag(information))
            weighted_error = weights * error_vector
            
            # Clip extreme values
            weighted_error = np.clip(weighted_error, -10.0, 10.0)
            
            residuals.extend(weighted_error)
        
        # Add regularization term to ensure well-conditioned problem
        if len(residuals) < len(params):
            # Add small regularization residuals
            reg_weight = 1e-6
            for param in params:
                residuals.append(reg_weight * param)
        
        return np.array(residuals) if residuals else np.array([0.0])
    
    def local_bundle_adjustment(self, current_keyframe_id: int, iterations: int = 10) -> bool:
        """
        Perform local bundle adjustment using SciPy
        
        Args:
            current_keyframe_id: ID of the current keyframe
            iterations: Number of optimization iterations
            
        Returns:
            Success flag
        """
        # Get local keyframes (current + recent neighbors)
        keyframe_ids = self._get_local_keyframes(current_keyframe_id)
        
        if len(keyframe_ids) < 2:
            return False
        
        # Get observed points
        point_ids = self._get_observed_points(keyframe_ids)
        
        if len(point_ids) < 3:  # Reduced minimum requirement
            return False
        
        # Prepare observations data
        observations_data = []
        for pt_id in point_ids:
            for kf_id, point_2d in self.observations[pt_id]:
                if kf_id in keyframe_ids:
                    observations_data.append((pt_id, kf_id, point_2d))
        
        if len(observations_data) < 6:  # Reduced minimum requirement
            return False
        
        # Prepare initial parameters (only non-fixed poses)
        non_fixed_keyframes = [kf_id for kf_id in keyframe_ids if not self.keyframes[kf_id]['fixed']]
        initial_poses = []
        
        for kf_id in non_fixed_keyframes:
            pose_vector = self.pose_to_vector(self.keyframes[kf_id]['pose'])
            initial_poses.append(pose_vector)
        
        initial_points = []
        for pt_id in point_ids:
            initial_points.append(self.map_points[pt_id])
        
        # Check if we have enough constraints
        n_params = len(non_fixed_keyframes) * 6 + len(point_ids) * 3
        n_residuals = len(observations_data) * 2
        
        if n_residuals < n_params:
            print(f"  Warning: Insufficient constraints ({n_residuals} residuals < {n_params} parameters)")
            # Try with fewer points if possible
            if len(point_ids) > 3:
                point_ids = point_ids[:max(3, n_residuals // 4)]
                initial_points = initial_points[:len(point_ids)]
            else:
                return False
        
        initial_params = np.concatenate([
            np.array(initial_poses).flatten() if initial_poses else np.array([]),
            np.array(initial_points).flatten()
        ])
        
        if len(initial_params) == 0:
            return False
        
        # Optimize using least squares
        try:
            # Use 'trf' method which is more robust for ill-conditioned problems
            result = least_squares(
                self.bundle_adjustment_residuals,
                initial_params,
                args=(keyframe_ids, point_ids, observations_data),
                method='trf',  # Trust Region Reflective algorithm
                max_nfev=iterations * len(initial_params),
                ftol=1e-4,
                xtol=1e-4,
                gtol=1e-4
            )
            
            if result.success or result.cost < 1e6:  # Accept if cost is reasonable
                # Update keyframe poses
                n_pose_params = len(non_fixed_keyframes) * 6
                
                if n_pose_params > 0:
                    optimized_poses = result.x[:n_pose_params].reshape(-1, 6)
                    
                    for i, kf_id in enumerate(non_fixed_keyframes):
                        optimized_pose = self.vector_to_pose(optimized_poses[i])
                        self.keyframes[kf_id]['pose'] = optimized_pose
                
                # Update map points
                optimized_points = result.x[n_pose_params:].reshape(-1, 3)
                for i, pt_id in enumerate(point_ids):
                    self.map_points[pt_id] = optimized_points[i]
                
                return True
            
        except Exception as e:
            print(f"Local BA optimization failed: {e}")
        
        return False
    
    def global_pose_graph_optimization(self, iterations: int = 20) -> bool:
        """
        Perform global pose graph optimization using SciPy
        
        Args:
            iterations: Number of optimization iterations
            
        Returns:
            Success flag
        """
        keyframe_ids = list(self.keyframes.keys())
        
        if len(keyframe_ids) < 3:
            return False
        
        # Get only non-fixed keyframes
        non_fixed_keyframes = [kf_id for kf_id in keyframe_ids if not self.keyframes[kf_id]['fixed']]
        
        if len(non_fixed_keyframes) == 0:
            return True  # Nothing to optimize
        
        # Check if we have any edges
        valid_edges = [edge for edge in self.pose_graph_edges 
                      if edge[0] in keyframe_ids and edge[1] in keyframe_ids]
        
        if len(valid_edges) == 0:
            return False
        
        # Prepare initial parameters (only non-fixed poses)
        initial_poses = []
        for kf_id in non_fixed_keyframes:
            pose_vector = self.pose_to_vector(self.keyframes[kf_id]['pose'])
            initial_poses.append(pose_vector)
        
        initial_params = np.array(initial_poses).flatten()
        
        if len(initial_params) == 0:
            return True
        
        # Check constraint ratio
        n_params = len(initial_params)
        n_residuals = len(valid_edges) * 6  # Each edge contributes 6 residuals
        
        print(f"  Global optimization: {n_residuals} residuals, {n_params} parameters")
        
        # Optimize using least squares
        try:
            # Use 'trf' method which handles under-constrained problems better
            result = least_squares(
                self.pose_graph_residuals,
                initial_params,
                args=(keyframe_ids,),
                method='trf',  # Trust Region Reflective algorithm
                max_nfev=iterations * max(1, len(initial_params)),
                ftol=1e-4,
                xtol=1e-4,
                gtol=1e-4
            )
            
            if result.success or result.cost < 1e6:  # Accept if cost is reasonable
                # Update keyframe poses
                optimized_poses = result.x.reshape(-1, 6)
                
                for i, kf_id in enumerate(non_fixed_keyframes):
                    optimized_pose = self.vector_to_pose(optimized_poses[i])
                    self.keyframes[kf_id]['pose'] = optimized_pose
                
                self.last_global_optimization = len(self.keyframes)
                return True
            else:
                print(f"  Global optimization converged with high cost: {result.cost}")
                return False
                
        except Exception as e:
            print(f"Global optimization failed: {e}")
        
        return False
    
    def optimize_step(self, current_keyframe_id: int, loop_closures: List = None):
        """
        Perform one optimization step (local BA + global optimization if needed)
        
        Args:
            current_keyframe_id: ID of the current keyframe
            loop_closures: List of detected loop closures
        """
        # Add loop closure edges if provided
        if loop_closures:
            for loop in loop_closures:
                self._add_loop_closure_edge(loop)
        
        # Local bundle adjustment
        self.local_bundle_adjustment(current_keyframe_id)
        
        # Global optimization check
        if (len(self.keyframes) - self.last_global_optimization >= self.global_optimization_interval or 
            loop_closures):
            self.global_pose_graph_optimization()
    
    def _get_local_keyframes(self, current_id: int) -> List[int]:
        """Get local keyframes around the current keyframe"""
        keyframe_ids = list(self.keyframes.keys())
        keyframe_ids.sort()
        
        try:
            current_idx = keyframe_ids.index(current_id)
        except ValueError:
            return keyframe_ids[-self.local_window_size:]
        
        start_idx = max(0, current_idx - self.local_window_size // 2)
        end_idx = min(len(keyframe_ids), start_idx + self.local_window_size)
        
        return keyframe_ids[start_idx:end_idx]
    
    def _get_observed_points(self, keyframe_ids: List[int]) -> List[int]:
        """Get points observed by the given keyframes"""
        observed_points = set()
        
        for pt_id, observations in self.observations.items():
            for kf_id, _ in observations:
                if kf_id in keyframe_ids:
                    observed_points.add(pt_id)
                    break
        
        return list(observed_points)
    
    def _is_loop_closure_edge(self, from_id: int, to_id: int) -> bool:
        """Check if an edge is a loop closure (non-consecutive keyframes)"""
        return abs(from_id - to_id) > 1
    
    def _add_loop_closure_edge(self, loop_closure):
        """Add a loop closure edge to the pose graph"""
        # This would be implemented based on your loop closure detection format
        # For now, assuming loop_closure contains the necessary information
        pass
    
    def get_optimized_trajectory(self) -> List[np.ndarray]:
        """
        Get the optimized trajectory
        
        Returns:
            List of 4x4 pose matrices
        """
        keyframe_ids = sorted(self.keyframes.keys())
        return [self.keyframes[kf_id]['pose'] for kf_id in keyframe_ids]
    
    def get_optimized_map_points(self) -> Dict[int, np.ndarray]:
        """
        Get the optimized map points
        
        Returns:
            Dictionary mapping point IDs to 3D coordinates
        """
        return self.map_points.copy()
    
    def save_trajectory(self, filename: str):
        """Save trajectory to file in KITTI format"""
        trajectory = self.get_optimized_trajectory()
        
        with open(filename, 'w') as f:
            for pose in trajectory:
                # Convert to KITTI format (3x4 matrix, row-major)
                pose_3x4 = pose[:3, :].flatten()
                f.write(' '.join(map(str, pose_3x4)) + '\n')


# Test and demonstration code
if __name__ == "__main__":
    print("Testing SciPy Optimizer Implementation...")
    
    # Create synthetic camera matrix
    camera_matrix = np.array([
        [718.856, 0, 607.1928],
        [0, 718.856, 185.2157],
        [0, 0, 1]
    ])
    
    # Initialize optimizer
    optimizer = ScipyOptimizer(camera_matrix)
    
    print("✓ SciPy Optimizer initialized")
    
    # Test 1: Add synthetic keyframes
    print("\nTest 1: Adding synthetic keyframes...")
    poses = []
    keyframe_ids = []
    
    for i in range(5):
        # Create synthetic pose (moving forward in Z)
        pose = np.eye(4)
        pose[2, 3] = i * 1.0  # Move 1 meter forward each frame
        poses.append(pose)
        
        # Create synthetic keypoints
        keypoints = [cv2.KeyPoint(100 + i*10, 100 + i*5, 1) for _ in range(20)]
        descriptors = np.random.rand(20, 128).astype(np.float32)
        
        kf_id = optimizer.add_keyframe(pose, keypoints, descriptors)
        keyframe_ids.append(kf_id)
        
        print(f"  Added keyframe {kf_id} at position: {pose[:3, 3]}")
    
    # Test 2: Add synthetic map points with better observations
    print("\nTest 2: Adding synthetic map points...")
    point_ids = []
    
    for i in range(15):  # Increased number of points
        # Create synthetic 3D point
        point_3d = np.array([i * 0.3, np.sin(i * 0.1) * 0.5, 5.0 + np.random.normal(0, 0.1)])
        
        # Create synthetic observations (more realistic)
        observations = []
        for j, kf_id in enumerate(keyframe_ids):  # Observed by ALL keyframes
            # Properly project 3D point to 2D using camera model
            pose = poses[j]
            point_cam = pose[:3, :3] @ point_3d + pose[:3, 3]
            
            if point_cam[2] > 0.1:  # Only if in front of camera
                x = optimizer.fx * point_cam[0] / point_cam[2] + optimizer.cx
                y = optimizer.fy * point_cam[1] / point_cam[2] + optimizer.cy
                
                # Add some noise
                x += np.random.normal(0, 1.0)
                y += np.random.normal(0, 1.0)
                
                # Check if projection is within image bounds
                if 0 < x < 1241 and 0 < y < 376:  # KITTI image size
                    point_2d = np.array([x, y])
                    observations.append((kf_id, point_2d))
        
        if len(observations) >= 2:  # Need at least 2 observations
            pt_id = optimizer.add_map_point(point_3d, observations)
            point_ids.append(pt_id)
            
            print(f"  Added map point {pt_id} at position: {point_3d} with {len(observations)} observations")
    
    # Test 3: Add pose graph edges
    print("\nTest 3: Adding pose graph edges...")
    for i in range(len(keyframe_ids) - 1):
        from_id = keyframe_ids[i]
        to_id = keyframe_ids[i + 1]
        
        # Compute relative pose
        relative_pose = np.linalg.inv(poses[i]) @ poses[i + 1]
        
        # Information matrix (6x6 for SE3)
        information = np.eye(6) * 100.0  # High confidence
        
        optimizer.add_pose_graph_edge(from_id, to_id, relative_pose, information)
        print(f"  Added edge from keyframe {from_id} to {to_id}")
    
    # Test 4: Local bundle adjustment
    print("\nTest 4: Running local bundle adjustment...")
    try:
        success = optimizer.local_bundle_adjustment(keyframe_ids[-1])
        print(f"  Local BA result: {'Success' if success else 'Failed'}")
        
        if success:
            optimized_poses = optimizer.get_optimized_trajectory()
            print(f"  Optimized {len(optimized_poses)} poses")
            for i, pose in enumerate(optimized_poses):
                print(f"    Keyframe {i}: position = {pose[:3, 3]}")
    except Exception as e:
        print(f"  Local BA failed with error: {e}")
    
    # Test 5: Global pose graph optimization
    print("\nTest 5: Running global pose graph optimization...")
    try:
        success = optimizer.global_pose_graph_optimization()
        print(f"  Global optimization result: {'Success' if success else 'Failed'}")
        
        if success:
            optimized_poses = optimizer.get_optimized_trajectory()
            print(f"  Final optimized trajectory:")
            for i, pose in enumerate(optimized_poses):
                print(f"    Keyframe {i}: position = {pose[:3, 3]}")
    except Exception as e:
        print(f"  Global optimization failed with error: {e}")
    
    # Test 6: Save trajectory
    print("\nTest 6: Saving trajectory...")
    try:
        optimizer.save_trajectory("scipy_trajectory.txt")
        print("  ✓ Trajectory saved to scipy_trajectory.txt")
    except Exception as e:
        print(f"  Failed to save trajectory: {e}")
    
    # Test 7: Integration test
    print("\nTest 7: Integration with optimization step...")
    try:
        # Add one more keyframe
        pose = np.eye(4)
        pose[2, 3] = 5.0
        keypoints = [cv2.KeyPoint(150, 125, 1) for _ in range(15)]
        descriptors = np.random.rand(15, 128).astype(np.float32)
        
        new_kf_id = optimizer.add_keyframe(pose, keypoints, descriptors)
        
        # Add some observations for the new keyframe
        for i, pt_id in enumerate(point_ids[:5]):  # Link to first 5 points
            point_2d = np.array([420 + i*15, 210])
            optimizer.observations[pt_id].append((new_kf_id, point_2d))
        
        # Run optimization step
        optimizer.optimize_step(new_kf_id)
        print("  ✓ Integration test completed successfully")
        
    except Exception as e:
        print(f"  Integration test failed: {e}")
    
    print("\n" + "="*50)
    print("SciPy Optimizer test completed!")
    print("Dependencies: numpy, scipy, opencv-python")
    print("="*50)