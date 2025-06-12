import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from enum import Enum
from dataclasses import dataclass

class VOMethod(Enum):
    """Visual Odometry method types"""
    FEATURE_BASED = "feature_based"
    DIRECT = "direct"
    HYBRID = "hybrid"

class FeatureDetector(Enum):
    """Feature detector types"""
    FAST = "fast"
    HARRIS = "harris"
    GFTT = "gftt"
    SIFT = "sift"
    SURF = "surf"
    ORB = "orb"

@dataclass
class CameraPose:
    """Camera pose representation"""
    rotation: np.ndarray  # 3x3 rotation matrix
    translation: np.ndarray  # 3x1 translation vector
    timestamp: float

@dataclass
class MapPoint:
    """3D map point representation"""
    position: np.ndarray  # 3D coordinates
    descriptor: np.ndarray  # Feature descriptor
    observations: List[int]  # List of keyframe IDs that observe this point

class FrontEndModule:
    """
    Visual Odometry Front End Module for VSLAM
    Implements feature-based, direct, and hybrid methods
    """
    
    def __init__(self, 
                 camera_matrix: np.ndarray,
                 distortion_coeffs: np.ndarray,
                 method: VOMethod = VOMethod.FEATURE_BASED,
                 feature_detector: FeatureDetector = FeatureDetector.ORB):
        """
        Initialize the Front End Module
        
        Args:
            camera_matrix: 3x3 camera intrinsic matrix
            distortion_coeffs: Camera distortion coefficients
            method: Visual odometry method to use
            feature_detector: Feature detector algorithm
        """
        self.camera_matrix = camera_matrix
        self.distortion_coeffs = distortion_coeffs
        self.method = method
        self.feature_detector_type = feature_detector
        
        # Internal state
        self.previous_frame = None
        self.current_pose = CameraPose(np.eye(3), np.zeros((3, 1)), 0.0)
        self.keyframes = []
        self.map_points = []
        self.frame_id = 0
        self.initialized = False
        
        # Initialize feature detector and descriptor
        self._init_feature_detector()
        
        # Parameters
        self.min_features = 100
        self.keyframe_threshold = 30  # Minimum features to create keyframe
        self.max_reprojection_error = 2.0
        
    def _init_feature_detector(self):
        """Initialize feature detector and descriptor based on selected type"""
        if self.feature_detector_type == FeatureDetector.ORB:
            self.detector = cv2.ORB_create(nfeatures=1000)
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        elif self.feature_detector_type == FeatureDetector.SIFT:
            self.detector = cv2.SIFT_create()
            self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        elif self.feature_detector_type == FeatureDetector.FAST:
            self.detector = cv2.FastFeatureDetector_create()
            self.descriptor = cv2.ORB_create()  # Use ORB for description
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        elif self.feature_detector_type == FeatureDetector.HARRIS:
            self.detector = cv2.goodFeaturesToTrack
            self.descriptor = cv2.ORB_create()
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        else:
            # Default to ORB
            self.detector = cv2.ORB_create(nfeatures=1000)
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    def process_frame(self, image: np.ndarray, timestamp: float = 0.0) -> np.ndarray:
        """
        Main processing function - routes to appropriate method
        
        Args:
            image: Input image frame
            timestamp: Frame timestamp
            
        Returns:
            Processed image with visual odometry results
        """
        self.frame_id += 1
        
        if self.method == VOMethod.FEATURE_BASED:
            return self.feature_based_vo(image, timestamp)
        elif self.method == VOMethod.DIRECT:
            return self.direct_vo(image, timestamp)
        elif self.method == VOMethod.HYBRID:
            return self.hybrid_vo(image, timestamp)
        else:
            raise ValueError(f"Unknown VO method: {self.method}")
    
    def feature_based_vo(self, image: np.ndarray, timestamp: float) -> np.ndarray:
        """
        Feature-based Visual Odometry implementation
        
        Args:
            image: Input image frame
            timestamp: Frame timestamp
            
        Returns:
            Image with feature-based VO visualization
        """
        # Step 1: Image preprocessing
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Step 2: Feature detection
        keypoints, descriptors = self.detect_features(gray)
        
        # Visualization image
        result_image = image.copy() if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        if not self.initialized:
            # Initialize with first frame
            self.previous_frame = {
                'image': gray,
                'keypoints': keypoints,
                'descriptors': descriptors,
                'timestamp': timestamp
            }
            self.initialized = True
            
            # Draw initial features
            result_image = self.draw_features(result_image, keypoints, (0, 255, 0))
            return result_image
        
        # Step 3: Feature matching
        if descriptors is not None and self.previous_frame['descriptors'] is not None:
            matches = self.match_features(self.previous_frame['descriptors'], descriptors)
            
            if len(matches) > self.min_features:
                # Step 4: Camera pose estimation
                pose_estimated, inlier_matches = self.estimate_pose_2d2d(
                    self.previous_frame['keypoints'], keypoints, matches
                )
                
                if pose_estimated:
                    # Step 5: Local mapping (simplified)
                    self.update_local_map(keypoints, descriptors, inlier_matches)
                    
                    # Step 6: Keyframe management
                    if self.should_create_keyframe(len(inlier_matches)):
                        self.create_keyframe(gray, keypoints, descriptors, timestamp)
                
                # Visualization
                result_image = self.draw_matches(result_image, self.previous_frame['keypoints'], 
                                               keypoints, inlier_matches)
        
        # Update previous frame
        self.previous_frame = {
            'image': gray,
            'keypoints': keypoints,
            'descriptors': descriptors,
            'timestamp': timestamp
        }
        
        return self.add_pose_info(result_image)
    
    def direct_vo(self, image: np.ndarray, timestamp: float) -> np.ndarray:
        """
        Direct Visual Odometry implementation
        
        Args:
            image: Input image frame
            timestamp: Frame timestamp
            
        Returns:
            Image with direct VO visualization
        """
        # Step 1: Image preprocessing
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        result_image = image.copy() if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        if not self.initialized:
            self.previous_frame = {
                'image': gray,
                'timestamp': timestamp
            }
            self.initialized = True
            return result_image
        
        # Step 2: Depth estimation (simplified for monocular)
        depth_map = self.estimate_depth_direct(gray)
        
        # Step 3: Camera motion estimation using photometric error
        motion_estimated = self.estimate_motion_photometric(
            self.previous_frame['image'], gray
        )
        
        if motion_estimated:
            # Step 4: Local map reconstruction
            self.update_dense_map(gray, depth_map)
        
        # Visualization - show high gradient areas
        result_image = self.visualize_direct_method(result_image, gray, depth_map)
        
        self.previous_frame = {
            'image': gray,
            'timestamp': timestamp
        }
        
        return self.add_pose_info(result_image)
    
    def hybrid_vo(self, image: np.ndarray, timestamp: float) -> np.ndarray:
        """
        Hybrid Visual Odometry implementation (SVO-style)
        
        Args:
            image: Input image frame
            timestamp: Frame timestamp
            
        Returns:
            Image with hybrid VO visualization
        """
        # Step 1: Image preprocessing
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        result_image = image.copy() if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        if not self.initialized:
            # Use feature-based initialization
            keypoints, descriptors = self.detect_features(gray)
            self.previous_frame = {
                'image': gray,
                'keypoints': keypoints,
                'descriptors': descriptors,
                'timestamp': timestamp
            }
            self.initialized = True
            return self.draw_features(result_image, keypoints, (255, 0, 255))
        
        # Hybrid approach: Feature detection + direct tracking
        # Step 2: Sparse feature detection
        keypoints, descriptors = self.detect_features(gray)
        
        # Step 3: Direct tracking around keypoints
        tracked_points = self.track_points_direct(
            self.previous_frame['image'], gray, self.previous_frame['keypoints']
        )
        
        # Step 4: Combine feature matching and direct tracking
        if descriptors is not None and self.previous_frame['descriptors'] is not None:
            feature_matches = self.match_features(self.previous_frame['descriptors'], descriptors)
            
            # Pose estimation using both methods
            pose_estimated = self.estimate_pose_hybrid(
                tracked_points, feature_matches, keypoints
            )
            
            if pose_estimated:
                # Probabilistic depth filter update
                self.update_depth_filters(tracked_points)
        
        # Visualization
        result_image = self.visualize_hybrid_method(result_image, keypoints, tracked_points)
        
        self.previous_frame = {
            'image': gray,
            'keypoints': keypoints,
            'descriptors': descriptors,
            'timestamp': timestamp
        }
        
        return self.add_pose_info(result_image)
    
    def detect_features(self, image: np.ndarray) -> Tuple[List, np.ndarray]:
        """Detect and describe features in image"""
        if self.feature_detector_type in [FeatureDetector.ORB, FeatureDetector.SIFT]:
            keypoints, descriptors = self.detector.detectAndCompute(image, None)
        elif self.feature_detector_type == FeatureDetector.FAST:
            keypoints = self.detector.detect(image, None)
            keypoints, descriptors = self.descriptor.compute(image, keypoints)
        elif self.feature_detector_type == FeatureDetector.HARRIS:
            corners = cv2.goodFeaturesToTrack(image, maxCorners=1000, qualityLevel=0.01, minDistance=10)
            keypoints = [cv2.KeyPoint(x[0][0], x[0][1], 1) for x in corners] if corners is not None else []
            keypoints, descriptors = self.descriptor.compute(image, keypoints)
        else:
            keypoints, descriptors = self.detector.detectAndCompute(image, None)
        
        return keypoints if keypoints is not None else [], descriptors
    
    def match_features(self, desc1: np.ndarray, desc2: np.ndarray) -> List:
        """Match features between two descriptor sets"""
        if desc1 is None or desc2 is None:
            return []
        
        matches = self.matcher.match(desc1, desc2)
        matches = sorted(matches, key=lambda x: x.distance)
        return matches[:min(len(matches), 500)]  # Limit matches
    
    def estimate_pose_2d2d(self, kp1: List, kp2: List, matches: List) -> Tuple[bool, List]:
        """Estimate camera pose from 2D-2D correspondences"""
        if len(matches) < 8:
            return False, []
        
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # Find essential matrix
        E, mask = cv2.findEssentialMat(pts1, pts2, self.camera_matrix, method=cv2.RANSAC)
        
        if E is None:
            return False, []
        
        # Recover pose
        _, R, t, mask = cv2.recoverPose(E, pts1, pts2, self.camera_matrix)
        
        # Update current pose
        self.current_pose.rotation = R @ self.current_pose.rotation
        self.current_pose.translation = self.current_pose.translation + self.current_pose.rotation @ t
        
        # Filter inlier matches
        inlier_matches = [matches[i] for i in range(len(matches)) if mask[i]]
        
        return True, inlier_matches
    
    def estimate_depth_direct(self, image: np.ndarray) -> np.ndarray:
        """Estimate depth map for direct methods (simplified)"""
        # Simplified depth estimation using gradient magnitude
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Normalize and invert (higher gradient = closer depth)
        depth_map = 255 - cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return depth_map
    
    def estimate_motion_photometric(self, prev_image: np.ndarray, curr_image: np.ndarray) -> bool:
        """Estimate camera motion using photometric error minimization"""
        # Simplified implementation using optical flow
        if self.previous_frame is None:
            return False
        
        # Calculate optical flow for high-gradient pixels
        lk_params = dict(winSize=(15, 15), maxLevel=2)
        
        # Select good features to track
        corners = cv2.goodFeaturesToTrack(prev_image, maxCorners=200, qualityLevel=0.01, minDistance=10)
        
        if corners is not None:
            # Calculate optical flow
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_image, curr_image, corners, None, **lk_params)
            
            # Filter good points
            good_new = next_pts[status == 1]
            good_old = corners[status == 1]
            
            if len(good_new) > 10:
                # Simplified pose update based on flow
                return True
        
        return False
    
    def track_points_direct(self, prev_image: np.ndarray, curr_image: np.ndarray, keypoints: List) -> List:
        """Track keypoints using direct method"""
        if not keypoints:
            return []
        
        # Convert keypoints to points
        prev_pts = np.float32([kp.pt for kp in keypoints]).reshape(-1, 1, 2)
        
        # Lucas-Kanade optical flow
        lk_params = dict(winSize=(15, 15), maxLevel=2)
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_image, curr_image, prev_pts, None, **lk_params)
        
        # Return tracked points
        tracked = []
        for i, (st, pt) in enumerate(zip(status, next_pts)):
            if st == 1:
                tracked.append((prev_pts[i][0], pt[0]))
        
        return tracked
    
    def estimate_pose_hybrid(self, tracked_points: List, feature_matches: List, keypoints: List) -> bool:
        """Estimate pose using hybrid information"""
        # Combine information from direct tracking and feature matching
        total_correspondences = len(tracked_points) + len(feature_matches)
        
        if total_correspondences > 20:
            # Simplified pose estimation
            return True
        return False
    
    def update_local_map(self, keypoints: List, descriptors: np.ndarray, matches: List):
        """Update local map with new observations"""
        # Simplified local mapping
        if descriptors is not None:
            for i, kp in enumerate(keypoints):
                if i < len(descriptors):
                    # Create new map point (simplified)
                    map_point = MapPoint(
                        position=np.array([kp.pt[0], kp.pt[1], 1.0]),  # Simplified 3D position
                        descriptor=descriptors[i],
                        observations=[self.frame_id]
                    )
                    self.map_points.append(map_point)
    
    def update_dense_map(self, image: np.ndarray, depth_map: np.ndarray):
        """Update dense map representation"""
        # Simplified dense mapping
        pass
    
    def update_depth_filters(self, tracked_points: List):
        """Update probabilistic depth filters"""
        # Simplified depth filter update
        pass
    
    def should_create_keyframe(self, num_matches: int) -> bool:
        """Determine if a new keyframe should be created"""
        # Simple keyframe creation criterion
        return num_matches < self.keyframe_threshold or len(self.keyframes) == 0
    
    def create_keyframe(self, image: np.ndarray, keypoints: List, descriptors: np.ndarray, timestamp: float):
        """Create a new keyframe"""
        keyframe = {
            'id': len(self.keyframes),
            'image': image.copy(),
            'keypoints': keypoints,
            'descriptors': descriptors,
            'pose': CameraPose(self.current_pose.rotation.copy(), 
                             self.current_pose.translation.copy(), timestamp),
            'timestamp': timestamp
        }
        self.keyframes.append(keyframe)
    
    # Visualization methods
    def draw_features(self, image: np.ndarray, keypoints: List, color: Tuple[int, int, int]) -> np.ndarray:
        """Draw detected features on image"""
        result = image.copy()
        for kp in keypoints:
            pt = (int(kp.pt[0]), int(kp.pt[1]))
            cv2.circle(result, pt, 3, color, -1)
        return result
    
    def draw_matches(self, image: np.ndarray, kp1: List, kp2: List, matches: List) -> np.ndarray:
        """Draw feature matches on image"""
        result = image.copy()
        
        # Draw current keypoints
        for kp in kp2:
            pt = (int(kp.pt[0]), int(kp.pt[1]))
            cv2.circle(result, pt, 3, (0, 255, 0), -1)
        
        # Draw matches as lines
        for match in matches[:50]:  # Limit visualization
            pt1 = kp1[match.queryIdx].pt
            pt2 = kp2[match.trainIdx].pt
            cv2.line(result, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), (0, 255, 255), 1)
        
        return result
    
    def visualize_direct_method(self, image: np.ndarray, gray: np.ndarray, depth_map: np.ndarray) -> np.ndarray:
        """Visualize direct method results"""
        result = image.copy()
        
        # Overlay depth map with transparency
        depth_colored = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)
        result = cv2.addWeighted(result, 0.7, depth_colored, 0.3, 0)
        
        return result
    
    def visualize_hybrid_method(self, image: np.ndarray, keypoints: List, tracked_points: List) -> np.ndarray:
        """Visualize hybrid method results"""
        result = image.copy()
        
        # Draw keypoints
        for kp in keypoints:
            pt = (int(kp.pt[0]), int(kp.pt[1]))
            cv2.circle(result, pt, 3, (255, 0, 255), -1)
        
        # Draw tracked points
        for prev_pt, curr_pt in tracked_points:
            cv2.line(result, (int(prev_pt[0]), int(prev_pt[1])), 
                    (int(curr_pt[0]), int(curr_pt[1])), (0, 255, 255), 2)
        
        return result
    
    def add_pose_info(self, image: np.ndarray) -> np.ndarray:
        """Add pose information to image"""
        result = image.copy()
        
        # Add text overlay with pose information
        pose_text = f"Frame: {self.frame_id}, Method: {self.method.value}"
        cv2.putText(result, pose_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        translation_text = f"T: [{self.current_pose.translation[0][0]:.2f}, {self.current_pose.translation[1][0]:.2f}, {self.current_pose.translation[2][0]:.2f}]"
        cv2.putText(result, translation_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        keyframes_text = f"Keyframes: {len(self.keyframes)}, Map Points: {len(self.map_points)}"
        cv2.putText(result, keyframes_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return result
    
    def get_current_pose(self) -> CameraPose:
        """Get current camera pose"""
        return self.current_pose
    
    def get_map_points(self) -> List[MapPoint]:
        """Get current map points"""
        return self.map_points
    
    def get_keyframes(self) -> List[Dict]:
        """Get current keyframes"""
        return self.keyframes
    
    def reset(self):
        """Reset the front end module"""
        self.previous_frame = None
        self.current_pose = CameraPose(np.eye(3), np.zeros((3, 1)), 0.0)
        self.keyframes = []
        self.map_points = []
        self.frame_id = 0
        self.initialized = False