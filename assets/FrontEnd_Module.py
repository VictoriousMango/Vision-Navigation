import cv2
import numpy as np
from io import StringIO
import random
from assets.Mapping import LocalMapper, MapPoint, KeyFrame
from assets.LoopDetection import BOVW

class transformations:
    def __init__(self):
        self.m = random.uniform(0.97, 1.03)
        self.alpha = np.deg2rad(random.uniform(-3, 3))
        self.theta = np.deg2rad(random.uniform(-3, 3))
        pass
    
    def T_Affine(self, frame):
        rows, cols = frame.shape[:2]
        A = np.float32([
            [self.m * np.cos(self.theta), -self.m * np.sin(self.theta) * np.cos(self.alpha), 0],
            [self.m * np.sin(self.theta) * np.sin(self.alpha), self.m * np.cos(self.alpha), 0]
        ])
        return cv2.warpAffine(frame, A, (cols, rows))
    
    def T_Normalize(self, frame):
        return cv2.normalize(frame, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

class CameraMatrices:
    def __init__(self):
        pass
    
    def Homography(self):
        pass
    
    def EstimatePose(self, pts_prev, pts_curr, K):
        # Ensure we have enough points
        if len(pts_prev) < 8 or len(pts_curr) < 8:
            return None, None, None
            
        # Compute Fundamental Matrix with stricter parameters
        F, mask = cv2.findFundamentalMat(
            pts_prev, pts_curr, 
            cv2.FM_RANSAC,
            ransacReprojThreshold=1.0,  # Stricter threshold
            confidence=0.99             # Higher confidence
        )
        
        if F is None or F.shape != (3, 3):
            return None, None, None, None, None
            
        if mask is None or len(mask) == 0:
            return None, None, None, None, None

        # Filter inliers
        prevPt_inliers = pts_prev[mask.ravel() == 1]
        currPt_inliers = pts_curr[mask.ravel() == 1]
        
        # Need at least 8 points for reliable pose estimation
        if len(prevPt_inliers) < 8 or len(currPt_inliers) < 8:
            return None, None, None, None, None
        
        # Compute Essential Matrix
        E, mask = cv2.findEssentialMat(prevPt_inliers, currPt_inliers, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        
        
        # Recover pose
        try:
            _, R, t, pose_mask = cv2.recoverPose(E, prevPt_inliers, currPt_inliers, K)
            return R, t, mask, E, F
        except cv2.error as e:
            print(f"Error in recoverPose: {e}")
            return None, None, None, None, None
    
    def triangulate_points(self, pose1, pose2, pts1, pts2, K):
        P1 = np.dot(K, pose1[:3, :])
        P2 = np.dot(K, pose2[:3, :])
        
        points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        points_3d = points_4d[:3, :] / points_4d[3, :]
        
        return points_3d.T

class featureDetection:
    def __init__(self):
        # Improved SIFT parameters for better feature quality
        self.SIFT_init = cv2.SIFT_create(
            nfeatures=2000,           # More features for better matching
            nOctaveLayers=3,
            contrastThreshold=0.03,   # Lower threshold for more features
            edgeThreshold=15,         # Higher edge threshold for stability
            sigma=1.6
        )
        self.FAST_init = cv2.FastFeatureDetector_create(threshold=20, nonmaxSuppression=True)
        self.ORB_init = cv2.ORB_create(
            nfeatures=2000,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=7,
            patchSize=7,
            scoreType=cv2.ORB_HARRIS_SCORE
        )
        self.BRISK_init = cv2.BRISK_create()

    def FD_SIFT(self, frame):
        if len(frame.shape) > 2:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization for better feature detection
        img = cv2.equalizeHist(img)
        
        keypoints, descriptors = self.SIFT_init.detectAndCompute(img, None)
        if descriptors is None:
            print("⚠️ SIFT returned None descriptors")
            return None, None, frame
        if keypoints is not None:
            frame = cv2.drawKeypoints(frame, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            return (keypoints, descriptors, frame)
        return None, None, frame
        
    def FD_AffineSIFT(self, frame):
        # Skip affine transformation unless specifically needed
        transform = transformations()
        frame = transform.T_Affine(frame)
        return self.FD_SIFT(frame)
        
    def FD_ORB(self, frame, color=True):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if color else frame
        img = cv2.equalizeHist(img)  # Improve contrast
        
        keypoints, descriptors = self.ORB_init.detectAndCompute(img, None)
        if keypoints is not None:
            frame = cv2.drawKeypoints(frame, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            return (keypoints, descriptors, frame)
        return None, None, frame
        
    def FD_BRISK(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img = cv2.equalizeHist(img)
        
        keypoints, descriptors = self.BRISK_init.detectAndCompute(img, None)
        if keypoints is not None:
            frame = cv2.drawKeypoints(frame, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            return (keypoints, descriptors, frame)
        return None, None, frame
    
    def FD_AffineORB(self, frame):
        transform = transformations()
        affine_frame = transform.T_Affine(frame)
        return self.FD_ORB(affine_frame)

class featureMatching:
    def __init__(self):
        self.matcher = None
    
    def FM_BF_NORM_Hamming(self):
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    def FM_BF_NORM_L2(self):
        self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    
    def BruteForce(self, prev_kp, prev_des, curr_kp, curr_des, max_matches=100):
        if prev_des is None or curr_des is None:
            return [], None, None
        
        if self.matcher is None:
            print("⚠️ Matcher not initialized")
            return [], None, None
            
        # Use knnMatch for ratio test
        matches = self.matcher.knnMatch(prev_des, curr_des, k=2)
        
        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.7 * n.distance:  # Ratio test threshold
                    good_matches.append(m)
        
        # Sort by distance and limit matches
        good_matches = sorted(good_matches, key=lambda x: x.distance)[:max_matches]
        
        if len(good_matches) < 8:
            return [], None, None
        
        # Extract matched keypoints
        pts_prev = np.float32([prev_kp[m.queryIdx].pt for m in good_matches])
        pts_curr = np.float32([curr_kp[m.trainIdx].pt for m in good_matches])

        return good_matches, pts_prev, pts_curr

class Pipeline(transformations, featureDetection, featureMatching, CameraMatrices, LocalMapper):
    def __init__(self):
        transformations.__init__(self)
        featureDetection.__init__(self)
        featureMatching.__init__(self)
        CameraMatrices.__init__(self)
        LocalMapper.__init__(self)
        self.loop_detector = BOVW()
        # LocalBundleAdjustment.__init__(self)
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.prev_frame = None
        self.prev_pose = np.eye(4)
        self.K = None
        self.current_pose = np.eye(4)  # Current absolute pose
        self.trajectory = [np.eye(4)[:3]]
        self.trajectory_path = []
        self.scale_factor = 1.0  # For scale correction
        self.frameID = 0  # Frame ID for tracking
        # Motion filtering parameters
        self.max_translation = 5.0  # Maximum translation per frame
        self.max_rotation = 0.5     # Maximum rotation per frame (radians)
    def reset(self):
        """Reset the pipeline state"""
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.prev_frame = None
        self.prev_pose = np.eye(4)
        self.trajectory = [np.eye(4)[:3]]
        self.trajectory_path = []
    def set_k_intrinsic(self, k_intrinsic):
        self.K = k_intrinsic
        self.camera_matrix = k_intrinsic
    
    def get_k_intrinsic(self):
        return self.K
    
    def validate_motion(self, R, t):
        """Validate if the estimated motion is reasonable"""
        # Check translation magnitude
        translation_norm = np.linalg.norm(t)
        if translation_norm > self.max_translation:
            return False
            
        # Check rotation magnitude
        rotation_angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))
        if rotation_angle > self.max_rotation:
            return False
            
        return True
    
    def estimate_scale(self, pts_prev, pts_curr, R, t):
        """Estimate scale using triangulation (simplified approach)"""
        # This is a simplified scale estimation
        # In practice, you might use more sophisticated methods
        if len(pts_prev) < 10:
            return 1.0
            
        # Use median depth change as scale estimate
        depths_prev = np.array([self.K[0, 0] / max(pt[0], 1e-6) for pt in pts_prev])
        depths_curr = np.array([self.K[0, 0] / max(pt[0], 1e-6) for pt in pts_curr])
        
        scale = np.median(depths_prev / np.clip(depths_curr, 1e-6, None))
        return np.clip(scale, 0.1, 10.0)  # Reasonable scale bounds
    
    # Correcting pose on the basis of loop detection
    def correct_trajectory_and_map(self, from_idx, to_idx):
        try:
            pose_i = self.keyframes[from_idx].pose
            pose_j = self.keyframes[to_idx].pose
            T_correction = pose_i @ np.linalg.inv(pose_j)

            # Update keyframe poses
            for k in range(to_idx, len(self.keyframes)):
                self.keyframes[k].pose = T_correction @ self.keyframes[k].pose

            # Update trajectory
            for k in range(to_idx, len(self.trajectory)):
                T = np.eye(4)
                T[:3] = self.trajectory[k]
                T_corr = T_correction @ T
                self.trajectory[k] = T_corr[:3]

            # Update trajectory path for plotting
            self.trajectory_path[to_idx:] = [
                (-kf.pose[0, 3], kf.pose[1, 3], kf.pose[2, 3])
                for kf in self.keyframes[to_idx:]
            ]

            self.current_pose = T_correction @ self.current_pose
            self.prev_pose = self.current_pose.copy()
            
            # Optional: Transform map points
            for i in range(to_idx, len(self.map_points)):
                if self.map_points[i] is not None:
                    self.map_points[i] = (T_correction[:3, :3] @ self.map_points[i].T).T + T_correction[:3, 3]
        except Exception as e:
            print(f"Error correcting trajectory and map: {e}")


    def VisualOdometry(self, frame, FeatureDetector=None, FeatureMatcher=None): 
        self.frameID += 1
        essential_matrix = fundamental_matrix = None
        pts_prev = pts_curr = matches = None

        # Feature Detection
        feature_detector = getattr(self, FeatureDetector, None)
        if feature_detector is None:
            print(f"⚠️ Feature detector '{FeatureDetector}' not found")
            return frame, None, None, None, FeatureDetector
        
        result = feature_detector(frame)
        if result is None or result[0] is None:
            print("⚠️ Feature detection failed")
            return frame, None, None, None, FeatureDetector
            
        keypoints, descriptors, processed_frame_fd = result

        # Loop Closure
        if descriptors is not None and len(descriptors) > 0:
            visual_word = self.loop_detector.Histogram(descriptors)
            self.loop_detector.historyOfBOVW(visual_word, descriptors, keypoints)
    
            loop_closures = self.loop_detector.LoopChecks()
            if loop_closures:
                print(f"🔁 Loop Detected between frames {loop_closures[-1][0]} and {loop_closures[-1][2]}")
                from_idx = loop_closures[-1][0]   # Earlier matching frame
                to_idx = loop_closures[-1][2]     # Current frame
                self.correct_trajectory_and_map(from_idx, to_idx)

        # Feature Matching
        if (self.prev_keypoints is not None and 
            self.prev_descriptors is not None and 
            descriptors is not None and
            len(keypoints) > 0):
            
            # Initialize matcher
            if self.matcher is None:
                if FeatureMatcher == 'FM_BF_NORM_Hamming':
                    self.FM_BF_NORM_Hamming()
                elif FeatureMatcher == 'FM_BF_NORM_L2':
                    self.FM_BF_NORM_L2()
                else:
                    print(f"⚠️ Unknown feature matcher: {FeatureMatcher}")
                    return processed_frame_fd, None, None, None, FeatureDetector
            
            matches, pts_prev, pts_curr = self.BruteForce(
                self.prev_keypoints, self.prev_descriptors, 
                keypoints, descriptors
            )
            processed_frame = None
            if len(matches) > 0:
                processed_frame = cv2.drawMatches(
                    self.prev_frame if self.prev_frame is not None else processed_frame_fd, self.prev_keypoints,
                    processed_frame_fd, keypoints,
                    matches[:20], None,
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
                    matchesThickness=2
                )
        else:
            processed_frame = processed_frame_fd


        # Update previous frame info
        triangulatedPoints = self.triangulate_points(
            self.current_pose, self.prev_pose, 
            pts_curr, pts_prev, self.K
        ) if pts_prev is not None and pts_curr is not None else None
        self.prev_keypoints = keypoints
        self.prev_descriptors = descriptors
        self.prev_frame = processed_frame_fd
        # Skip unnecessary transformations for better performance
        # processed_frame = self.T_Affine(processed_frame)
        # processed_frame = self.T_Normalize(processed_frame)

        # Pose Estimation
        if pts_prev is not None and pts_curr is not None and len(pts_prev) >= 8:
            pose_result = self.EstimatePose(pts_prev, pts_curr, self.K)
            
            if pose_result[0] is not None:
                R, t, mask, essential_matrix, fundamental_matrix = pose_result
                
                # Validate motion
                if self.validate_motion(R, t):
                    # Estimate and apply scale
                    scale = self.estimate_scale(pts_prev, pts_curr, R, t)
                    t_scaled = t * scale
                    t_scaled *= -1
                    
                    # Create transformation matrix
                    T_relative = np.eye(4)
                    T_relative[:3, :3] = R
                    T_relative[:3, 3] = t_scaled.squeeze()
                    
                    # FIXED: Correct pose composition
                    self.current_pose = self.current_pose @ T_relative
                    
                    # Added New => Storing KeyFrames for Surrounding Mapping
                    kf = KeyFrame(
                        pose=self.current_pose.copy(),
                        features=keypoints,
                        descriptors=descriptors,
                        frame_id=self.frameID
                    )
                    self.keyframes.append(kf)
                    self.map_points.append(triangulatedPoints)
                    # Store trajectory
                    self.trajectory.append(self.current_pose[:3])
                    
                    # Extract position for path
                    x, y, z = -self.current_pose[0, 3], self.current_pose[1, 3], self.current_pose[2, 3]
                    self.trajectory_path.append((x, y, z))
                else:
                    print("⚠️ Motion validation failed - skipping frame")
        # LBA
        # if self.frameID % self.keyframe_interval == 0:
        #     # Perform local bundle adjustment
        #     self.optimize_local_map(self.keyframes, self.map_points)
        self.prev_pose = self.current_pose.copy()  # Update previous pose for next frame
        return processed_frame, essential_matrix, fundamental_matrix, self.trajectory_path, FeatureDetector, self.map_points, self.prev_descriptors, self.prev_keypoints

class KITTIDataset:
    def __init__(self):
        pass
    
    @staticmethod
    def load_calib(file):
        if hasattr(file, 'readline'):
            line = file.readline()
        else:
            line = str(file).split('\n')[0]
            
        params = np.fromstring(line, dtype=np.float64, sep=' ')
        P = np.reshape(params, (3, 4))
        K = P[0:3, 0:3]
        return K, P

    @staticmethod
    def load_poses(file, batch_size):
        poses = []
        
        if hasattr(file, 'readlines'):
            lines = file.readlines()
        else:
            lines = str(file).split('\n')
            
        for line in lines:
            if line.strip():
                T = np.fromstring(line, dtype=np.float64, sep=' ')
                if len(T) == 12:
                    T = T.reshape(3, 4)
                    T = np.vstack((T, [0, 0, 0, 1]))
                    poses.append(T)
        return np.array(poses[:+batch_size])