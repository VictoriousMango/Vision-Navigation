import cv2
import numpy as np
from io import StringIO

class transformations:
    def __init__(self):
        self.m = 1.0
        self.alpha = 0.0 
        self.theta = 0.0
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
    def __init__(self):  # Fixed typo: 'seld' -> 'self'
        pass
    
    def Homography(self):
        pass
    
    def EstimatePose(self, pts_prev, pts_curr, K):
        # Ensure we have enough points
        if len(pts_prev) < 8 or len(pts_curr) < 8:
            return None, None, None
            
        # Compute Fundamental Matrix
        F, mask = cv2.findFundamentalMat(pts_prev, pts_curr, cv2.FM_RANSAC)
        
        # Check if F is valid and has the right shape
        if F is None or F.shape != (3, 3):
            return None, None, None
            
        # Check if mask is valid
        if mask is None or len(mask) == 0:
            return None, None, None
            
        prevPt_inliers = pts_prev[mask.ravel() == 1]
        currPt_inliers = pts_curr[mask.ravel() == 1]
        
        # Check if we have enough inliers
        if len(prevPt_inliers) < 5 or len(currPt_inliers) < 5:
            return None, None, None
        
        # Compute Essential Matrix
        E = K.T @ F @ K
        
        # Recover pose
        try:
            _, R, t, pose_mask = cv2.recoverPose(E, prevPt_inliers, currPt_inliers, K)
            return R, t, mask
        except cv2.error as e:
            print(f"Error in recoverPose: {e}")
            return None, None, None
    
    def triangulate_points(self, pose1, pose2, pts1, pts2, K):
        # Create projection matrices
        P1 = np.dot(K, pose1[:3, :])
        P2 = np.dot(K, pose2[:3, :])
        
        # Triangulate points
        points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        points_3d = points_4d[:3, :] / points_4d[3, :]
        
        return points_3d.T

class featureDetection:
    def __init__(self):
        """
        Feature That Can be used:
        | SIFT | AORB | ORB | BRISK | ASIFT |
        """
        self.SIFT_init = cv2.SIFT_create(nfeatures=1000,      # Maximum number of features
                            nOctaveLayers=3,     # Layers per octave
                            contrastThreshold=0.04,  # Contrast threshold
                            edgeThreshold=10,    # Edge threshold
                            sigma=1.6           # Gaussian blur sigma
                            )
        self.FAST_init = cv2.FastFeatureDetector_create(threshold=30, nonmaxSuppression=True)
        self.ORB_init = cv2.ORB_create(nfeatures=1000,      # Maximum features to retain
                            scaleFactor=1.2,     # Pyramid decimation ratio
                            nlevels=8,           # Number of pyramid levels
                            edgeThreshold=7,     # Border size for feature detection
                            patchSize=7,         # Patch size for oriented BRIEF
                            scoreType=cv2.ORB_HARRIS_SCORE  # Feature scoring method
                            )
        self.BRISK_init = cv2.BRISK_create()

    def FD_SIFT(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.SIFT_init.detectAndCompute(img, None)
        if descriptors is None:
            print("⚠️ SIFT returned None descriptors")
            return None, None, frame  # Return consistent format
        if keypoints is not None:
            frame = cv2.drawKeypoints(frame, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            return (keypoints, descriptors, frame)
        return None, None, frame  # Return consistent format
        
    def FD_AffineSIFT(self, frame):
        # Fixed: Create transformations instance properly
        transform = transformations()
        affine_frame = transform.T_Affine(frame)
        return self.FD_SIFT(affine_frame)
        
    def FD_ORB(self, frame, color=True):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if color else frame
        keypoints, descriptors = self.ORB_init.detectAndCompute(img, None)
        if keypoints is not None:
            frame = cv2.drawKeypoints(frame, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            return (keypoints, descriptors, frame)
        return None, None, frame  # Return consistent format
        
    def FD_BRISK(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.BRISK_init.detectAndCompute(img, None)
        if keypoints is not None:
            frame = cv2.drawKeypoints(frame, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            return (keypoints, descriptors, frame)
        return None, None, frame  # Return consistent format
    
    def FD_AffineORB(self, frame):
        # Fixed: Create transformations instance properly
        transform = transformations()
        affine_frame = transform.T_Affine(frame)
        return self.FD_ORB(affine_frame, color=False)

class featureMatching:
    def __init__(self):
        self.matcher = None
    
    def FM_BF_NORM_Hamming(self):
        """
        Initializes the Brute Force Matcher with Hamming distance.
        """
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    def FM_BF_NORM_L2(self):
        """
        Initializes the Brute Force Matcher with L2 distance.
        """
        self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    
    def BruteForce(self, prev_kp, prev_des, curr_kp, curr_des, max_matches=50):
        if prev_des is None or curr_des is None:
            return [], None, None  # Prevent matcher crash
        
        # Check if matcher is initialized
        if self.matcher is None:
            print("⚠️ Matcher not initialized. Call FM_BF_NORM_Hamming() or FM_BF_NORM_L2() first.")
            return [], None, None
            
        # Match descriptors
        matches = self.matcher.match(prev_des, curr_des)
        good_matches = sorted(matches, key=lambda x: x.distance)[:max_matches]
        
        # Extract matched keypoints
        pts_prev = np.float32([prev_kp[m.queryIdx].pt for m in good_matches])
        pts_curr = np.float32([curr_kp[m.trainIdx].pt for m in good_matches])

        return good_matches, pts_prev, pts_curr  # Return good_matches instead of all matches
    
class Pipeline(transformations, featureDetection, featureMatching, CameraMatrices):
    def __init__(self):
        transformations.__init__(self)
        featureDetection.__init__(self)
        featureMatching.__init__(self)
        CameraMatrices.__init__(self)  # Added missing initialization
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.prev_frame = None
        self.K = np.array([
                    [718.8560,   0.0000, 607.1928],
                    [  0.0000, 718.8560, 185.2157],
                    [  0.0000,   0.0000,   1.0000]
                ])
        self.trajectory = [np.eye(4)[:3]]  # Fixed spacing
        self.trajectory_path = []  # Initialize trajectory_path
    
    def set_k_intrinsic(self, k_intrinsic):
        """
        Sets the intrinsic camera matrix K.

        Parameters
        ----------
        k_intrinsic (ndarray): The intrinsic camera matrix.
        """
        self.K = k_intrinsic
    
    def get_k_intrinsic(self):
        """
        Returns the intrinsic camera matrix K.

        Returns
        -------
        k_intrinsic (ndarray): The intrinsic camera matrix.
        """
        return self.K
    
    def VisualOdometry(self, frame, FeatureDetector=None, FeatureMatcher=None):
        # Initialize
        essential_matrix = fundamental_matrix = None
        pts_prev = pts_curr = matches = None

        # --- Feature Detection ---
        feature_detector = getattr(self, FeatureDetector, None)
        if feature_detector is None:
            print(f"⚠️ Feature detector '{FeatureDetector}' not found")
            return frame, None, None, None, FeatureDetector
        
        result = feature_detector(frame)
        if result is None or result[0] is None:
            print("⚠️ Feature detection failed")
            return frame, None, None, None, FeatureDetector
            
        keypoints, descriptors, processed_frame = result

        # --- Feature Matching ---
        if (self.prev_keypoints is not None and 
            self.prev_descriptors is not None and 
            descriptors is not None and
            len(keypoints) > 0):
            
            # Initialize matcher if not already done
            if self.matcher is None:
                if FeatureMatcher == 'FM_BF_NORM_Hamming':
                    self.FM_BF_NORM_Hamming()
                elif FeatureMatcher == 'FM_BF_NORM_L2':
                    self.FM_BF_NORM_L2()
                else:
                    print(f"⚠️ Unknown feature matcher: {FeatureMatcher}")
                    return processed_frame, None, None, None, FeatureDetector
            
            matches, pts_prev, pts_curr = self.BruteForce(
                self.prev_keypoints, self.prev_descriptors, 
                keypoints, descriptors
            )
            
            if len(matches) > 0:
                processed_frame = cv2.drawMatches(
                    processed_frame, self.prev_keypoints,
                    processed_frame, keypoints,
                    matches[:10], None,
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
                    matchesThickness=1
                )

        # Update previous frame info
        self.prev_keypoints = keypoints
        self.prev_descriptors = descriptors
        self.prev_frame = processed_frame.copy()  # Make a copy to avoid reference issues

        # --- Transformations ---
        processed_frame = self.T_Affine(processed_frame)
        processed_frame = self.T_Normalize(processed_frame)

        # --- Compute Motion if enough matches ---
        if pts_prev is not None and pts_curr is not None and len(pts_prev) > 8:
            pose_result = self.EstimatePose(pts_prev, pts_curr, self.K)
            
            if pose_result[0] is not None:  # Check if pose estimation succeeded
                R, t, mask = pose_result
                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3] = t.squeeze()
                
                # Fixed trajectory computation
                if len(self.trajectory) > 0:
                    global_pose = self.trajectory[-1] @ np.linalg.inv(T)
                else:
                    global_pose = np.eye(4)[:3]
                    
                self.trajectory.append(global_pose)

                # Save trajectory coordinates (x, z)
                x, z = global_pose[0, 3], global_pose[2, 3]
                self.trajectory_path.append((x, z))

        # Return final frame and matrices - modified to match Streamlit expectations
        return processed_frame, essential_matrix, fundamental_matrix, self.trajectory_path, FeatureDetector

class KITTIDataset:
    def __init__(self):
        """
        Initializes the KITTI dataset loader

        Parameters
        ----------
        dataset_path (str): The path to the KITTI dataset
        """
        pass
    
    @staticmethod
    def load_calib(file):
        """
        Loads the calibration of the camera
        Parameters
        ----------
        file: File object or string content

        Returns
        -------
        K (ndarray): Intrinsic parameters
        P (ndarray): Projection matrix
        """
        # Handle both file objects and string content
        if hasattr(file, 'readline'):
            line = file.readline()
        else:
            line = str(file).split('\n')[0]
            
        params = np.fromstring(line, dtype=np.float64, sep=' ')
        P = np.reshape(params, (3, 4))
        K = P[0:3, 0:3]
        return K, P

    @staticmethod
    def load_poses(file):
        """
        Loads the GT poses

        Parameters
        ----------
        file: File object or string content

        Returns
        -------
        poses (ndarray): The GT poses
        """
        poses = []
        
        # Handle both file objects and string content
        if hasattr(file, 'readlines'):
            lines = file.readlines()
        else:
            lines = str(file).split('\n')
            
        for line in lines:
            if line.strip():  # Skip empty lines
                T = np.fromstring(line, dtype=np.float64, sep=' ')
                if len(T) == 12:  # Ensure we have 12 values
                    T = T.reshape(3, 4)
                    T = np.vstack((T, [0, 0, 0, 1]))
                    poses.append(T)
        return np.array(poses)