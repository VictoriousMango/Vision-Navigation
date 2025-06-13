import cv2
import numpy as np

class transformations:
    def __init__(self):
        self.m=1.0
        self.alpha=0.0 
        self.theta=0.0
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

class featureDetection:
    def __init__(self):
        """
        Feature That Can be used:
        | SIFT | FAST | ORB | BRISK | ASIFT |
        """
        self.SIFT_init = cv2.SIFT_create(nfeatures=500)
        self.FAST_init = cv2.FastFeatureDetector_create(threshold=30, nonmaxSuppression=True)
        self.ORB_init = cv2.ORB_create(nfeatures=25, scaleFactor=1.2, nlevels=8, edgeThreshold=31, firstLevel=0, WTA_K=2, scoreType=cv2.ORB_HARRIS_SCORE, patchSize=31)
        self.BRISK_init = cv2.BRISK_create(thresh=30)

    def FD_SIFT(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.SIFT_init.detectAndCompute(img, None)
        if keypoints is not None:
            frame = cv2.drawKeypoints(frame, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            return (keypoints, descriptors, frame)
        
    def FD_FAST(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints = self.FAST_init.detect(img, None)
        if keypoints is not None:
            frame = cv2.drawKeypoints(frame, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            return (keypoints, frame)
        
    def FD_ORB(self, frame, color=True):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if color else frame
        keypoints, descriptors = self.ORB_init.detectAndCompute(img, None)
        if keypoints is not None:
            frame = cv2.drawKeypoints(frame, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            return (keypoints, descriptors, frame)
        
    def FD_BRISK(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.BRISK_init.detectAndCompute(img, None)
        if keypoints is not None:
            frame = cv2.drawKeypoints(frame, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            return (keypoints, descriptors, frame)
    def FD_AffineORB(self, frame):
        return self.FD_ORB(transformations.T_Affine(self, frame), color=False)
        # return (keypoints, descriptors, cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))

class featureMatching:
    def __init__(self):
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    def FM_BruteForce(self, prev_kp, prev_des, curr_kp, curr_des, max_matches=50):
        # Match descriptors
        matches = self.matcher.match(prev_des, curr_des)
        # 
        matches = sorted(matches, key=lambda x: x.distance)[:max_matches]

        # Extract matched keypoints
        pts_prev = np.float32([prev_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts_curr = np.float32([curr_kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        return matches, pts_prev, pts_curr
    
class Pipeline(transformations, featureDetection, featureMatching):
    def __init__(self):
        transformations.__init__(self)
        featureDetection.__init__(self)
        featureMatching.__init__(self)
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.K = np.array([
                    [718.8560,   0.0000, 607.1928],
                    [  0.0000, 718.8560, 185.2157],
                    [  0.0000,   0.0000,   1.0000]
                ])
        self.trajectory = [ np . eye (4) [:3]]
        self.trajectory_path = []


    def VisualOdometry(self, frame, FeatureDetector=None, FeatureMatcher=None, Transformation1=None, Transformation2=None):
        matches, pts_prev, pts_curr = [], None, None
        essential_matrix, fundamental_matrix = None, None
        
        # Feature Detection
        FD = getattr(self, FeatureDetector, None)
        if FD is not None:
            keypoints, descriptors, frame = FD(frame)
            used_detector = FeatureDetector
        else:
            return frame, None, None, None, None

        # Feature Matching
        if self.prev_keypoints is not None and self.prev_descriptors is not None:
            FM = getattr(self, FeatureMatcher, None)
            if FM is not None:
                matches, pts_prev, pts_curr = FM(self.prev_keypoints, self.prev_descriptors, keypoints, descriptors)
                frame = cv2.drawMatches(frame, self.prev_keypoints, frame, keypoints, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        # Update previous keypoints and descriptors
        self.prev_keypoints = keypoints
        self.prev_descriptors = descriptors

        # Apply Affine Transform
        transform1 = getattr(self, Transformation1, None)
        if transform1 is not None:
            frame = transform1(frame)

        # Apply Normalization
        transform2 = getattr(self, Transformation2, None)
        if transform2 is not None:
            norm_img = transform2(frame)
        else:
            norm_img = frame

        if pts_prev is not None and len(matches) > 8:
            # Fundamental matrix
            F, mask_F = cv2.findFundamentalMat(pts_prev, pts_curr, cv2.FM_RANSAC)
            fundamental_matrix = F

            # Homography
            H, mask_H = cv2.findHomography(pts_prev, pts_curr, cv2.RANSAC, 5.0)
            inlier_ratio = np.sum(mask_H) / len(mask_H)
            matrix = H if inlier_ratio > 0.7 else F

            corrected_img = cv2.warpPerspective(norm_img, matrix, (frame.shape[1], frame.shape[0]))

            # Essential matrix and pose estimation
            E, mask_E = cv2.findEssentialMat(pts_prev, pts_curr, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
            essential_matrix = E

            if E is not None:
                _, R, t, _ = cv2.recoverPose(E, pts_prev, pts_curr, self.K)
                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3:] = t
                global_pose = self.trajectory[-1] @ np.linalg.inv(T)
                self.trajectory.append(global_pose)

                # Draw trajectory (x, z)
                x, z = int(global_pose[0, 3]*10 + 300), int(global_pose[2, 3]*10 + 300)
                
                self.trajectory_path.append((x, z))

        return frame, essential_matrix, fundamental_matrix, self.trajectory_path, used_detector



