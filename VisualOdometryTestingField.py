import numpy as np
import cv2
import os

class VisualOdometry:
    def __init__(self, data_path):
        self.images = self.load_images(data_path)
        self.detector = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        self.K = np.array([[718.856, 0, 607.1928],
                          [0, 718.856, 185.2157],
                          [0, 0, 1]])  # Camera intrinsics
        self.pose = np.eye(4)
        
    def load_images(self, path):
        images = []
        for filename in sorted(os.listdir(path)):
            if filename.endswith(".png"):
                img = cv2.imread(os.path.join(path, filename), 0)
                images.append(img)
        return images

    def get_matches(self, img1, img2):
        kp1, des1 = self.detector.detectAndCompute(img1, None)
        kp2, des2 = self.detector.detectAndCompute(img2, None)
        matches = self.matcher.match(des1, des2)
        return kp1, kp2, matches

    def estimate_motion(self, matches, kp1, kp2):
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
        
        E, mask = cv2.findEssentialMat(src_pts, dst_pts, self.K, cv2.RANSAC, 0.999, 1.0)
        _, R, t, _ = cv2.recoverPose(E, src_pts, dst_pts, self.K, mask=mask)
        return R, t

    def run(self):
        trajectory = []
        for i in range(1, len(self.images)):
            kp1, kp2, matches = self.get_matches(self.images[i-1], self.images[i])
            R, t = self.estimate_motion(matches, kp1, kp2)
            
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t.squeeze()
            
            self.pose = self.pose @ np.linalg.inv(T)
            trajectory.append(self.pose[:3, 3])
            
        return np.array(trajectory)

# Usage example:
vo = VisualOdometry("KITTI_sequence_1\\image_l\\")
trajectory = vo.run()
# Plotting the trajectory can be done using matplotlib or any other visualization library.
# Note: Replace "path/to/image_sequence/" with the actual path to your image sequence.
# The trajectory can be visualized using matplotlib or similar libraries.
# Example of plotting the trajectory
import matplotlib.pyplot as plt
def plot_trajectory(trajectory):
    plt.figure(figsize=(10, 10))
    plt.plot(trajectory[:, 0], trajectory[:, 2], marker='o', markersize=1, linestyle='-', color='b')
    plt.title("Camera Trajectory")
    plt.xlabel("X (m)")
    plt.ylabel("Z (m)")
    plt.axis('equal')
    plt.grid()
    plt.show()
plot_trajectory(trajectory)
# The above code defines a VisualOdometry class that processes a sequence of images to estimate the camera trajectory using SIFT feature detection and matching, essential matrix estimation, and pose recovery. The trajectory is then plotted using matplotlib.
# The code is designed to be run in a Python environment with OpenCV and matplotlib installed.
